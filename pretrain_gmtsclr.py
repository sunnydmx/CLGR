import torch
import model
import dataset
import loss
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.fft import rfft,rfftfreq
import argparse
import torch.nn.functional as F


def wgn(sequence, snr):
    Ps = torch.sum(abs(sequence)**2)/len(sequence)
    Pn = Ps/(10**((snr/10)))
    noise = torch.randn(len(sequence)) * torch.sqrt(Pn)
    signal_add_noise = sequence + noise
    return signal_add_noise.unsqueeze(0)

def wgn_multi(sequences, snr):
    sequences = sequences.T
    signals = torch.zeros(1)
    for seq in sequences:
        signal_add_noise = wgn(seq, snr)
        if signals.shape[-1] == 1:
            signals = signal_add_noise
        else:
            signals = torch.cat((signals, signal_add_noise), axis=0)
    return signals.T


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default='PeMSD8', help='Dataset Name', type=str)
args = parser.parse_args()
dataset_name = args.dataset_name
train_data, input_dim = dataset.get_mts_train_data(dataset_name)

ae = model.TSautoencoder(input_dim, train_data.shape[0], noise=True).to(device)
ae.load_state_dict(torch.load('./save_pth/'+dataset_name+'/vae.pth'))

train_data_aug_1 = train_data
train_data_aug_2 = train_data
train_data_fq = rfft(train_data_aug_1.T.numpy())
train_data_fq = torch.Tensor(train_data_fq).to(device)
train_data = train_data.T.to(device)
train_data_aug_1 = train_data_aug_1.T.to(device).float()
train_data_aug_2 = train_data_aug_2.T.to(device).float()
incep_target_encoder = model.SimpleConvGLU_double(train_data.shape[1], input_dim).to(device)
incep_online_encoder = model.SimpleConvGLU_double(train_data.shape[1], input_dim).to(device)
incep_online_encoder_fq = model.SimpleConvGLU_double(train_data_fq.shape[1], input_dim).to(device)

online_reward = model.SimpleConvGLU_double(train_data.shape[1], input_dim).to(device)
target_ema_updater = model.EMA(0.99)
ph = model.ProjectionHead_target(100).to(device)
ph_fq = model.ProjectionHead_target(100).to(device)
criterion = torch.nn.CrossEntropyLoss()
k = 5
epochs_num = 1000

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
optimizer = torch.optim.Adam([
                {'params': incep_online_encoder_fq.parameters(), 'lr': 1e-3},
                {'params': incep_online_encoder.parameters(), 'lr': 1e-3},
                {'params': ph.parameters(), 'lr': 1e-3},
                {'params': ph_fq.parameters(), 'lr': 1e-3}
            ], lr=1e-3)
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
losses_min = 1e+8

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

for epoch in range(epochs_num):
    train_data_aug_1, _ = ae(train_data.T)
    train_data_aug_1 = train_data_aug_1.T.to(device).float()
    
    online_pred_v = incep_online_encoder(train_data_aug_2)
    online_pred_one = ph(online_pred_v)
    target_proj_two = incep_target_encoder(train_data_aug_1)
    online_pred_v_fq = incep_online_encoder_fq(train_data_fq)
    online_pred_one_fq = ph_fq(online_pred_v_fq)
    l = loss_fn(online_pred_one, target_proj_two.detach()).mean()
    logits, labels = loss.info_nce_loss(online_pred_one, online_pred_one_fq)
    logits = logits.to(device)
    labels = labels.to(device)
    l_fq = criterion(logits, labels)
    l_total = 1*l + l_fq
    emb = torch.cat((online_pred_v_fq, online_pred_v),1)
    A = F.relu(torch.tanh(torch.matmul(emb, emb.T)))
    optimizer.zero_grad()
    l_total.backward()
    optimizer.step()
    update_moving_average(target_ema_updater, incep_target_encoder, incep_online_encoder)
    
    rewards = []
    for reward_epoch in range(k):
        for key in incep_online_encoder.state_dict():
            online_reward.state_dict()[key] = incep_online_encoder.state_dict()[key] + torch.normal(mean=0,std=0.3,size=incep_online_encoder.state_dict()[key].shape).to(device)
        r1 = online_reward(train_data_aug_1)
        r2 = online_reward(train_data_aug_2)
        z1 = ph(r1)
        z2 = ph(r2)
        distance, path = fastdtw(z1.cpu().detach().numpy(), z2.cpu().detach().numpy(), dist=euclidean)
        rewards.append(distance)
        torch.save(online_reward.state_dict(),'./save_pth/'+dataset_name+'/online_reward'+str(reward_epoch)+'.pth')
    rewards = np.array(rewards, dtype=float)
    rewards /= sum(rewards)
    for reward_epoch in range(k):
        online_reward.load_state_dict(torch.load('./save_pth/'+dataset_name+'/online_reward'+str(reward_epoch)+'.pth'))
        for key in incep_online_encoder.state_dict():
            if key.find('bn') >= 0:
                continue
            if reward_epoch == 0:
                incep_online_encoder.state_dict()[key] -= incep_online_encoder.state_dict()[key]
                incep_online_encoder.state_dict()[key] += rewards[reward_epoch] * online_reward.state_dict()[key]    
            else:
                incep_online_encoder.state_dict()[key] += rewards[reward_epoch] * online_reward.state_dict()[key]

        
    print('Epoch', epoch, ' loss:', l_total.item(), 'byol loss:', l.item(), 're loss:', l_fq.item())

    if l_total.item() < losses_min:
        np.save('emb_byol_'+dataset_name+'.npy', emb.detach().cpu().numpy())
        losses_min = l_total.item()
