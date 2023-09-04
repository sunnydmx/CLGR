from cgi import test
import torch
import dataset
import model
import numpy as np
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default='PeMSD8', help='Dataset Name', type=str)
args = parser.parse_args()
dataset_name = args.dataset_name
path = './save_pth/' + dataset_name
if not os.path.exists(path):
    os.mkdir(path)
train_data, input_dim = dataset.get_mts_train_data(dataset_name)
train_data = train_data.to(device)
ae = model.TSautoencoder(train_data.shape[1], train_data.shape[0]).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
losses_min = 1e+8

for epoch in range(1000):
    losses_train = []
    E, mu, log_var, rx = ae(train_data)
    mse_loss = criterion(train_data, rx)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = mse_loss + kld_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mean_loss = loss.item()
    if mean_loss < losses_min:
        np.save('emb_vae_'+dataset_name+'.npy', E.detach().cpu().numpy())
        torch.save(ae.state_dict(),'./save_pth/'+dataset_name+'/ae.pth')
        losses_min = mean_loss
    
