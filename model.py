from numpy import pad
import torch
import torch.nn as nn


class TSautoencoder(nn.Module):
	def __init__(self, input_dim, ts_len, noise=False):
		super(TSautoencoder, self).__init__()
		modules = []
		self.hidden_dims = [1000, 100]
		input_dim_tmp = ts_len
		for h_dim in self.hidden_dims:
			modules.append(
                nn.Sequential(
                    nn.Linear(input_dim_tmp, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU())
            )
			input_dim_tmp = h_dim
		self.encoder = nn.Sequential(*modules)
		self.fc_mu = nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1])
		self.fc_var = nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1])
			
		self.hidden_dims.reverse()
		modules = []
		# modules.append()
		for i in range(len(self.hidden_dims)-1):
			modules.append(
				nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                    nn.BatchNorm1d(self.hidden_dims[i+1]),
                    nn.ReLU())
            )
		
		self.decoder = nn.Sequential(*modules)
		self.decoder_input = nn.Linear(self.hidden_dims[-1], ts_len)
		self.noise = noise
	
	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return eps * std + mu

	def forward(self, x):
		x = x.T
		E = self.encoder(x)
		mu = self.fc_mu(E)
		log_var = self.fc_var(E)
		E_2 = self.reparameterize(mu, log_var)
		rx = self.decoder(E_2)
		rx = self.decoder_input(rx)
		if self.noise:
			E_1 = self.reparameterize(mu, log_var)
			rx1 = self.decoder(E_1)
			rx1 = self.decoder_input(rx1)
			return rx.T, rx1.T
		return E, mu, log_var, rx.T


class ProjectionHead_target(nn.Module):
	def __init__(self, input_dim):
		super(ProjectionHead_target, self).__init__()
		self.fc1 = nn.Linear(input_dim, input_dim)
		self.fc2 = nn.Linear(input_dim, input_dim)

	def forward(self, x):
		x = self.fc1(x)
		x = torch.relu(x)
		x = self.fc2(x)
		x = x.squeeze()
		return x
    

class SimpleConvGLU_double(nn.Module):
    def __init__(self, input_dim, num_nodes):
        super(SimpleConvGLU_double, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.conv1_left = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv1_right = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.glu1_conv = nn.GLU()
        self.conv2_left = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, stride=1, padding=4)
        self.conv2_right = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, stride=1, padding=4)
        self.glu2_conv = nn.GLU()
        self.conv_fc = nn.Conv1d(in_channels=input_dim, out_channels=100, kernel_size=1, stride=1)
        self.bn_conv = nn.BatchNorm1d(input_dim)
        self.bn_conv2 = nn.BatchNorm1d(input_dim)
        self.fc1_left = nn.Linear(input_dim, 1000)
        self.fc1_right = nn.Linear(input_dim, 1000)
        self.bn = nn.BatchNorm1d(1000)
        self.glu1 = nn.GLU()
        self.fc2_left = nn.Linear(1000, 100)
        self.fc2_right = nn.Linear(1000, 100)
        self.glu2 = nn.GLU()
    def forward(self, x):
        x = x.unsqueeze(1)
        x_left = self.conv1_left(x) # N * 1 * T
        x_right = self.conv1_right(x) # N * 1 * T
        x = self.glu1_conv(torch.cat((x_left, x_right),dim=-1))
        x_tmp = x.reshape(x.shape[0], self.input_dim, -1)
        x_tmp = self.bn_conv(x_tmp)
        x = x_tmp.reshape(x.shape[0], -1, self.input_dim)
        x = torch.relu(x)
        x_left = self.conv2_left(x) # N * 1 * T
        x_right = self.conv2_right(x) # N * 1 * T
        x = self.glu2_conv(torch.cat((x_left, x_right),dim=-1))
        x_tmp = x.reshape(x.shape[0], self.input_dim, -1)
        x_tmp = self.bn_conv2(x_tmp)
        x = x_tmp.reshape(x.shape[0], -1, self.input_dim)
        x = torch.relu(x)
        x = x.squeeze()
        i1_left = self.fc1_left(x)
        i1 = self.bn(i1_left)
        i1 = torch.relu(i1)
        i2_left = self.fc2_left(i1)
        return i2_left

    
class LR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = self.fc(x)
        return out


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new