import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        self.conv_1 = nn.Conv1d(22, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)

        self.linear_0 = nn.Linear(160, 435)
        self.linear_1 = nn.Linear(435, HIDDEN_DIM)  # mu
        self.linear_2 = nn.Linear(435, HIDDEN_DIM)  # logv

        self.linear_3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.gru = nn.GRU(HIDDEN_DIM, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, QM9_FEA)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, MAX_QM9_LEN, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = self.linear_4(out_reshape)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        # x_recon = self.fc_1(out_reshape)
        # x_recon = x_recon.contiguous().reshape(z.shape[0], -1, x_recon.shape[-1])
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 1)

        self.relu = nn.ReLU()

    def forward(self, z):
        out1 = self.relu(self.fc1(z))
        out2 = self.relu(self.fc2(out1))
        out = self.fc3(out2)

        return out.view(-1)  # , [out1, out2]
