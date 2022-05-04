import math

import numpy as np
import scipy.optimize

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
""""""
import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges

torch.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='VGAE')
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE']
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'VGAE': VGAE}



class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)



#
# channels = 16
# dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = kwargs[args.model](Encoder(dataset.num_features, channels)).to(dev)
# data.train_mask = data.val_mask = data.test_mask = data.y = None
# data = train_test_split_edges(data)
# x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#
# def train():
#     optimizer.zero_grad()
#     z = model.encode(x, train_pos_edge_index)
#     loss = model.recon_loss(z, train_pos_edge_index)
#     if args.model in ['VGAE']:
#         loss = loss + (1 / data.num_nodes) * model.kl_loss()
#     loss.backward()
#     optimizer.step()
#
#
# def test(pos_edge_index, neg_edge_index):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(x, train_pos_edge_index)
#     return model.test(z, pos_edge_index, neg_edge_index)
#
#
# for epoch in range(1, 401):
#     train()
#     auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
#     print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
# """"""

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
            Args
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
        '''
        super(VAE, self).__init__()

        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)

        self.con2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes+1) // 2
        self.vae = MLP_VAE_plain(input_dim*input_dim, latent_dim, output_dim)

        self.max_num_nodes = max_num_nodes
        self.pool = pool



class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r,z):
        z = torch.cat([z, r], 1)
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)



# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y


class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).cuda()
        z = eps*z_sgm + z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms