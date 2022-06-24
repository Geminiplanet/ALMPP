import argparse
import random

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from config import *
from dataset import load_qm9_dataset
from models import MolecularVAE, Predictor

parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
args = parser.parse_args()


# torch.backends.cudnn.enabled = False


def vae_loss(x, recon, mu, logvar, beta):
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss * beta
    # print(recon_loss, kl_loss, beta)
    return recon_loss + kl_loss


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data, test_data = load_qm9_dataset('data/qm9.csv', 5)
    indices = list(range(train_data.len))
    random.shuffle(indices)
    labeled_set = indices[:SUBSET]
    unlabeled_set = [x for x in indices if x not in labeled_set]
    train_loader = DataLoader(train_data, BATCH_SIZE, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
    test_loader = DataLoader(test_data, BATCH_SIZE)

    # train vae and predictor
    vae = MolecularVAE().to(device)
    optim_vae = optim.Adam(vae.parameters())

    predictor = Predictor().to(device)
    optim_pred = optim.Adam(predictor.parameters(), lr=LR)
    criterion = nn.L1Loss()

    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:SUBSET]
    vae.train()
    predictor.train()
    for epoch in range(101):
        train_loss = 0
        for data in tqdm(train_loader, leave=False):
            optim_vae.zero_grad()
            x, _, y = data
            x = x.float().to(device)
            label = y.float().to(device)
            recon_x, mu, logvar = vae(x)
            loss_vae = vae_loss(x, recon_x, mu, logvar, beta=1)
            train_loss += loss_vae
            # pred = predictor(mu)
            # loss_target = criterion(pred, label)

            # loss = loss_vae + loss_target


            # optim_pred.zero_grad()
            loss_vae.backward()
            optim_vae.step()
            # optim_pred.step()
        if epoch % 5 == 0:
            sample_input = x[0].cpu().argmax(axis=1)
            print('input: ', decode_smiles_from_indexes(sample_input, QM9_CHAR_LIST))
            sample_recon = recon_x[0].cpu().argmax(axis=1)
            print('recon: ', decode_smiles_from_indexes(sample_recon, QM9_CHAR_LIST))
        # print(f'epoch {epoch} loss: vae {loss_vae}, target {loss_target}')
        print(f'epoch {epoch} loss: {train_loss / len(train_loader.dataset)}')

    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x = x.float().to(device)
            mu, _ = vae.encode(x)
            output = predictor(mu)
