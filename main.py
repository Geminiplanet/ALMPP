import argparse

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import *
from load_dataset import UserDataset
from models import Net
from train_test import train_test

parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
args = parser.parse_args()
# torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    LR_milestones = [500, 1000]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # Load dataset and set training, testing dataset
    datadir = "data"
    train_data = UserDataset(datadir, "train")
    test_data = UserDataset(datadir, "test")
    print(f'train_data len: {len(train_data)}, test_data len: {len(test_data)}')
    # labeled_train_data = train_data.__getitem__(ADDEN)
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=2)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                 drop_last=True, num_workers=2)
    para = {'Nseq': NSEQ, 'Nfea': NFEA, 'hidden_dim': HIDDEN_DIM,
            'seed_dim': SEED_DIM, 'NLSTM_layer': NLSTM_LAYER, 'device': device}
    model = Net(para).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_AE = nn.CrossEntropyLoss()
    # AE_parameters = list(model.Enc.parameters()) + list(model.Dec.parameters())
    # optimizer_AE = optim.Adam(AE_parameters, lr=0.001)
    # optimizer_cri = optim.Adam(model.Cri.parameters(), lr=0.000002)

    # optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=0.001)
    train_test(model, criterion_AE, optimizer, scheduler, train_dataloader, test_dataloader, epochs=100, device=device)
