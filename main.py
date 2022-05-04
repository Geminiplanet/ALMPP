import argparse
import random

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE
from torch_geometric.transforms import RandomLinkSplit

from load_dataset import load_tox21_graph_data
from models import Encoder

parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
args = parser.parse_args()

if __name__ == '__main__':
    ADDEN = 500
    LR_milestones = [500, 1000]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and set training, testing dataset
    dataset = load_tox21_graph_data('data/nr-ar.sdf')
    # transform = RandomLinkSplit(is_undirected=True)
    # train_data, val_data, test_data = transform(dataset)
    TRAIN_NUM = int(0.7*len(dataset))
    train_data = dataset[:TRAIN_NUM]
    test_data = dataset[TRAIN_NUM:]
    indices = list(range(len(dataset)))
    # label_set =
    random.shuffle(dataset)
    print(f'train_data len: {len(train_data)}, test_data len: {len(test_data)}')
    random.shuffle(train_data)
    labeled_train_data = train_data[:ADDEN]
    train_dataloader = DataLoader(labeled_train_data, batch_size=16, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, pin_memory=True)

    model = VGAE(Encoder(2, 2)).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)
    train(model, optimizer, scheduler, labeled_train_data, epochs=100)

    print('ss')
