import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import *


def process_qm9_smiles_data(data_dir):
    """
    max smiles len: 42
    total smiles: 133885
    """
    char_dict = dict()
    i = 0
    for c in QM9_CHAR_LIST:
        char_dict[c] = i
        i += 1
    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in QM9_CHAR_LIST:
        if len(key) == 1:
            char_list1 += [key]
            char_dict1[key] = char_dict[key]
        elif len(key) == 2:
            char_list2 += [key]
            char_dict2[key] = char_dict[key]
        else:
            print("strange ", key)
    df = pd.read_csv(data_dir)
    target = df[QM9_TASKS].values
    pmax = np.array(target).max(axis=0)
    pmin = np.array(target).min(axis=0)
    # mean = np.array(target).mean(axis=0)
    # std = np.array(target).std(axis=0)
    smiles_list = df.smiles.values
    Xdata = []
    Ldata = []
    Pdata = []
    for i, smi in enumerate(smiles_list):
        smiles_len = len(smi)
        labels = (target[i] - pmin) / (pmax - pmin)  # normalization
        # labels = target[i] - mean / std
        Pdata.append(labels)
        X_d = np.zeros((MAX_QM9_LEN, len(QM9_CHAR_LIST)))  # one-hot
        # X_d = []
        j = 0
        istring = 0
        check = True
        while check:
            char2 = smi[j: j + 2]
            char1 = smi[j]
            if char2 in char_list2:
                index = char_dict2[char2]
                j += 2
                if j >= smiles_len:
                    check = False
            elif char1 in char_list1:
                index = char_dict1[char1]
                j += 1
                if j >= smiles_len:
                    check = False
            else:
                print(char1, char2, "error")
                sys.exit()
            X_d[istring, index] = 1
            # X_d.append(index)
            istring += 1
        Ldata.append(istring)
        for k in range(istring, MAX_QM9_LEN):
            X_d[k, 0] = 1
            # X_d.append(0)
        Xdata.append(X_d)
    X_data = np.asarray(Xdata, dtype="long")
    L_data = np.asarray(Ldata, dtype="long")
    P_data = np.asarray(Pdata, dtype="float")
    print(X_data.shape, L_data.shape, P_data.shape)
    # shape: X_data(133885, 42, 22) L_data(133885,) P_data(133885, 12)
    np.save('data/qm9/X_data.npy', X_data)
    np.save('data/qm9/L_data.npy', L_data)
    np.save('data/qm9/P_data.npy', P_data)


class QM9Dataset(Dataset):
    def __init__(self, data_dir, name, task_no):
        Xdata = torch.tensor(np.load(data_dir + f'X_data.npy'), dtype=torch.long)
        Ldata = torch.tensor(np.load(data_dir + f'L_data.npy'), dtype=torch.long)
        Pdata = torch.tensor(np.load(data_dir + f'P_data.npy'), dtype=torch.float)
        Pdata = Pdata[:, task_no]
        data_len = Xdata.shape[0]
        index = list(range(data_len))
        # random.shuffle(index)
        train_num = int(0.8 * data_len)
        if name == 'train':
            self.Xdata = Xdata[index[:train_num]]
            self.Ldata = Ldata[index[:train_num]]
            self.Pdata = Pdata[index[:train_num]]
            self.len = self.Xdata.shape[0]
        elif name == 'test':
            self.Xdata = Xdata[index[train_num:]]
            self.Ldata = Ldata[index[train_num:]]
            self.Pdata = Pdata[index[train_num:]]
            self.len = self.Xdata.shape[0]
        elif name == 'all':
            self.Xdata = Xdata
            self.Ldata = Ldata
            self.Pdata = Pdata
            self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        return self.Xdata[index], self.Ldata[index], self.Pdata[index]

    def __len__(self):
        return self.len


def load_qm9_dataset(datadir, task_no):
    # process_qm9_smiles_data(datadir)
    train_data = QM9Dataset('data/qm9/', 'train', task_no)
    test_data = QM9Dataset('data/qm9/', 'test', task_no)
    return train_data, test_data


if __name__ == '__main__':
    process_qm9_smiles_data('data/qm9.csv')
