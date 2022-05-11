import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm

from config import *
from utils import *

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE
    ]
}
tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
         'SR-HSE', 'SR-MMP', 'SR-p53']


def mol_to_graph(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atom
    num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                        [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_features)
    # data.x: shape [num_atoms, 2]
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_features = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                            [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_features)
            edges_list.append((j, i))
            edge_features_list.append(edge_features)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bond
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def load_tox21_graph_data(input_path):
    """
    param: input_path
    return: pyg_mol_graph_data_list
    """
    suppl = Chem.SDMolSupplier(input_path)
    data_list = []
    num_p = 0
    num_n = 0
    for mol in suppl:
        if mol is not None:
            if 'Active' not in mol.GetPropNames():
                print(f'no Active attr: {mol}')
            data = mol_to_graph(mol)
            data.y = torch.tensor(eval(mol.GetProp('Active')), dtype=torch.long)
            if data.y == 1:
                num_p += 1
            else:
                num_n += 1
            data_list.append(data)
    print(f"positive nums, negative nums: {(num_p, num_n)}")
    random.shuffle(data_list)
    return data_list


def process_tox21_smiles_data(input_path):
    """
    param: input_path
    return: smiles_mol_data_list
    """
    char_dict = dict()
    i = -1
    for c in CHAR_LIST:
        i += 1
        char_dict[c] = i
    # print(char_dict)
    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in CHAR_LIST:
        if len(key) == 1:
            char_list1 += [key]
            char_dict1[key] = char_dict[key]
        elif len(key) == 2:
            char_list2 += [key]
            char_dict2[key] = char_dict[key]
        else:
            print("strange ", key)
    data = pd.read_csv(input_path).values
    Xdata = []
    Ldata = []
    Pdata = []
    for line in data:
        # print(type(line[0]))
        if line[0] == 1 or line[0] == 0:
            # line[0]: NR-AR, line[13]: smiles
            smiles = line[13]
            Pdata += [line[0]]

            Nsmiles = len(smiles)
            X_d = np.zeros([NSEQ], dtype=int)
            X_d[0] = char_dict['<']
            i = 0
            istring = 0
            check = True
            flag = False
            while check:
                char2 = smiles[i: i + 2]
                char1 = smiles[i]
                if char2 in char_list2:
                    j = char_dict2[char2]
                    i += 2
                    if i >= Nsmiles:
                        check = False
                elif char1 in char_list1:
                    j = char_dict1[char1]
                    i += 1
                    if i >= Nsmiles:
                        check = False
                else:
                    # if line[0] == 0:
                    #     flag = True
                    #     break
                    print(char1, char2, "error")
                    sys.exit()
                X_d[istring + 1] = j
                istring += 1
            # if flag:
            #     continue
            for i in range(istring, NSEQ - 1):
                X_d[i + 1] = char_dict['>']
            Xdata += [X_d]
            Ldata += [istring + 1]
    train_num = int(0.1 * len(Xdata))
    test_num = int(0.8 * len(Xdata))
    X_train = np.asarray(Xdata[:train_num], dtype="long")
    L_train = np.asarray(Ldata[:train_num], dtype="long")
    P_train = np.asarray(Pdata[:train_num], dtype="long")
    X_test = np.asarray(Xdata[test_num:], dtype="long")
    L_test = np.asarray(Ldata[test_num:], dtype="long")
    P_test = np.asarray(Pdata[test_num:], dtype="long")
    print(X_train.shape, L_train.shape, P_train.shape)
    np.save('data/X_train.npy', X_train)
    np.save('data/L_train.npy', L_train)
    np.save('data/P_train.npy', P_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/L_test.npy', L_test)
    np.save('data/P_test.npy', P_test)
    return


# class UserDataset(Dataset):
#     def __init__(self, datadir, name):
#         Xdata_file = datadir + "/X_" + name + ".npy"
#         self.Xdata = torch.tensor(np.load(Xdata_file), dtype=torch.long)
#         Ldata_file = datadir + "/L_" + name + ".npy"
#         self.Ldata = torch.tensor(np.load(Ldata_file), dtype=torch.long)
#         Pdata_file = datadir + "/P_" + name + ".npy"
#         self.Pdata = torch.tensor(np.load(Pdata_file), dtype=torch.long)
#         #        PRdata_file=datadir+"/P"+dname+".npy"
#         #        Pdata_reg0=np.load(PRdata_file)
#
#         #        self.Pdata=torch.tensor(np.concatenate(
#         #            [Pdata_reg0[:,0:1],Pdata_reg0[:,3:4],Pdata_reg0[:,2:3]],
#         #            axis=1),dtype=torch.float32)
#         self.len = self.Xdata.shape[0]
#
#     def __getitem__(self, index):
#         return self.Xdata[index], self.Ldata[index], self.Pdata[index]
#
#     def __len__(self):
#         return self.len
#


def load_dataset_random(path, dataset, seed, tasks=None):
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test
    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, 'raw/{}.csv'.format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)
        except:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))

    df = df[df["smiles"].isin(remained_smiles)].reset_index()
    if dataset == 'sider' or dataset == 'clintox' or dataset == 'tox21' or dataset == 'ecoli' or dataset == 'AID1706_binarized_sars':
        train_size = int(0.8 * len(pyg_dataset))
        val_size = int(0.1 * len(pyg_dataset))
        test_size = len(pyg_dataset) - train_size - val_size
        pyg_dataset = pyg_dataset.shuffle()
        trn, val, test = pyg_dataset[:train_size], \
                         pyg_dataset[train_size:(train_size + val_size)], \
                         pyg_dataset[(train_size + val_size):]
        weights = []
        for i, task in enumerate(tasks):
            negative_df = df[df[task] == 0][["smiles", task]]
            positive_df = df[df[task] == 1][["smiles", task]]
            neg_len = len(negative_df)
            pos_len = len(positive_df)
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights

    elif dataset == 'esol' or dataset == 'freesolv' or dataset == 'lipophilicity':  # esol  freesolv lip support
        train_size = int(0.8 * len(pyg_dataset))
        val_size = int(0.1 * len(pyg_dataset))
        test_size = len(pyg_dataset) - train_size - val_size
        pyg_dataset = pyg_dataset.shuffle()
        trn, val, test = pyg_dataset[:train_size], \
                         pyg_dataset[train_size:(train_size + val_size)], \
                         pyg_dataset[(train_size + val_size):]
        trn.weights = 'regression task has no class weights!'
    else:
        print('This dataset should not use this split method')
    torch.save([trn, val, test], save_path)
    return load_dataset_random(path, dataset, seed, tasks)

def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        # if atom.GetDegree()>5:
        #     print(Chem.MolToSmiles(mol))
        #     print(atom.GetSymbol())
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding(atom.GetDegree(),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            #                 print(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')])
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, tasks, transform=None, pre_transform=None, pre_filter=None):
        self.tasks = tasks
        self.dataset = dataset

        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print("number of all smiles: ", len(smilesList))
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully processed smiles: ", len(remained_smiles))

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):

            mol = Chem.MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 6
                data.y = torch.LongTensor([label])
                if self.dataset == 'esol' or self.dataset == 'freesolv' or self.dataset == 'lipophilicity':
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        # pos = torch.FloatTensor(geom)
        data = Data(
            x=torch.FloatTensor(node_attr),
            # pos=pos,
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            y=None  # None as a placeholder
        )
        return data



def main():
    train_dataset, valid_dataset, test_dataset = load_dataset_random('data/', 'tox21', seed=66, tasks=tasks)

    # data = pd.read_csv('data/tox21.csv').values
    # for line in data:
    #     print(line[0])


if __name__ == "__main__":
    main()
