import random

from rdkit import Chem
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import numpy as np

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
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

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

# def load_tox21_smiles_data(input_path):
#     """
#     param: input_path
#     return: smiles_mol_data_list
#     """
#     suppl = Chem.SDMolSupplier(input_path)
#     fp_list = []
#     labels = []
#     num_p = 0
#     num_n = 0
#     for mol in suppl:
#         if mol is not None:
#             if 'Active' not in mol.GetPropNames():
#                 print(f'no Active attr: {mol}')
#             fp = Chem.RDKFingerprint(mol)
#             label = torch.tensor(eval(mol.GetProp('Active')), dtype=torch.long)
#             if label == 1:
#                 num_p += 1
#             else:
#                 num_n += 1
#             fp_list.append(fp)
#             labels.append(label)
#     print(f"positive nums, negative nums: {(num_p, num_n)}")
#     # dataset = TensorDataset(torch.tensor(fp_list, dtype=), labels)
#     return