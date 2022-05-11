import networkx as nx
import numpy as np
from rdkit import Chem

if __name__ == '__main__':
    # smiles_list = []
    # lens = []
    # with Chem.SDMolSupplier('data/nr-ar.sdf') as suppl:
    #     for mol in suppl:
    #         if mol is not None:
    #             # print(mol)
    #             smiles = Chem.MolToSmiles(mol)
    #             smiles_list.append(smiles)
    #             lens.append(len(smiles))
    # lens = np.array(lens)
    # lens.sort()
    # print(lens[-200: -300 :-1])
    graphs = []
    net = nx.grid_2d_graph(2, 3)
    graphs.append(net)
    print(net)