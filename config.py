CHAR_LIST = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti", "V",
             "Cr",
             "Mn", "Fe", "Ni", "Cu", "Zn", "Ge", "As", "Se", "Br", "Sr", "Zr", "Mo", "Pd", "Yb", "Ag", "Cd", "Sb", "I",
             "Ba", "Nd",
             "Dy", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
             "n", "c", "o", "s", "se",
             "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "(", ")", "[", "]",
             "-", "=", "#", "/", "\\", "+", "@", "<", ">", "."]
ADDEN = 500
BATCH_SIZE = 16
HIDDEN_DIM = 350
NLSTM_LAYER = 1
SEED_DIM = HIDDEN_DIM
NSEQ = 350
NFEA = len(CHAR_LIST)
STD0 = 0.2
STD00 = 0.02
STD_SEED = 0.25
STD_DECAY_RATIO = 0.99