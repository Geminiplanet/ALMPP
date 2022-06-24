QM9_CHAR_LIST = [" ", "H", "C", "N", "O", "F",
                 "1", "2", "3", "4", "5",
                 "(", ")", "[", "]",
                 "-", "=", "#", ":", "/", "\\", "+"]
QM9_TASKS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
MAX_QM9_LEN = 42
QM9_FEA = len(QM9_CHAR_LIST)

ADDEN = 500
SUBSET = 10000
CYCLE = 10

LR = 5e-4

BATCH_SIZE = 100
HIDDEN_DIM = 300