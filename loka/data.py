import numpy as np
from tqdm import tqdm

CHAR_MAP = {'.': 0, 'w': 1, 'b': -1}
OUTCOME_MAP = {'1': -1, '2': 1}


def load_breakthrough_states(filename):
    """Parse board-state lines into (X, Y) arrays of flattened boards and outcomes."""
    X = []
    Y = []

    with open(filename) as f:
        total = sum(1 for _ in f)

    with open(filename) as f:
        for line in tqdm(f, total=total, desc='Mapping breakthrough game states'):
            line = line.strip("\n")

            # last char shows result
            y = OUTCOME_MAP[line[-1]]

            # create board
            rows = line.split()[0].split("/")
            board = np.array([[CHAR_MAP[c] for c in row] for row in rows]).flatten()

            X.append(board)
            Y.append(y)

    return np.array(X), np.array(Y)
