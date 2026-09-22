import numpy as np

def load_breakthrough_states(filename):
    with open(filename) as f:
        parts = [line.split() for line in f if line.strip()]

    # (N, 25) array of single characters
    boards = np.array([list(p[0].replace("/", "")) for p in parts])
    outcomes = np.array([p[-1] for p in parts])

    X = np.concatenate([boards == "w", boards == "b"], axis=1).astype(np.uint8)
    Y = np.where(outcomes == "2", 1, -1).astype(np.int8)
    return X, Y
