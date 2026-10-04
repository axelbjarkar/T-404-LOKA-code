import numpy as np

def load_breakthrough_states(filename, two_channel: bool = False):
    with open(filename) as f:
        parts = [line.split() for line in f if line.strip()]

    # (N, 25) array of single characters
    boards = np.array([list(p[0].replace("/", "")) for p in parts])
    outcomes = np.array([p[-1] for p in parts])

    X = np.concatenate([boards == "w", boards == "b"], axis=1).astype(np.uint8)
    Y = np.where(outcomes == "2", 1, -1).astype(np.int8)

    if two_channel:
        X = X.reshape(-1, 2, 5, 5) # one channel for white, one for black

    return X, Y

def save_breakthrough_states(filename, X, Y):
    X = np.asarray(X).reshape(len(X), 2, 25)  # handles flat (N, 50) and two-channel (N, 2, 5, 5)
    boards = np.full((len(X), 25), ".")
    boards[X[:, 0] == 1] = "w"
    boards[X[:, 1] == 1] = "b"
    with open(filename, "w") as f:
        for b, y in zip(boards, Y):
            rows = "/".join("".join(b[r:r + 5]) for r in range(0, 25, 5))
            f.write(f"{rows} w   => 1 {2 if y == 1 else 1}\n")