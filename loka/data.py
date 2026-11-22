import numpy as np

def _side(n_cells):
    """Board side length from the number of cells (25 -> 5, 36 -> 6)."""
    side = int(round(n_cells ** 0.5))
    assert side * side == n_cells, f"{n_cells} cells is not a square board"
    return side

def _load_states(filename, two_channel, outcome_idx):
    with open(filename) as f:
        parts = [line.split() for line in f if line.strip()]

    # (N, side * side) array of single characters
    boards = np.array([list(p[0].replace("/", "")) for p in parts])
    outcomes = np.array([p[outcome_idx] for p in parts])

    X = np.concatenate([boards == "w", boards == "b"], axis=1).astype(np.uint8)
    Y = np.where(outcomes == "2", 1, -1).astype(np.int8)

    if two_channel:
        side = _side(X.shape[1] // 2)
        X = X.reshape(-1, 2, side, side) # one channel for white, one for black

    return X, Y, parts

def load_breakthrough_states(filename, two_channel: bool = False):
    X, Y, _ = _load_states(filename, two_channel, -1)
    return X, Y

def load_forgetting_states(filename, two_channel: bool = False):
    """Inverse of save_forgetting_states: returns X, Y, C."""
    X, Y, parts = _load_states(filename, two_channel, -2)
    C = np.array([int(p[-1]) for p in parts], dtype=np.int64)
    return X, Y, C

def _board_rows(X):
    """Convert X (flat (N, 2 s^2) or two-channel (N, 2, s, s)) to 'a/b/c/d/e' board strings (any square side s)."""
    X = np.asarray(X)
    n_cells = X.reshape(len(X), -1).shape[1] // 2
    side = _side(n_cells)
    X = X.reshape(len(X), 2, n_cells)
    boards = np.full((len(X), n_cells), ".")
    boards[X[:, 0] == 1] = "w"
    boards[X[:, 1] == 1] = "b"
    return ["/".join("".join(b[r:r + side]) for r in range(0, n_cells, side)) for b in boards]

def _write_states(filename, X, Y, extras=None, keep=None):
    rows = _board_rows(X)
    with open(filename, "w") as f:
        for i, (r, y) in enumerate(zip(rows, Y)):
            if keep is not None and not keep[i]:
                continue
            extra = f" {int(extras[i])}" if extras is not None else ""
            f.write(f"{r} w   => 1 {2 if y == 1 else 1}{extra}\n")

def save_breakthrough_states(filename, X, Y):
    _write_states(filename, X, Y)

def save_forgetting_states(filename, X, Y, C):
    """Like save_breakthrough_states, but appends the forgetting count c of each
    state after its label Y. Only states with c >= 0 are written."""
    C = np.asarray(C)
    _write_states(filename, X, Y, extras=C, keep=C >= 0)