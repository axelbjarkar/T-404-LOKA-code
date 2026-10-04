import numpy as np


def counts(X):
    # (N, 2): white count, black count per board
    return X.reshape(len(X), 2, -1).sum(axis=2).astype(np.uint8)


def near_goal(X, white_moves_up=True):
    """(N, 2) booleans: [white, black] has a piece on its goal row or the row before it."""
    B = X.reshape(len(X), 2, 5, 5).astype(bool)

    top = B[:, :, :2, :].any(axis=(2, 3))     # (N, 2): any piece in rows 0-1
    bottom = B[:, :, 3:, :].any(axis=(2, 3))  # (N, 2): any piece in rows 3-4

    if white_moves_up:  # white heads for row 0, black for row 4
        return np.stack([top[:, 0], bottom[:, 1]], axis=1)
    return np.stack([bottom[:, 0], top[:, 1]], axis=1)


def _flat(X):
    # (N, 50): positions as one row per board
    return X.reshape(len(X), -1)


def with_counts(X):
    """(N, 52): positions + [white count, black count]."""
    return np.concatenate([_flat(X), counts(X)], axis=1)


def with_near_goal(X, white_moves_up=True):
    """(N, 52): positions + [white near goal, black near goal]."""
    return np.concatenate([_flat(X), near_goal(X, white_moves_up)], axis=1)


def with_counts_and_near_goal(X, white_moves_up=True):
    """(N, 54): positions + counts + near-goal flags."""
    return np.concatenate([_flat(X), counts(X), near_goal(X, white_moves_up)], axis=1)