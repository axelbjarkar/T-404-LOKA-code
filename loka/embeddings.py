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

def _clear_lane(mine, theirs, moves_up):
    """(N,5,5) bool each -> (N,) True if some piece of `mine` has no `theirs` piece ahead in its column."""
    if moves_up:  # ahead = smaller row index
        seen = np.logical_or.accumulate(theirs, axis=1)
        ahead = np.concatenate([np.zeros_like(seen[:, :1]), seen[:, :-1]], axis=1)
    else:         # ahead = larger row index
        seen = np.logical_or.accumulate(theirs[:, ::-1], axis=1)[:, ::-1]
        ahead = np.concatenate([seen[:, 1:], np.zeros_like(seen[:, :1])], axis=1)
    return (mine & ~ahead).any(axis=(1, 2))


def clear_lane(X, white_moves_up=True):
    """(N, 2) booleans: [white, black] has a piece with no opponent anywhere ahead of it in its column."""
    B = X.reshape(len(X), 2, 5, 5).astype(bool)
    white, black = B[:, 0], B[:, 1]
    return np.stack([_clear_lane(white, black, white_moves_up),
                     _clear_lane(black, white, not white_moves_up)], axis=1)


def with_clear_lane(X, white_moves_up=True):
    """(N, 52): positions + [white clear lane, black clear lane]."""
    return np.concatenate([_flat(X), clear_lane(X, white_moves_up)], axis=1)