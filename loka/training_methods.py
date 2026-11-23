import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from loka.models import BreakthroughModel
from abc import ABC, abstractmethod


class TrainingMethod(ABC):  # abstract base class
    def __init__(self, model: BreakthroughModel, seed: int = 42):
        self.model = model
        self.seed = seed

    @abstractmethod
    def train(self,
              X: torch.Tensor,
              Y: torch.Tensor,
              epochs: int = 50,
              batch_size: int = 512,
              random_permuations: bool = True,

              # Hyperparams
              lr: float = 0.001,
              optimizer: str = 'Adam',
              loss_fn=nn.MSELoss(),

              device=torch.device("cpu"),
              pbar: tqdm = None,
              pbar_increment : int = 1
              ) -> None:
        ...

    def resolve_optimizer(self, name: str):
        try:
            return getattr(torch.optim, name)
        except AttributeError:
            raise ValueError(f"Unknown optimizer: {name}")


class ForgettingScores(TrainingMethod):
    def __init__(self, model: BreakthroughModel, seed: int = 42):
        super().__init__(model=model, seed=seed)

        self.C = None
        self._base = None
        self._prev_correct = None

    def train(self,
              X: torch.Tensor,
              Y: torch.Tensor,
              epochs: int = 50,
              batch_size: int = 512,
              random_permuations: bool = True,

              # Hyperparams
              lr: float = 0.001,
              optimizer: str = 'Adam',
              loss_fn=nn.MSELoss(),

              device=torch.device("cpu"),
              pbar: tqdm = None,
              pbar_increment : int = 1
              ) -> None:
        n = len(X)
        self.C = torch.zeros(n, dtype=torch.int32, device=device)
        self._base = torch.zeros(n, dtype=torch.int32, device=device)
        self._prev_correct = torch.zeros(n, dtype=torch.int32, device=device)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
        self.model = self.model.to(device)

        optimizer = self.resolve_optimizer(optimizer)(self.model.parameters(), lr=lr)

        # Seed once, so each epoch still gets a different shuffle
        torch.manual_seed(self.seed)

        for epoch in range(epochs):
            if random_permuations:
                perm = torch.randperm(n, device=device)  # new shuffle each epoch
            else:
                perm = torch.arange(n, device=device)

            self.model.train()
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_t[idx], Y_t[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            # Forgetting count
            self.model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    # Evaluate entire dataset in batches
                    preds.append(self.model(X_t[i:i + batch_size]))
            test_preds = torch.cat(preds)

            correct = (test_preds.clamp(-1, 1).round() == Y_t).int()  # -1 or 1

            # Increment forgetting events
            self.C += torch.maximum(self._prev_correct - correct, self._base)
            self._prev_correct = correct

            # Update progress (once per epoch)
            if pbar is not None:
                pbar.update(pbar_increment)

    def getForgettingCounts(self):
        return self.C


class CAL(TrainingMethod):
    def __init__(self, model: BreakthroughModel, seed: int = 42):
        super().__init__(model=model, seed=seed)

    def train(self, X, Y, epochs=50, batch_size=512, random_permuations=True,
              lr=0.001, optimizer='Adam', loss_fn=nn.MSELoss(),
              device=torch.device("cpu"), pbar=True) -> None:
        raise NotImplementedError


def make_forgettable_subset(
    E: torch.Tensor,
    Y: torch.Tensor,
    C: torch.Tensor,
    order: np.ndarray,
    S_size: int,
    forgettable_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a subset S of size `S_size` containing `forgettable_size` forgettable events.

    Takes the first `S_size` events in `order`. In the last `forgettable_size`
    slots, any unforgettable event (C == 0) is replaced by a randomly chosen
    forgettable event (C != 0) that is not already in S.

    Args:
        E: Events, indexed along axis 0.
        Y: Labels, aligned with `E`.
        C: Per-event forgettable flag (nonzero = forgettable).
        order: Event indices in priority order.
        S_size: Number of events in the subset.
        forgettable_size: Number of slots, counted from the end of S, to make forgettable.

    Returns:
        The events and labels of the subset, `(E[idx], Y[idx])`.
    """
    assert forgettable_size <= S_size

    idx: torch.Tensor = torch.as_tensor(order[:S_size], device=C.device).long().clone()
    tail: torch.Tensor = idx[S_size - forgettable_size:]    # view into idx

    bad: torch.Tensor = C[tail] == 0                        # unforgettable slots in the tail
    n: int = int(bad.sum())

    mask: torch.Tensor = C != 0
    mask[idx] = False                                       # exclude events already in S
    pool: torch.Tensor = mask.nonzero().squeeze(1)
    pool = pool[torch.randperm(len(pool), device=pool.device)[:n]]

    tail[bad] = pool
    idx_cpu = idx.cpu().numpy()
    return E[idx_cpu], Y[idx_cpu]


def _predict(model, X, bs=1024):
    return torch.cat([model(X[s:s + bs]).reshape(-1) for s in range(0, len(X), bs)])


@torch.no_grad()
def _penultimate(model, X, bs=1024):
    """Input of the model's last nn.Linear for every row of X (captured with a forward pre-hook; works for any model of the package)."""
    last = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    captured = []
    handle = last.register_forward_pre_hook(lambda mod, args: captured.append(args[0].reshape(len(args[0]), -1)))
    try:
        for s in range(0, len(X), bs):
            model(X[s:s + bs])
    finally:
        handle.remove()
    return torch.cat(captured)


@torch.no_grad()
def CAL_scores(model, X_label, X_pool, k_nearest, chunk=1024, bs=1024, space="raw"):
    """CAL score of every pool state: the mean |prediction of its k nearest labeled states - its own prediction|.
    Higher = the model disagrees more with its labeled neighbours. Labels of the pool are never used.

    space: "raw" measures distances on the inputs as given (as CAL_select always did),
           "penultimate" on the input of the model's last nn.Linear (e.g. 128-d for BreakthroughCNN).
    Returns a tensor of shape [len(X_pool)]; the model is left in eval mode."""
    if space not in ("raw", "penultimate"):
        raise ValueError(f"space must be 'raw' or 'penultimate', got {space!r}")
    model.eval()
    k = min(k_nearest, len(X_label))

    pred_label = _predict(model, X_label, bs)    # [L]
    pred_pool = _predict(model, X_pool, bs)      # [P]
    if space == "penultimate":
        X_label = _penultimate(model, X_label, bs)    # the pool's features are computed per chunk (they would not all fit on a small GPU)

    scores = torch.empty(len(X_pool), device=pred_pool.device)
    for s in range(0, len(X_pool), chunk):
        e = s + chunk
        block = X_pool[s:e] if space == "raw" else _penultimate(model, X_pool[s:e], bs)
        d = torch.cdist(block, X_label)
        idx = d.topk(k, largest=False).indices
        nb = pred_label[idx]
        scores[s:e] = (nb - pred_pool[s:e, None]).abs().mean(dim=1)
    return scores


@torch.no_grad()
def CAL_select(model, X_label, X_pool, size_Q, k_nearest, chunk=1024, bs=1024, space="raw"):
    """Indexes (into X_pool) of the size_Q pool states with the highest CAL score, see CAL_scores."""
    return CAL_scores(model, X_label, X_pool, k_nearest, chunk=chunk, bs=bs, space=space).topk(size_Q).indices
