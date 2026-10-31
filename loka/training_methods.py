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