import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

N_BOARD_FEATURES = 50  # 2 channels x 5 x 5


class BreakthroughModel(nn.Module):
    """Base class: preprocess(x) -> self.net -> (N, 1) squeezed to (N).

    Subclasses set `self.net` in __init__ and may override `preprocess`.
    """

    # Annotation, this attribute will be created by child classes
    net: nn.Module

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.preprocess(x)).squeeze(-1)  # (N, 1) -> (N)


class MLP(BreakthroughModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )


class BreakthroughCNN(BreakthroughModel):
    def __init__(self, n_extra: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            # 2 board channels + one constant channel per extra feature
            nn.Conv2d(2 + n_extra, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # (N, 64, 5, 5) -> (N, 1600)
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def preprocess(self, x):
        x = x.reshape(len(x), -1)                             # (N, 50 + n_extra)
        board = x[:, :N_BOARD_FEATURES].reshape(-1, 2, 5, 5)  # positions back to a board
        planes = x[:, N_BOARD_FEATURES:, None, None].expand(-1, -1, 5, 5)  # extra feature -> constant 5x5 plane
        return torch.cat([board, planes], dim=1)              # (N, 2 + n_extra, 5, 5)


def get_breakthrough_cnn(X):
    """Build a CNN whose channel count matches X: (N, 2, 5, 5), (N, 50), (N, 52) or (N, 54)."""
    return BreakthroughCNN(n_extra=X.reshape(len(X), -1).shape[1] - N_BOARD_FEATURES)

