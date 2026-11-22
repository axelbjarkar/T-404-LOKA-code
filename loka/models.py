import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import math

N_BOARD_FEATURES = 50  # 2 channels x 5 x 5 (the default 5x5 board; see board_features)


def board_features(board_size: int) -> int:
    """Number of position features of a board_size x board_size board: 2 channels (white, black)."""
    return 2 * board_size * board_size


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

import math

import torch
import torch.nn as nn


class AutoEncoder(BreakthroughModel):
    def __init__(self, encoder, decoder):
        """encoder, decoder: nn.Sequential (or a list of modules) for each half.
 
        The encoder maps input -> bottleneck and the decoder maps bottleneck -> input,
        e.g. Linear(50, 35), ReLU, ..., Linear(20, 10), ReLU for the encoder.
        """
        super().__init__()
        self._n_encoder_modules = len(encoder)
        self.net = nn.Sequential(*encoder, *decoder)
 
    def forward(self, x):
        return self.net(x)
 
    def train_model(self, X, optimizer, loss_fn, epochs, batch_size=256,
                    verbose=False, pbar=None, pbar_inc=1):
        """Train the autoencoder to reconstruct X (a tensor of shape [N, input_dim]).
 
        X should already be on the device you want to train on; the model is moved
        to X.device. optimizer and loss_fn are supplied by the caller. If verbose and
        pbar (a caller-owned tqdm bar) is given, it is advanced by pbar_inc per epoch
        and its postfix shows that epoch's mean training loss.
        """
        self.to(X.device)
        self.train()
        n = len(X)
 
        for _ in range(epochs):
            perm = torch.randperm(n, device=X.device)
            total = torch.zeros((), device=X.device)
            for i in range(0, n, batch_size):
                xb = X[perm[i:i + batch_size]]
 
                optimizer.zero_grad()
                batch_loss = loss_fn(self(xb), xb)  # target is the input itself
                batch_loss.backward()
                optimizer.step()
                total += batch_loss.detach() * len(xb)  # stays on device, no per-batch sync
 
            if verbose and pbar is not None:
                pbar.set_postfix(loss=f"{total.item() / n:.6f}")  # one sync per epoch
                pbar.update(pbar_inc)

 
    def extract_encoder(self):
        """Return the encoder half (input -> bottleneck) as a standalone nn.Sequential.
 
        It shares weights with this model, so further training of the
        autoencoder also updates the returned encoder. Use copy.deepcopy
        on the result if you want an independent copy.
        """
        return nn.Sequential(*list(self.net.children())[:self._n_encoder_modules])




class BreakthroughCNN(BreakthroughModel):
    def __init__(self, n_extra: int = 0, board_size: int = 5):
        super().__init__()
        self.board_size = board_size
        self.net = nn.Sequential(
            # 2 board channels + one constant channel per extra feature
            nn.Conv2d(2 + n_extra, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # (N, 64, b, b) -> (N, 64 * b * b)
            nn.Linear(64 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def preprocess(self, x):
        b, n_board = self.board_size, board_features(self.board_size)
        x = x.reshape(len(x), -1)                                    # (N, 2 b^2 + n_extra)
        board = x[:, :n_board].reshape(-1, 2, b, b)                  # positions back to a board
        planes = x[:, n_board:, None, None].expand(-1, -1, b, b)     # extra feature -> constant b x b plane
        return torch.cat([board, planes], dim=1)                     # (N, 2 + n_extra, b, b)


def get_breakthrough_cnn(X, board_size: int = 5):
    """Build a CNN whose channel count matches X: (N, 2, b, b), (N, 2 b^2) or (N, 2 b^2 + k) for a b x b board."""
    return BreakthroughCNN(n_extra=X.reshape(len(X), -1).shape[1] - board_features(board_size),
                           board_size=board_size)

class BreakthroughResNet18(BreakthroughModel):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1) # linear layer finally to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet18(x).squeeze(-1)
        

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        # two layers, see https://arxiv.org/html/1512.03385v1#S3.E2
        

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)

class BreakthroughResNet(BreakthroughModel):
    preprocess = BreakthroughCNN.preprocess          # same flat-features -> board conversion (uses self.board_size)

    def __init__(self, n_extra=0, channels=32, blocks=2, board_size=5,
                 head_ch=64, hidden=115):
        # hidden=115 gives the same parameter count as BreakthroughCNN on 5x5 (224,167 vs 224,161);
        # on 6x6 use hidden=119 (314,351 vs 314,273), see tests/test_models_board_size.py
        super().__init__()
        self.board_size = board_size
        in_ch = 2 + n_extra
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),     # stem
            nn.BatchNorm2d(channels), nn.ReLU(),
            *[ResBlock(channels) for _ in range(blocks)],             # tower
            nn.Conv2d(channels, head_ch, 1, bias=False),              # value head
            nn.BatchNorm2d(head_ch), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_ch * board_size * board_size, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )