import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze()  # (N,1) -> (N)

class BreakthroughCNN(nn.Module):
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

    def forward(self, x):
        x = x.reshape(len(x), -1)                             # (N, 50 + n_extra)
        board = x[:, :50].reshape(-1, 2, 5, 5)                # positions back to a board
        planes = x[:, 50:, None, None].expand(-1, -1, 5, 5)   # each extra feature -> constant 5x5 plane
        x = torch.cat([board, planes], dim=1)                 # (N, 2 + n_extra, 5, 5)
        return self.net(x).squeeze(-1)                        # (N, 1) -> (N)


def get_breakthrough_cnn(X):
    """Build a CNN whose channel count matches X: (N, 2, 5, 5), (N, 50), (N, 52) or (N, 54)."""
    return BreakthroughCNN(n_extra=X.reshape(len(X), -1).shape[1] - 50)
    
def get_breakthrough_resnet():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # replace with 2 channels
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # fc = final fully connected layer, maps the 2048 pooled features to the output
    # default is Linear(2048, 1000) for the ImageNet classes; replace with a single value
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Flatten(0),  # (N,1) -> (N) to match Y
    )
    return model
