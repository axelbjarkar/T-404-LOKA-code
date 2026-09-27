import torch.nn as nn
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # padding=1 keeps the board 5x5 through both conv layers
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # (N, 64, 5, 5) -> (N, 1600)
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (N,1) -> (N)
    
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
