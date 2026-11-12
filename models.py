import torch
from torch import nn
from torch.nn import functional as F


class ToyNetwork(nn.Module):
    """version0: score f(x) = -E(x) for 2-D points."""

    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(),  # smooth activations -> smooth energy landscape
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


class Network(nn.Module):
    """version1: score f(x) = -E(x) for 3x128x128 images (a plain CNN "critic")."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)  # 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 32
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)  # 16
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=2)  # 8
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.drop(x)
        return self.fc2(x)


if __name__ == "__main__":
    net = Network()
    print(net(torch.randn(1, 3, 128, 128)).shape)
