"""MLP classifier (784->256->64->10) and separate decoder (64->256->784)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        h1 = F.relu(self.fc1(x.flatten(1)))
        z = F.relu(self.fc2(h1))
        return self.fc3(z)

    @torch.no_grad()
    def latent(self, x):
        h1 = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(h1))  # z_MLP in R^64 (penultimate)

    def transpose_from_class(self, y_onehot):
        """Stage 2: top-down W^T projection of one-hot class vectors, no biases."""
        W1, W2, W3 = self.fc1.weight, self.fc2.weight, self.fc3.weight
        z_hat = F.relu(y_onehot @ W3)          # W3^T y_c
        h1_hat = F.relu(z_hat @ W2)            # W2^T z_hat
        return torch.tanh(h1_hat @ W1)         # W1^T h1_hat

    def transpose_from_latent(self, z):
        """Stage 3: direct W^T projection of real latent codes, no biases."""
        W1, W2 = self.fc1.weight, self.fc2.weight
        h1_hat = F.relu(z @ W2)
        return torch.tanh(h1_hat @ W1)


class MLPDecoder(nn.Module):
    """Stage 4 decoder trained against frozen z_MLP: 64->256->784, Tanh output."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)
