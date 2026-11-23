"""Vanilla AE and Sparse AE (both 64-d bottleneck; sparse adds L1 penalty on h)."""
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x.flatten(1))

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Tanh(),
        )

    def encode(self, x):
        h = torch.relu(self.fc1(x.flatten(1)))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_fn(x_hat, x, mu, logvar):
        # sum over pixels/dims per image, mean over batch (beta = 1)
        recon = torch.nn.functional.mse_loss(x_hat, x, reduction="none").flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
        return recon + kl, recon, kl
