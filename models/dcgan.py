"""DCGAN adapted for 28x28 grayscale: 7x7 -> 14x14 -> 28x28 feature maps."""
import torch.nn as nn


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.project = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0, bias=False),  # 1x1 -> 7x7
            nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # 7 -> 14
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),     # 14 -> 28
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),                 # 28 -> 28
        )

    def forward(self, z):
        h = self.project(z.view(-1, z.shape[1], 1, 1))
        return self.net(h)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),                            # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),              # 14 -> 7
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),             # 7 -> 7
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(256 * 7 * 7, 1)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.head(h).squeeze(1)


def dcgan_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
