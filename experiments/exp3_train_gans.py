"""Train Vanilla (MLP) GAN and DCGAN with BCEWithLogitsLoss on the shared split.

Run:  python -m experiments.exp3_train_gans
"""
import torch
import torch.nn as nn

from configs import config
from models.dcgan import DCGANDiscriminator, DCGANGenerator, dcgan_init
from models.gan import Discriminator, Generator
from utils.common import LossLogger, save_checkpoint
from utils.dataset import get_loaders
from utils.seed import set_seed


def train_gan(name, G, D, epochs, train_loader, device):
    G, D = G.to(device), D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=config.GAN_LR, betas=config.GAN_BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=config.GAN_LR, betas=config.GAN_BETAS)
    bce = nn.BCEWithLogitsLoss()
    logger = LossLogger(name)
    z_dim = config.GAN_Z_DIM
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        logger.start_epoch()
        run_g = run_d = 0.0
        for x, _ in train_loader:
            b = x.size(0)
            x = x.to(device)
            real_t = torch.ones(b, device=device)
            fake_t = torch.zeros(b, device=device)

            # Discriminator
            z = torch.randn(b, z_dim, device=device)
            with torch.no_grad():
                x_fake = G(z)
            opt_d.zero_grad()
            loss_d = bce(D(x), real_t) + bce(D(x_fake), fake_t)
            loss_d.backward()
            opt_d.step()

            # Generator
            z = torch.randn(b, z_dim, device=device)
            opt_g.zero_grad()
            loss_g = bce(D(G(z)), real_t)
            loss_g.backward()
            opt_g.step()

            run_g += loss_g.item()
            run_d += loss_d.item()
        n = len(train_loader)
        dt = logger.end_epoch(epoch, loss_g=run_g / n, loss_d=run_d / n)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:3d} loss_g={run_g / n:.4f} "
                  f"loss_d={run_d / n:.4f} ({dt:.1f}s)")
    save_checkpoint(G, f"{name}_G")
    save_checkpoint(D, f"{name}_D")


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    train_loader, _, _ = get_loaders()

    train_gan("vanilla_gan", Generator(config.GAN_Z_DIM), Discriminator(),
              config.GAN_EPOCHS, train_loader, device)

    G, D = DCGANGenerator(config.GAN_Z_DIM), DCGANDiscriminator()
    G.apply(dcgan_init)
    D.apply(dcgan_init)
    train_gan("dcgan", G, D, config.DCGAN_EPOCHS, train_loader, device)
    print("GAN training complete.")


if __name__ == "__main__":
    main()
