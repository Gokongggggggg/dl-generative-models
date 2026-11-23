"""Train Vanilla AE, Sparse AE (L1 on 64-d bottleneck), and VAE on the shared split.

Run:  python -m experiments.exp2_train_autoencoders
"""
import torch
import torch.nn.functional as F

from configs import config
from models.autoencoder import VAE, Autoencoder
from utils.common import LossLogger, save_checkpoint
from utils.dataset import get_loaders
from utils.seed import set_seed


def train_ae(name, sparse_lambda, train_loader, device):
    model = Autoencoder(config.LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.AE_LR)
    logger = LossLogger(name)
    for epoch in range(1, config.AE_EPOCHS + 1):
        model.train()
        logger.start_epoch()
        run_recon = run_l1 = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, h = model(x)
            recon = F.mse_loss(x_hat, x)
            l1 = h.abs().sum(1).mean()  # lambda * ||h||_1, per-sample L1 norm
            loss = recon + sparse_lambda * l1
            loss.backward()
            opt.step()
            run_recon += recon.item()
            run_l1 += l1.item()
        n = len(train_loader)
        dt = logger.end_epoch(epoch, train_mse=run_recon / n, train_l1=run_l1 / n)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:3d} mse={run_recon / n:.5f} "
                  f"l1={run_l1 / n:.4f} ({dt:.1f}s)")
    save_checkpoint(model, name)
    return model


def train_vae(train_loader, device):
    model = VAE(config.LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.VAE_LR)
    logger = LossLogger("vae")
    for epoch in range(1, config.VAE_EPOCHS + 1):
        model.train()
        logger.start_epoch()
        run_recon = run_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = VAE.loss_fn(x_hat, x, mu, logvar)
            loss.backward()
            opt.step()
            run_recon += recon.item()
            run_kl += kl.item()
        n = len(train_loader)
        dt = logger.end_epoch(epoch, train_recon_sum=run_recon / n, train_kl=run_kl / n)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[vae] epoch {epoch:3d} recon={run_recon / n:.2f} "
                  f"kl={run_kl / n:.2f} ({dt:.1f}s)")
    save_checkpoint(model, "vae")
    return model


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    train_loader, _, _ = get_loaders()
    train_ae("vanilla_ae", 0.0, train_loader, device)
    train_ae("sparse_ae", config.SPARSE_L1_LAMBDA, train_loader, device)
    train_vae(train_loader, device)
    print("Autoencoder family training complete.")


if __name__ == "__main__":
    main()
