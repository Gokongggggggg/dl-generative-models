"""Train DDPM (T=300, linear betas, U-Net eps-prediction, MSE).

Run:  python -m experiments.exp4_train_ddpm
"""
import torch

from configs import config
from models.ddpm import GaussianDiffusion, UNet
from utils.common import LossLogger, save_checkpoint
from utils.dataset import get_loaders
from utils.seed import set_seed


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    train_loader, _, _ = get_loaders()

    diffusion = GaussianDiffusion(
        UNet(config.DDPM_BASE_CH),
        timesteps=config.DDPM_TIMESTEPS,
        beta_start=config.DDPM_BETA_START,
        beta_end=config.DDPM_BETA_END,
    ).to(device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=config.DDPM_LR)
    logger = LossLogger("ddpm")

    for epoch in range(1, config.DDPM_EPOCHS + 1):
        diffusion.train()
        logger.start_epoch()
        run_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            loss = diffusion.p_losses(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
        n = len(train_loader)
        dt = logger.end_epoch(epoch, train_eps_mse=run_loss / n)
        print(f"[ddpm] epoch {epoch:3d} eps_mse={run_loss / n:.5f} ({dt:.1f}s)", flush=True)

    save_checkpoint(diffusion, "ddpm")
    print("DDPM training complete.")


if __name__ == "__main__":
    main()
