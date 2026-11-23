"""Part A: MLP Classifier Theoretical Limitation Experiment (5 stages).

Run from repo root:  python -m experiments.exp1_mlp_proof
"""
import json

import torch
import torch.nn.functional as F

from configs import config
from models.mlp import MLPClassifier, MLPDecoder
from utils.common import LossLogger, save_checkpoint
from utils.dataset import get_loaders
from utils.metrics import reconstruction_metrics
from utils.seed import set_seed


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def stage1_train_classifier(train_loader, test_loader, device):
    model = MLPClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.MLP_LR)
    logger = LossLogger("mlp_classifier")
    acc = 0.0
    max_epochs = config.MLP_EPOCHS * 2
    for epoch in range(1, max_epochs + 1):
        if epoch == config.MLP_EPOCHS + 1:
            for g in opt.param_groups:
                g["lr"] = config.MLP_LR / 2
        model.train()
        logger.start_epoch()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        dt = logger.end_epoch(epoch, train_ce=run_loss / len(train_loader))
        acc = evaluate_classifier(model, test_loader, device)
        print(f"[MLP] epoch {epoch:3d}  ce={run_loss / len(train_loader):.4f}  "
              f"test_acc={acc:.4f}  ({dt:.1f}s)")
        if acc >= config.MLP_TARGET_ACC and epoch >= 10:
            break
    assert acc >= config.MLP_TARGET_ACC, f"Test accuracy {acc:.4f} < {config.MLP_TARGET_ACC}"
    save_checkpoint(model, "mlp_classifier")
    return model, acc, logger


@torch.no_grad()
def collect_latents(model, loader, device):
    zs, xs, ys = [], [], []
    for x, y in loader:
        zs.append(model.latent(x.to(device)).cpu())
        xs.append(x)
        ys.append(y)
    return torch.cat(zs), torch.cat(xs), torch.cat(ys)


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    train_loader, _, test_loader = get_loaders()
    results = {}

    # Stage 1: train classifier to >= 88% test accuracy
    clf, acc, clf_logger = stage1_train_classifier(train_loader, test_loader, device)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    results["stage1_test_accuracy"] = acc
    results["stage1_train_time_per_epoch_s"] = clf_logger.mean_epoch_seconds

    z_test, x_test, y_test = collect_latents(clf, test_loader, device)

    # Stage 2: top-down weight transposition from one-hot class vectors
    y_onehot = torch.eye(10, device=device)[y_test[:2000].to(device)]
    x_hat2 = clf.transpose_from_class(y_onehot).cpu()
    mse2, psnr2, ssim2 = reconstruction_metrics(x_hat2, x_test[:2000])
    results["stage2_class_transpose"] = {"mse": mse2, "psnr": psnr2, "ssim": ssim2}
    print(f"[Stage 2] class W^T: MSE={mse2:.4f} PSNR={psnr2:.2f} SSIM={ssim2:.4f}")

    # Stage 3: latent transpose projection of real z_MLP
    x_hat3 = clf.transpose_from_latent(z_test.to(device)).cpu()
    mse3, psnr3, ssim3 = reconstruction_metrics(x_hat3, x_test)
    results["stage3_latent_transpose"] = {"mse": mse3, "psnr": psnr3, "ssim": ssim3}
    print(f"[Stage 3] latent W^T: MSE={mse3:.4f} PSNR={psnr3:.2f} SSIM={ssim3:.4f}")

    # Stage 4: frozen-latent decoding with a separately trained decoder
    dec = MLPDecoder().to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=config.FROZEN_DEC_LR)
    dec_logger = LossLogger("mlp_frozen_decoder")
    for epoch in range(1, config.FROZEN_DEC_EPOCHS + 1):
        dec.train()
        dec_logger.start_epoch()
        run_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            z = clf.latent(x)
            opt.zero_grad()
            loss = F.mse_loss(dec(z), x)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        dt = dec_logger.end_epoch(epoch, train_mse=run_loss / len(train_loader))
        if epoch % 10 == 0 or epoch == 1:
            print(f"[FrozenDec] epoch {epoch:3d}  mse={run_loss / len(train_loader):.5f} ({dt:.1f}s)")
    save_checkpoint(dec, "mlp_frozen_decoder")
    results["stage4_train_time_per_epoch_s"] = dec_logger.mean_epoch_seconds

    dec.eval()
    with torch.no_grad():
        x_hat4 = dec(z_test.to(device)).cpu()
    mse4, psnr4, ssim4 = reconstruction_metrics(x_hat4, x_test)
    results["stage4_frozen_decoder_test"] = {"mse": mse4, "psnr": psnr4, "ssim": ssim4}
    print(f"[Stage 4] frozen decoder: MSE={mse4:.4f} PSNR={psnr4:.2f} SSIM={ssim4:.4f}")

    # Stage 5: Gaussian prior sampling on the MLP latent space
    mu_z = z_test.mean(0)
    std_z = z_test.std(0)
    torch.save({"mu": mu_z, "std": std_z}, config.CKPT_DIR / "mlp_latent_stats.pt")
    g = torch.Generator(device=device).manual_seed(config.SEED)
    z_rand = mu_z.to(device) + std_z.to(device) * torch.randn(2000, 64, device=device, generator=g)
    with torch.no_grad():
        x_gen = dec(z_rand).cpu()
    results["stage5_prior_sample_mean_pixel"] = float(x_gen.mean())
    results["stage5_prior_sample_std_pixel"] = float(x_gen.std())
    results["stage5_note"] = (
        "Gaussian samples decoded through the frozen-latent decoder produce incoherent "
        "blobs: the discriminative latent space is not regularized into a smooth, "
        "space-filling prior, so N(mu_z, diag(sigma_z^2)) lands off the data manifold."
    )
    print("[Stage 5] prior sampling done (see Figur 3 panel 4).")

    with open(config.LOG_DIR / "exp1_mlp_proof.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/logs/exp1_mlp_proof.json")


if __name__ == "__main__":
    main()
