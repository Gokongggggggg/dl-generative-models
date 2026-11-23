"""Full evaluation: Test MSE/PSNR/SSIM, latency (batch 64), FID, Figures 1-5, summary.

Run:  python -m experiments.exp5_evaluate
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from configs import config
from models.autoencoder import VAE, Autoencoder
from models.dcgan import DCGANGenerator
from models.ddpm import GaussianDiffusion, UNet
from models.gan import Generator
from models.mlp import MLPClassifier, MLPDecoder
from utils.common import load_checkpoint
from utils.dataset import denorm, get_loaders
from utils.metrics import FIDAccumulator, measure_latency, reconstruction_metrics
from utils.seed import set_seed
from utils.visualize import panel_grid, save_fig, to_img


def load_models(device):
    m = {}
    m["mlp"] = load_checkpoint(MLPClassifier(), "mlp_classifier", device).eval()
    m["mlp_dec"] = load_checkpoint(MLPDecoder(), "mlp_frozen_decoder", device).eval()
    m["vanilla_ae"] = load_checkpoint(Autoencoder(), "vanilla_ae", device).eval()
    m["sparse_ae"] = load_checkpoint(Autoencoder(), "sparse_ae", device).eval()
    m["vae"] = load_checkpoint(VAE(), "vae", device).eval()
    m["vanilla_gan"] = load_checkpoint(Generator(config.GAN_Z_DIM), "vanilla_gan_G", device).eval()
    m["dcgan"] = load_checkpoint(DCGANGenerator(config.GAN_Z_DIM), "dcgan_G", device).eval()
    ddpm = GaussianDiffusion(UNet(config.DDPM_BASE_CH), config.DDPM_TIMESTEPS,
                             config.DDPM_BETA_START, config.DDPM_BETA_END)
    m["ddpm"] = load_checkpoint(ddpm, "ddpm", device).eval()
    stats = torch.load(config.CKPT_DIR / "mlp_latent_stats.pt",
                       map_location=device, weights_only=True)
    m["mlp_mu"], m["mlp_std"] = stats["mu"], stats["std"]
    for model in m.values():
        if isinstance(model, torch.nn.Module):
            for p in model.parameters():
                p.requires_grad = False
    return m


@torch.no_grad()
def recon_all(model_fn, test_loader, device):
    preds, targets = [], []
    for x, _ in test_loader:
        preds.append(model_fn(x.to(device)).cpu())
        targets.append(x)
    return torch.cat(preds), torch.cat(targets)


@torch.no_grad()
def generate(model_name, models, n, device, seed=config.SEED):
    """n prior samples in [-1,1]."""
    g = torch.Generator(device=device).manual_seed(seed)
    outs = []
    done = 0
    while done < n:
        b = min(128, n - done)
        if model_name == "vae":
            z = torch.randn(b, config.LATENT_DIM, device=device, generator=g)
            out = models["vae"].decode(z)
        elif model_name in ("vanilla_gan", "dcgan"):
            z = torch.randn(b, config.GAN_Z_DIM, device=device, generator=g)
            out = models[model_name](z)
        elif model_name == "ddpm":
            torch.manual_seed(seed + done)  # p_sample_loop uses global RNG
            out = models["ddpm"].p_sample_loop((b, 1, 28, 28), device)
        else:
            raise ValueError(model_name)
        outs.append(out.cpu())
        done += b
    return torch.cat(outs)[:n]


def epoch_time(name):
    path = config.LOG_DIR / f"{name}_losses.json"
    if not path.exists():
        return None
    h = json.load(open(path))
    return float(np.mean(h["epoch_seconds"]))


def compute_metrics(models, test_loader, device):
    rows = {}
    x_test = torch.cat([x for x, _ in test_loader])
    y_test = torch.cat([y for _, y in test_loader])

    # --- Reconstruction metrics (test set, [0,1] domain) ---
    recon_fns = {
        "Vanilla AE": lambda x: models["vanilla_ae"](x)[0],
        "Sparse AE": lambda x: models["sparse_ae"](x)[0],
        "VAE": lambda x: models["vae"].decode(models["vae"].encode(x)[0]),
        "Frozen MLP Decoder": lambda x: models["mlp_dec"](models["mlp"].latent(x)),
    }
    recon_cache = {}
    for name, fn in recon_fns.items():
        pred, tgt = recon_all(fn, test_loader, device)
        recon_cache[name] = pred
        mse, psnr, ssim = reconstruction_metrics(pred, tgt)
        rows[name] = {"test_mse": mse, "test_psnr_db": psnr, "test_ssim": ssim}
        print(f"[recon] {name}: MSE={mse:.5f} PSNR={psnr:.2f} SSIM={ssim:.4f}")

    # --- Latent dim + training time ---
    meta = {
        "Vanilla AE": (64, "vanilla_ae"),
        "Sparse AE": (64, "sparse_ae"),
        "VAE": (64, "vae"),
        "Vanilla GAN": (100, "vanilla_gan"),
        "DCGAN": (100, "dcgan"),
        "DDPM": (784, "ddpm"),
        "Frozen MLP Decoder": (64, "mlp_frozen_decoder"),
    }
    for name, (dim, log_name) in meta.items():
        rows.setdefault(name, {})
        rows[name]["latent_dim"] = dim
        rows[name]["train_time_per_epoch_s"] = epoch_time(log_name)

    # --- Inference latency, batch of 64 ---
    z64 = torch.randn(config.LATENCY_BATCH, config.LATENT_DIM, device=device)
    z100 = torch.randn(config.LATENCY_BATCH, config.GAN_Z_DIM, device=device)
    x64 = x_test[:config.LATENCY_BATCH].to(device)
    lat_fns = {
        "Vanilla AE": lambda b: models["vanilla_ae"](x64),
        "Sparse AE": lambda b: models["sparse_ae"](x64),
        "VAE": lambda b: models["vae"](x64),
        "Vanilla GAN": lambda b: models["vanilla_gan"](z100),
        "DCGAN": lambda b: models["dcgan"](z100),
        "Frozen MLP Decoder": lambda b: models["mlp_dec"](models["mlp"].latent(x64)),
        "DDPM": lambda b: models["ddpm"].p_sample_loop(
            (config.LATENCY_BATCH, 1, 28, 28), device),
    }
    for name, fn in lat_fns.items():
        repeats = 5 if name == "DDPM" else config.LATENCY_REPEATS
        mean_ms, std_ms = measure_latency(fn, repeats=repeats)
        rows[name]["latency_ms_batch64"] = mean_ms
        rows[name]["latency_std_ms"] = std_ms
        print(f"[latency] {name}: {mean_ms:.2f} +/- {std_ms:.2f} ms")

    # --- FID: 5000 real test vs 5000 generated ---
    real = x_test[:config.FID_NUM_SAMPLES].to(device)
    for name in ["VAE", "Vanilla GAN", "DCGAN", "DDPM"]:
        key = {"VAE": "vae", "Vanilla GAN": "vanilla_gan",
               "DCGAN": "dcgan", "DDPM": "ddpm"}[name]
        samples = generate(key, models, config.FID_NUM_SAMPLES, device)
        # fresh accumulator per model, re-adding the same real batch
        acc = FIDAccumulator(device)
        for i in range(0, len(real), 250):
            acc.update_real(real[i:i + 250])
        for i in range(0, len(samples), 250):
            acc.update_fake(samples[i:i + 250].to(device))
        fid = acc.compute()
        rows[name]["fid"] = fid
        print(f"[FID] {name}: {fid:.2f}")
        del acc
        torch.cuda.empty_cache()

    for name in rows:
        rows[name].setdefault("test_mse", None)
        rows[name].setdefault("test_psnr_db", None)
        rows[name].setdefault("test_ssim", None)
        rows[name].setdefault("fid", None)

    return rows, recon_cache, x_test, y_test


# ----------------------------- Figures -----------------------------

def figure1(models, recon_cache, x_test):
    n = 8
    fig = plt.figure(figsize=(12, 6.5))
    outer = GridSpec(4, 1, figure=fig, hspace=0.35)
    panel_grid(fig, outer[0], x_test[:n], 1, n, "Row 1: Ground Truth (Test Set)")
    for i, name in enumerate(["Vanilla AE", "Sparse AE", "VAE"], start=1):
        panel_grid(fig, outer[i], recon_cache[name][:n], 1, n,
                   f"Row {i + 1}: {name} Reconstruction")
    fig.suptitle("Figur 1: Reconstruction Comparison (Test Set)", fontsize=14)
    save_fig(fig, "Figur1_Reconstruction_Comparison.png")


def figure2(models, device):
    fig = plt.figure(figsize=(8, 16))
    outer = GridSpec(4, 1, figure=fig, hspace=0.25)
    panels = [
        ("Panel A: VAE  (z ~ N(0, I))", "vae"),
        ("Panel B: Vanilla GAN  (z ~ N(0, I))", "vanilla_gan"),
        ("Panel C: DCGAN  (z ~ N(0, I))", "dcgan"),
        ("Panel D: DDPM  (x_T ~ N(0, I), T=300 reverse steps)", "ddpm"),
    ]
    for i, (title, key) in enumerate(panels):
        samples = generate(key, models, 16, device)
        panel_grid(fig, outer[i], samples, 4, 4, title)
    fig.suptitle("Figur 2: Random Generative Sampling (16 samples per model)", fontsize=14)
    save_fig(fig, "Figur2_Random_Generative_Sampling.png")


def figure3(models, x_test, y_test, device):
    n = 8
    idx = torch.arange(n)
    x8 = x_test[idx].to(device)
    with torch.no_grad():
        z8 = models["mlp"].latent(x8)
        y_oh = torch.eye(10, device=device)[y_test[idx].to(device)]
        transpose_class = models["mlp"].transpose_from_class(y_oh).cpu()
        transpose_latent = models["mlp"].transpose_from_latent(z8).cpu()
        frozen_recon = models["mlp_dec"](z8).cpu()
        ae_recon = models["vanilla_ae"](x8)[0].cpu()
        g = torch.Generator(device=device).manual_seed(config.SEED)
        z_rand = models["mlp_mu"] + models["mlp_std"] * torch.randn(
            n, 64, device=device, generator=g)
        prior_samples = models["mlp_dec"](z_rand).cpu()

    fig = plt.figure(figsize=(12, 11))
    outer = GridSpec(4, 1, figure=fig, hspace=0.4)
    panel_grid(fig, outer[0], x_test[:n], 1, n, "(1) Ground Truth")
    panel_grid(fig, outer[1], torch.cat([transpose_class, transpose_latent]), 2, n,
               "(2) Direct Weight Transpose W^T (no training)",
               row_labels=["from class y_c", "from z_MLP"])
    panel_grid(fig, outer[2], torch.cat([frozen_recon, ae_recon]), 2, n,
               "(3) Trained Decoder on Frozen Latents vs Vanilla AE",
               row_labels=["Frozen z_MLP dec.", "Vanilla AE z_AE"])
    panel_grid(fig, outer[3], prior_samples, 1, n,
               "(4) Prior Sampling z_rand ~ N(mu_z, diag(sigma_z^2)) -> Frozen Decoder")
    fig.suptitle("Figur 3: MLP Classifier Limitation Proof", fontsize=14)
    save_fig(fig, "Figur3_MLP_Limitation_Proof.png")


def find_contrasting_z(models, clf, key, device, class_a=1, class_b=7, seed=config.SEED):
    """Search random z whose generated samples are confidently class_a / class_b."""
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1024, config.GAN_Z_DIM, device=device, generator=g)
    with torch.no_grad():
        imgs = torch.cat([models[key](z[i:i + 128]) for i in range(0, 1024, 128)])
        probs = F.softmax(clf(imgs), dim=1)
    conf, pred = probs.max(1)
    za = zb = None
    for cls, target in ((class_a, "a"), (class_b, "b")):
        mask = (pred == cls) & (conf > 0.9)
        if mask.any():
            cand = mask.nonzero().flatten()
            best = cand[probs[cand, cls].argmax()].item()
            if target == "a":
                za = z[best]
            else:
                zb = z[best]
    if za is None:
        za = z[(pred == class_a).nonzero()[0]] if (pred == class_a).any() else z[0]
    if zb is None:
        zb = z[(pred == class_b).nonzero()[0]] if (pred == class_b).any() else z[1]
    return za, zb


def figure4(models, x_test, y_test, device):
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # contrasting test images: Trouser (1) vs Sneaker (7)
    ia = (y_test == 1).nonzero()[0].item()
    ib = (y_test == 7).nonzero()[0].item()
    xa = x_test[ia:ia + 1].to(device)
    xb = x_test[ib:ib + 1].to(device)

    fig = plt.figure(figsize=(10, 10))
    outer = GridSpec(5, 1, figure=fig, hspace=0.35)
    ae_keys = [("Vanilla AE", "vanilla_ae"), ("Sparse AE", "sparse_ae"), ("VAE", "vae")]
    for r, (label, key) in enumerate(ae_keys):
        with torch.no_grad():
            model = models[key]
            if key == "vae":
                za, zb = model.encode(xa)[0][0], model.encode(xb)[0][0]
                imgs = [model.decode((1 - a) * za + a * zb)[0] for a in alphas]
            else:
                za, zb = model.encode(xa)[0], model.encode(xb)[0]
                imgs = [model.decode((1 - a) * za + a * zb)[0] for a in alphas]
        panel_grid(fig, outer[r], torch.stack(imgs), 1, 6,
                   f"{label}: z(a) = (1-a) z_A + a z_B   "
                   f"[A={config.CLASS_NAMES[1]}, B={config.CLASS_NAMES[7]}]")
    for r, (label, key) in enumerate([("Vanilla GAN", "vanilla_gan"), ("DCGAN", "dcgan")], start=3):
        za, zb = find_contrasting_z(models, models["mlp"], key, device)
        with torch.no_grad():
            imgs = [models[key](((1 - a) * za + a * zb).unsqueeze(0))[0] for a in alphas]
        panel_grid(fig, outer[r], torch.stack(imgs), 1, 6,
                   f"{label}: linear interpolation in z ~ N(0, I) "
                   f"(endpoints classified as {config.CLASS_NAMES[1]} / {config.CLASS_NAMES[7]})")
    # alpha labels on top row
    fig.suptitle("Figur 4: Latent Interpolation  (alpha = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)",
                 fontsize=14)
    save_fig(fig, "Figur4_Latent_Interpolation.png")


def figure5(models, device):
    fig = plt.figure(figsize=(16, 10))
    outer = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

    def load_hist(name):
        return json.load(open(config.LOG_DIR / f"{name}_losses.json"))

    def curve(ax, hist, keys, title):
        for k, lbl in keys:
            ax.plot(hist["epochs"], hist[k], label=lbl)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    curve(fig.add_subplot(outer[0, 0]), load_hist("vanilla_ae"),
          [("train_mse", "MSE")], "Vanilla AE")
    h = load_hist("sparse_ae")
    curve(fig.add_subplot(outer[0, 1]), h,
          [("train_mse", "MSE"), ("train_l1", "||h||_1 (per-sample L1 norm)")],
          "Sparse AE (lambda=1e-4)")
    curve(fig.add_subplot(outer[0, 2]), load_hist("vae"),
          [("train_recon_sum", "Recon (sum MSE)"), ("train_kl", "KL")], "VAE")
    curve(fig.add_subplot(outer[1, 0]), load_hist("vanilla_gan"),
          [("loss_g", "G loss"), ("loss_d", "D loss")], "Vanilla GAN")
    curve(fig.add_subplot(outer[1, 1]), load_hist("dcgan"),
          [("loss_g", "G loss"), ("loss_d", "D loss")], "DCGAN")
    curve(fig.add_subplot(outer[1, 2]), load_hist("ddpm"),
          [("train_eps_mse", "eps MSE")], "DDPM (T=300)")

    # DDPM reverse trajectory
    steps = [300, 250, 200, 150, 100, 60, 30, 10, 0]
    torch.manual_seed(config.SEED)
    with torch.no_grad():
        _, traj = models["ddpm"].p_sample_loop((1, 1, 28, 28), device,
                                               trajectory_idx=set(steps))
    inner_spec = outer[2, :]
    ax_t = fig.add_subplot(inner_spec, frame_on=False)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.set_title("DDPM Reverse Diffusion Trajectory:  x_T (pure noise) -> x_0",
                   fontsize=12, pad=12)
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    inner = GridSpecFromSubplotSpec(1, len(steps), subplot_spec=inner_spec, wspace=0.1)
    for i, s in enumerate(steps):
        ax = fig.add_subplot(inner[0, i])
        ax.imshow(to_img(traj[s][0]), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"t={s}" + (" (x_T)" if s == 300 else (" (x_0)" if s == 0 else "")),
                      fontsize=9)
    fig.suptitle("Figur 5: Training Loss Curves and DDPM Reverse Trajectory", fontsize=15)
    save_fig(fig, "Figur5_Loss_Curves_and_DDPM_Trajectory.png")


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    _, _, test_loader = get_loaders()
    models = load_models(device)

    rows, recon_cache, x_test, y_test = compute_metrics(models, test_loader, device)

    figure1(models, recon_cache, x_test)
    figure2(models, device)
    figure3(models, x_test, y_test, device)
    figure4(models, x_test, y_test, device)
    figure5(models, device)

    # Summary tables
    order = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN", "DDPM",
             "Frozen MLP Decoder"]
    cols = ["latent_dim", "train_time_per_epoch_s", "latency_ms_batch64",
            "test_mse", "test_psnr_db", "test_ssim", "fid"]
    df = pd.DataFrame([rows[k] for k in order], index=order)[cols]
    df.index.name = "model"
    df.to_csv(config.LOG_DIR / "final_metrics.csv")
    with open(config.LOG_DIR / "final_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(config.LOG_DIR / "summary_table.md", "w") as f:
        f.write("# Final Test-Set Metrics (7,000 images)\n\n")
        f.write(df.round(4).astype(object).where(df.notna(), "N/A").to_markdown())
        f.write("\n\nN/A: reconstruction metrics for GAN/DCGAN/DDPM (no encoder); "
                "FID for Vanilla/Sparse AE (not generative priors).\n")
    print(df.to_string())
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
