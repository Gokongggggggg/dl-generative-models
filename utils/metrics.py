"""Quantitative metrics: MSE / PSNR / SSIM on [0,1], latency, FID."""
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

from configs import config
from utils.dataset import denorm


def reconstruction_metrics(pred, target):
    """pred/target in [-1,1]; metrics computed on [0,1]. Returns (mse, psnr_db, ssim)."""
    p = denorm(pred).cpu().numpy()
    t = denorm(target).cpu().numpy()
    p = p.reshape(-1, 28, 28)
    t = t.reshape(-1, 28, 28)
    mse = float(np.mean((p - t) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = float(np.mean([
        structural_similarity(t[i], p[i], data_range=1.0) for i in range(len(p))
    ]))
    return mse, psnr, ssim


@torch.no_grad()
def measure_latency(fn, batch_size=config.LATENCY_BATCH,
                    warmup=config.LATENCY_WARMUP, repeats=config.LATENCY_REPEATS):
    """Mean latency in ms of fn(batch_size) with CUDA sync + perf_counter after warm-up."""
    for _ in range(warmup):
        fn(batch_size)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(batch_size)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def to_fid_uint8(x):
    """[-1,1] (B,1,28,28) -> uint8 RGB (B,3,28,28) in [0,255]."""
    x = (denorm(x) * 255.0).round().clamp(0, 255).to(torch.uint8)
    return x.repeat(1, 3, 1, 1)


class FIDAccumulator:
    def __init__(self, device):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def update_real(self, x):
        self.fid.update(to_fid_uint8(x), real=True)

    def update_fake(self, x):
        self.fid.update(to_fid_uint8(x), real=False)

    def compute(self):
        return float(self.fid.compute().item())
