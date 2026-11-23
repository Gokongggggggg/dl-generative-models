# D6421 Deep Learning Take-Home Project
## Comparative Study of Generative Models & MLP Classifier Limitation Proof (FashionMNIST)

Empirical comparison of **6 generative architectures** (Vanilla AE, Sparse AE, VAE,
Vanilla GAN, DCGAN, DDPM) plus a **5-stage experiment proving why a supervised MLP
classifier cannot directly reconstruct or generate images**, all implemented in
PyTorch under one identical data split and seed.

---

## 1. Hardware & Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, CUDA 11.8) |
| OS | Windows 10/11 (win32, build 26200) |
| Python | 3.13.4 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| torchmetrics | 1.9.0 (FID, InceptionV3 feature 2048) |

## 2. Setup

```bash
pip install -r requirements.txt
```

FashionMNIST downloads automatically on first run into `data/` (git-ignored).
The FID Inception weights download automatically on first evaluation.

## 3. Reproducibility

`utils/seed.py::set_seed(42)` fixes `random`, `numpy`, `torch`, `torch.cuda`
(all devices) and sets `torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`. DataLoader shuffling uses a dedicated
`torch.Generator` seeded with 42; all sampling routines use seeded generators.

## 4. Dataset & Preprocessing (identical for every model)

- `torchvision.datasets.FashionMNIST`, 28x28 grayscale, 10 classes.
- Original train (60k) + test (10k) merged into **70,000** images.
- `transforms.ToTensor()` + `transforms.Normalize((0.5,), (0.5,))` -> pixels in **[-1, 1]**.
- Deterministic `torch.randperm(70000, generator=seed 42)` split:
  **Train 56,000 (shuffle=True) / Validation 7,000 / Test 7,000 (shuffle=False)**.
- Batch size **128**. All final metrics are computed strictly on the 7,000-image test split.
- For speed, the transformed tensors are materialized once and cached
  (`data/fmnist_tensors.pt`); values are bit-identical to per-sample transforms.

## 5. Repository Layout

```
configs/config.py            # all hyperparameters & paths (single source of truth)
models/                      # mlp.py, autoencoder.py (AE+VAE), gan.py, dcgan.py, ddpm.py
utils/                       # seed.py, dataset.py, metrics.py, common.py, visualize.py
experiments/                 # exp1..exp5 (see below)
results/checkpoints/         # saved model weights (.pt)
results/figures/             # Figur 1-5, 300 DPI PNG
results/logs/                # per-epoch loss JSONs, final metrics JSON/CSV/Markdown
```

## 6. Execution (run from repo root, in order)

```bash
python -m experiments.exp1_mlp_proof            # Part A: 5-stage MLP limitation proof
python -m experiments.exp2_train_autoencoders   # Vanilla AE, Sparse AE, VAE
python -m experiments.exp3_train_gans           # Vanilla GAN + DCGAN
python -m experiments.exp4_train_ddpm           # DDPM (T=300)
python -m experiments.exp5_evaluate             # metrics + Figur 1-5 + summary tables
```

Each script is deterministic (seed 42) and writes checkpoints/loss logs incrementally.

## 7. Model Specifications

| Model | Architecture | Latent dim | Loss | Epochs |
|---|---|---|---|---|
| MLP classifier (Part A) | 784-256-64-10, ReLU | z_MLP = 64 | Cross-Entropy | until test acc >= 88% |
| Frozen MLP decoder | 64-256-784, Tanh | 64 | MSE (classifier frozen) | 50 |
| Vanilla AE | enc 784-256-64 / dec 64-256-784, Tanh | 64 | MSE | 50 |
| Sparse AE | same as Vanilla AE + **L1 penalty lambda=1e-4 on the 64-d bottleneck activations** | 64 | MSE + 1e-4 * \|\|h\|\|_1 (per-sample L1 norm) | 50 |
| VAE | enc 784-256-(mu 64, logvar 64), reparam., dec 64-256-784 Tanh | 64 | sum-MSE + KL(q(z\|x) \|\| N(0,I)), beta=1 | 50 |
| Vanilla GAN | G 100-256-512-784 Tanh; D 784-512-256-1 LeakyReLU(0.2) | z = 100 | BCEWithLogits (non-saturating G) | 100 |
| DCGAN | G: z100 -> 256x7x7 -> ConvT 14x14 -> ConvT 28x28 -> Conv, BN+ReLU; D: Conv 28->14->7, BN+LeakyReLU(0.2), linear head | z = 100 | BCEWithLogits | 80 |
| DDPM | U-Net (28->14->7, ResBlocks + GroupNorm, sinusoidal time emb., 1.78M params), eps-prediction | x_T = 784 | MSE(eps_theta, eps) | 60 |

DDPM schedule: **T = 300**, linear beta from 1e-4 to 0.02; full ancestral reverse
loop (x_T -> x_0) with posterior variance and clamping to [-1, 1].
Adam everywhere; GANs use lr 2e-4, betas (0.5, 0.999); DCGAN uses standard
N(0, 0.02) conv init.

## 8. Part A — MLP Limitation Proof (5 stages)

1. **Train classifier** 784-256-64-10 to test accuracy >= 88%; extract penultimate z_MLP (R^64).
2. **Top-down weight transposition** (no biases, no training):
   `z_hat = ReLU(W3^T y_c)`, `h1_hat = ReLU(W2^T z_hat)`, `x_hat = Tanh(W1^T h1_hat)`.
3. **Latent transpose projection**: `x_hat = Tanh(W1^T ReLU(W2^T z_MLP))` on real test latents.
4. **Frozen latent decoding**: all classifier weights `requires_grad=False`; train a separate
   64-256-784 Tanh decoder with MSE; compare Test MSE/PSNR/SSIM against Vanilla AE.
5. **Generative prior sampling**: fit `mu_z, sigma_z` on test latents, sample
   `z_rand ~ N(mu_z, diag(sigma_z^2))`, decode with the Stage-4 decoder.

**Why it fails (empirically confirmed):** a discriminative latent space only needs to be
*class-separable*, not *invertible* or *space-filling*. ReLU discards sign information, so
W^T cannot invert the forward map (Stages 2-3 yield near-zero SSIM). A trained decoder can
partially invert the frozen features (Stage 4) but stays worse than an AE trained end-to-end
for reconstruction. And because nothing regularizes the aggregate posterior toward a smooth
prior, Gaussian samples fall off the latent manifold and decode to incoherent blobs (Stage 5).

## 9. Metrics Protocol

- **MSE / PSNR / SSIM**: computed on [0,1]-rescaled images over the full 7,000-image test set
  (SSIM: `skimage`, `data_range=1.0`, per-image mean). VAE reconstruction uses mu (deterministic).
- **Latency**: batch of 64, `torch.cuda.synchronize()` + `time.perf_counter()`, 10 warm-up
  iterations then 50 timed repeats (DDPM: 5 repeats of the *full* 300-step reverse loop).
- **FID**: `torchmetrics.image.fid.FrechetInceptionDistance(feature=2048)`, 5,000 real test
  images vs 5,000 generated samples, grayscale replicated to 3-channel RGB uint8 [0,255].
- **Training time / epoch**: mean wall-clock seconds per epoch from the training logs.

## 10. Results (Test Set, 7,000 images)

| Model | Latent dim | Train s/epoch | Latency ms (batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla AE | 64 | 3.80 | 0.95 | 0.0069 | 21.60 | 0.8208 | N/A |
| Sparse AE | 64 | 3.20 | 1.19 | 0.0074 | 21.33 | 0.8108 | N/A |
| VAE | 64 | 4.09 | 1.20 | 0.0118 | 19.28 | 0.7271 | 79.42 |
| Vanilla GAN | 100 | 7.26 | 0.82 | N/A | N/A | N/A | 120.09 |
| DCGAN | 100 | 15.04 | 4.38 | N/A | N/A | N/A | **11.53** |
| DDPM | 784 | 34.20 | 2824.72 | N/A | N/A | N/A | 39.28 |
| Frozen MLP Decoder | 64 | 3.06 | 1.00 | 0.0155 | 18.11 | 0.6860 | N/A |

N/A = not applicable: GAN/DCGAN/DDPM have no encoder (no reconstruction); Vanilla/Sparse AE
have no regularized prior (FID not meaningful). DDPM latency is the **full 300-step reverse
loop** for a batch of 64; all others are a single forward pass. Latency measured with
`torch.cuda.synchronize()` + `time.perf_counter()` after 10 warm-up iterations.

### Part A — MLP limitation proof (`results/logs/exp1_mlp_proof.json`)

| Stage | Result |
|---|---|
| 1. Classifier test accuracy | **88.44%** (>= 88% requirement met, epoch 10) |
| 2. W^T from one-hot class y_c | MSE 0.1756, PSNR 7.56 dB, SSIM **0.053** (noise-like) |
| 3. W^T from real z_MLP | MSE 0.4147, PSNR 3.82 dB, SSIM **0.057** (noise-like) |
| 4. Trained decoder on frozen z_MLP | MSE 0.0155, PSNR 18.11 dB, SSIM 0.686 — worse than Vanilla AE (0.821) |
| 5. z_rand ~ N(mu_z, diag(sigma_z^2)) | incoherent blobs, no object structure (Figur 3, panel 4) |

### Empirical takeaways

1. **Discriminative latents are not invertible by transposition.** Stages 2-3 give SSIM ~0.05:
   ReLU destroys sign information and the classifier never learned a pixel-space inverse,
   so W^T projections are meaningless as images.
2. **A decoder can partially rescue frozen features, but not fully.** Stage 4 (SSIM 0.686)
   beats transposition by far yet loses to the end-to-end Vanilla AE (0.821): the frozen
   z_MLP discarded reconstruction-irrelevant detail that the classifier did not need.
3. **No prior regularization => no generation.** Stage 5 Gaussian samples decode to blobs;
   only the VAE (KL) and GANs/DDPM (adversarial / denoising objectives) shape a usable prior.
4. **Reconstruction vs generation trade-off.** Vanilla AE reconstructs best (SSIM 0.821) but
   cannot sample; VAE samples coherently at some reconstruction cost (0.727); DCGAN achieves
   the best sample quality (FID 11.5); DDPM is second (39.3) with sharp, diverse samples but
   ~3 s per batch of 64 (300 sequential network evaluations) vs ~4 ms for DCGAN.
5. **Sparsity costs little reconstruction.** The L1-norm penalty (lambda=1e-4) shrinks
   ||h||_1 from ~130 to ~17 with SSIM 0.811 vs 0.821 — a mild, controlled trade-off.

Full precision values: `results/logs/final_metrics.csv` / `.json`,
`results/logs/summary_table.md`, per-epoch curves in `results/logs/*_losses.json`.

## 11. Figures (results/figures/, 300 DPI)

| File | Content |
|---|---|
| `Figur1_Reconstruction_Comparison.png` | 8 test images: GT / Vanilla AE / Sparse AE / VAE |
| `Figur2_Random_Generative_Sampling.png` | 4x4 prior samples: VAE, Vanilla GAN, DCGAN, DDPM |
| `Figur3_MLP_Limitation_Proof.png` | GT vs W^T transposes vs frozen decoder vs AE vs prior sampling |
| `Figur4_Latent_Interpolation.png` | z(alpha) = (1-alpha) z_A + alpha z_B, alpha in {0, .2, .4, .6, .8, 1}, 5 models |
| `Figur5_Loss_Curves_and_DDPM_Trajectory.png` | loss curves for all 6 models (G/D separate) + DDPM x_T -> x_0 trajectory |

## 12. Git History

Incremental commits per milestone (local repository, no remote):

```
a230dfc Scaffold project: configs, seeded utils, 80/10/10 FashionMNIST split, metrics helpers
6d0bfc4 Part A: MLP limitation proof, 5 stages (test acc 88.44%, W^T SSIM~0.05, frozen decoder SSIM 0.686)
8a0ee31 Train Vanilla AE, Sparse AE (L1-norm lambda=1e-4 on 64-d bottleneck), VAE (sum-MSE + KL)
0e74a5c Train Vanilla GAN (MLP, 100 epochs) and DCGAN (conv 28x28, 80 epochs) with BCEWithLogits
b9af92b Evaluation script + figures 1/3/4 layout; fix checkpoint device placement and label clipping
d4ed391 Train DDPM: U-Net + sinusoidal time emb, T=300 linear betas, 60 epochs
cff2a55 Fix DDPM sampler: clamp only final x_0 (per-step clamping truncated the noise distribution)
ba21fb6 Full evaluation: metrics, FID, Figur 1-5 @300DPI, summary tables
(+ final documentation commit adding this README)
```

Note on the DDPM fix: clamping intermediate x_t to [-1,1] at every reverse step truncates
the ~N(0,1) noise distribution at high t and collapses sample quality (FID 197); clamping
only the final x_0 restores correct ancestral sampling (FID 39.3).
