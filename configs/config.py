"""Central configuration for all experiments (D6421 take-home project)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
DEVICE = "cuda"

# Data
DATA_DIR = ROOT / "data"
BATCH_SIZE = 128
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1  # 56k / 7k / 7k of 70k
NUM_WORKERS = 0  # Windows: avoid worker spawn overhead / nondeterminism

# Paths
RESULTS_DIR = ROOT / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

# Latent / noise dimensions
LATENT_DIM = 64          # AE / Sparse AE / VAE / MLP penultimate
GAN_Z_DIM = 100          # Vanilla GAN + DCGAN prior

# Part A: MLP limitation proof
MLP_EPOCHS = 40
MLP_LR = 1e-3
MLP_TARGET_ACC = 0.88
FROZEN_DEC_EPOCHS = 50
FROZEN_DEC_LR = 1e-3

# Part B models
AE_EPOCHS = 50
AE_LR = 1e-3
SPARSE_L1_LAMBDA = 1e-4   # L1 penalty on the 64-d bottleneck activations
VAE_EPOCHS = 50
VAE_LR = 1e-3

GAN_EPOCHS = 100
GAN_LR = 2e-4
DCGAN_EPOCHS = 80
DCGAN_LR = 2e-4
GAN_BETAS = (0.5, 0.999)

DDPM_EPOCHS = 60
DDPM_LR = 2e-4
DDPM_TIMESTEPS = 300
DDPM_BETA_START = 1e-4
DDPM_BETA_END = 0.02
DDPM_BASE_CH = 64

# Evaluation
FID_NUM_SAMPLES = 5000    # real (first 5000 of test) vs 5000 generated
LATENCY_BATCH = 64
LATENCY_WARMUP = 10
LATENCY_REPEATS = 50
FIG_DPI = 300

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

for d in (CKPT_DIR, FIG_DIR, LOG_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
