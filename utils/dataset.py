"""FashionMNIST: merge train+test (70k), normalize to [-1,1], split 80/10/10 with seed 42."""
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from configs import config
from utils.seed import seed_worker, set_seed


def get_datasets():
    set_seed(config.SEED)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_part = datasets.FashionMNIST(config.DATA_DIR, train=True, download=True, transform=tf)
    test_part = datasets.FashionMNIST(config.DATA_DIR, train=False, download=True, transform=tf)
    full = ConcatDataset([train_part, test_part])  # 70,000 images
    assert len(full) == 70000

    n_train = int(config.TRAIN_FRAC * len(full))   # 56,000
    n_val = int(config.VAL_FRAC * len(full))       # 7,000
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(config.SEED)).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return Subset(full, train_idx), Subset(full, val_idx), Subset(full, test_idx)


def get_loaders():
    train_ds, val_ds, test_ds = get_datasets()
    g = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, worker_init_fn=seed_worker, generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader


def denorm(x):
    """[-1,1] -> [0,1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0
