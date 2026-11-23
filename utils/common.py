"""Shared training-loop helpers: epoch timing, loss logging, checkpointing."""
import json
import time

import torch

from configs import config


class LossLogger:
    """Records per-epoch losses and wall-clock seconds; persists to results/logs."""

    def __init__(self, name):
        self.name = name
        self.history = {"epochs": [], "epoch_seconds": []}
        self._t0 = None

    def start_epoch(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def end_epoch(self, epoch, **losses):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.history["epochs"].append(epoch)
        self.history["epoch_seconds"].append(dt)
        for k, v in losses.items():
            self.history.setdefault(k, []).append(float(v))
        self.save()
        return dt

    def save(self):
        path = config.LOG_DIR / f"{self.name}_losses.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    @property
    def mean_epoch_seconds(self):
        s = self.history["epoch_seconds"]
        return float(sum(s) / len(s)) if s else None


def save_checkpoint(model, name, extra=None):
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, config.CKPT_DIR / f"{name}.pt")


def load_checkpoint(model, name, device=None, key="state_dict"):
    device = device or config.DEVICE
    payload = torch.load(config.CKPT_DIR / f"{name}.pt",
                         map_location=device, weights_only=True)
    model.load_state_dict(payload[key] if key in payload else payload)
    return model.to(device)
