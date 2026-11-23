"""DDPM: U-Net with sinusoidal time embeddings, linear beta schedule, eps-prediction."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """28x28 -> 14x14 -> 7x7 U-Net with residual blocks and time conditioning."""

    def __init__(self, base_ch=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.time_dim = base_ch
        self.inc = nn.Conv2d(1, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, time_dim)          # 28
        self.ds1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)            # -> 14
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)      # 14
        self.ds2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)    # -> 7
        self.mid1 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.mid2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.up2 = ResBlock(base_ch * 4, base_ch, time_dim)        # 14 (cat skip)
        self.up1 = ResBlock(base_ch * 2, base_ch, time_dim)        # 28 (cat skip)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h0 = self.inc(x)
        s1 = self.down1(h0, t_emb)
        s2 = self.down2(self.ds1(s1), t_emb)
        h = self.ds2(s2)
        h = self.mid2(self.mid1(h, t_emb), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(torch.cat([h, s2], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(torch.cat([h, s1], dim=1), t_emb)
        return self.out(h)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=300, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", ac)
        self.register_buffer("alphas_cumprod_prev", F.pad(ac[:-1], (1, 0), value=1.0))
        self.register_buffer("sqrt_ac", ac.sqrt())
        self.register_buffer("sqrt_one_minus_ac", (1.0 - ac).sqrt())
        self.register_buffer("sqrt_recip_alpha", (1.0 / alphas).sqrt())
        self.register_buffer("posterior_var",
                             betas * (1.0 - F.pad(ac[:-1], (1, 0), value=1.0)) / (1.0 - ac))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_ac[t][:, None, None, None]
        s2 = self.sqrt_one_minus_ac[t][:, None, None, None]
        return s1 * x0 + s2 * noise, noise

    def p_losses(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        return F.mse_loss(self.model(xt, t), noise)

    @torch.no_grad()
    def p_sample_loop(self, shape, device, trajectory_idx=None):
        """Full reverse loop x_T -> x_0. Optionally record intermediate states."""
        x = torch.randn(shape, device=device)
        trajectory = {self.T: x.clone()} if trajectory_idx is not None else None
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            mean = self.sqrt_recip_alpha[i] * (
                x - self.betas[i] / self.sqrt_one_minus_ac[i] * eps)
            if i > 0:
                x = mean + self.posterior_var[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
            # intermediate x_t are ~N(0,1)-scaled at high t; clamping them
            # truncates the noise distribution and biases the trajectory
            if trajectory_idx is not None and i in trajectory_idx:
                trajectory[i] = x.clone()
        x = x.clamp(-1, 1)
        if trajectory_idx is not None:
            trajectory[0] = x.clone()
            return x, trajectory
        return x
