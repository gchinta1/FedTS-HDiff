"""The conditional 1-D U-Net noise predictor of the original FedTS-HDiff scripts (unchanged)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.lin2(F.silu(self.lin1(emb)))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.block1 = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv1d(in_ch, out_ch, 3, padding=1))
        self.block2 = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv1d(out_ch, out_ch, 3, padding=1))
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb).view(t_emb.size(0), -1, 1)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet1D(nn.Module):
    """Input channels: [noisy future, past]; output: predicted noise on the future.

    Parameter registration order is unchanged from the original, so ``out_conv`` (the
    FedRep "head") is the final contiguous block of parameters.
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 1, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv1d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock1D(base_ch, base_ch * 2, time_dim)
        self.down2 = ResBlock1D(base_ch * 2, base_ch * 4, time_dim)
        self.pool = nn.MaxPool1d(2)
        self.mid1 = ResBlock1D(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock1D(base_ch * 4, base_ch * 4, time_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ResBlock1D(base_ch * 8, base_ch * 2, time_dim)
        self.up2 = ResBlock1D(base_ch * 4, base_ch, time_dim)
        self.out_conv = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv1d(base_ch, out_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        m = self.mid2(self.mid1(self.pool(d2), t_emb), t_emb)
        u1 = self.up1(torch.cat([self.up(m), d2], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.up(u1), d1], dim=1), t_emb)
        return self.out_conv(u2)


class UNetEps(nn.Module):
    """eps(x_t_future, past, t) wrapper with the original channel layout."""

    head_prefix = "net.out_conv"

    def __init__(self, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        self.net = UNet1D(in_ch=2, out_ch=1, base_ch=base_ch, time_dim=time_dim)

    def forward(self, x_t: torch.Tensor, past: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, past], dim=1), t)
