"""CSDI noise predictor (Tashiro et al., NeurIPS 2021) for univariate forecasting.

Follows the official ``diff_CSDI`` design: input projection of [conditional observation,
noisy target], diffusion-step embedding, side information (time-position embedding,
feature embedding and the conditional mask), and residual layers with a temporal
transformer and gated activation.  With a single feature the feature transformer of
the official code is skipped, exactly as the official code does for K == 1.  The fixed
sinusoidal tables are computed on the fly instead of being stored as buffers; initialisation
(``Conv1d_with_init``) and the time embedding follow the official ``diff_models.py``/``main_model.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def _conv1d_with_init(cin: int, cout: int) -> nn.Conv1d:
    """Official ``Conv1d_with_init``: 1x1 convolution with Kaiming-normal weights."""
    layer = nn.Conv1d(cin, cout, 1)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def csdi_time_embedding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """Official ``CSDI_base.time_embedding``: interleaved sin/cos, div_term = 10000^(-2i/d)."""
    pe = torch.zeros(positions.shape[0], d_model, device=positions.device)
    div = 1.0 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=positions.device) / d_model)
    ang = positions.float().unsqueeze(1) * div
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim: int = 128, proj_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.projection1 = nn.Linear(dim, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # official table: [sin(t * 10^(4j/(d/2-1))... ) | cos(...)] over d/2 frequencies
        half = self.dim // 2
        freqs = 10.0 ** (torch.arange(half, device=t.device) / (half - 1) * 4.0)
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        x = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return F.silu(self.projection2(F.silu(self.projection1(x))))


class ResidualLayer(nn.Module):
    def __init__(self, side_dim: int, channels: int, diff_dim: int, heads: int):
        super().__init__()
        self.diffusion_projection = nn.Linear(diff_dim, channels)
        self.cond_projection = _conv1d_with_init(side_dim, 2 * channels)
        self.mid_projection = _conv1d_with_init(channels, 2 * channels)
        self.output_projection = _conv1d_with_init(channels, 2 * channels)
        self.time_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu",
                                       batch_first=True),
            num_layers=1, enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor, side: torch.Tensor, demb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x + self.diffusion_projection(demb).unsqueeze(-1)
        y = self.time_layer(y.transpose(1, 2)).transpose(1, 2)
        y = self.mid_projection(y) + self.cond_projection(side)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        res, skip = torch.chunk(self.output_projection(y), 2, dim=1)
        return (x + res) / math.sqrt(2.0), skip


class CSDIEps(nn.Module):
    def __init__(self, past_len: int, future_len: int, channels: int = 64, layers: int = 4, heads: int = 8,
                 diff_dim: int = 128, time_emb: int = 128, feat_emb: int = 16):
        super().__init__()
        self.past_len, self.future_len, self.time_emb_dim = past_len, future_len, time_emb
        self.diffusion_embedding = DiffusionEmbedding(diff_dim, diff_dim)
        self.embed_layer = nn.Embedding(1, feat_emb)
        side_dim = time_emb + feat_emb + 1
        self.input_projection = _conv1d_with_init(2, channels)
        self.residual_layers = nn.ModuleList(ResidualLayer(side_dim, channels, diff_dim, heads) for _ in range(layers))
        self.output_projection1 = _conv1d_with_init(channels, channels)
        self.output_projection2 = _conv1d_with_init(channels, 1)
        nn.init.zeros_(self.output_projection2.weight)  # official: weight only

    def forward(self, x_t: torch.Tensor, past: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        n, P, Fl = x_t.size(0), self.past_len, self.future_len
        L = P + Fl
        zeros_f = torch.zeros((n, 1, Fl), device=x_t.device, dtype=x_t.dtype)
        zeros_p = torch.zeros((n, 1, P), device=x_t.device, dtype=x_t.dtype)
        cond_obs = torch.cat([past, zeros_f], dim=2)
        noisy = torch.cat([zeros_p, x_t], dim=2)
        h = F.relu(self.input_projection(torch.cat([cond_obs, noisy], dim=1)))
        pos = csdi_time_embedding(torch.arange(L, device=x_t.device), self.time_emb_dim).T  # [time_emb, L]
        feat = self.embed_layer.weight[0].unsqueeze(-1).expand(-1, L)  # [feat_emb, L]
        mask = torch.cat([torch.ones(1, P, device=x_t.device), torch.zeros(1, Fl, device=x_t.device)], dim=1)
        side = torch.cat([pos, feat, mask], dim=0).unsqueeze(0).expand(n, -1, -1)
        demb = self.diffusion_embedding(t)
        skips = []
        for layer in self.residual_layers:
            h, s = layer(h, side, demb)
            skips.append(s)
        out = torch.stack(skips).sum(0) / math.sqrt(len(self.residual_layers))
        out = self.output_projection2(F.relu(self.output_projection1(out)))
        return out[:, :, P:]
