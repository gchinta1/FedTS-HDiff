"""Flat-parameter machinery: the model's parameters and gradients are views into two
contiguous vectors, so loading a client/edge/global model, averaging models and
applying optimiser updates are single vector operations.

:class:`FlatAdamW` reproduces ``torch.optim.AdamW`` (decoupled weight decay, bias
correction, no AMSGrad) on segments of the flat vector, with the optimiser state
kept per client so that the original "one persistent AdamW per client" protocol is
preserved while only one model lives on the device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..models.forecaster import SegmentSpec


class FlatModel:
    def __init__(self, module: nn.Module, device: torch.device):
        if any(True for _ in module.buffers()):
            raise ValueError("flat models must keep all state in parameters (found buffers)")
        params = list(module.named_parameters())
        self.numel = sum(p.numel() for _, p in params)
        self.flat = torch.zeros(self.numel, device=device)
        self.grad = torch.zeros(self.numel, device=device)
        self.slices: dict[str, slice] = {}
        o = 0
        for name, p in params:
            n = p.numel()
            self.flat[o:o + n].copy_(p.data.reshape(-1))
            p.data = self.flat[o:o + n].view_as(p)
            p.grad = self.grad[o:o + n].view_as(p)
            self.slices[name] = slice(o, o + n)
            o += n
        self.module = module
        self.names = [n for n, _ in params]

    def segment_slice(self, spec: SegmentSpec) -> slice:
        if spec.prefixes == ("",):
            return slice(0, self.numel)
        hits = [self.slices[n] for n in self.names if any(n.startswith(pre) for pre in spec.prefixes)]
        if not hits:
            raise ValueError(f"segment {spec.name}: no parameter matches {spec.prefixes}")
        start, stop = hits[0].start, hits[-1].stop
        if sum(s.stop - s.start for s in hits) != stop - start:
            raise ValueError(f"segment {spec.name} is not contiguous")
        return slice(start, stop)

    def load(self, vec: torch.Tensor) -> None:
        self.flat.copy_(vec)

    def snapshot(self) -> torch.Tensor:
        return self.flat.detach().clone()

    def grads_are_views(self) -> bool:
        return all(p.grad is not None and p.grad.data_ptr() == self.grad[self.slices[n]].data_ptr()
                   for n, p in self.module.named_parameters())


@dataclass
class OptState:
    m: torch.Tensor
    v: torch.Tensor
    steps: dict[str, int] = field(default_factory=dict)

    @classmethod
    def zeros(cls, numel: int, device: torch.device) -> "OptState":
        return cls(torch.zeros(numel, device=device), torch.zeros(numel, device=device))


class FlatAdamW:
    def __init__(self, fm: FlatModel, segments: list[SegmentSpec], lr: float, eps: float):
        self.fm, self.lr, self.eps = fm, lr, eps
        self.specs = {s.name: s for s in segments}
        self.slices = {s.name: fm.segment_slice(s) for s in segments}

    def update(self, state: OptState, segment: str, grad: torch.Tensor) -> None:
        spec, sl = self.specs[segment], self.slices[segment]
        b1, b2, wd, lr = spec.beta1, spec.beta2, spec.weight_decay, self.lr
        m, v, p = state.m[sl], state.v[sl], self.fm.flat[sl]
        step = state.steps.get(segment, 0) + 1
        state.steps[segment] = step
        if wd:
            p.mul_(1.0 - lr * wd)
        m.mul_(b1).add_(grad, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
        p.addcdiv_(m, denom, value=-lr / bc1)
