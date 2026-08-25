from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportTransform(nn.Module):
    """Interface for support-respecting observation transforms."""

    def forward(self, x_native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@dataclass
class TransformConfig:
    kind: str
    lo: float | None = None
    hi: float | None = None
    eps: float = 1e-5


class IdentityTransform(SupportTransform):
    def forward(self, x_native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(x_native.shape[:-1], device=x_native.device, dtype=x_native.dtype)
        return x_native, zeros

    def inverse(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(u.shape[:-1], device=u.device, dtype=u.dtype)
        return u, zeros


class PositiveLogTransform(SupportTransform):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x_native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_native.clamp_min(self.eps)
        u = x.log()
        logabsdet = -u.sum(dim=-1)
        return u, logabsdet

    def inverse(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = u.exp()
        logabsdet = u.sum(dim=-1)
        return x, logabsdet


class BoundedLogitTransform(SupportTransform):
    def __init__(self, lo: float = 0.01, hi: float = 1.0, eps: float = 1e-5):
        super().__init__()
        if hi <= lo:
            raise ValueError(f"Expected hi > lo, got lo={lo}, hi={hi}")
        self.lo = float(lo)
        self.hi = float(hi)
        self.eps = float(eps)

    def forward(self, x_native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = ((x_native - self.lo) / (self.hi - self.lo)).clamp(self.eps, 1.0 - self.eps)
        u = torch.logit(scaled)
        log_sig = F.logsigmoid(u)
        log_one_minus_sig = F.logsigmoid(-u)
        logabsdet = (
            -(torch.log(torch.as_tensor(self.hi - self.lo, device=u.device, dtype=u.dtype)))
            - log_sig
            - log_one_minus_sig
        ).sum(dim=-1)
        return u, logabsdet

    def inverse(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sig = torch.sigmoid(u)
        x = self.lo + (self.hi - self.lo) * sig
        log_sig = F.logsigmoid(u)
        log_one_minus_sig = F.logsigmoid(-u)
        logabsdet = (
            torch.log(torch.as_tensor(self.hi - self.lo, device=u.device, dtype=u.dtype))
            + log_sig
            + log_one_minus_sig
        ).sum(dim=-1)
        return x, logabsdet


def build_support_transform(config: TransformConfig | dict | None = None) -> SupportTransform:
    if config is None:
        return BoundedLogitTransform()
    if isinstance(config, dict):
        config = TransformConfig(**config)

    if config.kind == "identity":
        return IdentityTransform()
    if config.kind == "positive_log":
        return PositiveLogTransform(eps=config.eps)
    if config.kind == "bounded_logit":
        if config.lo is None or config.hi is None:
            raise ValueError("bounded_logit requires lo and hi")
        return BoundedLogitTransform(lo=config.lo, hi=config.hi, eps=config.eps)
    raise ValueError(f"Unknown support transform kind: {config.kind}")


def iv_to_unconstrained(
    x: torch.Tensor,
    lo: float = 0.01,
    hi: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    return build_support_transform({"kind": "bounded_logit", "lo": lo, "hi": hi, "eps": eps}).forward(x)[0]


def unconstrained_to_iv(
    u: torch.Tensor,
    lo: float = 0.01,
    hi: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    return build_support_transform({"kind": "bounded_logit", "lo": lo, "hi": hi, "eps": eps}).inverse(u)[0]
