"""Track B narrative-conditioning adapter for the frozen SNI generator.

Maps a high-dim narrative embedding to an *additive* memory_dim context, injected
parallel to the model's existing ``risk_context`` channel
(``generic_state_aware_normalized_innovation_flow_matching.py``:
``memory_states = memory_states + risk_context[:, None, :]``). That channel is the
proof-of-propagation: it already drives the risk-state/width allocation that the
2026-06-17 recap showed the frozen 734a responds to faithfully.

Design constraints (evidence-grounded):
- **Zero-init final layer** => exact zero context at init, so the conditioned model is
  byte-identical to frozen 734a until the adapter is trained (non-regression + the
  null-narrative metamorphic pass come for free).
- **Presence-gated** => an absent narrative yields zero context => the start-only baseline.
- **No low-dim bottleneck** (``hidden >= memory_dim``): Exp-153a's window-specific steering
  failed precisely because the condition was squeezed through a 128-d bottleneck too small
  for the output. The narrative enters at high dim and is never compressed below memory_dim.

The backbone stays frozen; only this adapter (and, for the width variant, the existing
``risk_context_proj``) is trained, on a SEPARATE checkpoint. 734a is never modified.
"""

from __future__ import annotations

import torch
from torch import nn


class NarrativeConditioningAdapter(nn.Module):
    """High-dim narrative embedding -> additive memory_dim context (zero-init, gated)."""

    def __init__(self, narrative_dim: int, memory_dim: int, hidden: int = 512) -> None:
        super().__init__()
        if hidden < memory_dim:
            raise AssertionError(
                f"avoid the 153a low-dim bottleneck: hidden ({hidden}) must be "
                f">= memory_dim ({memory_dim})"
            )
        self.narrative_dim = int(narrative_dim)
        self.memory_dim = int(memory_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(narrative_dim),
            nn.Linear(narrative_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, memory_dim),
        )
        # zero-init the output projection => exact no-op at init (frozen-734a identity).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, narrative_emb: torch.Tensor, present: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return additive context of shape ``(B, memory_dim)``.

        ``present`` is a boolean/0-1 tensor of shape ``(B,)``; rows where it is false
        receive a zero context (the frozen baseline). When ``present`` is None every
        row is treated as present.
        """
        ctx = self.net(narrative_emb)  # (B, memory_dim)
        if present is None:
            return ctx
        gate = present.to(ctx.dtype).reshape(-1, 1)  # (B, 1)
        return ctx * gate
