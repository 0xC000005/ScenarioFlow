from __future__ import annotations

import torch
import torch.nn as nn


class StructuredResidualActivityGeometry(nn.Module):
    """
    Minimal geometry/group abstraction for latent residual activity.

    Current v0:
      - one 5x5 IV surface => one group by default

    Future extension:
      - multiple factor groups can be added by supplying `group_ids`
      - the same pooling/broadcast interface then works over a grouped graph
    """

    def __init__(
        self,
        n_frames: int,
        n_cells: int,
        group_ids: list[int] | None = None,
        adjacency_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_frames = int(n_frames)
        self.n_cells = int(n_cells)
        if group_ids is None:
            group_ids = [0] * self.n_cells
        if len(group_ids) != self.n_cells:
            raise ValueError(f"group_ids length {len(group_ids)} != n_cells {self.n_cells}")
        group_tensor = torch.tensor(group_ids, dtype=torch.long)
        n_groups = int(group_tensor.max().item()) + 1
        membership = torch.zeros(n_groups, self.n_cells, dtype=torch.float32)
        membership[group_tensor, torch.arange(self.n_cells)] = 1.0
        counts = membership.sum(dim=1, keepdim=True).clamp_min(1.0)
        membership_norm = membership / counts

        self.n_groups = n_groups
        self.register_buffer("group_ids", group_tensor, persistent=False)
        self.register_buffer("group_membership", membership, persistent=False)
        self.register_buffer("group_membership_norm", membership_norm, persistent=False)
        self.register_buffer("group_weights", counts.squeeze(1) / counts.sum().clamp_min(1.0), persistent=False)

        if adjacency_matrix is None:
            side = int(round(self.n_cells ** 0.5))
            if side * side == self.n_cells:
                adjacency = torch.zeros(self.n_cells, self.n_cells, dtype=torch.float32)
                for idx in range(self.n_cells):
                    r, c = divmod(idx, side)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < side and 0 <= cc < side:
                            adjacency[idx, rr * side + cc] = 1.0
            else:
                adjacency = torch.eye(self.n_cells, dtype=torch.float32)
        else:
            adjacency = adjacency_matrix.to(dtype=torch.float32)
            if adjacency.shape != (self.n_cells, self.n_cells):
                raise ValueError(
                    f"adjacency_matrix shape {tuple(adjacency.shape)} != ({self.n_cells}, {self.n_cells})"
                )
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.register_buffer("adjacency_with_self", torch.maximum(adjacency, torch.eye(self.n_cells)), persistent=False)

    def pool_group(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] -> pooled [B, T, G]
        """
        return torch.einsum("btc,gc->btg", x, self.group_membership_norm.to(device=x.device, dtype=x.dtype))

    def pool_group_global(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] -> pooled [B, G]
        """
        pooled = self.pool_group(x)
        return pooled.mean(dim=1)

    def broadcast_group(self, g: torch.Tensor) -> torch.Tensor:
        """
        g: [B, G] or [B, T, G] -> [B, T, C]
        """
        membership = self.group_membership.to(device=g.device, dtype=g.dtype)
        if g.ndim == 2:
            return torch.einsum("bg,gc->bc", g, membership).unsqueeze(1).expand(-1, self.n_frames, -1)
        if g.ndim == 3:
            return torch.einsum("btg,gc->btc", g, membership)
        raise ValueError(f"Expected [B,G] or [B,T,G], got {tuple(g.shape)}")

    def local_adjacency(self, include_self: bool = False) -> torch.Tensor:
        if include_self:
            return self.adjacency_with_self
        return self.adjacency

    def aggregate_local(self, weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        weights: [B, T, C, C], x: [B, T, C] -> [B, T, C]
        """
        return torch.einsum("btij,btj->bti", weights, x)
