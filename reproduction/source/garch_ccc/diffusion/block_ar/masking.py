"""MCVD Multi-Task Masking for Block-AR Diffusion."""
from enum import IntEnum
from typing import Optional, Tuple

import torch


class MCVDTask(IntEnum):
    FORWARD = 0       # past visible, future masked
    BACKWARD = 1      # past masked, future visible
    INTERPOLATION = 2 # both visible
    UNCONDITIONAL = 3 # both masked


# Mask lookup: task -> (mask_past, mask_future)
_TASK_MASKS = {
    MCVDTask.FORWARD:       (False, True),
    MCVDTask.BACKWARD:      (True, False),
    MCVDTask.INTERPOLATION: (False, False),
    MCVDTask.UNCONDITIONAL: (True, True),
}


def sample_mcvd_masks(
    batch_size: int,
    p_mask: float = 0.5,
    device: torch.device = None,
    task_probs: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample MCVD task masks for multi-task training.

    Two modes:
      1. Legacy (task_probs=None): independent Bernoulli on past/future with
         probability p_mask.  Gives coupled distribution:
           fwd = p(1-p), bwd = p(1-p), interp = (1-p)^2, uncond = p^2.
      2. Explicit (task_probs set): direct multinomial over the 4 tasks.
         Decouples each task's probability.  Tuple order:
           (forward, backward, interpolation, unconditional).

    Args:
        batch_size: B
        p_mask: legacy Bernoulli probability (ignored if task_probs set)
        device: torch device
        task_probs: (p_fwd, p_bwd, p_interp, p_uncond), must sum to ~1.0

    Returns:
        mask_past: (B,) bool - True means mask past (use null embedding)
        mask_future: (B,) bool - True means mask future
    """
    if task_probs is not None:
        probs = torch.tensor(task_probs, dtype=torch.float32, device=device)
        tasks = torch.multinomial(probs, batch_size, replacement=True)
        mask_past = torch.zeros(batch_size, dtype=torch.bool, device=device)
        mask_future = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for task_id, (mp, mf) in _TASK_MASKS.items():
            sel = tasks == task_id
            if mp:
                mask_past[sel] = True
            if mf:
                mask_future[sel] = True
        return mask_past, mask_future

    # Legacy: independent Bernoulli
    mask_past = torch.rand(batch_size, device=device) < p_mask
    mask_future = torch.rand(batch_size, device=device) < p_mask
    return mask_past, mask_future


def get_task_types(
    mask_past: torch.Tensor,
    mask_future: torch.Tensor,
) -> torch.Tensor:
    """Determine MCVD task type per batch item from masks.

    Returns:
        tasks: (B,) int tensor of MCVDTask values
    """
    tasks = torch.zeros(mask_past.shape[0], dtype=torch.long, device=mask_past.device)
    # FORWARD: past visible (not masked), future masked
    tasks[(~mask_past) & mask_future] = MCVDTask.FORWARD
    # BACKWARD: past masked, future visible
    tasks[mask_past & (~mask_future)] = MCVDTask.BACKWARD
    # INTERPOLATION: both visible
    tasks[(~mask_past) & (~mask_future)] = MCVDTask.INTERPOLATION
    # UNCONDITIONAL: both masked
    tasks[mask_past & mask_future] = MCVDTask.UNCONDITIONAL
    return tasks
