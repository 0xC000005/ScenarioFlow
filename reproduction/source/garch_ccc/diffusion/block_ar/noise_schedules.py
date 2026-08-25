"""Task-Adaptive Diffusion Forcing Noise Schedules."""
import torch

from .masking import MCVDTask, get_task_types


def sample_task_adaptive_noise(
    task: MCVDTask,
    batch_size: int,
    block_size: int,
    n_steps: int,
    jitter_std: float = 0.15,
    device: torch.device = None,
) -> torch.Tensor:
    """Sample per-frame noise levels with task-adaptive structure.

    Each frame's noise level = structured mean + Gaussian jitter.
    The structured mean depends on task type and frame position.

    Args:
        task: MCVDTask type
        batch_size: B
        block_size: number of frames in block
        n_steps: total diffusion steps
        jitter_std: fraction of n_steps for jitter (0.15 = +/-15 steps for n_steps=100)
        device: torch device

    Returns:
        noise_levels: (B, block_size) int64 in [0, n_steps)
    """
    pos = torch.linspace(0, 1, block_size, device=device)  # (block_size,)

    if task == MCVDTask.FORWARD:
        # Near past -> low noise; far from past -> high noise
        mean_frac = pos
    elif task == MCVDTask.BACKWARD:
        # Far from future -> high noise; near future -> low noise
        mean_frac = 1.0 - pos
    elif task == MCVDTask.INTERPOLATION:
        # Both edges anchored (low noise), middle uncertain (high noise)
        mean_frac = 1.0 - 2.0 * torch.abs(pos - 0.5)
    elif task == MCVDTask.UNCONDITIONAL:
        # No anchors -> independent uniform per frame
        return torch.randint(0, n_steps, (batch_size, block_size), device=device)
    else:
        raise ValueError(f"Unknown task type: {task}")

    # Structured mean: (1, block_size) -> (B, block_size)
    k_mean = (mean_frac * (n_steps - 1)).unsqueeze(0).expand(batch_size, -1)

    # Add per-frame Gaussian jitter for training diversity
    jitter = torch.randn(batch_size, block_size, device=device) * (jitter_std * n_steps)
    k = (k_mean + jitter).long().clamp(0, n_steps - 1)

    return k


def sample_batch_task_adaptive_noise(
    mask_past: torch.Tensor,
    mask_future: torch.Tensor,
    block_size: int,
    n_steps: int,
    jitter_std: float = 0.15,
) -> torch.Tensor:
    """Sample noise for a batch with mixed per-item task types.

    Vectorized implementation -- no Python loops over batch items.

    Args:
        mask_past: (B,) bool
        mask_future: (B,) bool
        block_size: frames per block
        n_steps: diffusion steps
        jitter_std: jitter scale

    Returns:
        noise_levels: (B, block_size) int64 in [0, n_steps)
    """
    device = mask_past.device
    B = mask_past.shape[0]

    # 1. Compute task types from masks: (B,)
    tasks = get_task_types(mask_past, mask_future)

    # 2. Position fractions: (block_size,)
    pos = torch.linspace(0, 1, block_size, device=device)

    # 3. Build per-task mean_frac profiles: (4, block_size)
    #    Index 0=FORWARD, 1=BACKWARD, 2=INTERPOLATION, 3=UNCONDITIONAL
    profiles = torch.zeros(4, block_size, device=device)
    profiles[MCVDTask.FORWARD] = pos                              # 0 -> 1
    profiles[MCVDTask.BACKWARD] = 1.0 - pos                      # 1 -> 0
    profiles[MCVDTask.INTERPOLATION] = 1.0 - 2.0 * torch.abs(pos - 0.5)  # tent
    profiles[MCVDTask.UNCONDITIONAL] = 0.5  # placeholder, will be overridden

    # 4. Gather mean_frac per batch item: (B, block_size)
    mean_frac = profiles[tasks]  # advanced indexing: tasks is (B,) -> (B, block_size)

    # 5. Compute structured mean noise levels
    k_mean = mean_frac * (n_steps - 1)  # (B, block_size)

    # 6. Add Gaussian jitter
    jitter = torch.randn(B, block_size, device=device) * (jitter_std * n_steps)
    k = (k_mean + jitter).long().clamp(0, n_steps - 1)

    # 7. Override UNCONDITIONAL items with uniform random
    uncond_mask = (tasks == MCVDTask.UNCONDITIONAL)  # (B,)
    if uncond_mask.any():
        n_uncond = uncond_mask.sum().item()
        k[uncond_mask] = torch.randint(
            0, n_steps, (n_uncond, block_size), device=device
        )

    return k
