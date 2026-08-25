"""
Wrapper for CLN E2E models (161a, 163a, 159b etc.) to implement the
sample_batched() interface expected by test_block_ar_requirements_v2.py.

Translates between:
  - v2 test interface: sample_batched(history, n_samples) → (B, K, T, 5, 5) in [0,1]
  - CLN E2E internals: encoder + MeanPredictor + CLN transformer
"""

import torch
import torch.nn as nn

from experiments.backfill.block_ar.train_cond_oneshot_flow import normalize_iv


class CLNEndToEndWrapper(nn.Module):
    """Wraps encoder + optional mean head + CLN transformer for v2 test suite."""

    def __init__(self, encoder, mean_pred, cln_model, config, device="cuda"):
        super().__init__()
        self.encoder = encoder
        self.mean_pred = mean_pred
        self.cln_model = cln_model
        self._config = config
        self._device = device

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cuda"):
        """Load from a CLN E2E checkpoint dict."""
        cfg = checkpoint["config"]
        model_type = cfg.get("type", "")

        # Load encoder
        from experiments.backfill.block_ar.train_cond_oneshot_flow import load_encoder
        encoder_path = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
        encoder, cond_dim = load_encoder(encoder_path, device)
        encoder.load_state_dict(checkpoint["encoder_state"])
        encoder.eval()

        T = cfg.get("n_frames", 30)
        C = cfg.get("n_cells", 25)
        mean_pred = None
        if "mean_pred_state" in checkpoint:
            from experiments.backfill.block_ar.train_158a_end_to_end import MeanPredictor

            mean_pred = MeanPredictor(cond_dim=cond_dim, n_frames=T, n_cells=C)
            mean_pred.load_state_dict(checkpoint["mean_pred_state"])
            mean_pred.eval()

        # Load CLN transformer (detect no-LN variant)
        is_no_ln = "no_ln" in model_type
        if is_no_ln:
            from experiments.backfill.block_ar.train_159a_no_ln_cln import (
                NoLNCLNResidualTransformer
            )
            cln_model = NoLNCLNResidualTransformer(
                n_frames=T, n_cells=C,
                d_model=cfg.get("d_model", 128),
                n_heads=cfg.get("n_heads", 4),
                n_layers=cfg.get("n_layers", 4),
                cond_dim=cond_dim,
                noise_dim=cfg.get("noise_dim", 32),
            )
        else:
            from experiments.backfill.block_ar.train_155d_cln_transformer import (
                CLNResidualTransformer
            )
            cln_model = CLNResidualTransformer(
                n_frames=T, n_cells=C,
                d_model=cfg.get("d_model", 128),
                n_heads=cfg.get("n_heads", 4),
                n_layers=cfg.get("n_layers", 4),
                cond_dim=cond_dim,
                noise_dim=cfg.get("noise_dim", 32),
            )
        cln_model.load_state_dict(checkpoint["cln_state"])
        cln_model.eval()

        wrapper = cls(encoder, mean_pred, cln_model, cfg, device)
        wrapper = wrapper.to(device)
        return wrapper

    def sample_batched(self, history, n_samples=50, **kwargs):
        """Generate samples compatible with v2 test suite.

        Args:
            history: (B, T_hist, 5, 5) in [-1, 1] (normalized)

        Returns:
            samples: (B, n_samples, T_future, 5, 5) in [0, 1] (denormalized)
        """
        B = history.shape[0]
        T = self._config.get("n_frames", 30)
        C = self._config.get("n_cells", 25)
        noise_dim = self._config.get("noise_dim", 32)
        device = history.device

        # Chunk to avoid OOM (transformer is memory-heavy with B*K inputs)
        CHUNK = 10

        with torch.no_grad():
            cond = self.encoder(history)  # (B, cond_dim)
            last_frame = (history[:, -1].reshape(B, C) + 1.0) / 2.0
            if self.mean_pred is None:
                base = last_frame.unsqueeze(1).expand(B, T, C)
            else:
                base = self.mean_pred(cond, last_frame).reshape(B, T, C)

            all_residuals = []
            for start in range(0, n_samples, CHUNK):
                k = min(CHUNK, n_samples - start)
                noise = torch.randn(B * k, noise_dim, device=device)
                cond_K = cond.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
                res = self.cln_model(cond_K, noise).reshape(B, k, T, C)
                all_residuals.append(res)

            residual = torch.cat(all_residuals, dim=1)  # (B, n_samples, T, C)
            combined = (base.unsqueeze(1) + residual).clamp(0, 1)
            samples = combined.reshape(B, n_samples, T, 5, 5)

        return samples
