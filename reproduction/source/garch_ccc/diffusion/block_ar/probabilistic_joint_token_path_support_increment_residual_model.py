from __future__ import annotations

import torch

from diffusion.block_ar.probabilistic_joint_token_path_support_residual_model import (
    ProbabilisticJointTokenPathSupportResidualModel,
    ProbabilisticJointTokenPathSupportResidualModelConfig as ProbabilisticJointTokenPathSupportIncrementResidualModelConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


class ProbabilisticJointTokenPathSupportIncrementResidualModel(
    ProbabilisticJointTokenPathSupportResidualModel
):
    """293h-v0: scaffold-increment residual daily law."""

    def scaffold_coords_for_training(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
        coarse_scaffold: torch.Tensor,
    ) -> torch.Tensor:
        hist_window = self._flatten_history(history_norm).clone()
        future = self._flatten_history(future_norm)
        prev_scaffold = hist_window[:, -1]
        coords = []
        for step in range(self.cfg.future_len):
            scaffold_level = coarse_scaffold[:, step]
            raw_change = scaffold_level - prev_scaffold
            coord = self.transform_change(raw_change, hist_window)
            coords.append(coord)
            next_level = future[:, step]
            hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
            prev_scaffold = scaffold_level
        return torch.stack(coords, dim=1)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten_history(history_norm)
        bsz = history_norm.shape[0]
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        temp = max(temp, 1e-4)

        outputs = []
        remaining = int(n_samples)
        while remaining > 0:
            sample_chunk = min(chunk_size, remaining)
            hist_window = (
                history_norm.unsqueeze(1)
                .expand(-1, sample_chunk, -1, -1)
                .reshape(bsz * sample_chunk, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            memory = self.encode_history(hist_window)
            coarse_logits = self.coarse_head(memory.mean(dim=1)) / temp
            coarse_probs = torch.softmax(coarse_logits, dim=-1)
            coarse_ids = torch.multinomial(coarse_probs, num_samples=1).squeeze(-1)
            coarse_vecs = self.coarse_vectors(coarse_ids)
            coarse_scaffold = self.expand_coarse_scaffold(coarse_vecs)

            curr = hist_window[:, -1]
            prev_scaffold = curr.clone()
            pred_vecs: list[torch.Tensor] = []
            pred_levels: list[torch.Tensor] = []
            bos = torch.zeros(
                hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
            )
            for step in range(self.cfg.future_len):
                if pred_vecs:
                    prefix = torch.cat([bos, torch.stack(pred_vecs, dim=1)], dim=1)
                else:
                    prefix = bos
                dec = self._decode_all(memory, prefix, coarse_scaffold)
                logits = self.out_head(dec[:, -1]) / temp
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                residual_vec = self.token_vectors(next_ids)
                scaffold_delta = coarse_scaffold[:, step] - prev_scaffold
                scaffold_coord = self.transform_change(scaffold_delta, hist_window)
                total_coord = scaffold_coord + residual_vec
                raw_change = self.inverse_transform_change(total_coord, hist_window)
                next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
                pred_levels.append(next_level)
                pred_vecs.append(residual_vec)
                hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
                curr = next_level
                prev_scaffold = coarse_scaffold[:, step]
            future_norm = torch.stack(pred_levels, dim=1)
            future_01 = denormalize_iv(future_norm)
            if self.cfg.n_cells == 25:
                future_01 = future_01.view(bsz, sample_chunk, self.cfg.future_len, 5, 5)
            else:
                future_01 = future_01.view(bsz, sample_chunk, self.cfg.future_len, self.cfg.n_cells)
            outputs.append(future_01)
            remaining -= sample_chunk
        return torch.cat(outputs, dim=1).contiguous()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ProbabilisticJointTokenPathSupportIncrementResidualModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticJointTokenPathSupportIncrementResidualModelConfig(**payload["config"])
    model = ProbabilisticJointTokenPathSupportIncrementResidualModel(
        cfg,
        payload["residual_codebook"].to(device),
        payload["coarse_codebook"].to(device),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
