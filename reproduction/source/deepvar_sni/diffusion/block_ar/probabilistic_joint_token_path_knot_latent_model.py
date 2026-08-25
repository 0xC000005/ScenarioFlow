from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class ProbabilisticJointTokenPathKnotLatentModelConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    d_model: int = 192
    nhead: int = 6
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1

    change_coord: str = "asinh_local_scale"
    change_scale_eps: float = 1e-3

    codebook_size: int = 256
    n_knots: int = 5
    latent_dim: int = 64
    kl_weight: float = 1e-3
    label_smoothing: float = 0.0
    sample_temperature: float = 1.0


class ProbabilisticJointTokenPathKnotLatentModel(nn.Module):
    """293c-v0: fixed-horizon joint-token law with interpolated knot latents."""

    def __init__(self, cfg: ProbabilisticJointTokenPathKnotLatentModelConfig, codebook: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        if codebook.shape != (cfg.codebook_size, cfg.n_cells):
            raise ValueError(
                f"Expected codebook {(cfg.codebook_size, cfg.n_cells)}, got {tuple(codebook.shape)}"
            )

        self.history_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.token_proj = nn.Linear(cfg.n_cells, cfg.d_model)
        self.history_pos = nn.Embedding(cfg.history_len, cfg.d_model)
        self.future_pos = nn.Embedding(cfg.future_len + 1, cfg.d_model)
        self.knot_embed = nn.Embedding(cfg.n_knots, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)
        self.future_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)
        self.out_head = nn.Linear(cfg.d_model, cfg.codebook_size)

        self.prior_attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        self.post_attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True)
        self.prior_mu = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.latent_dim),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.latent_dim),
        )
        self.post_mu = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.latent_dim),
        )
        self.post_logvar = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.latent_dim),
        )
        self.latent_to_model = nn.Linear(cfg.latent_dim, cfg.d_model)

        knot_steps, step_weights = self._build_knot_interpolation(cfg.future_len, cfg.n_knots)
        self.register_buffer("knot_steps", knot_steps)
        self.register_buffer("step_interp_weights", step_weights)
        self.register_buffer("codebook", codebook.clone())

    @staticmethod
    def _build_knot_interpolation(future_len: int, n_knots: int) -> tuple[torch.Tensor, torch.Tensor]:
        knot_steps = torch.round(torch.linspace(0, future_len - 1, steps=n_knots)).long()
        weights = torch.zeros(future_len, n_knots, dtype=torch.float32)
        for step in range(future_len):
            if step <= int(knot_steps[0].item()):
                weights[step, 0] = 1.0
                continue
            if step >= int(knot_steps[-1].item()):
                weights[step, -1] = 1.0
                continue
            for knot_idx in range(n_knots - 1):
                left = int(knot_steps[knot_idx].item())
                right = int(knot_steps[knot_idx + 1].item())
                if left <= step <= right:
                    alpha = float(step - left) / max(right - left, 1)
                    weights[step, knot_idx] = 1.0 - alpha
                    weights[step, knot_idx + 1] = alpha
                    break
        return knot_steps, weights

    @staticmethod
    def _flatten_history(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def compute_change_scale(self, history_norm: torch.Tensor) -> torch.Tensor:
        hist = self._flatten_history(history_norm)
        hist_change = hist[:, 1:] - hist[:, :-1]
        scale = hist_change.pow(2).mean(dim=1, keepdim=True).sqrt()
        return scale.clamp_min(self.cfg.change_scale_eps)

    def transform_change(self, raw_change: torch.Tensor, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return raw_change
        if self.cfg.change_coord == "asinh_local_scale":
            scale = self.compute_change_scale(history_norm)
            if raw_change.ndim == 2:
                scale = scale.squeeze(1)
            return torch.asinh(raw_change / scale)
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def inverse_transform_change(self, model_change: torch.Tensor, history_norm: torch.Tensor) -> torch.Tensor:
        if self.cfg.change_coord == "raw":
            return model_change
        if self.cfg.change_coord == "asinh_local_scale":
            scale = self.compute_change_scale(history_norm)
            if model_change.ndim == 2:
                scale = scale.squeeze(1)
            return torch.sinh(model_change) * scale
        raise ValueError(f"Unknown change_coord={self.cfg.change_coord}")

    def encode_history(self, history_norm: torch.Tensor) -> torch.Tensor:
        hist = self._flatten_history(history_norm)
        pos = torch.arange(self.cfg.history_len, device=hist.device).unsqueeze(0)
        x = self.history_proj(hist) + self.history_pos(pos)
        return self.encoder(x)

    def target_coords(self, history_norm: torch.Tensor, future_norm: torch.Tensor) -> torch.Tensor:
        hist_window = self._flatten_history(history_norm).clone()
        future = self._flatten_history(future_norm)
        curr = hist_window[:, -1]
        coords = []
        for step in range(self.cfg.future_len):
            next_level = future[:, step]
            raw_change = next_level - curr
            coord = self.transform_change(raw_change, hist_window)
            coords.append(coord)
            hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
            curr = next_level
        return torch.stack(coords, dim=1)

    def assign_tokens(self, coords: torch.Tensor) -> torch.Tensor:
        flat = coords.reshape(-1, self.cfg.n_cells)
        dists = torch.cdist(flat, self.codebook)
        ids = dists.argmin(dim=-1)
        return ids.view(coords.shape[0], coords.shape[1])

    def token_vectors(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.codebook[token_ids]

    def encode_future_tokens(self, token_vecs: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(token_vecs.shape[1], device=token_vecs.device).unsqueeze(0)
        x = self.token_proj(token_vecs) + self.future_pos(pos + 1)
        return self.future_encoder(x)

    def _knot_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        knot_ids = torch.arange(self.cfg.n_knots, device=device)
        knot_steps = self.knot_steps.to(device)
        query = self.knot_embed(knot_ids) + self.future_pos(knot_steps + 1)
        return query.unsqueeze(0).expand(batch_size, -1, -1)

    def infer_knot_latents(
        self,
        history_memory: torch.Tensor,
        future_token_vecs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        queries = self._knot_queries(history_memory.shape[0], history_memory.device)
        prior_ctx, _ = self.prior_attn(queries, history_memory, history_memory, need_weights=False)
        prior_mu = self.prior_mu(prior_ctx)
        prior_logvar = self.prior_logvar(prior_ctx).clamp(min=-8.0, max=8.0)

        if future_token_vecs is None:
            eps = torch.randn_like(prior_mu)
            z = prior_mu + torch.exp(0.5 * prior_logvar) * eps
            return z, (prior_mu, prior_logvar)

        future_memory = self.encode_future_tokens(future_token_vecs)
        post_ctx, _ = self.post_attn(queries, future_memory, future_memory, need_weights=False)
        post_in = torch.cat([prior_ctx, post_ctx], dim=-1)
        post_mu = self.post_mu(post_in)
        post_logvar = self.post_logvar(post_in).clamp(min=-8.0, max=8.0)
        eps = torch.randn_like(post_mu)
        z = post_mu + torch.exp(0.5 * post_logvar) * eps
        return z, (prior_mu, prior_logvar, post_mu, post_logvar)

    def expand_step_latents(self, knot_latents: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tk,bkd->btd", self.step_interp_weights.to(knot_latents.device), knot_latents)

    def _decode_all(
        self,
        memory: torch.Tensor,
        prefix_vecs: torch.Tensor,
        step_latents: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = prefix_vecs.shape[1]
        pos = torch.arange(prefix_len, device=prefix_vecs.device).unsqueeze(0)
        x = self.token_proj(prefix_vecs) + self.future_pos(pos)
        x = x + self.latent_to_model(step_latents[:, :prefix_len])
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(prefix_len, device=prefix_vecs.device)
        return self.decoder(x, memory, tgt_mask=tgt_mask)

    def decode_teacher_forced_logits(
        self,
        history_norm: torch.Tensor,
        target_ids: torch.Tensor,
        step_latents: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode_history(history_norm)
        bos = torch.zeros(
            target_ids.shape[0], 1, self.cfg.n_cells, device=target_ids.device, dtype=torch.float32
        )
        prefix = torch.cat([bos, self.token_vectors(target_ids[:, :-1])], dim=1)
        dec = self._decode_all(memory, prefix, step_latents)
        return self.out_head(dec)

    def _kl_div(
        self, prior_mu: torch.Tensor, prior_logvar: torch.Tensor, post_mu: torch.Tensor, post_logvar: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * torch.mean(
            prior_logvar - post_logvar
            + (torch.exp(post_logvar) + (post_mu - prior_mu).pow(2)) / torch.exp(prior_logvar)
            - 1.0
        )

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = self._flatten_history(future_norm)
        target_coords = self.target_coords(history_norm, future)
        target_ids = self.assign_tokens(target_coords)
        target_vecs = self.token_vectors(target_ids)

        memory = self.encode_history(history_norm)
        knot_latents, stats = self.infer_knot_latents(memory, target_vecs)
        prior_mu, prior_logvar, post_mu, post_logvar = stats
        step_latents = self.expand_step_latents(knot_latents)
        logits = self.decode_teacher_forced_logits(history_norm, target_ids, step_latents)
        token_nll = F.cross_entropy(
            logits.reshape(-1, self.cfg.codebook_size),
            target_ids.reshape(-1),
            label_smoothing=self.cfg.label_smoothing,
        )
        kl = self._kl_div(prior_mu, prior_logvar, post_mu, post_logvar)
        loss = token_nll + self.cfg.kl_weight * kl

        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)
        acc = (pred_ids == target_ids).float().mean()
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        metrics = {
            "total": loss.detach(),
            "token_nll": token_nll.detach(),
            "kl": kl.detach(),
            "token_acc": acc.detach(),
            "token_entropy": entropy.detach(),
        }
        return loss, metrics

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
            knot_latents, _ = self.infer_knot_latents(memory)
            step_latents = self.expand_step_latents(knot_latents)
            curr = hist_window[:, -1]
            pred_vecs: list[torch.Tensor] = []
            pred_levels: list[torch.Tensor] = []
            bos = torch.zeros(
                hist_window.shape[0], 1, self.cfg.n_cells, device=hist_window.device, dtype=hist_window.dtype
            )
            for _step in range(self.cfg.future_len):
                if pred_vecs:
                    prefix = torch.cat([bos, torch.stack(pred_vecs, dim=1)], dim=1)
                else:
                    prefix = bos
                dec = self._decode_all(memory, prefix, step_latents)
                logits = self.out_head(dec[:, -1]) / temp
                probs = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_vec = self.token_vectors(next_ids)
                raw_change = self.inverse_transform_change(next_vec, hist_window)
                next_level = torch.clamp(curr + raw_change, -1.0, 1.0)
                pred_levels.append(next_level)
                pred_vecs.append(next_vec)
                hist_window = torch.cat([hist_window[:, 1:], next_level.unsqueeze(1)], dim=1)
                curr = next_level
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
) -> tuple[ProbabilisticJointTokenPathKnotLatentModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ProbabilisticJointTokenPathKnotLatentModelConfig(**payload["config"])
    model = ProbabilisticJointTokenPathKnotLatentModel(cfg, payload["codebook"].to(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ProbabilisticJointTokenPathKnotLatentModel,
    cfg: ProbabilisticJointTokenPathKnotLatentModelConfig,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "codebook": model.codebook.detach().cpu(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
