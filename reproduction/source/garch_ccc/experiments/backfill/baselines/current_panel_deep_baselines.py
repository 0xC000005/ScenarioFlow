"""Neural baselines for the current real-VIX 39-state panel.

These baselines intentionally share the same state-increment interface as the
current classical baselines:

    sample_joint(history_increment, n_samples) -> future state increments

They instantiate the published baseline mechanisms under the common
current-panel scenario-generation protocol: DeepVAR low-rank Gaussian AR,
TimeGrad autoregressive diffusion, and CSDI conditional path diffusion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class CurrentPanelDeepVAR(nn.Module):
    """DeepVAR AR LSTM with low-rank Gaussian emissions."""

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rank: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.rank = int(rank)
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.mu_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.diag_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.lowrank_head = nn.Linear(self.hidden_dim, self.input_dim * self.rank)

    def _distribution(self, h_last: torch.Tensor) -> LowRankMultivariateNormal:
        mu = self.mu_head(h_last)
        diag = F.softplus(self.diag_head(h_last)) + 1e-3
        cov_factor = self.lowrank_head(h_last).reshape(-1, self.input_dim, self.rank)
        return LowRankMultivariateNormal(mu, cov_factor, diag)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.lstm(history)
        prev = history[:, -1:, :]
        loss = future.new_tensor(0.0)
        for step in range(future.shape[1]):
            _, (h, c) = self.lstm(prev, (h, c))
            dist = self._distribution(h[-1])
            target = future[:, step, :]
            loss = loss - dist.log_prob(target).mean()
            prev = target.unsqueeze(1)
        return loss / future.shape[1]

    @torch.no_grad()
    def sample_trajectory(self, history: torch.Tensor, n_future: int) -> torch.Tensor:
        _, (h, c) = self.lstm(history)
        prev = history[:, -1:, :]
        out = []
        for _step in range(int(n_future)):
            _, (h, c) = self.lstm(prev, (h, c))
            sample = self._distribution(h[-1]).rsample()
            out.append(sample)
            prev = sample.unsqueeze(1)
        return torch.stack(out, dim=1)


class TimeGradDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        diff_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(diff_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(diff_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(
            torch.cat([x_noisy, self.time_proj(self.time_emb(t)), self.cond_proj(cond)], dim=-1)
        )


class CurrentPanelTimeGrad(nn.Module):
    """TimeGrad AR denoising diffusion over 39-dimensional increments."""

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 128,
        n_diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.denoiser = TimeGradDenoiser(self.input_dim, self.hidden_dim)
        betas = torch.linspace(float(beta_start), float(beta_end), self.n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(history)
        h_state = h.squeeze(0)
        loss = future.new_tensor(0.0)
        for step in range(future.shape[1]):
            x0 = future[:, step, :]
            t = torch.randint(0, self.n_diffusion_steps, (x0.shape[0],), device=x0.device)
            noise = torch.randn_like(x0)
            x_noisy = (
                self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x0
                + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise
            )
            loss = loss + F.mse_loss(self.denoiser(x_noisy, t, h_state), noise)
            _, h = self.gru(x0.unsqueeze(1), h_state.unsqueeze(0))
            h_state = h.squeeze(0)
        return loss / future.shape[1]

    @torch.no_grad()
    def _sample_next(self, h_state: torch.Tensor, n_ddim_steps: int) -> torch.Tensor:
        bsz = h_state.shape[0]
        x = torch.randn(bsz, self.input_dim, device=h_state.device)
        step_indices = torch.linspace(
            0,
            self.n_diffusion_steps - 1,
            int(n_ddim_steps) + 1,
            device=h_state.device,
        ).long()
        timesteps = step_indices.flip(0)[:-1]
        for i, t_val in enumerate(timesteps):
            t = torch.full((bsz,), int(t_val.item()), device=h_state.device, dtype=torch.long)
            noise_pred = self.denoiser(x, t, h_state)
            alpha_t = self.alphas_cumprod[t_val]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0, device=h_state.device)
            )
            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
        return x

    @torch.no_grad()
    def sample_trajectory(
        self,
        history: torch.Tensor,
        n_future: int,
        n_ddim_steps: int = 10,
    ) -> torch.Tensor:
        _, h = self.gru(history)
        h_state = h.squeeze(0)
        out = []
        for _step in range(int(n_future)):
            sample = self._sample_next(h_state, n_ddim_steps)
            out.append(sample)
            _, h = self.gru(sample.unsqueeze(1), h_state.unsqueeze(0))
            h_state = h.squeeze(0)
        return torch.stack(out, dim=1)


class CurrentPanelPathDiffusion(nn.Module):
    """CSDI conditional diffusion over the whole future path."""

    def __init__(
        self,
        input_dim: int = 39,
        future_len: int = 30,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        n_diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_len = int(future_len)
        self.path_dim = self.input_dim * self.future_len
        self.cond_dim = int(cond_dim)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.encoder = nn.GRU(self.input_dim, self.cond_dim, batch_first=True)
        self.time_emb = SinusoidalPositionEmbeddings(64)
        self.time_proj = nn.Sequential(nn.Linear(64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_proj = nn.Linear(self.cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.path_dim + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.path_dim),
        )
        betas = torch.linspace(1e-4, 0.02, self.n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def _condition(self, history: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(history)
        return h.squeeze(0)

    def _predict_noise(self, x_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(
            torch.cat([x_noisy, self.time_proj(self.time_emb(t)), self.cond_proj(cond)], dim=-1)
        )

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        cond = self._condition(history)
        x0 = future.reshape(future.shape[0], -1)
        t = torch.randint(0, self.n_diffusion_steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_noisy = (
            self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x0
            + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise
        )
        return F.mse_loss(self._predict_noise(x_noisy, t, cond), noise)

    @torch.no_grad()
    def sample_trajectory(self, history: torch.Tensor, n_ddim_steps: int = 20) -> torch.Tensor:
        cond = self._condition(history)
        bsz = history.shape[0]
        x = torch.randn(bsz, self.path_dim, device=history.device)
        step_indices = torch.linspace(
            0,
            self.n_diffusion_steps - 1,
            int(n_ddim_steps) + 1,
            device=history.device,
        ).long()
        timesteps = step_indices.flip(0)[:-1]
        for i, t_val in enumerate(timesteps):
            t = torch.full((bsz,), int(t_val.item()), device=history.device, dtype=torch.long)
            noise_pred = self._predict_noise(x, t, cond)
            alpha_t = self.alphas_cumprod[t_val]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else torch.tensor(1.0, device=history.device)
            )
            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred
        return x.reshape(bsz, self.future_len, self.input_dim)


class CurrentPanelDeepBaseline:
    """Wrapper exposing the current-panel sample_joint interface."""

    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        model_type: str,
        device: str | torch.device = "cpu",
        future_len: int = 30,
        sample_steps: int = 10,
        chunk_size: int = 512,
    ) -> None:
        self.model = model.to(device).eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.device = torch.device(device)
        self.model_type = model_type
        self.future_len = int(future_len)
        self.sample_steps = int(sample_steps)
        self.chunk_size = int(chunk_size)

    def eval(self) -> "CurrentPanelDeepBaseline":
        self.model.eval()
        return self

    def sample_joint(
        self,
        history_changes: np.ndarray,
        n_samples: int = 64,
        **_: Any,
    ) -> np.ndarray:
        history = np.asarray(history_changes, dtype=np.float32)
        bsz, hist_len, dim = history.shape
        history_z = (history - self.mean[None, None, :]) / self.std[None, None, :]
        repeated = np.repeat(history_z[:, None, :, :], int(n_samples), axis=1).reshape(
            bsz * int(n_samples),
            hist_len,
            dim,
        )
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, repeated.shape[0], self.chunk_size):
                h = torch.from_numpy(repeated[start : start + self.chunk_size]).to(self.device)
                if self.model_type == "path_diffusion":
                    sample_z = self.model.sample_trajectory(h, n_ddim_steps=self.sample_steps)
                else:
                    sample_z = self.model.sample_trajectory(
                        h,
                        n_future=self.future_len,
                        n_ddim_steps=self.sample_steps,
                    ) if self.model_type == "timegrad" else self.model.sample_trajectory(
                        h,
                        n_future=self.future_len,
                    )
                outputs.append(sample_z.detach().cpu().numpy())
        out_z = np.concatenate(outputs, axis=0).reshape(bsz, int(n_samples), self.future_len, dim)
        out = out_z * self.std[None, None, None, :] + self.mean[None, None, None, :]
        return out.astype(np.float32)


def build_model(model_type: str, input_dim: int, future_len: int = 30, **kwargs: Any) -> nn.Module:
    if model_type == "deepvar":
        return CurrentPanelDeepVAR(
            input_dim=input_dim,
            hidden_dim=int(kwargs.get("hidden_dim", 128)),
            num_layers=int(kwargs.get("num_layers", 2)),
            rank=int(kwargs.get("rank", 5)),
        )
    if model_type == "timegrad":
        return CurrentPanelTimeGrad(
            input_dim=input_dim,
            hidden_dim=int(kwargs.get("hidden_dim", 128)),
            n_diffusion_steps=int(kwargs.get("n_diffusion_steps", 50)),
        )
    if model_type == "path_diffusion":
        return CurrentPanelPathDiffusion(
            input_dim=input_dim,
            future_len=future_len,
            hidden_dim=int(kwargs.get("hidden_dim", 256)),
            cond_dim=int(kwargs.get("cond_dim", 128)),
            n_diffusion_steps=int(kwargs.get("n_diffusion_steps", 50)),
        )
    raise ValueError(f"unknown current-panel deep baseline {model_type!r}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_type: str,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    payload = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "config": config,
        "best_val_loss": float(best_val_loss),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_current_panel_deep_baseline(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    sample_steps: int | None = None,
    chunk_size: int = 512,
) -> CurrentPanelDeepBaseline:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    model_type = ckpt["model_type"]
    model_kwargs = {
        k: v for k, v in config.items() if k not in {"model_type", "input_dim", "future_len"}
    }
    model = build_model(
        model_type,
        input_dim=int(config["input_dim"]),
        future_len=int(config.get("future_len", 30)),
        **model_kwargs,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return CurrentPanelDeepBaseline(
        model,
        ckpt["mean"],
        ckpt["std"],
        model_type=model_type,
        device=device,
        future_len=int(config.get("future_len", 30)),
        sample_steps=int(sample_steps if sample_steps is not None else config.get("sample_steps", 10)),
        chunk_size=int(chunk_size),
    )
