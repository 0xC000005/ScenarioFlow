"""Final DeepVAR, TimeGrad, and CSDI-style current-panel baselines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torch.distributions import LowRankMultivariateNormal


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        embedded = time[:, None].float() * frequencies[None, :]
        return torch.cat([embedded.sin(), embedded.cos()], dim=-1)


class CurrentPanelDeepVAR(nn.Module):
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
        self.rank = int(rank)
        self.lstm = nn.LSTM(
            self.input_dim,
            int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.mean_head = nn.Linear(int(hidden_dim), self.input_dim)
        self.diagonal_head = nn.Linear(int(hidden_dim), self.input_dim)
        self.low_rank_head = nn.Linear(int(hidden_dim), self.input_dim * self.rank)

    def _distribution(self, hidden: torch.Tensor) -> LowRankMultivariateNormal:
        mean = self.mean_head(hidden)
        diagonal = functional.softplus(self.diagonal_head(hidden)) + 1e-3
        factor = self.low_rank_head(hidden).reshape(-1, self.input_dim, self.rank)
        return LowRankMultivariateNormal(mean, factor, diagonal)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, state = self.lstm(history)
        previous = history[:, -1:, :]
        loss = future.new_tensor(0.0)
        for step in range(future.shape[1]):
            _, state = self.lstm(previous, state)
            loss = (
                loss - self._distribution(state[0][-1]).log_prob(future[:, step]).mean()
            )
            previous = future[:, step : step + 1]
        return loss / future.shape[1]

    @torch.no_grad()
    def sample_trajectory(self, history: torch.Tensor, n_future: int) -> torch.Tensor:
        _, state = self.lstm(history)
        previous = history[:, -1:, :]
        output = []
        for _ in range(int(n_future)):
            _, state = self.lstm(previous, state)
            sample = self._distribution(state[0][-1]).rsample()
            output.append(sample)
            previous = sample[:, None, :]
        return torch.stack(output, dim=1)


class _Denoiser(nn.Module):
    def __init__(
        self, input_dim: int, condition_dim: int, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(64)
        self.time_projection = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self, noisy: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.network(
            torch.cat(
                [
                    noisy,
                    self.time_projection(self.time_embedding(time)),
                    self.condition_projection(condition),
                ],
                dim=-1,
            )
        )


class CurrentPanelTimeGrad(nn.Module):
    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 128,
        n_diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.gru = nn.GRU(self.input_dim, int(hidden_dim), batch_first=True)
        self.denoiser = _Denoiser(self.input_dim, int(hidden_dim))
        betas = torch.linspace(1e-4, 0.02, self.n_diffusion_steps)
        cumulative = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(cumulative))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - cumulative)
        )

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history)
        state = hidden.squeeze(0)
        loss = future.new_tensor(0.0)
        for step in range(future.shape[1]):
            target = future[:, step]
            time = torch.randint(
                0, self.n_diffusion_steps, (target.shape[0],), device=target.device
            )
            noise = torch.randn_like(target)
            noisy = (
                self.sqrt_alphas_cumprod[time, None] * target
                + self.sqrt_one_minus_alphas_cumprod[time, None] * noise
            )
            loss = loss + functional.mse_loss(self.denoiser(noisy, time, state), noise)
            _, hidden = self.gru(target[:, None, :], state[None, :, :])
            state = hidden.squeeze(0)
        return loss / future.shape[1]

    @torch.no_grad()
    def _sample_next(self, state: torch.Tensor, steps: int) -> torch.Tensor:
        value = torch.randn(state.shape[0], self.input_dim, device=state.device)
        times = (
            torch.linspace(
                0, self.n_diffusion_steps - 1, int(steps) + 1, device=state.device
            )
            .long()
            .flip(0)[:-1]
        )
        for position, time_value in enumerate(times):
            time = torch.full(
                (state.shape[0],),
                int(time_value),
                device=state.device,
                dtype=torch.long,
            )
            predicted = self.denoiser(value, time, state)
            alpha = self.alphas_cumprod[time_value]
            previous = (
                self.alphas_cumprod[times[position + 1]]
                if position + 1 < len(times)
                else value.new_tensor(1.0)
            )
            clean = (value - torch.sqrt(1 - alpha) * predicted) / torch.sqrt(alpha)
            value = torch.sqrt(previous) * clean + torch.sqrt(1 - previous) * predicted
        return value

    @torch.no_grad()
    def sample_trajectory(
        self, history: torch.Tensor, n_future: int, n_ddim_steps: int = 10
    ) -> torch.Tensor:
        _, hidden = self.gru(history)
        state = hidden.squeeze(0)
        output = []
        for _ in range(int(n_future)):
            sample = self._sample_next(state, n_ddim_steps)
            output.append(sample)
            _, hidden = self.gru(sample[:, None, :], state[None, :, :])
            state = hidden.squeeze(0)
        return torch.stack(output, dim=1)


class CurrentPanelPathDiffusion(nn.Module):
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
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.encoder = nn.GRU(self.input_dim, int(cond_dim), batch_first=True)
        self.denoiser = _Denoiser(self.path_dim, int(cond_dim), int(hidden_dim))
        betas = torch.linspace(1e-4, 0.02, self.n_diffusion_steps)
        cumulative = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(cumulative))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - cumulative)
        )

    def _condition(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)[1].squeeze(0)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        target = future.reshape(future.shape[0], -1)
        time = torch.randint(
            0, self.n_diffusion_steps, (target.shape[0],), device=target.device
        )
        noise = torch.randn_like(target)
        noisy = (
            self.sqrt_alphas_cumprod[time, None] * target
            + self.sqrt_one_minus_alphas_cumprod[time, None] * noise
        )
        return functional.mse_loss(
            self.denoiser(noisy, time, self._condition(history)), noise
        )

    @torch.no_grad()
    def sample_trajectory(
        self, history: torch.Tensor, n_ddim_steps: int = 20
    ) -> torch.Tensor:
        condition = self._condition(history)
        value = torch.randn(history.shape[0], self.path_dim, device=history.device)
        times = (
            torch.linspace(
                0,
                self.n_diffusion_steps - 1,
                int(n_ddim_steps) + 1,
                device=history.device,
            )
            .long()
            .flip(0)[:-1]
        )
        for position, time_value in enumerate(times):
            time = torch.full(
                (history.shape[0],),
                int(time_value),
                device=history.device,
                dtype=torch.long,
            )
            predicted = self.denoiser(value, time, condition)
            alpha = self.alphas_cumprod[time_value]
            previous = (
                self.alphas_cumprod[times[position + 1]]
                if position + 1 < len(times)
                else value.new_tensor(1.0)
            )
            clean = (value - torch.sqrt(1 - alpha) * predicted) / torch.sqrt(alpha)
            value = torch.sqrt(previous) * clean + torch.sqrt(1 - previous) * predicted
        return value.reshape(history.shape[0], self.future_len, self.input_dim)


def build_model(
    model_type: str, input_dim: int, future_len: int = 30, **kwargs: Any
) -> nn.Module:
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
    raise ValueError(f"unknown deep baseline {model_type!r}")


class CurrentPanelDeepBaseline:
    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        model_type: str,
        device: str | torch.device,
        future_len: int,
        sample_steps: int,
        chunk_size: int,
    ) -> None:
        self.model = model.to(device).eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.model_type = model_type
        self.device = torch.device(device)
        self.future_len = int(future_len)
        self.sample_steps = int(sample_steps)
        self.chunk_size = int(chunk_size)

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 64, **_: Any
    ) -> np.ndarray:
        history = np.asarray(history_changes, dtype=np.float32)
        batch, length, dimension = history.shape
        normalized = (history - self.mean[None, None, :]) / self.std[None, None, :]
        repeated = np.repeat(normalized[:, None], int(n_samples), axis=1).reshape(
            -1, length, dimension
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, repeated.shape[0], self.chunk_size):
                chunk = torch.from_numpy(repeated[start : start + self.chunk_size]).to(
                    self.device
                )
                if self.model_type == "path_diffusion":
                    sample = self.model.sample_trajectory(
                        chunk, n_ddim_steps=self.sample_steps
                    )
                elif self.model_type == "timegrad":
                    sample = self.model.sample_trajectory(
                        chunk, n_future=self.future_len, n_ddim_steps=self.sample_steps
                    )
                else:
                    sample = self.model.sample_trajectory(
                        chunk, n_future=self.future_len
                    )
                outputs.append(sample.cpu().numpy())
        values = np.concatenate(outputs).reshape(
            batch, int(n_samples), self.future_len, dimension
        )
        return (
            values * self.std[None, None, None, :] + self.mean[None, None, None, :]
        ).astype(np.float32)


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
            "config": config,
            "best_val_loss": float(best_val_loss),
        },
        path,
    )


def load_current_panel_deep_baseline(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    sample_steps: int | None = None,
    chunk_size: int = 512,
) -> CurrentPanelDeepBaseline:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    model_type = str(checkpoint["model_type"])
    model_kwargs = {
        key: value
        for key, value in config.items()
        if key
        not in {"model_type", "input_dim", "future_len", "history_len", "sample_steps"}
    }
    model = build_model(
        model_type,
        input_dim=int(config["input_dim"]),
        future_len=int(config.get("future_len", 30)),
        **model_kwargs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return CurrentPanelDeepBaseline(
        model,
        checkpoint["mean"],
        checkpoint["std"],
        model_type=model_type,
        device=device,
        future_len=int(config.get("future_len", 30)),
        sample_steps=int(sample_steps or config.get("sample_steps", 10)),
        chunk_size=chunk_size,
    )
