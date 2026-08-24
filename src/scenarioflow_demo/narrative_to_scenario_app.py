"""Gradio API surface for the narrative-to-scenario React demo."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from scenarioflow.artifacts import ArtifactResolver
from scenarioflow.model import load_model
from scenarioflow.state import UnifiedVariableSpec, reconstruct_state_from_increments

from .narrative_to_scenario import (
    SupportSelection,
    allocate_weighted_samples,
    build_prefix_feature_matrix,
    fit_state_alignment,
    path_quantiles,
    reconstruct_prefix_from_features,
    retrieve_support,
    terminal_delta_summary,
)
from .openai_conditioning import OpenAINarrativeConditioner


@dataclass(frozen=True, slots=True)
class NarrativeToScenarioRuntime:
    bridge_adapter: Path
    full_start_arrays: Path
    full_start_report: Path
    support_arrays: Path
    support_report: Path
    stage2_checkpoint: Path

    @classmethod
    def from_artifacts(
        cls,
        *,
        resolver: ArtifactResolver,
        artifact_root: str | Path,
    ) -> "NarrativeToScenarioRuntime":
        return cls(
            bridge_adapter=resolver.resolve("nl.bridge-adapter", artifact_root),
            full_start_arrays=resolver.resolve(
                "nl.full-start-bridge-arrays", artifact_root
            ),
            full_start_report=resolver.resolve(
                "nl.full-start-bridge-report", artifact_root
            ),
            support_arrays=resolver.resolve("nl.support-bank", artifact_root),
            support_report=resolver.resolve("nl.support-bank-report", artifact_root),
            stage2_checkpoint=resolver.resolve("core.stage2", artifact_root),
        )


class NarrativeToScenarioServiceLike(Protocol):
    def start_choices(self) -> tuple[tuple[str, int], ...]: ...

    def context(self, window_index: int) -> dict[str, tuple[float, ...]]: ...

    def generate(
        self, story: str, window_index: int, *, samples: int = 48
    ) -> dict[str, Any]: ...


CONTEXT_COLUMNS = {
    "SPX": 25,
    "USDJPY": 27,
    "DXY": 28,
    "CRUDE_OIL": 31,
    "US2Y": 32,
    "US10Y": 33,
    "AAA_OAS": 34,
    "BBB_OAS": 35,
    "GOLD": 37,
    "VIX": 38,
}


class NarrativeToScenarioService:
    """Lazy ScenarioFlow-NL runtime with no file paths accepted from the browser."""

    def __init__(
        self,
        runtime: NarrativeToScenarioRuntime,
        *,
        conditioner: Any | None = None,
        model_loader: Callable[..., tuple[Any, dict[str, Any]]] = load_model,
        decoder_steps: int = 1000,
        decoder_hidden_dim: int = 256,
        top_k: int = 8,
        min_index_gap: int = 30,
        device: str | torch.device | None = None,
        seed: int = 791,
    ) -> None:
        self.runtime = runtime
        self.conditioner = conditioner
        self.model_loader = model_loader
        self.decoder_steps = int(decoder_steps)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.top_k = int(top_k)
        self.min_index_gap = int(min_index_gap)
        requested = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(
            requested if requested == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.seed = int(seed)
        self._metadata: list[dict[str, Any]] | None = None
        self._arrays: dict[str, np.ndarray] | None = None
        self._alignment: Any | None = None
        self._layout: Any | None = None
        self._model: Any | None = None
        self._specs: list[UnifiedVariableSpec] | None = None

    def _ensure_metadata(self) -> list[dict[str, Any]]:
        if self._metadata is None:
            payload = json.loads(
                self.runtime.full_start_report.read_text(encoding="utf-8")
            )
            raw = payload.get("window_metadata", [])
            if not isinstance(raw, list) or not raw:
                raise ValueError("the start-state artifact has no window metadata")
            self._metadata = [dict(row) for row in raw if isinstance(row, dict)]
        return self._metadata

    def _ensure_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            required = {
                "memory_targets",
                "history_level",
                "history_norm",
                "history_raw",
                "center",
                "scale",
                "drift_feature",
                "train_indices",
                "test_indices",
                "support_indices",
            }
            with np.load(self.runtime.support_arrays, allow_pickle=False) as payload:
                missing = sorted(required.difference(payload.files))
                if missing:
                    raise ValueError(
                        f"support-bank artifact is missing arrays: {missing}"
                    )
                self._arrays = {name: np.asarray(payload[name]) for name in required}
            if self._arrays["history_level"].shape[0] != len(self._ensure_metadata()):
                raise ValueError("support-bank rows do not match start-state metadata")
        return self._arrays

    def _ensure_generator(self) -> None:
        if self._model is not None:
            return
        arrays = self._ensure_arrays()
        with np.load(self.runtime.full_start_arrays, allow_pickle=False) as payload:
            if "memory_targets" not in payload:
                raise ValueError("state-alignment artifact is missing memory_targets")
            alignment_memory = np.asarray(payload["memory_targets"], dtype=np.float32)
        support_memory = np.asarray(arrays["memory_targets"], dtype=np.float32)
        if alignment_memory.shape != support_memory.shape or not np.array_equal(
            alignment_memory, support_memory
        ):
            raise ValueError("state-alignment and support memories do not match")
        features, self._layout = build_prefix_feature_matrix(
            arrays["history_level"],
            arrays["history_norm"],
            arrays["center"],
            arrays["scale"],
            arrays["drift_feature"],
        )
        self._alignment = fit_state_alignment(
            alignment_memory,
            arrays["history_level"][:, -1, :],
            features,
            train_indices=arrays["train_indices"],
            test_indices=arrays["test_indices"],
            hidden_dim=self.decoder_hidden_dim,
            steps=self.decoder_steps,
            batch_size=64,
            lr=1e-3,
            seed=self.seed,
            device=self.device,
        )
        self._model, payload = self.model_loader(
            self.runtime.stage2_checkpoint, self.device
        )
        raw_specs = payload.get("state_specs", [])
        if not isinstance(raw_specs, list) or len(raw_specs) != 39:
            raise ValueError("Stage 2 checkpoint must contain 39 state specifications")
        self._specs = [UnifiedVariableSpec(**dict(row)) for row in raw_specs]
        if hasattr(self._model, "eval"):
            self._model.eval()
        if self.conditioner is None:
            self.conditioner = OpenAINarrativeConditioner(self.runtime.bridge_adapter)

    def start_choices(self) -> tuple[tuple[str, int], ...]:
        rows = []
        for position, metadata in enumerate(self._ensure_metadata()):
            date = str(metadata.get("calendar_end_date", "")).strip()
            index = int(metadata.get("window_index", position))
            if date:
                rows.append((date, index))
        return tuple(rows)

    def context(self, window_index: int) -> dict[str, tuple[float, ...]]:
        arrays = self._ensure_arrays()
        index = int(window_index)
        if index < 0 or index >= arrays["history_raw"].shape[0]:
            raise ValueError("starting market state is outside the released range")
        history = np.asarray(arrays["history_raw"][index], dtype=np.float32)
        return {
            market: tuple(float(value) for value in history[:, column])
            for market, column in CONTEXT_COLUMNS.items()
        }

    def _select_sparse_components(
        self, selection: SupportSelection, counts: np.ndarray
    ) -> list[int]:
        ranked = sorted(
            range(len(selection.window_indices)),
            key=lambda position: (selection.weights[position], -position),
            reverse=True,
        )
        chosen: list[int] = []
        cumulative = 0.0
        for position in ranked:
            if int(counts[position]) <= 0:
                continue
            chosen.append(position)
            cumulative += max(float(selection.weights[position]), 0.0)
            if len(chosen) >= 3 or cumulative >= 0.90:
                break
        return sorted(chosen)

    def _rollout(
        self,
        selection: SupportSelection,
        *,
        start_index: int,
        samples: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]], float]:
        self._ensure_generator()
        assert self._arrays is not None
        assert self._alignment is not None
        assert self._layout is not None
        assert self._model is not None
        assert self._specs is not None
        arrays = self._arrays
        counts = allocate_weighted_samples(np.asarray(selection.weights), samples)
        requested_level = np.asarray(
            arrays["history_level"][start_index, -1], dtype=np.float32
        )
        requested_raw = np.asarray(
            arrays["history_raw"][start_index, -1], dtype=np.float32
        )
        pieces: list[np.ndarray] = []
        slices: list[tuple[int, int]] = []
        cursor = 0
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            for window_index, count in zip(
                selection.window_indices, counts, strict=True
            ):
                count = int(count)
                if count <= 0:
                    slices.append((cursor, cursor))
                    continue
                memory = np.asarray(
                    arrays["memory_targets"][int(window_index)], dtype=np.float32
                )[None, :]
                features = self._alignment.predict_features(
                    memory, requested_level[None, :]
                )
                prefix = reconstruct_prefix_from_features(
                    features,
                    start_state=requested_level[None, :],
                    layout=self._layout,
                )
                tensor = lambda value: torch.from_numpy(value).to(self.device)
                n_steps = min(30, int(self._model.cfg.future_len))
                with torch.no_grad():
                    increments = self._model.sample_batched(
                        tensor(prefix["history_level"]),
                        tensor(prefix["history_norm"]),
                        tensor(prefix["center"]),
                        tensor(prefix["scale"]),
                        drift_feature=tensor(prefix["drift_feature"]),
                        n_samples=count,
                        n_steps=n_steps,
                        chunk_size=min(16, count),
                        temperature=0.5,
                    )
                states = reconstruct_state_from_increments(
                    requested_raw,
                    increments.detach().cpu().numpy()[0],
                    self._specs,
                )
                pieces.append(states)
                slices.append((cursor, cursor + count))
                cursor += count
        if not pieces:
            raise RuntimeError("support retrieval produced no scenario paths")
        pooled = np.concatenate(pieces, axis=0).astype(np.float32)
        delta = pooled - requested_raw[None, None, :]
        mean = np.nanmean(delta, axis=0, keepdims=True)
        pooled = (requested_raw[None, None, :] + mean + 3.5 * (delta - mean)).astype(
            np.float32
        )
        chosen = self._select_sparse_components(selection, counts)
        selected_indices = [
            sample for position in chosen for sample in range(*slices[position])
        ]
        selected_states = pooled[np.asarray(selected_indices, dtype=np.int64)]
        selected_mass = sum(float(selection.weights[position]) for position in chosen)
        metadata = self._ensure_metadata()
        support_rows: list[dict[str, Any]] = []
        for rank, position in enumerate(
            sorted(chosen, key=lambda item: selection.weights[item], reverse=True),
            start=1,
        ):
            window_index = int(selection.window_indices[position])
            candidate = dict(selection.candidates[position])
            window = metadata[window_index]
            base_weight = float(selection.weights[position])
            support_rows.append(
                {
                    **candidate,
                    "rank": rank,
                    "window_id": str(
                        window.get("window_id", f"joint39_train_{window_index:04d}")
                    ),
                    "history_start_date": str(window.get("calendar_start_date", "")),
                    "base_support_weight": base_weight,
                    "posterior_weight": base_weight / max(selected_mass, 1e-12),
                    "component_sample_count": int(counts[position]),
                }
            )
        return selected_states, support_rows, selected_mass

    def _generation(
        self,
        selection: SupportSelection,
        *,
        start_index: int,
        samples: int,
    ) -> dict[str, Any]:
        assert self._arrays is not None
        assert self._specs is not None
        states, support, weight_mass = self._rollout(
            selection, start_index=start_index, samples=samples
        )
        start = np.asarray(
            self._arrays["history_raw"][start_index, -1], dtype=np.float32
        )
        spec_names = [spec.name for spec in self._specs]
        return {
            "path_quantiles": path_quantiles(states, start, spec_names),
            "terminal_delta_summary": terminal_delta_summary(states, start, spec_names),
            "posterior_ensemble": {
                "mode": "nearest_similar_main_regime_top3_90",
                "selected_component_count": len(support),
                "posterior_sample_count": int(states.shape[0]),
                "base_weight_mass": float(weight_mass),
                "selected_support": support,
            },
        }

    def generate(
        self, story: str, window_index: int, *, samples: int = 48
    ) -> dict[str, Any]:
        self._ensure_generator()
        assert self._arrays is not None
        assert self._specs is not None
        index = int(window_index)
        if index < 0 or index >= self._arrays["history_level"].shape[0]:
            raise ValueError("starting market state is outside the released range")
        conditioned = self.conditioner.condition(story)
        implications = [
            item.model_dump()
            for item in (
                conditioned.grounding.current_market_state_implications
                + conditioned.grounding.recent_regime_implications
            )
        ]
        common = {
            "query_memory": conditioned.query_memory,
            "memory_targets": self._arrays["memory_targets"],
            "history_level": self._arrays["history_level"],
            "train_indices": self._arrays["support_indices"],
            "query_start_state": self._arrays["history_level"][index, -1],
            "top_k": self.top_k,
            "min_index_gap": self.min_index_gap,
        }
        narrative = retrieve_support(
            **common,
            mode="narrative",
            implications=implications,
            spec_names=[spec.name for spec in self._specs],
        )
        baseline = retrieve_support(**common, mode="start_only")
        generation = self._generation(narrative, start_index=index, samples=samples)
        baseline_generation = self._generation(
            baseline, start_index=index, samples=samples
        )
        generation["start_only_baseline"] = {
            "memory_prior_mode": "soft_topk_start_only",
            **baseline_generation,
        }
        return {
            "status": "ok",
            "window_index": index,
            "conditioning": conditioned.grounding.model_dump(),
            "generation": generation,
        }


def build_narrative_to_scenario_app(service: NarrativeToScenarioServiceLike):
    """Expose the two endpoints consumed by the React client under ``/g1``."""
    import gradio as gr

    choices = service.start_choices()
    if not choices:
        raise ValueError("no narrative-to-scenario starting dates are available")

    def generate_endpoint(
        _fan_market: str,
        _analogue_scope: str,
        story: str,
        window_index: float | int,
    ) -> tuple[str, str, str, str, str, str, str, str]:
        try:
            report = service.generate(story, int(window_index), samples=48)
        except Exception:
            report = {
                "status": "error",
                "message": "Scenario generation could not complete.",
            }
        return ("", "", "", "", "", "", "", json.dumps(report))

    def context_endpoint(window_index: float | int) -> str:
        try:
            return json.dumps(service.context(int(window_index)))
        except Exception:
            return "{}"

    with gr.Blocks(title="ScenarioFlow narrative to scenario") as app:
        start = gr.Dropdown(
            choices=list(choices),
            value=choices[0][1],
            label="Starting market state",
        )
        fan_market = gr.Textbox(value="SPX", visible=False)
        analogue_scope = gr.Textbox(value="ALL", visible=False)
        story = gr.Textbox(visible=False)
        outputs = [gr.Textbox(visible=False) for _ in range(8)]
        context = gr.Textbox(visible=False)
        generate_button = gr.Button(visible=False)
        context_button = gr.Button(visible=False)
        generate_button.click(
            generate_endpoint,
            inputs=[fan_market, analogue_scope, story, start],
            outputs=outputs,
            api_name="run_live_openai_prefix_for_app",
        )
        context_button.click(
            context_endpoint,
            inputs=[start],
            outputs=[context],
            api_name="start_window_context",
        )
    return app
