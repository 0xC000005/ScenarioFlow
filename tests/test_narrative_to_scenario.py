"""Cached and numerical tests for the narrative-to-scenario demo."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scenarioflow_demo.narrative_to_scenario import (
    NarrativeAdapter,
    PrefixFeatureLayout,
    allocate_weighted_samples,
    build_prefix_feature_matrix,
    fit_state_alignment,
    path_quantiles,
    reconstruct_prefix_from_features,
    retrieve_support,
    terminal_delta_summary,
)
from scenarioflow_demo.openai_conditioning import (
    ConditionedNarrative,
    ConditionMarketImplication,
    ConditionOnlyGroundingResult,
    OpenAINarrativeConditioner,
    grounding_messages,
)
from scenarioflow_demo.narrative_to_scenario_app import (
    NarrativeToScenarioRuntime,
    NarrativeToScenarioService,
    build_narrative_to_scenario_app,
)


def _spec_names() -> list[str]:
    return [f"iv:{index:02d}" for index in range(25)] + [
        "factor:spx",
        "factor:us2y",
        "factor:us10y",
        "factor:bbb_oas",
        "factor:aaa_oas",
        "factor:usdjpy",
        "factor:dxy",
        "factor:gold",
        "factor:crude_oil",
        "factor:vix",
        "factor:unused_1",
        "factor:unused_2",
        "factor:unused_3",
        "factor:unused_4",
    ]


def test_prefix_features_round_trip_and_pin_the_requested_start() -> None:
    rng = np.random.default_rng(17)
    history_level = rng.normal(size=(3, 4, 5)).astype(np.float32)
    history_norm = rng.normal(size=(3, 4, 5)).astype(np.float32)
    center = rng.normal(size=(3, 5)).astype(np.float32)
    scale = np.exp(rng.normal(size=(3, 5))).astype(np.float32)
    drift = rng.normal(size=(3, 5)).astype(np.float32)
    features, layout = build_prefix_feature_matrix(
        history_level, history_norm, center, scale, drift
    )
    assert isinstance(layout, PrefixFeatureLayout)

    requested_start = rng.normal(size=(3, 5)).astype(np.float32)
    decoded = reconstruct_prefix_from_features(
        features, start_state=requested_start, layout=layout
    )
    assert np.allclose(decoded["history_level"][:, -1], requested_start)
    assert np.allclose(
        decoded["history_level"] - requested_start[:, None, :],
        history_level - history_level[:, -1:, :],
    )
    assert np.allclose(decoded["history_norm"], history_norm)
    assert np.allclose(decoded["center"], center)
    assert np.allclose(decoded["scale"], scale)
    assert np.allclose(decoded["drift_feature"], drift)


def test_state_alignment_learns_memory_and_start_to_prefix_mapping() -> None:
    rng = np.random.default_rng(19)
    memory = rng.normal(size=(24, 3)).astype(np.float32)
    start = rng.normal(size=(24, 2)).astype(np.float32)
    features = np.concatenate(
        [memory, start, memory[:, :1] + start[:, :1]], axis=1
    ).astype(np.float32)
    fitted = fit_state_alignment(
        memory,
        start,
        features,
        train_indices=np.arange(20),
        test_indices=np.arange(20, 24),
        hidden_dim=16,
        steps=120,
        batch_size=12,
        lr=5e-3,
        seed=31,
    )
    predicted = fitted.predict_features(memory[20:], start[20:])
    assert predicted.shape == (4, features.shape[1])
    assert np.isfinite(predicted).all()
    assert fitted.loss_last < fitted.loss_first


def test_openai_conditioner_projects_structured_grounding_without_a_live_call(
    tmp_path: Path,
) -> None:
    adapter = NarrativeAdapter(embedding_dim=4, condition_dim=2, hidden_dim=3)
    checkpoint = tmp_path / "bridge.pt"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "embedding_dim": 4,
            "condition_dim": 2,
        },
        checkpoint,
    )
    grounding = ConditionOnlyGroundingResult(
        narrative_frame="Dollar liquidity pressure",
        current_market_state_summary="The dollar is firm.",
        recent_regime_summary="Credit spreads recently widened.",
        cleaned_conditioning_text="A firm dollar accompanies wider credit spreads.",
        current_market_state_implications=[
            ConditionMarketImplication(
                market="DXY",
                direction="up",
                magnitude="medium",
                confidence="high",
                horizon="current_state",
                evidence=["the dollar is firm"],
                rationale="The narrative directly describes the dollar.",
            )
        ],
    )
    observed: dict[str, object] = {}

    class Responses:
        def parse(self, **kwargs):
            observed["grounding_model"] = kwargs["model"]
            observed["text_format"] = kwargs["text_format"]
            return type("Response", (), {"output_parsed": grounding, "id": "cached"})()

    class Embeddings:
        def create(self, **kwargs):
            observed["embedding_model"] = kwargs["model"]
            observed["embedding_input"] = kwargs["input"]
            item = type(
                "Embedding", (), {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}
            )()
            return type("EmbeddingResponse", (), {"data": [item]})()

    client = type(
        "Client", (), {"responses": Responses(), "embeddings": Embeddings()}
    )()
    result = OpenAINarrativeConditioner(checkpoint, client=client).condition(
        "Dollar liquidity tightens."
    )
    assert result.query_memory.shape == (2,)
    assert np.isfinite(result.query_memory).all()
    assert observed["grounding_model"] == "gpt-5.4-mini"
    assert observed["embedding_model"] == "text-embedding-3-small"
    assert observed["text_format"] is ConditionOnlyGroundingResult
    assert "CURRENT_MARKET_STATE_SUMMARY" in str(observed["embedding_input"])
    assert "Dollar liquidity tightens." not in str(observed["embedding_input"])


def test_grounding_prompt_separates_current_conditions_from_forward_language() -> None:
    messages = grounding_messages("SPX is down. It will rebound next month.")
    prompt = messages[1]["content"]
    conditioning, warning = prompt.split("Warning-only forward or forecast sentences.")
    assert "- SPX is down." in conditioning
    assert "It will rebound next month." not in conditioning
    assert "- It will rebound next month." in warning


def test_retrieval_separates_narrative_and_start_only_support() -> None:
    memory = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    history_level = np.zeros((4, 2, 2), dtype=np.float32)
    history_level[:, -1] = np.asarray(
        [[8.0, 8.0], [9.0, 9.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32
    )
    train = np.arange(4, dtype=np.int64)
    query_start = np.asarray([0.0, 0.0], dtype=np.float32)

    narrative = retrieve_support(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=memory,
        history_level=history_level,
        train_indices=train,
        query_start_state=query_start,
        mode="narrative",
        top_k=2,
        min_index_gap=0,
    )
    start_only = retrieve_support(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=memory,
        history_level=history_level,
        train_indices=train,
        query_start_state=query_start,
        mode="start_only",
        top_k=2,
        min_index_gap=30,
    )
    assert narrative.window_indices[0] == 0
    assert start_only.window_indices == (2, 3)
    assert np.isclose(sum(narrative.weights), 1.0)
    assert np.isclose(sum(start_only.weights), 1.0)


def test_weighted_sample_allocation_is_deterministic_and_exact() -> None:
    first = allocate_weighted_samples(np.asarray([0.55, 0.30, 0.15]), 48)
    second = allocate_weighted_samples(np.asarray([0.55, 0.30, 0.15]), 48)
    assert np.array_equal(first, second)
    assert first.sum() == 48
    assert first.tolist() == [27, 14, 7]


def test_path_and_terminal_summaries_have_the_frontend_markets() -> None:
    rng = np.random.default_rng(23)
    start = np.linspace(0.2, 39.2, 39, dtype=np.float32)
    states = start[None, None, :] + rng.normal(size=(12, 3, 39)).astype(np.float32)
    rows = path_quantiles(states, start, _spec_names(), max_paths=4)
    terminal = terminal_delta_summary(states, start, _spec_names())
    markets = {row["market"] for row in rows}
    assert {
        "IV_SURFACE",
        "IV_ATM_3M",
        "IV_ATM_1Y",
        "IV_OTM_PUT_1Y",
        "IV_WING_6M_K130",
        "SPX",
        "VIX",
    }.issubset(markets)
    assert len(rows) == 15
    assert len(terminal) == 11
    assert all(len(row["p50"]) == 3 for row in rows)
    assert all(len(row["sample_paths"]) <= 4 for row in rows)
    assert all("terminal_probability_up" in row for row in terminal)
    assert all("terminal_probability_flat" in row for row in terminal)
    assert all(row["terminal_sample_count"] == 12 for row in terminal)


def test_runtime_resolves_only_semantic_artifact_ids(tmp_path: Path) -> None:
    ids = {
        "nl.bridge-adapter",
        "nl.full-start-bridge-arrays",
        "nl.full-start-bridge-report",
        "nl.support-bank",
        "nl.support-bank-report",
        "core.stage2",
    }

    class Resolver:
        def resolve(self, artifact_id: str, artifact_root: str | Path) -> Path:
            assert artifact_id in ids
            return Path(artifact_root) / f"{artifact_id}.bin"

    runtime = NarrativeToScenarioRuntime.from_artifacts(
        resolver=Resolver(), artifact_root=tmp_path
    )
    assert runtime.stage2_checkpoint.name == "core.stage2.bin"
    assert runtime.bridge_adapter.name == "nl.bridge-adapter.bin"


def test_service_runs_bridge_retrieval_alignment_and_stage2_without_openai(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(43)
    n_windows, history_len, n_cells = 8, 2, 39
    memory = rng.normal(size=(n_windows, 2)).astype(np.float32)
    memory[0] = [1.0, 0.0]
    history_level = rng.normal(size=(n_windows, history_len, n_cells)).astype(
        np.float32
    )
    history_norm = rng.normal(size=(n_windows, history_len, n_cells)).astype(np.float32)
    history_raw = np.abs(history_level) + 10.0
    center = np.zeros((n_windows, n_cells), dtype=np.float32)
    scale = np.ones((n_windows, n_cells), dtype=np.float32)
    drift = np.zeros((n_windows, n_cells), dtype=np.float32)
    support = tmp_path / "support.npz"
    np.savez(
        support,
        memory_targets=memory,
        history_level=history_level,
        history_norm=history_norm,
        history_raw=history_raw,
        center=center,
        scale=scale,
        drift_feature=drift,
        train_indices=np.arange(6, dtype=np.int64),
        test_indices=np.arange(6, 8, dtype=np.int64),
        support_indices=np.arange(8, dtype=np.int64),
    )
    full_arrays = tmp_path / "full.npz"
    np.savez(full_arrays, memory_targets=memory, condition_vectors=memory)
    metadata = [
        {
            "window_index": index,
            "window_id": f"window-{index}",
            "calendar_start_date": f"2016-03-{index + 1:02d}",
            "calendar_end_date": f"2016-04-{index + 1:02d}",
        }
        for index in range(n_windows)
    ]
    full_report = tmp_path / "full.json"
    full_report.write_text(json.dumps({"window_metadata": metadata}), encoding="utf-8")
    support_report = tmp_path / "support.json"
    support_report.write_text(
        json.dumps({"window_metadata": metadata}), encoding="utf-8"
    )
    unused = tmp_path / "unused.bin"
    unused.write_bytes(b"unused")
    runtime = NarrativeToScenarioRuntime(
        bridge_adapter=unused,
        full_start_arrays=full_arrays,
        full_start_report=full_report,
        support_arrays=support,
        support_report=support_report,
        stage2_checkpoint=unused,
    )
    specs = [
        {
            "name": name,
            "source_column": name,
            "source_index": index,
            "transform": "diff_level",
            "reference_increment_column": None,
            "reference_increment_index": None,
            "lower_bound": None,
            "upper_bound": None,
        }
        for index, name in enumerate(_spec_names())
    ]

    class Model:
        cfg = SimpleNamespace(future_len=3)

        def sample_batched(self, history, _norm, _center, _scale, **kwargs):
            return torch.full(
                (
                    history.shape[0],
                    int(kwargs["n_samples"]),
                    int(kwargs["n_steps"]),
                    history.shape[-1],
                ),
                0.1,
                dtype=torch.float32,
                device=history.device,
            )

    grounding = ConditionOnlyGroundingResult(
        narrative_frame="Dollar pressure",
        current_market_state_summary="The dollar is firm.",
        recent_regime_summary="No separate recent regime is stated.",
        cleaned_conditioning_text="The current dollar level is firm.",
    )

    class Conditioner:
        calls = 0

        def condition(self, _story: str):
            self.calls += 1
            return ConditionedNarrative(
                grounding=grounding,
                query_text="cached condition",
                query_memory=np.asarray(memory[7], dtype=np.float32),
            )

    conditioner = Conditioner()
    service = NarrativeToScenarioService(
        runtime,
        conditioner=conditioner,
        model_loader=lambda _path, _device: (Model(), {"state_specs": specs}),
        decoder_steps=12,
        decoder_hidden_dim=12,
        top_k=3,
        min_index_gap=0,
        device="cpu",
    )
    assert service.start_choices()[0] == ("2016-04-01", 0)
    assert len(service.context(0)["SPX"]) == history_len
    report = service.generate("Dollar liquidity tightens.", 0, samples=12)
    assert conditioner.calls == 1
    assert len(report["generation"]["path_quantiles"]) == 15
    assert len(report["generation"]["terminal_delta_summary"]) == 11
    assert report["generation"]["start_only_baseline"]["path_quantiles"]
    assert 1 <= len(report["generation"]["posterior_ensemble"]["selected_support"]) <= 3
    assert "window-7" in {
        row["window_id"]
        for row in report["generation"]["posterior_ensemble"]["selected_support"]
    }


def test_cached_app_response_uses_four_inputs_and_makes_no_api_call() -> None:
    pytest.importorskip("gradio")
    cached = json.loads(
        Path("tests/fixtures/demo/narrative_to_scenario.json").read_text(
            encoding="utf-8"
        )
    )

    class CachedService:
        def start_choices(self):
            return (("2016-03-22", 120),)

        def context(self, window_index: int):
            assert window_index == 120
            return cached["context"]

        def generate(self, story: str, window_index: int, *, samples: int = 48):
            assert story == "Dollar liquidity tightens."
            assert window_index == 120
            assert samples == 48
            return cached["report"]

    app = build_narrative_to_scenario_app(CachedService())
    dependencies = {
        dependency["api_name"]: dependency
        for dependency in app.config["dependencies"]
        if dependency.get("api_name")
    }
    assert {"run_live_openai_prefix_for_app", "start_window_context"}.issubset(
        dependencies
    )
    assert len(dependencies["run_live_openai_prefix_for_app"]["inputs"]) == 4
    dropdown = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("label") == "Starting market state"
    )
    assert [list(choice) for choice in dropdown["props"]["choices"]] == [
        ["2016-03-22", 120]
    ]


def test_app_error_does_not_expose_runtime_paths() -> None:
    pytest.importorskip("gradio")

    class BrokenService:
        def start_choices(self):
            return (("2016-03-22", 120),)

        def context(self, _window_index: int):
            raise RuntimeError("SECRET_RUNTIME_PATH/data.npz")

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("SECRET_RUNTIME_PATH/model.pt")

    app = build_narrative_to_scenario_app(BrokenService())
    functions = {block.api_name: block.fn for block in app.fns.values()}
    generated = functions["run_live_openai_prefix_for_app"](
        "SPX", "ALL", "Dollar pressure", 120
    )
    context = functions["start_window_context"](120)
    assert "SECRET_RUNTIME_PATH" not in str(generated)
    assert json.loads(generated[7]) == {
        "status": "error",
        "message": "Scenario generation could not complete.",
    }
    assert context == "{}"
