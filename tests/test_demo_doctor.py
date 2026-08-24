"""Capability tests for the non-invasive optional demo doctor."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import subprocess

from scenarioflow.artifacts import ArtifactResolver
from scenarioflow.demo_doctor import inspect_demo_capabilities


def _resolver_with_runtime(
    tmp_path: Path, *, present: bool
) -> tuple[ArtifactResolver, Path]:
    artifacts = (
        ("nl.episode-cards", "nl/runtime/episode_cards/cards.jsonl", "nl-runtime,all"),
        ("nl.bridge-adapter", "nl/runtime/bridge/bridge_adapter.pt", "nl-runtime,all"),
        (
            "nl.full-start-bridge-arrays",
            "nl/runtime/state_alignment/arrays.npz",
            "nl-runtime,all",
        ),
        (
            "nl.full-start-bridge-report",
            "nl/runtime/state_alignment/report.json",
            "nl-runtime,all",
        ),
        ("nl.support-bank", "nl/runtime/support_bank/arrays.npz", "nl-runtime,all"),
        (
            "nl.support-bank-report",
            "nl/runtime/support_bank/report.json",
            "nl-runtime,all",
        ),
        ("core.stage2", "models/core/stage2/best_model.pt", "core,all"),
    )
    manifest = tmp_path / "manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "schema_version",
                "artifact_id",
                "profile",
                "category",
                "relative_path",
                "bytes",
                "sha256",
                "frozen_source_path",
                "url_path",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for artifact_id, relative, profile in artifacts:
            payload = artifact_id.encode()
            writer.writerow(
                {
                    "schema_version": "scenarioflow.artifacts.v1",
                    "artifact_id": artifact_id,
                    "profile": profile,
                    "category": (
                        "checkpoint" if artifact_id == "core.stage2" else "nl-runtime"
                    ),
                    "relative_path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "frozen_source_path": f"frozen/{artifact_id}.bin",
                    "url_path": relative,
                }
            )
    root = tmp_path / "cache"
    if present:
        for artifact_id, relative, _profile in artifacts:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifact_id.encode())
    return ArtifactResolver.from_tsv(manifest), root


def test_missing_optional_requirements_are_scoped_to_each_demo(tmp_path: Path) -> None:
    """One missing demo dependency must not be reported as a package-wide failure."""
    resolver, root = _resolver_with_runtime(tmp_path, present=False)
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "account detail", "token detail")

    report = inspect_demo_capabilities(
        resolver=resolver,
        artifact_root=root,
        environ={},
        find_spec=lambda _name: None,
        which=lambda _name: None,
        run=run,
    )
    assert report["numerical_workflows"] == "available"
    assert report["demos"]["narrative-to-scenario"]["available"] is False
    assert set(report["demos"]["narrative-to-scenario"]["missing"]) == {
        "OPENAI_API_KEY",
        "demo-openai extra",
        "demo-server extra",
        "nl-runtime artifacts",
        "Stage 2 checkpoint",
        "demo backend",
    }
    assert report["demos"]["scenario-to-narrative"]["available"] is False
    assert set(report["demos"]["scenario-to-narrative"]["missing"]) == {
        "Codex CLI",
        "demo-server extra",
        "nl-runtime artifacts",
        "demo backend",
    }
    assert calls == []
    assert "account detail" not in str(report)
    assert "token detail" not in str(report)


def test_codex_login_status_exit_one_means_demo_two_is_unavailable(
    tmp_path: Path,
) -> None:
    """An installed but logged-out Codex CLI must produce the login action only."""
    resolver, root = _resolver_with_runtime(tmp_path, present=True)

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "private account", "")

    report = inspect_demo_capabilities(
        resolver=resolver,
        artifact_root=root,
        environ={"OPENAI_API_KEY": "secret-value"},
        find_spec=lambda _name: object(),
        which=lambda name: "/usr/bin/codex" if name == "codex" else None,
        run=run,
    )
    demo_two = report["demos"]["scenario-to-narrative"]
    assert demo_two["available"] is False
    assert demo_two["missing"] == ["Codex CLI login"]
    assert demo_two["actions"] == ["Run `codex login`."]
    assert "secret-value" not in str(report)
    assert "private account" not in str(report)


def test_both_demos_can_be_reported_available_without_model_requests(
    tmp_path: Path,
) -> None:
    """Satisfied local probes must not call an API or interactive login."""
    resolver, root = _resolver_with_runtime(tmp_path, present=True)
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "logged in", "")

    report = inspect_demo_capabilities(
        resolver=resolver,
        artifact_root=root,
        environ={"OPENAI_API_KEY": "secret-value"},
        find_spec=lambda _name: object(),
        which=lambda name: "/usr/bin/codex" if name == "codex" else None,
        run=run,
    )
    assert report == {
        "numerical_workflows": "available",
        "demos": {
            "narrative-to-scenario": {
                "available": True,
                "missing": [],
                "actions": [],
            },
            "scenario-to-narrative": {
                "available": True,
                "missing": [],
                "actions": [],
            },
        },
    }
    assert commands == [["/usr/bin/codex", "login", "status"]]
