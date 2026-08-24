"""Contracts for the frozen, reader-facing paper evidence release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scenarioflow.paper.authority import (
    AuthorityError,
    load_authorities,
    project_portable_json_v1,
    verify_release_authorities,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "FROZEN_AUTHORITY.tsv"
SCHEMA_VERSION = "scenarioflow.frozen-authority.v1"

PORTABLE_SOURCES = {
    "results/fmts_multiseed_20260805/aggregate/summary.json",
    "results/fmts_multiseed_20260805/command_dag.json",
    "results/fmts_multiseed_20260805/protocol_manifest.json",
    "results/fmts_reproduction_20260805/reproduction_report.json",
    "results/foundation_baselines_current_panel_20260810/benchmark_manifest.json",
    "results/fmts_reproduction_20260805/reports/layer_a_garch_ccc/garch_ccc_contract_report.json",
}
PORTABLE_PROJECTION_PROOFS = {
    "results/fmts_multiseed_20260805/aggregate/summary.json": (
        701_189,
        "2082647c920f6cdff88faf15ed37515dfe9fba3e7e0697df38f9203d2d0954d1",
        "results/frozen/multiseed/summary.json",
        692_525,
        "4ebb4d26666cf3d05f8f85c03f678584c23be799479e94f328078fd6783eb3fb",
        "50fbf182f6f349234d45718d110259674b800fd28b18b64bb11c4f144bf0dab0",
    ),
    "results/fmts_multiseed_20260805/command_dag.json": (
        955_426,
        "d5f987f597e93d132c19c9c7c047d7224ce83c94d200c2a704c640b60418675d",
        "results/frozen/multiseed/command-dag.json",
        910_790,
        "6e50e4b911358a702980e1d5929eb24fe6d64593d5f0b42696df7df118a268b2",
        "ecf90958d519226e280c3f35cfc10bfbed067e068fb5f381b2edb4c343e83844",
    ),
    "results/fmts_multiseed_20260805/protocol_manifest.json": (
        86_010,
        "33f832dfc799e1c3619f012b7286791dc2e4fe9cf1c9acb52e55e136206397c8",
        "results/frozen/multiseed/protocol-manifest.json",
        76_765,
        "f0111cb5fe42ebdd8dedc8f468ec4ca411015495b727b431801407a75bee7808",
        "e71d93605342bb41da12189ac533be419749454377ebb6c525163222a486d004",
    ),
    "results/fmts_reproduction_20260805/reproduction_report.json": (
        540_179,
        "39213591e2b61e352c8aaa087da5417489520c19e1ebf5b4acf9dc8b370ae64f",
        "results/frozen/benchmark/checkpoint-replay/reproduction-report.json",
        539_195,
        "26403729fa56647ba65f6b686c8d1b0c9bff786d7c3dc33e62cad96033f67d8b",
        "a56b27ee32ba1403c13aad108f1a1803a2ba70cac10052c583b07b1fc9bec9ea",
    ),
    "results/foundation_baselines_current_panel_20260810/benchmark_manifest.json": (
        65_089,
        "30c75646bdd00de4531b58962af8411189bbfcc1049d7b9538d9b98a58d93f30",
        "results/frozen/foundation/benchmark-manifest.json",
        64_865,
        "8f1e4bd59f0ab38a36fffdafbbba59fa27442aba48946469350a11c9b9a91f19",
        "96653d45b79475999e7b965aa5b540384277891ef341eac6c374bb23040a11b5",
    ),
    "results/fmts_reproduction_20260805/reports/layer_a_garch_ccc/garch_ccc_contract_report.json": (
        40_286,
        "fd61802d5f0e12acd6d34dd2fb37bb8422a4313e677cd41bf45ecfb73687b128",
        "results/frozen/benchmark/checkpoint-replay/gaussian-ccc/contract-report.json",
        39_950,
        "70c9ac714a4f81b182746810791658bea35d14c4d28bfde762e0377eb5f3190a",
        "9870ee9c967ab4ae264721613d712d3ab2786188546e8f067f1aa16eea868246",
    ),
}

EXPECTED_RELEASED_PATHS = {
    "metadata/models/iv-only/args.json",
    "metadata/models/iv-only/train-summary.json",
    "metadata/models/iv-only/training-history.json",
    "metadata/models/stage1/args.json",
    "metadata/models/stage1/train-summary.json",
    "metadata/models/stage1/training-history.json",
    "metadata/models/stage2/args.json",
    "metadata/models/stage2/train-summary.json",
    "metadata/models/stage2/training-history.json",
    "metadata/models/adaptive-graph/args.json",
    "metadata/models/adaptive-graph/train-summary.json",
    "metadata/models/adaptive-graph/training-history.json",
    "metadata/models/baselines/current-panel-deep.json",
    "metadata/models/baselines/g2-history-scale.json",
    "metadata/models/foundation/moirai-training.json",
    "metadata/models/foundation/toto-training.json",
    "results/frozen/benchmark/current-panel/summary.json",
    "results/frozen/benchmark/current-panel/scenarioflow/results.json",
    "results/frozen/benchmark/current-panel/scenarioflow-graph/results.json",
    "results/frozen/benchmark/current-panel/g2-deepvar-history-scale/results.json",
    "results/frozen/benchmark/checkpoint-replay/baseline-summary.json",
    "results/frozen/benchmark/checkpoint-replay/scenarioflow/results.json",
    "results/frozen/benchmark/checkpoint-replay/scenarioflow-graph/results.json",
    "results/frozen/benchmark/checkpoint-replay/gaussian-ccc/summary.json",
    "results/frozen/benchmark/checkpoint-replay/gaussian-ccc/manifest.json",
    "results/frozen/benchmark/checkpoint-replay/gaussian-ccc/results.json",
    "results/frozen/benchmark/checkpoint-replay/gaussian-ccc/contract-report.json",
    "results/frozen/benchmark/checkpoint-replay/reproduction-report.json",
    "results/frozen/multiseed/metric-summaries.csv",
    "results/frozen/multiseed/paired-comparisons.csv",
    "results/frozen/multiseed/per-run-metrics.csv",
    "results/frozen/multiseed/per-run-thresholds.csv",
    "results/frozen/multiseed/summary.json",
    "results/frozen/multiseed/summary.md",
    "results/frozen/multiseed/command-dag.json",
    "results/frozen/multiseed/protocol-manifest.json",
    "results/frozen/foundation/benchmark-manifest.json",
    "results/frozen/foundation/canonical-eval/manifest.json",
    "results/frozen/foundation/canonical-eval/summary.json",
    "results/frozen/foundation/canonical-eval/moirai/results.json",
    "results/frozen/foundation/canonical-eval/toto/results.json",
    "results/frozen/dollar-liquidity/narrative-report.json",
    "results/frozen/dollar-liquidity/start-only-report.json",
    "results/frozen/dollar-liquidity/fixed-start-comparison.json",
    "results/frozen/dollar-liquidity/matched-evaluation.json",
    "results/frozen/graph/triscope/fan-selection.json",
    "results/frozen/graph/triscope/joint-factor-fans.png",
    "results/frozen/graph/triscope/anchor-panel.json",
    "results/frozen/graph/triscope/framework-manifest.json",
    "results/frozen/graph/triscope/iv-train.json",
    "results/frozen/graph/triscope/iv-validation.json",
    "results/frozen/graph/triscope/joint-iv-validation.json",
    "results/frozen/graph/triscope/joint-panel-validation.json",
    "results/frozen/graph/triscope/scorecard.json",
    "results/frozen/graph/management/iv-cross-cell-sensitivity.png",
    "results/frozen/graph/management/iv-marginal-daily-changes.png",
    "results/frozen/graph/management/manifest.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_authority_manifest(
    path: Path,
    *,
    released_path: str,
    payload: bytes,
) -> None:
    fields = (
        "schema_version",
        "authority_id",
        "category",
        "released_path",
        "released_bytes",
        "released_sha256",
        "frozen_source_path",
        "frozen_bytes",
        "frozen_sha256",
        "transformation",
    )
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "schema_version": SCHEMA_VERSION,
                "authority_id": "fixture.authority",
                "category": "result",
                "released_path": released_path,
                "released_bytes": len(payload),
                "released_sha256": digest,
                "frozen_source_path": "results/frozen-fixture.json",
                "frozen_bytes": len(payload),
                "frozen_sha256": digest,
                "transformation": "identity",
            }
        )


def test_release_contains_the_exact_frozen_authority_set() -> None:
    """Dropping, adding, or remapping one frozen authority must fail."""
    authorities = load_authorities(MANIFEST)

    assert len(authorities) == 57
    assert sum(item.frozen_bytes for item in authorities) == 19_804_662
    assert {item.released_path.as_posix() for item in authorities} == (
        EXPECTED_RELEASED_PATHS
    )
    assert len({item.authority_id for item in authorities}) == 57
    assert len({item.frozen_source_path for item in authorities}) == 57
    frozen_identities = sorted(
        (item.frozen_source_path, item.frozen_bytes, item.frozen_sha256)
        for item in authorities
    )
    assert (
        hashlib.sha256(
            json.dumps(frozen_identities, separators=(",", ":")).encode()
        ).hexdigest()
        == "45eec7aa039be790370bec9747876b2b7712196e168456d32697400e7eebdce6"
    )

    identity = [item for item in authorities if item.transformation == "identity"]
    portable = [
        item for item in authorities if item.transformation == "portable-json-v1"
    ]
    assert len(identity) == 51
    assert len(portable) == 6
    assert {item.frozen_source_path for item in portable} == PORTABLE_SOURCES
    assert all(
        (item.released_bytes, item.released_sha256)
        == (item.frozen_bytes, item.frozen_sha256)
        for item in identity
    )

    assert verify_release_authorities(ROOT, MANIFEST) == list(authorities)


def test_portable_projection_proofs_are_independently_pinned() -> None:
    """A scientific edit plus a matching TSV edit must still fail this gate."""
    seen_sources = set()
    for item in load_authorities(MANIFEST):
        if item.transformation != "portable-json-v1":
            continue
        seen_sources.add(item.frozen_source_path)
        expected = PORTABLE_PROJECTION_PROOFS[item.frozen_source_path]
        assert (
            item.frozen_bytes,
            item.frozen_sha256,
            item.released_path.as_posix(),
            item.released_bytes,
            item.released_sha256,
        ) == expected[:5]
        document = json.loads((ROOT / item.released_path).read_text(encoding="utf-8"))
        semantic_digest = hashlib.sha256(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        assert semantic_digest == expected[5]
    assert seen_sources == set(PORTABLE_PROJECTION_PROOFS)


def test_portable_projection_changes_only_recognized_path_strings() -> None:
    """Portable projection must preserve records and scientific values."""
    donor_root = "/" + "home/max/Documents/vol-surface-vae-pub"
    backup_root = donor_root + "-paper-backups"
    cache_root = "/" + "home/max/.cache/vol-surface-vae-pub"
    temporary_root = "/" + "tmp/fmts-window-scores-paper-25b962"
    private_path = (
        "/" + "home/max/.claude/projects/-home-max-Documents-vol-surface-vae-pub/"
        "run/subagents/workflows/worker/agent.jsonl"
    )
    source = {
        "scientific": {"score": 0.0125, "seed": 7345, "valid": True},
        "path_key_map": {f"{donor_root}/source.py": "unchanged-value"},
        "paths": [
            f"{donor_root}/results/run.json",
            f"{backup_root}/snapshot/table.tex",
            f"{cache_root}/models/model.bin",
            f"{temporary_root}/data/panel.parquet",
            f"evidence exists: {private_path}",
        ],
    }
    projected_bytes = project_portable_json_v1(
        json.dumps(source, separators=(",", ":")).encode()
    )
    projected = json.loads(projected_bytes)
    private_digest = hashlib.sha256(private_path.encode()).hexdigest()

    assert projected == {
        "scientific": {"score": 0.0125, "seed": 7345, "valid": True},
        "path_key_map": {"frozen-source://source.py": "unchanged-value"},
        "paths": [
            "frozen-source://results/run.json",
            "paper-authority://snapshot/table.tex",
            "artifact-cache://models/model.bin",
            "legacy-worktree://fmts-window-scores-paper-25b962/data/panel.parquet",
            f"evidence exists: private-provenance://{private_digest}",
        ],
    }
    assert project_portable_json_v1(projected_bytes) == projected_bytes


def test_portable_projection_rejects_key_collisions() -> None:
    """Two provenance keys must never silently collapse into one record."""
    donor_path = "/" + "home/max/Documents/vol-surface-vae-pub/result.json"
    payload = json.dumps(
        {donor_path: {"score": 1.0}, "frozen-source://result.json": {"score": 2.0}}
    ).encode()

    with pytest.raises(AuthorityError, match="key collision"):
        project_portable_json_v1(payload)


@pytest.mark.parametrize(
    "value",
    [
        "/" + "home/another-user/private/result.json",
        "/" + "tmp/unrecognized-run/result.json",
        "/" + "root/private/result.json",
        "/" + "var/lib/scenarioflow/result.json",
        "/" + "mnt/research/result.json",
        "/" + "workspace/output/result.json",
        "/" + "opt/scenarioflow/result.json",
        "/" + "srv/scenarioflow/result.json",
        "/" + "équipe/private/result.json",
        "file:" + ("/" * 3) + "root/private/result.json",
        "file:" + "/" + "var/lib/result.json",
        "custom:" + ("/" * 3) + "mnt/data/result.json",
        "C:" + r"\Users\researcher\result.json",
        "D:" + r"\research\result.json",
        "E:" + "/research/result.json",
        ("\\" * 2) + r"server\share\result.json",
    ],
)
def test_portable_projection_fails_closed_on_unknown_host_paths(value: str) -> None:
    """An unrecognized host path must never survive a public projection."""
    documents = (
        {"nested": [{"path": value}]},
        {value: {"score": 1.0}},
    )
    for document in documents:
        with pytest.raises(AuthorityError, match="unrecognized absolute host path"):
            project_portable_json_v1(json.dumps(document).encode())


def test_portable_projection_allows_approved_provenance_uris() -> None:
    """The five public provenance schemes must remain stable and readable."""
    uris = [
        "frozen-source://results/run.json",
        "paper-authority://snapshot/table.tex",
        "artifact-cache://models/model.bin",
        "legacy-worktree://run-id/data/panel.parquet",
        "private-provenance://" + "a" * 64,
    ]
    payload = json.dumps({uri: {"source": uri} for uri in uris}).encode()

    assert json.loads(project_portable_json_v1(payload)) == json.loads(payload)


def test_portable_projection_rejects_nested_duplicate_json_keys() -> None:
    """Duplicate JSON keys must not be collapsed before path projection."""
    payload = b'{"outer":{"score":1,"score":2}}'

    with pytest.raises(AuthorityError, match="duplicate JSON key: score"):
        project_portable_json_v1(payload)


def test_verifier_rejects_missing_and_extra_managed_files(tmp_path: Path) -> None:
    """The verifier must reject both incomplete and contaminated frozen trees."""
    payload = b'{"ok":true}\n'
    relative = "results/frozen/fixture/result.json"
    manifest = tmp_path / "results" / "FROZEN_AUTHORITY.tsv"
    _write_authority_manifest(manifest, released_path=relative, payload=payload)

    with pytest.raises(AuthorityError, match="missing released authority"):
        verify_release_authorities(tmp_path, manifest)

    destination = tmp_path / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    assert len(verify_release_authorities(tmp_path, manifest)) == 1

    (destination.parent / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AuthorityError, match="unexpected managed release file"):
        verify_release_authorities(tmp_path, manifest)


def test_verifier_rejects_tampered_payload(tmp_path: Path) -> None:
    """Changing a released byte must break its frozen identity check."""
    payload = b'{"ok":true}\n'
    relative = "metadata/models/fixture/args.json"
    manifest = tmp_path / "results" / "FROZEN_AUTHORITY.tsv"
    _write_authority_manifest(manifest, released_path=relative, payload=payload)
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b'{"ok":false}\n')

    with pytest.raises(AuthorityError, match="size or SHA-256 mismatch"):
        verify_release_authorities(tmp_path, manifest)


def test_manifest_loader_rejects_unsafe_or_ambiguous_rows(tmp_path: Path) -> None:
    """Unsafe paths and duplicate destinations must not enter the authority map."""
    payload = b"{}\n"
    manifest = tmp_path / "authority.tsv"
    _write_authority_manifest(
        manifest,
        released_path="results/frozen/fixture/result.json",
        payload=payload,
    )
    rows = manifest.read_text(encoding="utf-8").splitlines()
    duplicate = rows[1].split("\t")
    duplicate[1] = "fixture.second"
    manifest.write_text("\n".join([*rows, "\t".join(duplicate)]) + "\n")

    with pytest.raises(AuthorityError, match="duplicate released path"):
        load_authorities(manifest)

    _write_authority_manifest(
        manifest,
        released_path="../outside.json",
        payload=payload,
    )
    with pytest.raises(AuthorityError, match="normalized relative path"):
        load_authorities(manifest)


def test_frozen_manifest_records_match_released_file_hashes() -> None:
    """Every public identity must be independently checkable from the TSV."""
    for item in load_authorities(MANIFEST):
        released = ROOT / item.released_path
        assert released.stat().st_size == item.released_bytes
        assert _sha256(released) == item.released_sha256
