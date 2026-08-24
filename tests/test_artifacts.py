"""Behavioral tests for the checksum-pinned release artifact resolver."""

from __future__ import annotations

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import pytest

from scenarioflow.artifacts import ArtifactError, ArtifactResolver


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest-v1.tsv"
GITHUB_AUTHORITY = ROOT / "artifacts" / "github-authority-v1.tsv"
DATA_FILENAMES = {
    "multi_factor_data.npz",
    "multi_factor_levels.parquet",
    "multi_factor_returns.parquet",
    "spx_vol_surface_history_full_data_fixed.parquet",
    "vol_surface_with_ret.npz",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    path: Path,
    *,
    payload: bytes = b"artifact-payload",
    artifact_id: str = "core.fixture",
    profile: str = "core,all",
    category: str = "checkpoint",
    relative_path: str = "models/core/fixture/best_model.pt",
    url_path: str | None = None,
    digest: str | None = None,
    size: int | None = None,
) -> Path:
    fields = (
        "schema_version",
        "artifact_id",
        "profile",
        "category",
        "relative_path",
        "bytes",
        "sha256",
        "frozen_source_path",
        "url_path",
    )
    row = {
        "schema_version": "scenarioflow.artifacts.v1",
        "artifact_id": artifact_id,
        "profile": profile,
        "category": category,
        "relative_path": relative_path,
        "bytes": str(len(payload) if size is None else size),
        "sha256": digest or _sha256(payload),
        "frozen_source_path": "frozen/source.bin",
        "url_path": relative_path if url_path is None else url_path,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    return path


def test_release_manifest_is_the_exact_external_subset() -> None:
    """Adding Git data/evidence or dropping a frozen payload must fail the release gate."""
    resolver = ArtifactResolver.from_tsv(MANIFEST)
    records = resolver.records
    assert len(records) == 41
    assert sum(record.expected_bytes for record in records) == 1_186_583_031
    assert (
        sum(
            record.category
            in {
                "nl-runtime",
                "checkpoint",
                "multiseed-checkpoint",
                "foundation-checkpoint",
            }
            for record in records
        )
        == 33
    )
    assert (
        sum(
            record.category in {"nl-paper-arrays", "foundation-evidence"}
            for record in records
        )
        == 8
    )
    assert {record.artifact_id for record in resolver.for_profile("core")} == {
        "core.stage1",
        "core.stage2",
        "core.adaptive-graph",
    }
    assert set(resolver.profiles) >= {
        "core",
        "baselines",
        "multiseed",
        "foundation",
        "nl-runtime",
        "evidence",
        "all",
    }
    assert {
        profile: len(resolver.for_profile(profile))
        for profile in (
            "core",
            "baselines",
            "multiseed",
            "foundation",
            "nl-runtime",
            "evidence",
            "all",
        )
    } == {
        "core": 3,
        "baselines": 5,
        "multiseed": 12,
        "foundation": 2,
        "nl-runtime": 10,
        "evidence": 8,
        "all": 41,
    }
    assert not DATA_FILENAMES.intersection(
        {Path(record.relative_path).name for record in records}
    )
    assert all(record.category != "data" for record in records)
    assert all(record.url_path == record.relative_path for record in records)
    frozen_identities = sorted(
        (record.frozen_source_path, record.expected_bytes, record.sha256)
        for record in records
    )
    authority_digest = hashlib.sha256(
        json.dumps(frozen_identities, separators=(",", ":")).encode()
    ).hexdigest()
    assert (
        authority_digest
        == "962092c18342aa7c9e56ed645f7d2ec3ba6d6853ab0a90faa99be4ad3f062471"
    )

    with GITHUB_AUTHORITY.open(encoding="utf-8", newline="") as handle:
        authorities = list(csv.DictReader(handle, delimiter="\t"))
    assert {
        (row["relative_path"], int(row["bytes"]), row["sha256"]) for row in authorities
    } == {
        (
            "results/fmts_reproduction_20260805/checkpoint_replay/garch_ccc/manifest.json",
            13_850,
            "ea16b3a2f59fc066ab6c394190ac15fd70a66e47c2aa5d70b07111700646882d",
        ),
        (
            "results/fmts_reproduction_20260805/checkpoint_replay/garch_ccc/garch_ccc/results_current_panel.json",
            91_037,
            "4749ef068a1510595a68e2227e9fe1be8cabc6927fcb8ad39b4b0b1271369f5a",
        ),
        (
            "results/fmts_reproduction_20260805/reports/layer_a_garch_ccc/garch_ccc_contract_report.json",
            40_286,
            "fd61802d5f0e12acd6d34dd2fb37bb8422a4313e677cd41bf45ecfb73687b128",
        ),
    }
    assert not {row["relative_path"] for row in authorities}.intersection(
        {record.relative_path for record in records}
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"relative_path": "../escape.bin"}, "relative path"),
        (
            {"relative_path": "/absolute.bin", "url_path": "/absolute.bin"},
            "relative path",
        ),
        (
            {
                "relative_path": "models//core/fixture/best_model.pt",
                "url_path": "models/core/fixture/best_model.pt",
            },
            "relative path",
        ),
        ({"sha256": "ABC"}, "SHA-256"),
        ({"bytes": "0"}, "positive"),
        (
            {
                "category": "data",
                "relative_path": "data/input.bin",
                "url_path": "data/input.bin",
            },
            "data",
        ),
        ({"url_path": "other/member.bin"}, "URL path"),
        ({"profile": "unknown"}, "profile"),
    ],
)
def test_manifest_rejects_unsafe_or_malformed_rows(
    tmp_path: Path, changes: dict[str, str], message: str
) -> None:
    """Weakening any manifest authority field must fail before filesystem I/O."""
    path = _write_manifest(tmp_path / "manifest.tsv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        fields = tuple(rows[0])
    rows[0].update(changes)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ArtifactError, match=message):
        ArtifactResolver.from_tsv(path)


def test_manifest_rejects_duplicate_ids_and_paths(tmp_path: Path) -> None:
    """Two manifest rows must never alias an identity or destination."""
    path = _write_manifest(tmp_path / "manifest.tsv")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate artifact ID"):
        ArtifactResolver.from_tsv(path)

    path = _write_manifest(tmp_path / "paths.tsv")
    lines = path.read_text(encoding="utf-8").splitlines()
    duplicate = lines[1].replace("core.fixture", "core.second", 1)
    path.write_text("\n".join(lines + [duplicate]) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate artifact path"):
        ArtifactResolver.from_tsv(path)


def test_local_bundle_fetch_is_verified_atomic_and_idempotent(tmp_path: Path) -> None:
    """A changed local member or second bundle read must not authenticate a final file."""
    payload = b"verified local artifact"
    manifest = _write_manifest(tmp_path / "manifest.tsv", payload=payload)
    resolver = ArtifactResolver.from_tsv(manifest)
    bundle = tmp_path / "bundle"
    source = bundle / "models/core/fixture/best_model.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    root = tmp_path / "cache"

    fetched = resolver.fetch(["core.fixture"], root, bundle_dir=bundle)
    assert fetched == [root / "models/core/fixture/best_model.pt"]
    assert fetched[0].read_bytes() == payload
    assert not fetched[0].with_name(fetched[0].name + ".part").exists()
    source.unlink()
    assert resolver.fetch(["core.fixture"], root, bundle_dir=bundle) == fetched
    assert resolver.resolve("core.fixture", root) == fetched[0]
    assert resolver.status("core.fixture", root).state == "present"


def test_corrupt_final_requires_repair_and_survives_failed_repair(
    tmp_path: Path,
) -> None:
    """A bad replacement must never destroy the existing corrupt file or claim success."""
    payload = b"correct bytes"
    manifest = _write_manifest(tmp_path / "manifest.tsv", payload=payload)
    resolver = ArtifactResolver.from_tsv(manifest)
    root = tmp_path / "cache"
    final = root / "models/core/fixture/best_model.pt"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"old corrupt bytes")
    bundle = tmp_path / "bundle"
    source = bundle / "models/core/fixture/best_model.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wrong replacement")

    assert resolver.status("core.fixture", root).state == "corrupt"
    with pytest.raises(ArtifactError, match="--repair"):
        resolver.fetch(["core.fixture"], root, bundle_dir=bundle)
    with pytest.raises(ArtifactError, match="checksum"):
        resolver.fetch(["core.fixture"], root, bundle_dir=bundle, repair=True)
    assert final.read_bytes() == b"old corrupt bytes"
    source.write_bytes(payload)
    resolver.fetch(["core.fixture"], root, bundle_dir=bundle, repair=True)
    assert final.read_bytes() == payload


def test_local_bundle_missing_member_has_safe_actionable_error(tmp_path: Path) -> None:
    """A missing bundle member must identify the artifact without exposing host paths."""
    manifest = _write_manifest(tmp_path / "manifest.tsv")
    resolver = ArtifactResolver.from_tsv(manifest)
    with pytest.raises(ArtifactError) as caught:
        resolver.fetch(
            ["core.fixture"], tmp_path / "cache", bundle_dir=tmp_path / "bundle"
        )
    message = str(caught.value)
    assert "core.fixture" in message
    assert "models/core/fixture/best_model.pt" in message
    assert resolver.record("core.fixture").sha256 in message
    assert str(tmp_path) not in message


def test_symlink_destination_is_never_followed(tmp_path: Path) -> None:
    """Replacing a destination component with a symlink must not write outside the root."""
    payload = b"artifact"
    resolver = ArtifactResolver.from_tsv(
        _write_manifest(tmp_path / "manifest.tsv", payload=payload)
    )
    root = tmp_path / "cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "models").symlink_to(outside, target_is_directory=True)
    bundle = tmp_path / "bundle"
    source = bundle / "models/core/fixture/best_model.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    with pytest.raises(ArtifactError, match="symlink"):
        resolver.fetch(["core.fixture"], root, bundle_dir=bundle)
    assert not list(outside.iterdir())


class _RangeServer:
    def __init__(self, body: bytes, mode: str = "good") -> None:
        self.body = body
        self.mode = mode
        self.requests: list[str | None] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                requested = self.headers.get("Range")
                owner.requests.append(requested)
                if requested and owner.mode == "good":
                    start = int(requested.removeprefix("bytes=").removesuffix("-"))
                    chunk = owner.body[start:]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{len(owner.body) - 1}/{len(owner.body)}",
                    )
                elif requested and owner.mode == "bad-range":
                    chunk = owner.body
                    self.send_response(206)
                    self.send_header(
                        "Content-Range",
                        f"bytes 0-{len(owner.body) - 1}/{len(owner.body)}",
                    )
                    owner.mode = "good"
                elif requested and owner.mode == "range-not-satisfiable":
                    chunk = b""
                    self.send_response(416)
                    owner.mode = "good"
                elif owner.mode == "missing":
                    chunk = b""
                    self.send_response(404)
                elif owner.mode == "interrupt":
                    chunk = owner.body[:7]
                    self.send_response(200)
                else:
                    chunk = owner.body
                    self.send_response(200)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_RangeServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/"


@pytest.mark.parametrize(
    ("mode", "expected_requests"),
    [
        ("good", ["bytes=7-"]),
        ("ignore", ["bytes=7-"]),
        ("bad-range", ["bytes=7-", None]),
        ("range-not-satisfiable", ["bytes=7-", None]),
    ],
)
def test_http_partial_resume_and_restart_rules(
    tmp_path: Path, mode: str, expected_requests: list[str | None]
) -> None:
    """Wrong Range handling must restart safely instead of appending duplicate bytes."""
    payload = b"complete remote artifact"
    resolver = ArtifactResolver.from_tsv(
        _write_manifest(tmp_path / "manifest.tsv", payload=payload)
    )
    root = tmp_path / "cache"
    part = root / "models/core/fixture/best_model.pt.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:7])
    with _RangeServer(payload, mode=mode) as server:
        result = resolver.fetch(["core.fixture"], root, base_url=server.base_url)
        assert server.requests == expected_requests
    assert result[0].read_bytes() == payload


def test_http_digest_mismatch_never_installs_payload(tmp_path: Path) -> None:
    """A complete remote response with wrong bytes must remain unauthenticated."""
    expected = b"expected"
    resolver = ArtifactResolver.from_tsv(
        _write_manifest(tmp_path / "manifest.tsv", payload=expected)
    )
    root = tmp_path / "cache"
    final = root / "models/core/fixture/best_model.pt"
    with _RangeServer(b"tampered") as server:
        with pytest.raises(ArtifactError, match="checksum"):
            resolver.fetch(["core.fixture"], root, base_url=server.base_url)
    assert not final.exists()
    assert final.with_name(final.name + ".part.failed").exists()


def test_incomplete_remote_body_remains_resumable(tmp_path: Path) -> None:
    """A short transfer must remain a .part and resume on the next invocation."""
    payload = b"complete remote artifact"
    resolver = ArtifactResolver.from_tsv(
        _write_manifest(tmp_path / "manifest.tsv", payload=payload)
    )
    root = tmp_path / "cache"
    final = root / "models/core/fixture/best_model.pt"
    part = final.with_name(final.name + ".part")
    with _RangeServer(payload, mode="interrupt") as server:
        with pytest.raises(ArtifactError, match="interrupted"):
            resolver.fetch(["core.fixture"], root, base_url=server.base_url)
    assert part.read_bytes() == payload[:7]
    assert not final.exists()
    with _RangeServer(payload, mode="good") as server:
        resolver.fetch(["core.fixture"], root, base_url=server.base_url)
        assert server.requests == ["bytes=7-"]
    assert final.read_bytes() == payload


def test_remote_missing_member_uses_the_same_safe_authority_fields(
    tmp_path: Path,
) -> None:
    """A remote 404 must identify only the pinned ID, member, and digest."""
    resolver = ArtifactResolver.from_tsv(_write_manifest(tmp_path / "manifest.tsv"))
    with _RangeServer(b"", mode="missing") as server:
        with pytest.raises(ArtifactError) as caught:
            resolver.fetch(
                ["core.fixture"],
                tmp_path / "cache",
                base_url=server.base_url.replace("http://", "http://user:secret@"),
            )
    message = str(caught.value)
    record = resolver.record("core.fixture")
    assert record.artifact_id in message
    assert record.url_path in message
    assert record.sha256 in message
    assert "user" not in message
    assert "secret" not in message
    assert str(tmp_path) not in message


def test_offline_mode_never_attempts_a_network_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline miss must fail locally even if a base URL environment variable exists."""
    resolver = ArtifactResolver.from_tsv(_write_manifest(tmp_path / "manifest.tsv"))
    monkeypatch.setenv("SCENARIOFLOW_ARTIFACT_BASE_URL", "http://127.0.0.1:1/private")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network opened in offline mode")

    monkeypatch.setattr("scenarioflow.artifacts.urlopen", forbidden)
    with pytest.raises(ArtifactError, match="offline"):
        resolver.fetch(["core.fixture"], tmp_path / "cache", offline=True)
