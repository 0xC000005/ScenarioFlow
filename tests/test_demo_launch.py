"""Tests for the lazy local demo launchers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenarioflow_demo import launch


class _FakeApp:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def launch(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


@pytest.mark.parametrize(
    ("demo", "port"),
    (("narrative-to-scenario", 7860), ("scenario-to-narrative", 7861)),
)
def test_launch_demo_builds_only_the_selected_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, demo: str, port: int
) -> None:
    """Launching one demo must not construct or start the other."""
    app = _FakeApp()
    built: list[tuple[str, object]] = []

    monkeypatch.setattr(
        launch,
        "_build_narrative_to_scenario",
        lambda **kwargs: built.append(("narrative-to-scenario", kwargs)) or app,
    )
    monkeypatch.setattr(
        launch,
        "_build_scenario_to_narrative",
        lambda **kwargs: built.append(("scenario-to-narrative", kwargs)) or app,
    )

    result = launch.launch_demo(
        demo=demo,
        manifest=tmp_path / "manifest.tsv",
        artifact_root=tmp_path / "cache",
        data_dir=tmp_path / "data",
        host="127.0.0.1",
        port=port,
        root_path="/demo",
    )

    assert result == 0
    assert built == [
        (
            demo,
            {
                "manifest": tmp_path / "manifest.tsv",
                "artifact_root": tmp_path / "cache",
                "data_dir": tmp_path / "data",
            },
        )
    ]
    assert app.calls == [
        {
            "server_name": "127.0.0.1",
            "server_port": port,
            "root_path": "/demo",
            "share": False,
            "show_error": False,
        }
    ]


def test_unknown_demo_is_rejected_before_server_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown demo"):
        launch.launch_demo(
            demo="unknown",
            manifest=tmp_path / "manifest.tsv",
            artifact_root=tmp_path / "cache",
            data_dir=tmp_path / "data",
            host="127.0.0.1",
            port=9000,
        )
