"""Native command surface for the ScenarioFlow research release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

from .artifacts import (
    ArtifactError,
    ArtifactResolver,
    ArtifactStatus,
    default_manifest_path,
)
from .demo_doctor import inspect_demo_capabilities


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow", description=__doc__)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, help_text in (
        ("setup", "check the native installation"),
        ("fetch", "fetch checksum-pinned release artifacts"),
        ("artifacts", "inspect artifact integrity"),
        ("paper", "render numerical paper assets from released evidence"),
        ("demo", "inspect or launch optional demos"),
    ):
        commands.add_parser(name, help=help_text, add_help=False)
    return parser


def _selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile")
    selection.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cache"))


def _fetch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow fetch")
    _selection_arguments(parser)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--base-url")
    source.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow artifacts verify")
    _selection_arguments(parser)
    parser.add_argument("--json", action="store_true")
    return parser


def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow setup")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=_repository_root() / "data")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--json", action="store_true")
    return parser


def _doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow demo doctor")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--json", action="store_true")
    return parser


def _launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow demo launch")
    parser.add_argument(
        "demo",
        choices=("narrative-to-scenario", "scenario-to-narrative"),
        nargs="?",
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--data-dir", type=Path, default=_repository_root() / "data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    return parser


def _paper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenarioflow paper")
    parser.add_argument(
        "mode", choices=("all", "tables", "plots"), nargs="?", default="all"
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/paper-reproduction")
    )
    return parser


def _selected_ids(
    resolver: ArtifactResolver, profile: str | None, artifact_ids: Sequence[str]
) -> list[str]:
    return [
        record.artifact_id
        for record in resolver.select(profile=profile, artifact_ids=artifact_ids)
    ]


def _print_statuses(statuses: Sequence[ArtifactStatus], *, as_json: bool) -> None:
    documents = [status.as_dict() for status in statuses]
    if as_json:
        print(json.dumps({"artifacts": documents}, sort_keys=True))
        return
    for document in documents:
        print(
            f"{document['state']:11} {document['artifact_id']}  {document['relative_path']}"
        )


def _check_processed_data(data_dir: Path) -> bool:
    checksums = data_dir / "CHECKSUMS.tsv"
    try:
        with checksums.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error):
        return False
    if len(rows) != 5:
        return False
    expected_names = {Path(row["path"]).name for row in rows}
    actual_names = {
        path.name for path in data_dir.iterdir() if path.is_file() and path != checksums
    }
    if actual_names != expected_names:
        return False
    for row in rows:
        path = data_dir / Path(row["path"]).name
        try:
            if path.stat().st_size != int(row["bytes"]):
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                return False
        except (OSError, ValueError):
            return False
    return True


def _run_setup(arguments: list[str]) -> int:
    parser = _setup_parser()
    args = parser.parse_args(arguments)
    if not args.check:
        parser.error("setup is non-mutating; use --check")
    resolver = ArtifactResolver.from_tsv(args.manifest)
    statuses = resolver.verify(
        [record.artifact_id for record in resolver.records], args.artifact_root
    )
    document = {
        "package": "ok",
        "processed_data": "ok" if _check_processed_data(args.data_dir) else "invalid",
        "artifact_manifest": "ok",
        "external_artifacts": {
            "present": sum(status.state == "present" for status in statuses),
            "total": len(statuses),
        },
    }
    if args.json:
        print(json.dumps(document, sort_keys=True))
    else:
        print(f"package: {document['package']}")
        print(f"processed data: {document['processed_data']}")
        print(f"artifact manifest: {document['artifact_manifest']}")
        print(
            "external artifacts: "
            f"{document['external_artifacts']['present']}/{document['external_artifacts']['total']} present"
        )
    return 0 if document["processed_data"] == "ok" else 1


def _run_fetch(arguments: list[str]) -> int:
    args = _fetch_parser().parse_args(arguments)
    resolver = ArtifactResolver.from_tsv(args.manifest)
    artifact_ids = _selected_ids(resolver, args.profile, args.artifact)
    resolver.fetch(
        artifact_ids,
        args.artifact_root,
        base_url=args.base_url,
        bundle_dir=args.bundle_dir,
        resume=not args.no_resume,
        repair=args.repair,
        offline=args.offline,
    )
    statuses = resolver.verify(artifact_ids, args.artifact_root)
    _print_statuses(statuses, as_json=args.json)
    return 0


def _run_verify(arguments: list[str]) -> int:
    args = _verify_parser().parse_args(arguments)
    resolver = ArtifactResolver.from_tsv(args.manifest)
    artifact_ids = _selected_ids(resolver, args.profile, args.artifact)
    statuses = resolver.verify(artifact_ids, args.artifact_root)
    _print_statuses(statuses, as_json=args.json)
    return 0 if all(status.state == "present" for status in statuses) else 1


def _run_demo(arguments: list[str]) -> int:
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="scenarioflow demo")
        commands = parser.add_subparsers(dest="command")
        commands.add_parser("doctor", help="check optional local capabilities")
        commands.add_parser("launch", help="launch an optional demo")
        parser.print_help()
        return 0
    if arguments[0] == "doctor":
        args = _doctor_parser().parse_args(arguments[1:])
        resolver = ArtifactResolver.from_tsv(args.manifest)
        report = inspect_demo_capabilities(
            resolver=resolver, artifact_root=args.artifact_root
        )
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            for name, status in report["demos"].items():
                label = "available" if status["available"] else "unavailable"
                print(f"{name}: {label}")
                for action in status["actions"]:
                    print(f"  {action}")
        return 0
    if arguments[0] == "launch":
        args = _launch_parser().parse_args(arguments[1:])
        if args.demo is None:
            _launch_parser().print_help()
            return 0
        resolver = ArtifactResolver.from_tsv(args.manifest)
        report = inspect_demo_capabilities(
            resolver=resolver,
            artifact_root=args.artifact_root,
        )
        status = report["demos"][args.demo]
        if not status["available"]:
            for action in status["actions"]:
                print(action, file=sys.stderr)
            return 2
        from scenarioflow_demo.launch import launch_demo

        return launch_demo(
            demo=args.demo,
            manifest=args.manifest,
            artifact_root=args.artifact_root,
            data_dir=args.data_dir,
            host=args.host,
            port=(
                args.port
                if args.port is not None
                else 7860 if args.demo == "narrative-to-scenario" else 7861
            ),
        )
    raise ArtifactError(f"unknown demo command: {arguments[0]}")


def _run_paper(arguments: list[str]) -> int:
    args = _paper_parser().parse_args(arguments)
    from .paper.render import PaperRenderError, render_paper_assets, render_report_json

    try:
        report = render_paper_assets(
            mode=args.mode,
            artifact_manifest=args.manifest,
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
        )
    except PaperRenderError as error:
        raise ArtifactError(str(error)) from error
    print(render_report_json(report))
    return 0


def main(argv: list[str] | None = None) -> int | None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _root_parser().print_help()
        return 0
    command, tail = arguments[0], arguments[1:]
    try:
        if command == "setup":
            return _run_setup(tail)
        if command == "fetch":
            return _run_fetch(tail)
        if command == "artifacts":
            if not tail or tail[0] in {"-h", "--help"}:
                print("usage: scenarioflow artifacts verify [options]")
                return 0
            if tail[0] != "verify":
                raise ArtifactError(f"unknown artifacts command: {tail[0]}")
            return _run_verify(tail[1:])
        if command == "demo":
            return _run_demo(tail)
        if command == "paper":
            return _run_paper(tail)
        raise ArtifactError(f"unknown command: {command}")
    except ArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
