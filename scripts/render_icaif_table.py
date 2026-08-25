#!/usr/bin/env python3
"""Call the preserved ICAIF table function against a staged source tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    namespace = runpy.run_path(str(args.source.resolve()))
    function = namespace["make_classical_baseline_table"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    function.__globals__["OUT"] = args.output_dir.resolve()
    function(frame="insample")


if __name__ == "__main__":
    main()
