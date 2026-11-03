# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
"""CLI entry point: ``python -m tools.sailify``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CUDA sources to PPU sources with Sailify."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Write the converted tree to this directory (out-of-place). "
            "Omit to convert in place."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if the tree is up to date.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Path to the PyTorch source tree. Defaults to CWD.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root) if args.source_root else Path.cwd()

    from tools.sailify.convert import run_full_sailify

    try:
        in_place = run_full_sailify(
            source_root,
            output_dir=args.output_dir,
            force=args.force,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if in_place:
        print("Conversion complete (in-place). Ready to build.")
    else:
        print(
            f"Conversion complete. Build in the output tree:\n"
            f"    cd {args.output_dir}\n"
            f"    pip install . --no-build-isolation -v"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
