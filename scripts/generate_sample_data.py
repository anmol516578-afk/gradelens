"""Generate the demo data in sample_data/.

Usage:
    python scripts/generate_sample_data.py            # writes to ./sample_data
    python scripts/generate_sample_data.py --out DIR --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sample_data import write_sample_files  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GradeLens sample data")
    parser.add_argument("--out", type=Path, default=ROOT / "sample_data", help="output directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    args = parser.parse_args(argv)
    paths = write_sample_files(args.out, seed=args.seed)
    for key, path in paths.items():
        print(f"{key:<11} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
