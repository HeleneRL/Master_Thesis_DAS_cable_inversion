from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    "detector_bulk.py",
    "build_transmitter_positions.py",
    "build_prior_cable_3d_consistent.py",
    "build_trust_map.py",
    "build_inversion_dataset.py",
    "invert_cable_diagnostics.py",
    "make_plots_relative_only.py"
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DAS inversion pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    for step in STEPS:
        cmd = [sys.executable, str(src_dir / step), "--config", str(args.config)]
        print(f"\n=== Running {step} ===")
        subprocess.run(cmd, check=True)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
