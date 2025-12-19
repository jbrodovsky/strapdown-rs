#!/usr/bin/env python3
"""
Run every predefined simulation configuration against every CSV in data/input/.
Outputs are written to <output_root>/<config_name>/<input_basename>.csv

Usage:
  python scripts/run_all_configs.py [--bin /path/to/strapdown-sim] [--input data/input/] [--output-root data/] [--jobs N]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

DEFAULT_BIN = Path("target/release/strapdown-sim")
DEFAULT_INPUT = Path("data/input")
DEFAULT_OUTPUT_ROOT = Path("data")

CONFIGS: List[Tuple[str, List[str]]] = [
    ("baseline", ["--sched", "passthrough", "--fault", "none", "--seed", "42"]),
    (
        "degraded_fullrate",
        [
            "--sched",
            "passthrough",
            "--fault",
            "degraded",
            "--rho-pos",
            "0.99",
            "--sigma-pos-m",
            "3",
            "--rho-vel",
            "0.95",
            "--sigma-vel-mps",
            "0.3",
            "--r-scale",
            "5",
            "--seed",
            "42",
        ],
    ),
    (
        "degraded_5s",
        [
            "--sched",
            "fixed",
            "--interval-s",
            "5",
            "--phase-s",
            "0.0",
            "--fault",
            "degraded",
            "--rho-pos",
            "0.99",
            "--sigma-pos-m",
            "3",
            "--rho-vel",
            "0.95",
            "--sigma-vel-mps",
            "0.3",
            "--r-scale",
            "5",
            "--seed",
            "42",
        ],
    ),
    (
        "slowbias",
        [
            "--sched",
            "passthrough",
            "--fault",
            "slowbias",
            "--drift-n-mps",
            "0.02",
            "--drift-e-mps",
            "0.00",
            "--q-bias",
            "1e-6",
            "--rotate-omega-rps",
            "0.0",
            "--seed",
            "42",
        ],
    ),
    (
        "slowbias_rot",
        [
            "--sched",
            "passthrough",
            "--fault",
            "slowbias",
            "--drift-n-mps",
            "0.02",
            "--drift-e-mps",
            "0.00",
            "--q-bias",
            "5e-6",
            "--rotate-omega-rps",
            "1e-3",
            "--seed",
            "42",
        ],
    ),
    (
        "hijack",
        [
            "--sched",
            "passthrough",
            "--fault",
            "hijack",
            "--hijack-start-s",
            "120",
            "--hijack-duration-s",
            "60",
            "--hijack-offset-n-m",
            "50",
            "--hijack-offset-e-m",
            "0",
            "--seed",
            "42",
        ],
    ),
    (
        "sched_10s",
        [
            "--sched",
            "fixed",
            "--interval-s",
            "10",
            "--phase-s",
            "0.0",
            "--fault",
            "none",
            "--seed",
            "42",
        ],
    ),
    (
        "duty_10on_2off",
        [
            "--sched",
            "duty",
            "--on-s",
            "10",
            "--off-s",
            "2",
            "--phase-s",
            "1",
            "--fault",
            "none",
            "--seed",
            "42",
        ],
    ),
    (
        "combo",
        [
            "--sched",
            "fixed",
            "--interval-s",
            "5",
            "--phase-s",
            "0.0",
            "--fault",
            "degraded",
            "--rho-pos",
            "0.995",
            "--sigma-pos-m",
            "4",
            "--rho-vel",
            "0.97",
            "--sigma-vel-mps",
            "0.35",
            "--r-scale",
            "5",
            "--seed",
            "42",
        ],
    ),
    (
        "combo_duty_hijack",
        [
            "--sched",
            "duty",
            "--on-s",
            "10",
            "--off-s",
            "3",
            "--phase-s",
            "0",
            "--fault",
            "hijack",
            "--hijack-start-s",
            "150",
            "--hijack-duration-s",
            "120",
            "--hijack-offset-n-m",
            "10",
            "--hijack-offset-e-m",
            "10",
            "--seed",
            "42",
        ],
    ),
]


def find_binary(bin_path: Path | None) -> Path:
    if bin_path:
        if bin_path.exists() and bin_path.is_file():
            return bin_path
        raise FileNotFoundError(f"Provided binary not found: {bin_path}")

    # env var
    env_bin = shutil.which("strapdown-sim")
    if env_bin:
        return Path(env_bin)

    # common cargo path
    cand = DEFAULT_BIN
    if cand.exists():
        return cand

    raise FileNotFoundError(
        "strapdown-sim binary not found. Build it or pass --bin /path/to/strapdown-sim"
    )


def build_cmd(
    bin_path: Path, input_file: Path, output_file: Path, cfg_args: List[str]
) -> List[str]:
    # Use the same invocation style as the notebook: <bin> -i <input> -o <output> closed-loop <cfg_args...>
    cmd = [
        str(bin_path),
        "-i",
        str(input_file),
        "-o",
        str(output_file),
        "closed-loop",
    ] + cfg_args
    return cmd


def run_job(
    bin_path: Path, input_path: Path, out_root: Path, cfg_name: str, cfg_args: List[str]
) -> int:
    out_dir = out_root / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / input_path.name

    cmd = build_cmd(bin_path, input_path, out_path, cfg_args)
    print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode
    except Exception as e:
        print(f"Error running job for {input_path} with config {cfg_name}: {e}")
        return 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bin", type=Path, default=None, help="Path to strapdown-sim binary"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing input CSVs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args(argv)

    try:
        bin_path = find_binary(args.bin)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    input_dir = args.input
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 3

    inputs = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not inputs:
        print(f"No CSV input files found in {input_dir}", file=sys.stderr)
        return 4

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    # prepare jobs
    jobs = []
    for cfg_name, cfg_args in CONFIGS:
        for input_path in inputs:
            jobs.append((cfg_name, input_path, cfg_args))

    # run jobs sequentially or in parallel
    if args.jobs <= 1:
        failures = 0
        for cfg_name, input_path, cfg_args in jobs:
            rc = run_job(bin_path, input_path, out_root, cfg_name, cfg_args)
            if rc != 0:
                print(f"Job failed (rc={rc}): {cfg_name} {input_path}")
                failures += 1
        print(f"Finished. {failures} jobs failed.")
        return 0 if failures == 0 else 5
    else:
        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            future_to_job = {
                ex.submit(
                    run_job, bin_path, input_path, out_root, cfg_name, cfg_args
                ): (cfg_name, input_path)
                for (cfg_name, input_path, cfg_args) in jobs
            }
            for fut in concurrent.futures.as_completed(future_to_job):
                cfg_name, input_path = future_to_job[fut]
                try:
                    rc = fut.result()
                    if rc != 0:
                        print(f"Job failed (rc={rc}): {cfg_name} {input_path}")
                        failures += 1
                except Exception as e:
                    print(f"Job raised exception: {cfg_name} {input_path} -> {e}")
                    failures += 1
        print(f"Finished parallel run. {failures} jobs failed.")
        return 0 if failures == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
