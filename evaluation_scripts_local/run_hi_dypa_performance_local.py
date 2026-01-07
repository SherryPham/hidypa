#!/usr/bin/env python
"""
Convenience driver to run hi_dypa performance evaluations locally.

This script sequentially invokes `evaluation_scripts/evaluate_multiuser_performance.py`
for all nine configurations (naive baseline + eight hi_dypa splits) and forwards
shared arguments such as the prompts file, number of prompts, model selection, and user ID.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "evaluation_scripts", "evaluate_multiuser_performance.py"
)

CONFIGS = [
    {"label": "Configuration 1: Naive (L=8)", "scheme": "naive", "l_bits": 8},
    {
        "label": "Configuration 2: Hi-DyPa G=1, U=7",
        "scheme": "hi_dypa",
        "group_bits": 1,
        "user_bits": 7,
        "l_bits": 8,
    },
    {
        "label": "Configuration 3: Hi-DyPa G=2, U=6",
        "scheme": "hi_dypa",
        "group_bits": 2,
        "user_bits": 6,
        "l_bits": 8,
    },
    {
        "label": "Configuration 4: Hi-DyPa G=3, U=5",
        "scheme": "hi_dypa",
        "group_bits": 3,
        "user_bits": 5,
        "l_bits": 8,
    },
    {
        "label": "Configuration 5: Hi-DyPa G=4, U=4",
        "scheme": "hi_dypa",
        "group_bits": 4,
        "user_bits": 4,
        "l_bits": 8,
    },
    {
        "label": "Configuration 6: Hi-DyPa G=5, U=3",
        "scheme": "hi_dypa",
        "group_bits": 5,
        "user_bits": 3,
        "l_bits": 8,
    },
    {
        "label": "Configuration 7: Hi-DyPa G=6, U=2",
        "scheme": "hi_dypa",
        "group_bits": 6,
        "user_bits": 2,
        "l_bits": 8,
    },
    {
        "label": "Configuration 8: Hi-DyPa G=7, U=1",
        "scheme": "hi_dypa",
        "group_bits": 7,
        "user_bits": 1,
        "l_bits": 8,
    },
    {
        "label": "Configuration 9: Hi-DyPa G=8, U=0",
        "scheme": "hi_dypa",
        "group_bits": 8,
        "user_bits": 0,
        "l_bits": 8,
    },
]


def build_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all hi_dypa performance configurations locally using the "
            "evaluate_multiuser_performance.py script."
        )
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to use for child processes (default: current).",
    )
    parser.add_argument(
        "--prompts-file",
        default="assets/prompts.txt",
        help="Path to prompts file relative to repo root.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=100,
        help="Number of prompts per configuration (default: 100).",
    )
    parser.add_argument(
        "--users-file",
        default="assets/users.csv",
        help="Path to the users CSV relative to repo root.",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        choices=["gpt2", "gpt-oss-20b", "gpt-oss-120b"],
        help="Model to use for embedding/detection.",
    )
    parser.add_argument(
        "--delta", type=float, default=3.5, help="Watermark strength (default: 3.5)."
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=2.5,
        help="Entropy threshold (default: 2.5).",
    )
    parser.add_argument(
        "--hashing-context",
        type=int,
        default=5,
        help="Hashing context window (default: 5).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help="Detection z-score threshold (default: 4.0).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens for generation (default: 512).",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/multiuser_performance",
        help="Base output directory (default: evaluation/multiuser_performance).",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=64,
        help="User ID to use for all evaluations (default: 64).",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag appended to each configuration directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--seeds-file",
        default=None,
        help="Path to seeds.txt file to read existing seeds from (optional). If provided, seeds will be reused for each configuration.",
    )
    return parser


def build_command(args: argparse.Namespace, config: dict, run_tag: str) -> List[str]:
    cmd = [args.python_bin, EVAL_SCRIPT]
    cmd.extend(["--prompts-file", args.prompts_file])
    cmd.extend(["--max-prompts", str(args.max_prompts)])
    cmd.extend(["--users-file", args.users_file])
    cmd.extend(["--model", args.model])
    cmd.extend(["--delta", str(args.delta)])
    cmd.extend(["--entropy-threshold", str(args.entropy_threshold)])
    cmd.extend(["--hashing-context", str(args.hashing_context)])
    cmd.extend(["--z-threshold", str(args.z_threshold)])
    cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    cmd.extend(["--l-bits", str(config["l_bits"])])
    cmd.extend(["--user-id", str(args.user_id)])
    
    # Create output directory with scheme info
    if config["scheme"] == "naive":
        scheme_output_dir = os.path.join(args.output_dir, "naive", f"L{config['l_bits']}")
    else:
        scheme_output_dir = os.path.join(
            args.output_dir,
            "hi_dypa",
            f"G{config['group_bits']}_U{config['user_bits']}",
        )
    
    if run_tag:
        scheme_output_dir = os.path.join(scheme_output_dir, run_tag)
    
    cmd.extend(["--output-dir", scheme_output_dir])

    # Scheme-specific parameters
    if config["scheme"] == "hi_dypa":
        cmd.extend(["--group-bits", str(config["group_bits"])])
        cmd.extend(["--user-bits", str(config["user_bits"])])

    # Optional seeds file
    if args.seeds_file:
        cmd.extend(["--seeds-file", args.seeds_file])

    return cmd


def main():
    parser = build_common_parser()
    args = parser.parse_args()

    # Normal mode: run evaluations
    print("=" * 80)
    print("Running hi_dypa performance evaluations locally")
    print(f"Evaluator script: {EVAL_SCRIPT}")
    print(f"Prompts file: {args.prompts_file} | Users file: {args.users_file}")
    print(f"Number of prompts per config: {args.max_prompts}")
    print(f"User ID: {args.user_id}")
    print("=" * 80)

    # Determine run_tag early
    run_tag = args.run_tag
    if not run_tag:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_tag = f"local_{timestamp}"

    for config in CONFIGS:
        print("\n" + "=" * 42)
        print(config["label"])
        print("=" * 42)
        cmd = build_command(args, config, run_tag)
        print("Command:", " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(
                f"\nConfiguration failed with exit code {result.returncode}. "
                "Stopping early."
            )
            sys.exit(result.returncode)

    print("\n" + "=" * 80)
    print("All configurations completed successfully!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"   Run tag: {run_tag}")


if __name__ == "__main__":
    main()

