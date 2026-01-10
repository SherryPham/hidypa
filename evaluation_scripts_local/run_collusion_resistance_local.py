#!/usr/bin/env python
"""
Convenience driver to run all collusion-resistance configurations locally.

This mirrors the pattern of other local evaluation scripts but is designed
for a single workstation. It sequentially invokes
`evaluation_scripts/compare_collusion_resistance.py` for all nine schemes
(naive baseline + eight hi_dypa splits) and forwards shared arguments
such as the prompts file, number of prompts, and model selection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import List

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "evaluation_scripts", "compare_collusion_resistance.py"
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
            "Run all collusion-resistance configurations locally using the "
            "compare_collusion_resistance.py script."
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
        "--num-prompts",
        type=int,
        default=300,
        help="Number of prompts per configuration (default: 300).",
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
        default=400,
        help="Maximum new tokens for generation per user (default: 400).",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/collusion_resistance",
        help="Base output directory (default: evaluation/collusion_resistance).",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag appended to each configuration directory.",
    )
    parser.add_argument(
        "--save-raw-results",
        action="store_true",
        help="If set, also save detailed attack records as raw_results.jsonl.gz",
    )
    parser.add_argument(
        "--raw-results-file",
        type=str,
        default="raw_results.jsonl.gz",
        help="Filename for the raw results artifact (default: raw_results.jsonl.gz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: auto-generated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--generate-csv-summary",
        action="store_true",
        help="After all configs complete, generate a CSV file aggregating all summary.json results.",
    )
    parser.add_argument(
        "--csv-summary-path",
        default=None,
        help="Path for the aggregated CSV summary (default: <output-dir>/summary_all_configs.csv).",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Skip running evaluations and only generate CSV from existing summary.json files.",
    )
    parser.add_argument(
        "--run-tag-filter",
        default=None,
        help="When using --csv-only, filter summaries by this run_tag (default: include all).",
    )
    parser.add_argument(
        "--seeds-file",
        default=None,
        help="Path to seeds.txt file to read existing seeds from (optional). If provided, seeds will be reused for each configuration.",
    )
    return parser


def build_command(args: argparse.Namespace, config: dict, run_tag: str) -> List[str]:
    cmd = [args.python_bin, EVAL_SCRIPT]
    cmd.extend(["--scheme", config["scheme"]])
    cmd.extend(["--prompts-file", args.prompts_file])
    cmd.extend(["--num-prompts", str(args.num_prompts)])
    cmd.extend(["--users-file", args.users_file])
    cmd.extend(["--model", args.model])
    cmd.extend(["--delta", str(args.delta)])
    cmd.extend(["--entropy-threshold", str(args.entropy_threshold)])
    cmd.extend(["--hashing-context", str(args.hashing_context)])
    cmd.extend(["--z-threshold", str(args.z_threshold)])
    cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    cmd.extend(["--output-dir", args.output_dir])
    if run_tag:
        cmd.extend(["--run-tag", run_tag])

    # Scheme-specific parameters
    if config["scheme"] == "naive":
        cmd.extend(["--l-bits", str(config["l_bits"])])
    else:
        cmd.extend(["--group-bits", str(config["group_bits"])])
        cmd.extend(["--user-bits", str(config["user_bits"])])
        cmd.extend(["--l-bits", str(config["l_bits"])])

    # Optional arguments
    if args.save_raw_results:
        cmd.append("--save-raw-results")
    if args.raw_results_file != "raw_results.jsonl.gz":
        cmd.extend(["--raw-results-file", args.raw_results_file])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.seeds_file:
        cmd.extend(["--seeds-file", args.seeds_file])

    return cmd


def collect_and_aggregate_summaries(
    output_dir: str, run_tag: str | None, csv_path: str | None = None
) -> str:
    """
    Collect all summary.json files from the output directory structure and
    aggregate them into a single CSV file.
    
    Note: This script creates two summary.json files per configuration:
    - One in <config_dir>/2_colluders/summary.json
    - One in <config_dir>/3_colluders/summary.json
    
    We'll create separate rows for each, with a 'num_colluders' column.
    
    Args:
        output_dir: Base output directory (e.g., "evaluation/collusion_resistance")
        run_tag: Optional run tag to filter summaries (None = include all)
        csv_path: Optional custom path for CSV output
    
    Returns:
        Path to the generated CSV file
    """
    output_dir_abs = os.path.join(REPO_ROOT, output_dir)
    
    if csv_path is None:
        csv_path = os.path.join(output_dir_abs, "summary_all_configs.csv")
    else:
        csv_path = os.path.join(REPO_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
    
    # Find all summary.json files (including those in 2_colluders and 3_colluders subdirs)
    summary_files = []
    for root, dirs, files in os.walk(output_dir_abs):
        if "summary.json" in files:
            summary_path = os.path.join(root, "summary.json")
            # Only include summaries from this run_tag if specified
            if run_tag and run_tag in summary_path:
                summary_files.append(summary_path)
            elif not run_tag:
                # If no run_tag filter, include all summaries
                summary_files.append(summary_path)
    
    if not summary_files:
        print(f"Warning: No summary.json files found in {output_dir_abs}")
        if run_tag:
            print(f"  (filtered by run_tag: {run_tag})")
        return csv_path
    
    print(f"\nFound {len(summary_files)} summary.json files. Aggregating...")
    
    # Load and flatten all summaries
    rows = []
    for summary_path in summary_files:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            
            # Determine num_colluders from the path or summary
            num_colluders = summary.get("num_colluders", None)
            if num_colluders is None:
                # Try to infer from path
                if "2_colluders" in summary_path:
                    num_colluders = 2
                elif "3_colluders" in summary_path:
                    num_colluders = 3
            
            # Create a flat row
            row = {
                "scheme": summary.get("scheme", ""),
                "model": summary.get("model", ""),
                "run_tag": summary.get("run_tag", ""),
                "l_bits": summary.get("l_bits", ""),
                "group_bits": summary.get("group_bits", ""),
                "user_bits": summary.get("user_bits", ""),
                "num_prompts": summary.get("num_prompts", ""),
                "num_colluders": num_colluders,
                "random_seed": summary.get("random_seed", ""),
                "generated_utc": summary.get("generated_utc", ""),
            }
            
            # Flatten success_rates into the row
            success_rates = summary.get("success_rates", {})
            for case_type, stats in success_rates.items():
                if isinstance(stats, dict):
                    row[f"{case_type}_successful"] = stats.get("successful", "")
                    row[f"{case_type}_total"] = stats.get("total", "")
                    row[f"{case_type}_success_rate"] = stats.get("success_rate", "")
                    # Store false positives if available
                    if "false_positives" in stats:
                        row[f"{case_type}_false_positives"] = stats.get("false_positives", 0)
                    if "false_positive_rate" in stats:
                        row[f"{case_type}_false_positive_rate"] = stats.get("false_positive_rate", 0.0)
                else:
                    row[f"{case_type}_success_rate"] = stats
            
            rows.append(row)
        except Exception as e:
            print(f"  Warning: Failed to load {summary_path}: {e}")
            continue
    
    if not rows:
        print("  Warning: No valid summaries could be loaded.")
        return csv_path
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Calculate aggregate success rates for each row
    # 2-colluder cases
    case_types_2 = ['same_group_2', 'cross_group_2']
    # 3-colluder cases
    case_types_3 = ['same_group_3', 'cross_group_3', 'mixed_2same_1diff']
    
    # Calculate success rates for each row
    overall_successful = []
    overall_total = []
    success_2_successful = []
    success_2_total = []
    success_3_successful = []
    success_3_total = []
    
    for idx, row in df.iterrows():
        # Overall (all cases)
        total_successful = 0
        total_count = 0
        
        # 2-colluder cases
        successful_2 = 0
        total_2 = 0
        
        # 3-colluder cases
        successful_3 = 0
        total_3 = 0
        
        # Check all case types
        all_case_types = case_types_2 + case_types_3
        for case_type in all_case_types:
            successful_col = f"{case_type}_successful"
            total_col = f"{case_type}_total"
            
            if successful_col in df.columns and total_col in df.columns:
                successful_val = row.get(successful_col)
                total_val = row.get(total_col)
                
                # Check if values are valid (not NaN, not empty string)
                if pd.notna(successful_val) and pd.notna(total_val) and successful_val != "" and total_val != "":
                    try:
                        successful_val = float(successful_val)
                        total_val = float(total_val)
                        total_successful += successful_val
                        total_count += total_val
                        
                        # Add to 2 or 3 colluder totals
                        if case_type in case_types_2:
                            successful_2 += successful_val
                            total_2 += total_val
                        elif case_type in case_types_3:
                            successful_3 += successful_val
                            total_3 += total_val
                    except (ValueError, TypeError):
                        pass
        
        # Calculate rates
        overall_successful.append(total_successful)
        overall_total.append(total_count)
        success_2_successful.append(successful_2)
        success_2_total.append(total_2)
        success_3_successful.append(successful_3)
        success_3_total.append(total_3)
    
    # Add success rate columns to dataframe
    df['overall_success_rate'] = [
        (s / t * 100.0) if t > 0 else 0.0 
        for s, t in zip(overall_successful, overall_total)
    ]
    
    df['success_rate_2_colluders'] = [
        (s / t * 100.0) if t > 0 else 0.0 
        for s, t in zip(success_2_successful, success_2_total)
    ]
    
    df['success_rate_3_colluders'] = [
        (s / t * 100.0) if t > 0 else 0.0 
        for s, t in zip(success_3_successful, success_3_total)
    ]
    
    # Calculate false positive rates from summary data
    # Note: false_positive_rate is now "percentage of prompts with at least one false positive"
    # We calculate weighted averages based on the number of prompts in each case type
    overall_fp_rates = []
    fp_rates_2 = []
    fp_rates_3 = []
    
    for idx, row in df.iterrows():
        # Calculate weighted false positive rate for 2-colluder cases
        fp_rate_2 = 0.0
        total_prompts_2 = 0
        for case_type in case_types_2:
            fp_rate_col = f"{case_type}_false_positive_rate"
            total_col = f"{case_type}_total"
            if fp_rate_col in df.columns and total_col in df.columns:
                fp_rate_val = row.get(fp_rate_col)
                total_val = row.get(total_col)
                if pd.notna(fp_rate_val) and pd.notna(total_val) and fp_rate_val != "" and total_val != "":
                    try:
                        fp_rate_val = float(fp_rate_val)
                        total_val = float(total_val)
                        # Weight by number of prompts
                        fp_rate_2 += fp_rate_val * total_val
                        total_prompts_2 += total_val
                    except (ValueError, TypeError):
                        pass
        
        # Calculate weighted false positive rate for 3-colluder cases
        fp_rate_3 = 0.0
        total_prompts_3 = 0
        for case_type in case_types_3:
            fp_rate_col = f"{case_type}_false_positive_rate"
            total_col = f"{case_type}_total"
            if fp_rate_col in df.columns and total_col in df.columns:
                fp_rate_val = row.get(fp_rate_col)
                total_val = row.get(total_col)
                if pd.notna(fp_rate_val) and pd.notna(total_val) and fp_rate_val != "" and total_val != "":
                    try:
                        fp_rate_val = float(fp_rate_val)
                        total_val = float(total_val)
                        # Weight by number of prompts
                        fp_rate_3 += fp_rate_val * total_val
                        total_prompts_3 += total_val
                    except (ValueError, TypeError):
                        pass
        
        # Calculate overall weighted false positive rate
        fp_rate_overall = 0.0
        total_prompts_overall = 0
        all_case_types = case_types_2 + case_types_3
        for case_type in all_case_types:
            fp_rate_col = f"{case_type}_false_positive_rate"
            total_col = f"{case_type}_total"
            if fp_rate_col in df.columns and total_col in df.columns:
                fp_rate_val = row.get(fp_rate_col)
                total_val = row.get(total_col)
                if pd.notna(fp_rate_val) and pd.notna(total_val) and fp_rate_val != "" and total_val != "":
                    try:
                        fp_rate_val = float(fp_rate_val)
                        total_val = float(total_val)
                        # Weight by number of prompts
                        fp_rate_overall += fp_rate_val * total_val
                        total_prompts_overall += total_val
                    except (ValueError, TypeError):
                        pass
        
        # Calculate weighted averages
        fp_rates_2.append(fp_rate_2 / total_prompts_2 if total_prompts_2 > 0 else 0.0)
        fp_rates_3.append(fp_rate_3 / total_prompts_3 if total_prompts_3 > 0 else 0.0)
        overall_fp_rates.append(fp_rate_overall / total_prompts_overall if total_prompts_overall > 0 else 0.0)
    
    # Add false positive rate columns
    df['overall_false_positive_rate'] = overall_fp_rates
    df['false_positive_rate_2_colluders'] = fp_rates_2
    df['false_positive_rate_3_colluders'] = fp_rates_3
    
    # Calculate average wrongly accused users per prompt for each case type
    all_case_types = case_types_2 + case_types_3
    for case_type in all_case_types:
        fp_col = f"{case_type}_false_positives"
        total_col = f"{case_type}_total"
        avg_col = f"{case_type}_avg_wrongly_accused_per_prompt"
        
        if fp_col in df.columns and total_col in df.columns:
            avg_values = []
            for idx, row in df.iterrows():
                fp_val = row.get(fp_col)
                total_val = row.get(total_col)
                if pd.notna(fp_val) and pd.notna(total_val) and fp_val != "" and total_val != "":
                    try:
                        fp_val = float(fp_val)
                        total_val = float(total_val)
                        avg = fp_val / total_val if total_val > 0 else 0.0
                        avg_values.append(avg)
                    except (ValueError, TypeError):
                        avg_values.append(0.0)
                else:
                    avg_values.append(0.0)
            df[avg_col] = avg_values
    
    # Sort by scheme, then group_bits, then user_bits, then num_colluders for consistent ordering
    sort_columns = ["scheme", "group_bits", "user_bits", "num_colluders"]
    available_sort_columns = [col for col in sort_columns if col in df.columns]
    if available_sort_columns:
        df = df.sort_values(
            by=available_sort_columns,
            na_position="last",
        )
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"  Aggregated CSV saved to: {csv_path}")
    print(f"  Contains {len(df)} rows (configurations × colluder counts) with {len(df.columns)} columns")
    
    return csv_path


def print_avg_wrongly_accused_summary(csv_path: str):
    """
    Print a summary table of average wrongly accused users per prompt from the CSV.
    
    Args:
        csv_path: Path to the CSV file containing the aggregated results
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: Could not read CSV file {csv_path}: {e}")
        return
    
    print("\n" + "=" * 80)
    print(" " * 20 + "AVERAGE WRONGLY ACCUSED USERS PER PROMPT")
    print("=" * 80)
    
    # Case types to check
    case_types_2 = ['same_group_2', 'cross_group_2']
    case_types_3 = ['same_group_3', 'cross_group_3', 'mixed_2same_1diff']
    all_case_types = case_types_2 + case_types_3
    
    # Group by scheme and configuration
    for scheme in df['scheme'].unique():
        scheme_df = df[df['scheme'] == scheme]
        print(f"\n{scheme.upper()} Scheme:")
        print("-" * 80)
        
        for idx, row in scheme_df.iterrows():
            # Build config label
            if scheme == 'naive':
                config_label = f"L{row.get('l_bits', '?')}"
            else:
                g_bits = row.get('group_bits', '?')
                u_bits = row.get('user_bits', '?')
                config_label = f"G{g_bits}_U{u_bits}"
            
            num_colluders = row.get('num_colluders', '?')
            print(f"\n  Config: {config_label} | {num_colluders} Colluders")
            
            # Print averages for each case type
            for case_type in all_case_types:
                avg_col = f"{case_type}_avg_wrongly_accused_per_prompt"
                fp_col = f"{case_type}_false_positives"
                total_col = f"{case_type}_total"
                
                if avg_col in df.columns:
                    avg_val = row.get(avg_col)
                    fp_val = row.get(fp_col)
                    total_val = row.get(total_col)
                    
                    if pd.notna(avg_val) and pd.notna(fp_val) and pd.notna(total_val):
                        try:
                            avg_val = float(avg_val)
                            fp_val = float(fp_val)
                            total_val = float(total_val)
                            if total_val > 0:
                                print(f"    {case_type:25s}: {avg_val:6.2f} users/prompt (total: {fp_val:.0f} across {total_val:.0f} prompts)")
                        except (ValueError, TypeError):
                            pass
    
    print("\n" + "=" * 80)


def main():
    parser = build_common_parser()
    args = parser.parse_args()

    # CSV-only mode: skip evaluations and just aggregate existing summaries
    if args.csv_only:
        print("=" * 80)
        print("CSV Aggregation Mode: Generating CSV from existing summary.json files")
        print("=" * 80)
        
        # Use run_tag_filter if provided, otherwise use run_tag, otherwise None (include all)
        filter_tag = args.run_tag_filter if args.run_tag_filter is not None else args.run_tag
        
        csv_path = collect_and_aggregate_summaries(
            args.output_dir, filter_tag, args.csv_summary_path
        )
        print(f"\nCSV summary: {csv_path}")
        
        # Print average wrongly accused users per prompt summary
        print_avg_wrongly_accused_summary(csv_path)
        return

    # Normal mode: run evaluations
    print("=" * 80)
    print("Running collusion resistance evaluations locally")
    print(f"Evaluator script: {EVAL_SCRIPT}")
    print(f"Prompts file: {args.prompts_file} | Users file: {args.users_file}")
    print(f"Number of prompts per config: {args.num_prompts}")
    print("=" * 80)

    # Determine run_tag early so we can use it for CSV aggregation
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
    
    # Generate CSV summary if requested
    if args.generate_csv_summary and not args.dry_run:
        csv_path = collect_and_aggregate_summaries(
            args.output_dir, run_tag, args.csv_summary_path
        )
        print(f"\nCSV summary: {csv_path}")
        
        # Print average wrongly accused users per prompt summary
        print_avg_wrongly_accused_summary(csv_path)


if __name__ == "__main__":
    main()

