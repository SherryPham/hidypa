#!/usr/bin/env python3
"""Collect the clean-text detection L-sweep into one comparison table.

Reads every summary.json written by evaluate_hi_dypa_detection.py under an
evaluation directory and prints one row per (scheme, codeword length L), so the
cost of a longer payload can be read off directly and Hi-DyPa can be compared
against the flat (naive) baseline at each L.

Group/user accuracy only exist for hi_dypa; flat schemes show "--" there.

Usage:
    python3 helper_scripts/collect_lbit_detection_rows.py \
        --evaluation-dir evaluation/hi_dypa_lbit_detection \
        --model gpt2 --run-tag job_1234567 --latex
"""

import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Column order matches the paper's clean-detection table.
METRIC_COLUMNS = [
    ('group_accuracy', 'Group acc'),
    ('user_accuracy', 'User acc'),
    ('full_identity_accuracy', 'Full ID acc'),
    ('lbit_accuracy', 'L-bit acc'),
    ('false_positive_rate', 'FPR'),
    ('false_negative_rate', 'FNR'),
]

# hi_dypa first, then the flat baselines, so the table reads as
# "the scheme, then what it is being compared against".
SCHEME_ORDER = {'hi_dypa': 0, 'naive': 1, 'segment': 2}


def find_summaries(evaluation_dir: str) -> list[str]:
    """Return every summary.json path beneath evaluation_dir."""
    paths = []
    for root, _dirs, files in os.walk(evaluation_dir):
        if 'summary.json' in files:
            paths.append(os.path.join(root, 'summary.json'))
    return sorted(paths)


def load_summary(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Warning: could not read {path}: {exc}", file=sys.stderr)
        return None


def format_row(summary: dict) -> dict:
    metrics = summary.get('metrics', {})
    row = {
        'scheme': summary.get('scheme'),
        'l_bits': summary.get('l_bits'),
        'group_bits': summary.get('group_bits'),
        'user_bits': summary.get('user_bits'),
        'num_users': summary.get('num_users'),
        'num_prompts': summary.get('num_prompts'),
        'seed': summary.get('random_seed'),
        'run_tag': summary.get('run_tag'),
    }
    for key, _label in METRIC_COLUMNS:
        row[key] = metrics.get(key)
    return row


def bits_label(row: dict) -> str:
    """G/U split for hi_dypa, or 'flat' for the single-codeword schemes."""
    if row['scheme'] == 'hi_dypa' and row['group_bits'] is not None:
        return f"{row['group_bits']}/{row['user_bits']}"
    return 'flat'


def print_table(rows: list[dict]) -> None:
    header = f"{'Scheme':<9} {'L':>4} {'G/U':>7} {'Users':>7} {'Prompts':>8}"
    for _key, label in METRIC_COLUMNS:
        header += f" {label:>12}"
    print(header)
    print('-' * len(header))

    previous_scheme = None
    for row in rows:
        if previous_scheme is not None and row['scheme'] != previous_scheme:
            print('-' * len(header))
        previous_scheme = row['scheme']

        users = row['num_users'] if row['num_users'] is not None else '?'
        line = (
            f"{row['scheme']:<9} {row['l_bits']:>4} {bits_label(row):>7} "
            f"{users:>7} {row['num_prompts']:>8}"
        )
        for key, _label in METRIC_COLUMNS:
            value = row[key]
            line += f" {value:>12.4f}" if isinstance(value, float) else f" {'--':>12}"
        print(line)


def print_latex(rows: list[dict], model: str) -> None:
    print()
    print(r"% Clean-text detection accuracy vs. codeword length L "
          f"({model}, capacity-scaled population)")
    print(r"\begin{tabular}{l r r r " + "r " * len(METRIC_COLUMNS) + r"}")
    print(r"\toprule")
    labels = ' & '.join(label for _key, label in METRIC_COLUMNS)
    print(rf"Scheme & $L$ & $L_g/L_u$ & $N$ & {labels} \\")
    print(r"\midrule")
    previous_scheme = None
    for row in rows:
        if previous_scheme is not None and row['scheme'] != previous_scheme:
            print(r"\midrule")
        previous_scheme = row['scheme']

        cells = []
        for key, _label in METRIC_COLUMNS:
            value = row[key]
            cells.append(f"{value:.3f}" if isinstance(value, float) else '--')
        scheme_name = row['scheme'].replace('_', r'\_')
        print(
            f"{scheme_name} & {row['l_bits']} & {bits_label(row)} & "
            f"{row['num_users'] if row['num_users'] is not None else '--'} & "
            + ' & '.join(cells) + r" \\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect clean-text detection results across codeword lengths"
    )
    parser.add_argument(
        '--evaluation-dir',
        type=str,
        default='evaluation/hi_dypa_lbit_detection',
        help='Directory holding the sweep results (default: evaluation/hi_dypa_lbit_detection)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Only include runs from this model (e.g. gpt2)'
    )
    parser.add_argument(
        '--run-tag',
        type=str,
        default=None,
        help='Only include runs with this run tag (e.g. job_1234567)'
    )
    parser.add_argument(
        '--scheme',
        type=str,
        default=None,
        help='Only include this scheme (hi_dypa, naive, segment)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Also write the table to this CSV path'
    )
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Also print a LaTeX tabular of the same table'
    )
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir
    if not os.path.isabs(evaluation_dir):
        evaluation_dir = os.path.join(parent_dir, evaluation_dir)

    if not os.path.isdir(evaluation_dir):
        print(f"Error: evaluation directory not found: {evaluation_dir}", file=sys.stderr)
        return 1

    rows = []
    for path in find_summaries(evaluation_dir):
        summary = load_summary(path)
        if summary is None:
            continue
        if args.scheme and summary.get('scheme') != args.scheme:
            continue
        if args.model and summary.get('model') != args.model:
            continue
        if args.run_tag and summary.get('run_tag') != args.run_tag:
            continue
        rows.append(format_row(summary))

    if not rows:
        print(f"No matching summaries found under {evaluation_dir}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (
        SCHEME_ORDER.get(r['scheme'], 99),
        r['l_bits'] if r['l_bits'] is not None else 0,
    ))

    print()
    print("=" * 96)
    print(" Clean-text detection accuracy vs. codeword length")
    if args.model:
        print(f" Model: {args.model}")
    if args.run_tag:
        print(f" Run tag: {args.run_tag}")
    print("=" * 96)
    print_table(rows)

    if args.latex:
        print_latex(rows, args.model or 'gpt2')

    if args.csv:
        import csv as csv_module
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(parent_dir, csv_path)
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        fieldnames = (
            ['scheme', 'l_bits', 'group_bits', 'user_bits', 'num_users',
             'num_prompts', 'seed', 'run_tag']
            + [key for key, _label in METRIC_COLUMNS]
        )
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to: {csv_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
