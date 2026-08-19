#!/usr/bin/env python3
"""
Collect the Segment-WM baseline results into paper-ready table rows.

Reads the summary.json files written by the evaluation scripts under
`evaluation/<experiment>/segment/L8/<run_tag>/` and prints one row per table
(II-XIII) in the same format and column order as the Hi-DyPa paper, so the
numbers can be pasted straight into the LaTeX tables.

Usage
-----
    python helper_scripts/collect_segment_rows.py --evaluation-dir evaluation --model gpt2
    python helper_scripts/collect_segment_rows.py --evaluation-dir evaluation --model deepseek-llm-7b --latex

If several run tags exist for an experiment, the most recently modified one is
used unless --run-tag is given.
"""

import argparse
import glob
import json
import os
import re
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

SCHEME_LABEL = "Segment-WM"
CONFIG_LABEL = "(0,8)"

RATIOS = [0.05, 0.10, 0.15, 0.20]

# experiment directory -> (paper table for GPT-2, paper table for DeepSeek)
EXPERIMENTS = {
    "hi_dypa_detection": ("II", "II"),
    "collusion_resistance": ("III / IV", "III / IV"),
    "framing_attack": ("V", "V"),
    "robustness": ("VI", "VII"),
    "paraphrasing_attack": ("VIII", "IX"),
    "synonym_attack": ("X", "XI"),
    "rewrite_attack": ("XII", "XIII"),
}

VARIANT_PREFIX = {
    "robustness": "deletion",
    "paraphrasing_attack": "paraphrase",
    "synonym_attack": "synonym",
    "rewrite_attack": "rewrite",
}


def summary_model(path):
    """Read the model name recorded inside a summary.json, if any."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("model")
    except Exception:
        return None


def find_summary(evaluation_dir, experiment, run_tag, filename="summary.json", model=None):
    """
    Locate the segment summary for one experiment.

    GPT-2 and DeepSeek runs write to the same directory tree, so picking purely
    by modification time would silently report one model's numbers under the
    other's table number. Every summary.json records the model it was produced
    with, so filter on that and only fall back to newest-wins when the field is
    missing (older result files).
    """
    base = os.path.join(evaluation_dir, experiment, "segment", "L8")
    if not os.path.isdir(base):
        return None

    if run_tag:
        candidates = [os.path.join(base, run_tag, filename)]
    else:
        candidates = sorted(
            glob.glob(os.path.join(base, "*", filename)) + [os.path.join(base, filename)],
            key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
            reverse=True,
        )

    existing = [p for p in candidates if os.path.exists(p)]
    if not existing:
        return None

    if model and not run_tag:
        # The nested collusion summaries live one level down; check the parent.
        for path in existing:
            probe = path
            if os.path.basename(os.path.dirname(path)).endswith("_colluders"):
                probe = os.path.join(os.path.dirname(os.path.dirname(path)), "summary.json")
            recorded = summary_model(path) or summary_model(probe)
            if recorded == model:
                return path
        matched_any = any(
            summary_model(p) is not None for p in existing
        )
        if matched_any:
            return None      # results exist, but none for this model

    return existing[0]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detection_row(evaluation_dir, run_tag, model=None):
    path = find_summary(evaluation_dir, "hi_dypa_detection", run_tag, model=model)
    if not path:
        return None, None
    data = load(path)
    metrics = data["metrics"]
    return {
        "Acc.": metrics.get("full_identity_accuracy"),
        "FP": metrics.get("false_positive_rate"),
    }, path


def collusion_row(evaluation_dir, run_tag, num_colluders, model=None):
    path = find_summary(
        evaluation_dir, "collusion_resistance", run_tag,
        os.path.join(f"{num_colluders}_colluders", "summary.json"), model=model,
    )
    if not path:
        return None, None
    data = load(path)
    rates = data.get("success_rates", {})

    successful = sum(c.get("successful", 0) for c in rates.values())
    total = sum(c.get("total", 0) for c in rates.values())
    wrong = sum(c.get("false_positives", 0) for c in rates.values())

    return {
        "Acc.": (successful / total * 100.0) if total else None,
        "Wrong Acc.": (wrong / total) if total else None,
    }, path


def framing_row(evaluation_dir, run_tag, k, model=None):
    path = find_summary(evaluation_dir, "framing_attack", run_tag, model=model)
    if not path:
        return None, None
    data = load(path)
    entry = data.get("crr_by_k", {}).get(str(k))
    if entry is None:
        available = ", ".join(sorted(data.get("crr_by_k", {}), key=int))
        return {"CRR (%)": None, "note": f"k={k} not evaluated (have: {available})"}, path
    return {"CRR (%)": entry["mean"]}, path


def model_from_slurm_log(perf_dir):
    """
    Recover the model for a performance run that predates the `model` column.

    The directory is named segment_L8_job_<jobid>, and every SLURM script prints
    "Model      : <name>" in its header, so the log identifies the run.
    """
    match = re.search(r"segment_L8_job_(\d+)$", os.path.basename(perf_dir.rstrip(os.sep)))
    if not match:
        return None
    log = os.path.join(parent_dir, "slurm_out", f"slurm-{match.group(1)}.out")
    if not os.path.exists(log):
        return None
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                found = re.match(r"\s*Model\s*:\s*(\S+)", line)
                if found:
                    return found.group(1)
                if line.startswith("==="):
                    continue
    except Exception:
        return None
    return None


def read_perf_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        values = f.readline().strip().split(",")
    return dict(zip(header, values))


def performance_row(evaluation_dir, run_tag, model=None):
    """Table XIV: tracing time in milliseconds."""
    base = os.path.join(evaluation_dir, "multiuser_performance")
    if not os.path.isdir(base):
        return None, None

    pattern = f"segment_L8_{run_tag}" if run_tag else "segment_L8_*"
    candidates = sorted(
        glob.glob(os.path.join(base, pattern, "performance_summary.csv")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not candidates:
        return None, None

    path = candidates[0]
    if model and not run_tag:
        chosen = None
        for candidate in candidates:
            record = read_perf_csv(candidate)
            recorded = record.get("model") or model_from_slurm_log(os.path.dirname(candidate))
            if recorded == model:
                chosen = candidate
                break
            if recorded is None and chosen is None:
                chosen = candidate      # unidentifiable: keep as last resort
        if chosen is None:
            return None, None
        path = chosen

    record = read_perf_csv(path)
    try:
        trace_ms = float(record["trace_time_sec"]) * 1000.0
    except (KeyError, ValueError):
        return None, path
    return {"Trace (ms)": trace_ms}, path


def attack_row(evaluation_dir, experiment, run_tag, model=None):
    """Average each metric over the four positional modes, per perturbation ratio."""
    path = find_summary(evaluation_dir, experiment, run_tag, model=model)
    if not path:
        return None, None
    data = load(path)
    by_variant = data.get("metrics_by_variant", {})
    prefix = VARIANT_PREFIX[experiment]
    ratio_key = f"{prefix}_percent" if experiment == "robustness" else f"{prefix}_ratio"

    row = {}
    for ratio in RATIOS:
        accs, fps = [], []
        for entry in by_variant.values():
            value = entry.get(ratio_key)
            if value is None or abs(float(value) - ratio) > 1e-9:
                continue
            accs.append(entry.get("full_identity_accuracy"))
            fps.append(entry.get("false_positive_rate"))
        accs = [a for a in accs if a is not None]
        fps = [f for f in fps if f is not None]
        row[f"{ratio:.2f} Acc."] = sum(accs) / len(accs) if accs else None
        row[f"{ratio:.2f} FP"] = sum(fps) / len(fps) if fps else None
    return row, path


def fmt(value, decimals=3):
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--evaluation-dir", type=str, default="evaluation",
                        help="Root of the evaluation results tree")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model the results were produced with (used to pick the table number)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Specific run tag; default is the most recent")
    parser.add_argument("--framing-k", type=int, default=50,
                        help="Number of colluding trials to report for Table V (default: 50)")
    parser.add_argument("--latex", action="store_true",
                        help="Also emit the LaTeX row for each table")
    parser.add_argument("--table", type=str, nargs="+", default=None,
                        metavar="ROMAN",
                        help="Show only these tables, e.g. --table II  or  --table III IV. "
                             "Default: all of them.")
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir
    if not os.path.isabs(evaluation_dir):
        evaluation_dir = os.path.join(parent_dir, evaluation_dir)
    if not os.path.isdir(evaluation_dir):
        sys.exit(f"Evaluation directory not found: {evaluation_dir}")

    is_deepseek = "deepseek" in args.model.lower()
    print()
    print("=" * 78)
    print(f" Segment-WM baseline rows   model={args.model}   config (Lg, Lu) = (0, 8)")
    print("=" * 78)

    rows = []

    row, path = detection_row(evaluation_dir, args.run_tag, args.model)
    rows.append(("II", "Table II  (clean-text detection)", row, path,
                 ["Acc.", "FP"], 3))

    for k in (2, 3):
        row, path = collusion_row(evaluation_dir, args.run_tag, k, args.model)
        table = "III" if k == 2 else "IV"
        rows.append((table, f"Table {table}  ({k}-colluder collusion)", row, path,
                     ["Acc.", "Wrong Acc."], 2))

    row, path = framing_row(evaluation_dir, args.run_tag, args.framing_k, args.model)
    rows.append(("V", f"Table V   (framing, k={args.framing_k})", row, path,
                 ["CRR (%)"], 1))

    for experiment in ("robustness", "paraphrasing_attack", "synonym_attack", "rewrite_attack"):
        table = EXPERIMENTS[experiment][1 if is_deepseek else 0]
        row, path = attack_row(evaluation_dir, experiment, args.run_tag, args.model)
        columns = [f"{r:.2f} {m}" for r in RATIOS for m in ("Acc.", "FP")]
        label = experiment.replace("_", " ")
        rows.append((table, f"Table {table:<4}({label})", row, path, columns, 3))

    row, path = performance_row(evaluation_dir, args.run_tag, args.model)
    rows.append(("XIV", "Table XIV (tracing time)", row, path, ["Trace (ms)"], 3))

    if args.table:
        wanted = {t.upper() for t in args.table}
        rows = [r for r in rows if r[0].upper() in wanted]
        if not rows:
            sys.exit(f"No known table matches {sorted(wanted)}. "
                     f"Valid: II III IV V VI VII VIII IX X XI XII XIII XIV")

    for _table_id, title, row, path, columns, decimals in rows:
        print()
        print(f"{title}")
        if row is None:
            print("  (no results found -- run the corresponding SLURM script first)")
            continue
        relative = os.path.relpath(path, parent_dir)
        print(f"  source: {path if relative.startswith('..') else relative}")
        width = max(len(c) for c in columns)
        header = "  " + "  ".join(f"{c:>{width}}" for c in columns)
        values = "  " + "  ".join(f"{fmt(row.get(c), decimals):>{width}}" for c in columns)
        print(header)
        print(values)
        if row.get("note"):
            print(f"  note: {row['note']}")
        if args.latex:
            cells = " & ".join(fmt(row.get(c), decimals) for c in columns)
            print(f"  LaTeX: {SCHEME_LABEL} & {CONFIG_LABEL} & {cells} \\\\")

    print()
    print("Notes:")
    print("  * Acc. is full identity (user) accuracy; FP is the false positive rate.")
    print("  * Attack rows are averaged over the four positional modes for each ratio,")
    print("    matching the paper's 'averaged over positional modes' convention.")
    print("  * Table III/IV 'Wrong Acc.' is wrongly accused non-colluders per prompt.")
    print()


if __name__ == "__main__":
    main()
