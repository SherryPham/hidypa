#!/usr/bin/env python3
"""
show_hierarchical_results.py

Renders the summary JSON written by evaluate_hierarchical_gpt2.py (and the
tracing-time benchmark) as readable tables, with --markdown / --csv output for
dropping straight into a paper or thesis.

Usage:
    # everything in the results directory
    python helper_scripts/show_hierarchical_results.py evaluation/hierarchical_gpt2

    # one file, as a markdown table
    python helper_scripts/show_hierarchical_results.py \
        evaluation/hierarchical_gpt2/calibrate_gpt2_job_16312504_summary.json --markdown

    # tracing-time benchmark results
    python helper_scripts/show_hierarchical_results.py \
        evaluation/tracing_time/results_local.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def pct(value):
    return "-" if value is None else f"{value * 100:.1f}%"


def num(value, spec=".1f"):
    return "-" if value is None else format(value, spec)


def render(rows, headers, aligns=None, markdown=False):
    if not rows:
        return ""
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    aligns = aligns or [">"] * len(headers)
    aligns = [a if i else "<" for i, a in enumerate(aligns)]

    def line(cells, pad="|" if markdown else " "):
        body = pad.join(
            f" {str(c):{a}{w}} " for c, a, w in zip(cells, aligns, widths)
        )
        return f"|{body}|" if markdown else "  " + body

    out = [line(headers)]
    if markdown:
        out.append("|" + "|".join(
            (":" + "-" * (w + 1) if a == "<" else "-" * (w + 1) + ":")
            for a, w in zip(aligns, widths)) + "|")
    else:
        out.append("  " + "-" * (sum(widths) + 3 * len(widths)))
    out.extend(line(r) for r in rows)
    return "\n".join(out)


# --------------------------------------------------------------- gpt2 summaries

def show_gpt2(payload, path, markdown=False):
    cfg = payload.get("config", {})
    summaries = payload.get("summaries", [])
    if not summaries:
        return None

    mode = summaries[0].get("mode", "?")
    print(f"\n{'=' * 100}")
    print(f"{os.path.basename(path)}   mode={mode}   model={cfg.get('model')}   "
          f"tag={cfg.get('run_tag')}")
    print(f"{'=' * 100}")

    headers = ["config", "blk/bit", "erase", "collide", "exact cw",
               "exact id", "trace us", "gen s", "trials"]
    rows = []
    for s in summaries:
        label = (f"L={s['L']} tok={s.get('max_new_tokens')}" if mode == "calibrate"
                 else f"{s.get('scheme')} (L={s['L']})")
        rows.append([
            label,
            num(s.get("blocks_per_bit_mean")),
            pct(s.get("erasure_rate")),
            pct(s.get("collision_rate")),
            pct(s.get("exact_codeword_rate")),
            pct(s.get("exact_identity_rate")),
            num(s.get("trace_us_median"), ".1f"),
            num(s.get("gen_seconds_mean"), ".1f"),
            s.get("trials", "-"),
        ])
    print(render(rows, headers, markdown=markdown))

    # per-level detail, hierarchical schemes only
    detail = [s for s in summaries if s.get("prefix_accuracy")]
    if detail:
        print("\n  Per-level (prefix accuracy = levels 1..k all correct):")
        depth = max(len(s["prefix_accuracy"]) for s in detail)
        headers2 = ["config"] + [f"L{i + 1}" for i in range(depth)] + \
                   ["containment", "cands/trace"]
        rows2 = []
        for s in detail:
            label = (f"L={s['L']} tok={s.get('max_new_tokens')}" if mode == "calibrate"
                     else f"{s.get('scheme')}")
            cells = [label]
            for i in range(depth):
                acc = s["prefix_accuracy"][i] if i < len(s["prefix_accuracy"]) else None
                cells.append(pct(acc))
            cells.append(num(s.get("containment_size_mean")) + " users")
            cells.append(num(s.get("candidates_evaluated_mean"), ".0f"))
            rows2.append(cells)
        print(render(rows2, headers2, markdown=markdown))

    if mode == "calibrate":
        usable = [s for s in summaries if (s.get("erasure_rate") or 1.0) < 0.05]
        print("\n  Decision rule: largest L reaching < 5% erasure.")
        if usable:
            best = max(usable, key=lambda s: (s["L"], -s["max_new_tokens"]))
            print(f"  -> USE L={best['L']} at max_new_tokens={best['max_new_tokens']} "
                  f"(erasure {pct(best['erasure_rate'])}, "
                  f"exact identity {pct(best['exact_identity_rate'])})")
        else:
            best = max(summaries, key=lambda s: s.get("exact_identity_rate") or 0)
            print(f"  -> No config reached < 5% erasure. Best exact identity: "
                  f"L={best['L']} tok={best['max_new_tokens']} "
                  f"({pct(best['exact_identity_rate'])}). Report the L=12 fallback.")
    return summaries


# ------------------------------------------------------- tracing-time benchmark

def show_tracing(payload, path, markdown=False):
    records = payload.get("results", [])
    if not records:
        return None
    cfg = payload.get("config", {})
    print(f"\n{'=' * 100}")
    print(f"{os.path.basename(path)}   tracing-stage benchmark   L={cfg.get('L')}   "
          f"tag={cfg.get('run_tag')}")
    print(f"{'=' * 100}")

    for n_users in sorted({r["num_users"] for r in records}):
        for rate in sorted({r["erasure_rate"] for r in records}):
            subset = [r for r in records
                      if r["num_users"] == n_users and r["erasure_rate"] == rate]
            if not subset:
                continue
            print(f"\n  N = {n_users}, erasure rate = {rate}")
            rows = [[r["scheme"], num(r["us_per_trace_median"], ".2f"),
                     f"{r['traces_per_second']:,.0f}",
                     pct(r["exact_identification_rate"])] for r in subset]
            print(render(rows, ["scheme", "us/trace", "traces/s", "exact id"],
                         markdown=markdown))
            ours = next((r for r in subset if r["scheme"] == "hier_3layer_tables"), None)
            if ours:
                speed = "  ".join(
                    f"{r['scheme']} x{r['us_per_trace_median'] / ours['us_per_trace_median']:.1f}"
                    for r in subset if r["scheme"] != "hier_3layer_tables")
                print(f"    speedup of hier_3layer_tables: {speed}")
    return records


# -------------------------------------------------------------------- csv export

def write_csv(rows, path):
    if not rows:
        print("Nothing to write.")
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
    print(f"\nWrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("path", help="summary JSON, or a directory of them")
    parser.add_argument("--markdown", action="store_true",
                        help="emit markdown tables for pasting into a paper")
    parser.add_argument("--csv", metavar="OUT", help="also write a flat CSV")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.json")))
    else:
        files = sorted(glob.glob(args.path))
    if not files:
        print(f"No JSON files found at {args.path}")
        return 1

    collected = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {path}: {exc}")
            continue
        if "summaries" in payload:
            rows = show_gpt2(payload, path, args.markdown)
        elif "results" in payload:
            rows = show_tracing(payload, path, args.markdown)
        else:
            print(f"  ! {os.path.basename(path)}: unrecognised format")
            continue
        if rows:
            collected.extend(rows)

    if args.csv:
        write_csv(collected, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
