#!/usr/bin/env python3
"""Figure: clean-text detection accuracy vs. codeword length L.

Two panels, sized for a double-column (figure*) slot:

  (a) Identification accuracy vs. L, Hi-DyPa against the flat/naive baseline.
  (b) Mean detection z-score vs. L on log-log axes, with a least-squares fit of
      z = a/L + c and the z = 4 decision threshold. The fit is what turns the
      accuracy collapse in (a) into a derived capacity limit.

Panel (b) is deliberately a separate panel rather than a second y-axis on (a):
two measures on different scales never share an axis.

Usage:
    python3 helper_scripts/plot_lbit_detection.py                       # accuracy panel only
    python3 helper_scripts/plot_lbit_detection.py --csv evaluation/hi_dypa_lbit_detection/lbit_table.csv
    python3 helper_scripts/plot_lbit_detection.py --metric lbit_accuracy
    python3 helper_scripts/plot_lbit_detection.py --out fig/lbit.pdf --png

Reading --csv (the file written by collect_lbit_detection_rows.py --csv) is the
reproducible path: it overrides the inline accuracy numbers so the figure always
matches the run. Mean z-scores are not in summary.json, so they stay inline --
regenerate them with:

    for d in evaluation/hi_dypa_lbit_detection/hi_dypa/*/job_<ID>; do
      python3 -c "
    import gzip,json,statistics
    z=[json.loads(l)['z_score'] for l in gzip.open('$d/raw_results.jsonl.gz','rt')]
    print('$d'.split('/')[-2], '%.2f'%statistics.mean(z))"
    done
"""

from __future__ import annotations

import argparse
import csv as csv_module
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Data — GPT-2, 300 prompts, 512 tokens, capacity-scaled population
# (job array 15706204). Override the accuracy rows with --csv.
# =============================================================================
L_VALUES = [8, 12, 16, 32, 48]

# np.nan marks a run whose number is not in hand yet; the line simply breaks
# there rather than being drawn through a fabricated point.
ACCURACY = {
    "hi_dypa": {
        "full_identity_accuracy": [0.9167, 0.4700, 0.0100, 0.0000, 0.0000],
        "lbit_accuracy":          [0.8733, 0.1233, 0.0000, 0.0000, 0.0000],
        "group_accuracy":         [0.9400, 0.6567, 0.0300, 0.0000, 0.0000],
    },
    "naive": {
        "full_identity_accuracy": [0.8667, 0.3467, 0.0067, 0.0000, np.nan],
        "lbit_accuracy":          [0.8533, 0.1300, 0.0033, 0.0000, np.nan],
        "group_accuracy":         [np.nan] * 5,
    },
    # Segment-WM baseline: filled in by --csv once the segment array finishes.
    "segment": {
        "full_identity_accuracy": [np.nan] * 5,
        "lbit_accuracy":          [np.nan] * 5,
        "group_accuracy":         [np.nan] * 5,
    },
}

# Mean of the per-prompt z_score field, Hi-DyPa runs.
MEAN_Z = [6.96, 4.60, 3.55, 1.79, 1.35]

Z_THRESHOLD = 4.0

# =============================================================================
# Style — validated categorical slots, checked all-pairs (three lines can each
# sit next to either other, so the adjacent-only pairlist is not enough):
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#4a3aa7" --mode light --pairs all
#   -> worst CVD dE 13.0, worst normal-vision dE 16.3, contrast all >= 3:1: PASS.
# Violet is the third slot rather than the palette's aqua (contrast 2.74:1 on a
# white surface, which would need label relief) or green (CVD dE 3.2 against
# orange -- a fail). Line style and marker shape repeat the identity so the
# figure survives grayscale printing, where hue alone would not.
# =============================================================================
SERIES = {
    "hi_dypa": {"color": "#2a78d6", "marker": "o", "linestyle": "-",  "label": "Hi-DyPa"},
    "naive":   {"color": "#eb6834", "marker": "s", "linestyle": "--", "label": "Flat (naive)"},
    "segment": {"color": "#4a3aa7", "marker": "^", "linestyle": ":",  "label": "Segment-WM"},
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e2"
AXIS = "#c9c8c4"
SURFACE = "#ffffff"

METRIC_LABELS = {
    "full_identity_accuracy": "Full-identity accuracy",
    "lbit_accuracy": "Exact $L$-bit recovery",
    "group_accuracy": "Group accuracy",
}


def apply_style() -> None:
    """Recessive chrome: hairline solid grid, no top/right spines, serif text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Solid hairline grid: dashing reads as "threshold" and is reserved for
        # the actual threshold line in panel (b).
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def load_csv(path: str, metric: str) -> dict:
    """Pull {scheme: [value per L]} out of the collector's CSV."""
    table = {scheme: {} for scheme in SERIES}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            scheme = row.get("scheme")
            if scheme not in table:
                continue
            try:
                l_bits = int(row["l_bits"])
            except (KeyError, TypeError, ValueError):
                continue
            raw = row.get(metric, "")
            try:
                table[scheme][l_bits] = float(raw)
            except (TypeError, ValueError):
                table[scheme][l_bits] = np.nan

    missing = [s for s, vals in table.items() if not vals]
    if missing:
        print(f"  Warning: no rows for {', '.join(missing)} in {path}", file=sys.stderr)

    return {
        scheme: [vals.get(L, np.nan) for L in L_VALUES]
        for scheme, vals in table.items()
    }


def fit_inverse_l(l_values, z_values):
    """Least squares z = a/L + c. Returns (a, c, L where z crosses threshold)."""
    x = 1.0 / np.asarray(l_values, dtype=float)
    y = np.asarray(z_values, dtype=float)
    a, c = np.polyfit(x, y, 1)
    l_star = a / (Z_THRESHOLD - c) if Z_THRESHOLD > c else float("inf")
    return a, c, l_star


def panel_accuracy(ax, values: dict, metric: str, title: str | None = None) -> None:
    for scheme, style in SERIES.items():
        y = np.asarray(values[scheme], dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(
            L_VALUES, y,
            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
            linewidth=1.8, markersize=6,
            # Surface ring keeps the two markers readable where both sit at 0.
            markeredgecolor=SURFACE, markeredgewidth=1.0,
            label=style["label"], clip_on=False, zorder=3,
        )

    ax.set_xlabel("Codeword length $L$ (bits)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylim(0, 1.02)
    # Log x for the same reason as panel (b): on a linear axis the 8-16 region,
    # where everything happens, is squeezed into the left fifth while two flat
    # zero lines take the rest. It also keeps both panels on one x treatment.
    ax.set_xscale("log")
    ax.set_xlim(7, 55)
    ax.set_xticks(L_VALUES)
    ax.set_xticklabels([str(v) for v in L_VALUES])
    ax.minorticks_off()
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(loc="upper right", handlelength=2.2, borderaxespad=0.2)
    if title:
        ax.set_title(title, loc="left", color=INK_PRIMARY, pad=6)


def panel_zscore(ax, l_values, z_values, title: str | None = None):
    a, c, l_star = fit_inverse_l(l_values, z_values)

    dense = np.linspace(min(l_values) * 0.85, max(l_values) * 1.15, 300)
    ax.plot(dense, a / dense + c, color=INK_MUTED, linewidth=1.0, zorder=2)

    ax.axhline(Z_THRESHOLD, color=INK_SECONDARY, linestyle="--", linewidth=0.9,
               dashes=(4, 3), zorder=1)

    ax.plot(
        l_values, z_values,
        color=SERIES["hi_dypa"]["color"], marker="o", linestyle="none",
        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.0,
        clip_on=False, zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Codeword length $L$ (bits)")
    ax.set_ylabel("Mean detection $z$-score")
    ax.set_xlim(6.5, 60)
    ax.set_ylim(1.0, 10)
    ax.set_xticks(l_values)
    ax.set_xticklabels([str(v) for v in l_values])
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # A single series needs no legend box; the fit and the threshold are
    # reference marks, so they are named where they sit. The three labels are
    # kept apart in x so none of them lands on the crossing point.
    ax.text(max(l_values) * 1.05, Z_THRESHOLD * 1.10,
            f"detection threshold $z={Z_THRESHOLD:.0f}$",
            fontsize=7.5, color=INK_SECONDARY, ha="right", va="bottom")
    ax.text(26, (a / 26 + c) * 1.22,
            rf"$z \approx {a:.0f}/L$", fontsize=8, color=INK_SECONDARY,
            ha="left", va="bottom")

    if np.isfinite(l_star) and min(l_values) < l_star < max(l_values):
        ax.plot([l_star], [Z_THRESHOLD], marker="v", markersize=5,
                color=INK_SECONDARY, clip_on=False, zorder=4)
        ax.annotate(
            rf"$L^\ast \approx {l_star:.0f}$ bits",
            xy=(l_star, Z_THRESHOLD * 0.93), xytext=(8.4, 1.9),
            fontsize=7.5, color=INK_SECONDARY, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.7,
                            shrinkA=3, shrinkB=2),
        )

    if title:
        ax.set_title(title, loc="left", color=INK_PRIMARY, pad=6)
    return a, c, l_star


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot clean-text detection accuracy vs. codeword length"
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="lbit_table.csv from collect_lbit_detection_rows.py")
    parser.add_argument("--metric", type=str, default="full_identity_accuracy",
                        choices=sorted(METRIC_LABELS), help="Metric for panel (a)")
    parser.add_argument("--out", type=str, default="figures/lbit_detection.pdf",
                        help="Output path (PDF keeps it vector for LaTeX)")
    parser.add_argument("--png", action="store_true",
                        help="Also write a 300 dpi PNG beside the PDF")
    parser.add_argument("--panel", type=str, default="accuracy",
                        choices=["accuracy", "z", "both"],
                        help="Which panel to draw (default: accuracy only)")
    parser.add_argument("--width", type=float, default=None,
                        help="Figure width in inches "
                             "(default: 3.4 single-column, 7.0 for --panel both)")
    parser.add_argument("--height", type=float, default=None,
                        help="Figure height in inches (default: 2.6, or 2.9 for --panel both)")
    args = parser.parse_args()

    two_panel = args.panel == "both"
    width = args.width if args.width is not None else (7.0 if two_panel else 3.4)
    height = args.height if args.height is not None else (2.9 if two_panel else 2.6)

    values = {s: list(ACCURACY[s][args.metric]) for s in SERIES}
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
            return 1
        values = load_csv(args.csv, args.metric)
        print(f"  Loaded {args.metric} from {args.csv}")

    apply_style()
    fit = None

    if two_panel:
        # Panel titles only earn their space when there are panels to tell apart;
        # for a lone chart the caption already names it.
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(width, height))
        panel_accuracy(ax_a, values, args.metric, title="(a) Identification accuracy")
        fit = panel_zscore(ax_b, L_VALUES, MEAN_Z,
                           title="(b) Detection strength and capacity limit")
        fig.tight_layout(pad=0.6, w_pad=2.4)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        if args.panel == "z":
            fit = panel_zscore(ax, L_VALUES, MEAN_Z)
        else:
            panel_accuracy(ax, values, args.metric)
        fig.tight_layout(pad=0.6)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"  Wrote {args.out}")

    if args.png:
        png_path = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"  Wrote {png_path}")

    if fit is not None:
        a, c, l_star = fit
        print(f"  Fit: z = {a:.1f}/L + {c:.2f}   ->   z=4 crossing at L = {l_star:.1f} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
