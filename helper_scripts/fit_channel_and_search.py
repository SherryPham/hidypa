#!/usr/bin/env python3
"""
fit_channel_and_search.py

Pick a hierarchy configuration using the MEASURED channel instead of an i.i.d.
model.

Why this exists
---------------
An independent per-bit erasure model at the mean rate badly under-predicts real
accuracy. On the GPT-2 A100 run the mean erasure rate at L=12 was 18.1%, and an
i.i.d. model at 18.1% predicts ~35% exact identification -- but the measured
value was 78%. The reason is over-dispersion: erasure count is driven by how many
high-entropy blocks a particular generation happens to contain, so many texts
come back nearly clean while a minority come back mostly erased. Averaging that
into one rate destroys the structure that decoding actually exploits.

This script therefore resamples the EMPIRICAL distribution of per-trial erasure
counts taken from the run's records file, rather than assuming a binomial.

Usage
-----
    # summarise the measured channel
    python helper_scripts/fit_channel_and_search.py \
        evaluation/hierarchical_gpt2/full_gpt2_job_16312504_records.jsonl.gz --describe

    # search configurations against that channel
    python helper_scripts/fit_channel_and_search.py \
        evaluation/hierarchical_gpt2/full_gpt2_job_16312504_records.jsonl.gz \
        --num-users 1000 --trials 4000
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from src.hierarchy import (  # noqa: E402
    HierarchyIndex,
    HierarchySpec,
    LevelSpec,
    codebook_capacity,
    decode_path,
)

ERASE = "⊥"


# ------------------------------------------------------------------- channel

def load_channel(paths: list[str]) -> dict:
    """Empirical per-trial erasure/collision counts, keyed by payload width."""
    by_L: dict[int, dict] = {}
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cw = rec.get("recovered_codeword")
                if not cw:
                    continue
                L = len(cw)
                slot = by_L.setdefault(L, {"erasures": [], "collisions": [],
                                           "flips": [], "trials": 0,
                                           "schemes": collections.Counter()})
                slot["erasures"].append(rec.get("num_erasures", cw.count(ERASE)))
                slot["collisions"].append(rec.get("num_collisions", cw.count("*")))
                if rec.get("bit_errors_among_decided") is not None:
                    slot["flips"].append(rec["bit_errors_among_decided"])
                slot["trials"] += 1
                slot["schemes"][rec.get("scheme", "?")] += 1
    return by_L


def describe(by_L: dict):
    for L in sorted(by_L):
        slot = by_L[L]
        counts = collections.Counter(slot["erasures"])
        total = slot["trials"]
        mean = sum(slot["erasures"]) / total
        print(f"\nL = {L}   trials = {total}   mean erasures = {mean:.2f} "
              f"({mean / L:.1%} per bit)")
        print(f"  schemes: {dict(slot['schemes'])}")
        flips = slot["flips"]
        if flips:
            print(f"  bit errors among decided positions: mean {sum(flips) / len(flips):.3f}")
        print("  erasure-count distribution:")
        for k in sorted(counts):
            bar = "#" * max(1, round(60 * counts[k] / total))
            print(f"    {k:>3} erased  {counts[k]:>5}  {counts[k] / total:>6.1%}  {bar}")
        clean = counts[0] / total
        print(f"  -> {clean:.1%} of trials came back with zero erasures. "
              f"An i.i.d. model at {mean / L:.1%} would predict "
              f"{(1 - mean / L) ** L:.1%}.")


# -------------------------------------------------------------------- search

def evaluate(levels, erasure_pool, num_users, trials, seed=17):
    try:
        spec = HierarchySpec.from_levels(levels)
    except ValueError:
        return None
    if spec.capacity() < num_users:
        return None
    index = HierarchyIndex(spec, num_users)
    rng = random.Random(seed)
    L = spec.L
    exact = contained = 0
    csize = 0
    for _ in range(trials):
        row = rng.randrange(num_users)
        word = list(index.codeword_of_row(row))
        k = min(rng.choice(erasure_pool), L)
        for pos in rng.sample(range(L), k):
            word[pos] = ERASE
        result = decode_path("".join(word), index)
        if result.row == row:
            exact += 1
        true_path = index.path_of_row(row)
        cp = result.containment_path
        if cp == tuple(true_path[: len(cp)]):
            contained += 1
            csize += len(index.rows_under(cp))
    return {
        "spec": spec,
        "exact": exact / trials,
        "contained": contained / trials,
        "csize": csize / max(contained, 1),
        "capacity": spec.capacity(),
    }


def candidate_specs(L: int):
    """1-, 2- and 3-level layouts using constructible codebooks."""
    out = []
    dists = (1, 2, 4)
    out.append([LevelSpec("flat", L, 1 << L, 1)])
    for b1 in range(2, L - 1):
        for d1 in dists:
            if d1 > b1:
                continue
            f1 = codebook_capacity(b1, d1)
            b2 = L - b1
            for d2 in dists:
                if d2 <= b2:
                    out.append([LevelSpec("l1", b1, f1, d1),
                                LevelSpec("l2", b2, codebook_capacity(b2, d2), d2)])
            for b2 in range(2, L - b1):
                b3 = L - b1 - b2
                if b3 < 2:
                    continue
                for d2 in dists:
                    for d3 in dists:
                        if d2 > b2 or d3 > b3:
                            continue
                        out.append([LevelSpec("l1", b1, f1, d1),
                                    LevelSpec("l2", b2, codebook_capacity(b2, d2), d2),
                                    LevelSpec("l3", b3, codebook_capacity(b3, d3), d3)])
    return out


def describe_spec(spec):
    return " + ".join(f"{lv.bits}b/d{lv.min_distance}/x{lv.fanout}" for lv in spec.levels)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("records", nargs="+",
                        help="*_records.jsonl.gz from evaluate_hierarchical_gpt2.py")
    parser.add_argument("--describe", action="store_true",
                        help="only print the measured channel, do not search")
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    paths = []
    for pattern in args.records:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("No records files found.")
        return 1

    by_L = load_channel(paths)
    if not by_L:
        print("No usable records.")
        return 1

    print("=" * 92)
    print("MEASURED CHANNEL")
    print("=" * 92)
    describe(by_L)
    if args.describe:
        return 0

    for L in sorted(by_L):
        pool = by_L[L]["erasures"]
        print(f"\n{'=' * 92}")
        print(f"CONFIGURATION SEARCH at L = {L}  "
              f"(resampling {len(pool)} measured erasure counts, "
              f"{args.num_users} users, {args.trials} trials)")
        print("=" * 92)
        results = [r for r in (evaluate(c, pool, args.num_users, args.trials, args.seed)
                               for c in candidate_specs(L)) if r]
        if not results:
            print("  no configuration reaches the required capacity")
            continue
        results.sort(key=lambda r: (-r["exact"], r["csize"]))
        print(f"  {'configuration':<44} {'cap':>6} {'exact':>7} {'contain':>8} {'size':>7}")
        print("  " + "-" * 76)
        for r in results[: args.top]:
            print(f"  {describe_spec(r['spec']):<44} {r['capacity']:>6} "
                  f"{r['exact']:>6.1%} {r['contained']:>7.1%} {r['csize']:>7.1f}")
        flat = next((r for r in results if r["spec"].depth == 1), None)
        best = results[0]
        if flat and flat is not best:
            print(f"\n  best hierarchy beats flat by "
                  f"{best['exact'] - flat['exact']:+.1%} "
                  f"({best['exact']:.1%} vs {flat['exact']:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
