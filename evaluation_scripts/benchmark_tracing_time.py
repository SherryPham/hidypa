#!/usr/bin/env python3
"""
benchmark_tracing_time.py

Measures the *tracing stage* only: the cost of mapping an already-recovered
codeword to a user identity. Detection (the 2L zero-bit passes over the token
sequence) is identical for every L-bit scheme and dominates end-to-end wall
clock, so it is deliberately excluded -- the tracing stage is the part that
scales with the number of users.

Schemes compared at L = 16:

  naive              flat 16-bit codeword, O(N) scan               (src/watermark.py)
  hi_dypa_2layer     existing (G=8, U=8) two-stage trace           (src/watermark.py)
  hier_2layer_optC   8+8, d=(4,2), hierarchical decode             (src/hierarchy.py)
  hier_3layer_scan   8+4+4, d=(4,2,2), bitwise coarse-to-fine      (src/hierarchy.py)
  hier_3layer_tables 8+4+4, d=(4,2,2), precomputed decode tables   (src/hierarchy.py)
  segment_rs_ml      Segment-WM RS(6,4,4) nearest-codeword + match (src/reedsolomon.py)
  segment_rs_synd    Segment-WM RS(6,4,4) syndrome decode + match  (src/reedsolomon.py)

The two segment rows reproduce SegmentMultiUserWatermarker.trace_from_codeword:
SegmentWatermarker.detect() returns the RS-decoded payload as an L-bit string and
the inherited NaiveMultiUserWatermarker then matches it against every user, so
the segment tracing cost is "RS decode + O(N) naive match".

Usage:
    python evaluation_scripts/benchmark_tracing_time.py \
        --users-file assets/users.csv --num-users 1000 \
        --trials 2000 --repeat 5 --erasure-rates 0.0,0.05,0.10,0.20 \
        --output evaluation/tracing_time/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The recovered-codeword symbols (U+22A5 and '*') are non-ASCII and several
# existing print() calls in src/watermark.py emit them. On a non-UTF-8 stdout
# (Windows cp1252, or a container without a UTF-8 locale) that raises
# UnicodeEncodeError mid-run, so force UTF-8 before anything prints.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


import pandas as pd  # noqa: E402

from src.hierarchy import (  # noqa: E402
    HierarchyIndex,
    HierarchySpec,
    LevelSpec,
    TableDecoder,
    decode_path,
    load_spec,
)
from src.reedsolomon import (  # noqa: E402
    ReedSolomon,
    ReedSolomonCodebook,
    payload_to_symbols,
)

L_BITS = 16
SEGMENT_RS = (6, 4, 4)          # DEFAULT_RS_PARAMS[16] in src/segment_watermark.py


# --------------------------------------------------------------------- channel

def corrupt_bits(codeword: str, erasure_rate: float, flip_rate: float,
                 rng: random.Random) -> str:
    """Independent per-bit erasure / flip, matching the permuted L-bit channel."""
    out = []
    for bit in codeword:
        roll = rng.random()
        if roll < erasure_rate:
            out.append("⊥")
        elif roll < erasure_rate + flip_rate:
            out.append("1" if bit == "0" else "0")
        else:
            out.append(bit)
    return "".join(out)


def corrupt_symbols(symbols: list[int], erasure_rate: float, rng: random.Random,
                    symbol_bits: int) -> list[int]:
    """
    Segment-WM sees whole symbols. A symbol is damaged when any of its bits is
    damaged, so the matched per-symbol rate is 1 - (1 - p)^symbol_bits. A damaged
    symbol becomes a uniformly random wrong symbol, since the segment detector
    takes an argmax and cannot emit an erasure.
    """
    per_symbol = 1.0 - (1.0 - erasure_rate) ** symbol_bits
    space = 1 << symbol_bits
    out = []
    for symbol in symbols:
        if rng.random() < per_symbol:
            wrong = rng.randrange(space - 1)
            out.append(wrong if wrong < symbol else wrong + 1)
        else:
            out.append(symbol)
    return out


# ---------------------------------------------------------------------- tracers

class Tracer:
    """A named tracing strategy: codeword -> user id (or None)."""

    name = "base"
    kind = "bits"                 # "bits" or "symbols"
    setup_seconds = 0.0
    note = ""

    def codeword_for(self, row: int):
        raise NotImplementedError

    def trace(self, payload):
        raise NotImplementedError


class NaiveTracer(Tracer):
    """Flat 16-bit codeword, linear scan -- src/watermark.py NaiveMultiUserWatermarker."""

    name = "naive"
    note = "O(N) scan over binary user IDs"

    def __init__(self, muw):
        self.muw = muw

    def codeword_for(self, row):
        return self.muw.get_codeword_for_user(row)

    def trace(self, payload):
        hits = self.muw.trace_from_codeword(payload)
        return hits[0]["user_id"] if len(hits) == 1 else None


class LegacyHiDyPaTracer(Tracer):
    """Existing two-stage Hi-DyPa trace -- src/watermark.py HiDyPaMultiUserWatermarker."""

    name = "hi_dypa_2layer"
    note = "existing G=8/U=8 two-stage trace"

    def __init__(self, muw):
        self.muw = muw

    def codeword_for(self, row):
        return self.muw.get_codeword_for_user(row)

    def trace(self, payload):
        hits = self.muw.trace_from_codeword(payload)
        return hits[0]["user_id"] if len(hits) == 1 else None


class HierTracer(Tracer):
    """New hierarchical decode, either bitwise scan or precomputed tables."""

    def __init__(self, name, spec: HierarchySpec, num_users: int, use_tables: bool,
                 note: str = ""):
        self.name = name
        self.index = HierarchyIndex(spec, num_users)
        self.use_tables = use_tables
        self.note = note
        self.decoder = None
        if use_tables:
            started = time.perf_counter()
            self.decoder = TableDecoder(self.index)
            self.setup_seconds = time.perf_counter() - started

    def codeword_for(self, row):
        return self.index.codeword_of_row(row)

    def trace(self, payload):
        result = (self.decoder.decode(payload) if self.decoder
                  else decode_path(payload, self.index))
        return result.row


class NaivePureTracer(Tracer):
    """
    Flat 16-bit codeword with an O(N) scan.

    Reproduces NaiveMultiUserWatermarker._match_users_from_codeword exactly
    (best match over non-erased positions, ties rejected) without importing
    torch, so the baseline is available on any machine.
    """

    name = "naive"
    note = "O(N) scan over binary user IDs"

    def __init__(self, num_users: int):
        self.num_users = num_users
        self.user_codewords = [format(u, f"0{L_BITS}b") for u in range(num_users)]

    def codeword_for(self, row):
        return self.user_codewords[row]

    def trace(self, payload):
        valid = [i for i, bit in enumerate(payload) if bit not in ("⊥", "*", "?")]
        if not valid:
            return None
        best_score = -1
        best_user = None
        ties = 0
        for user_id, candidate in enumerate(self.user_codewords):
            score = 0
            for i in valid:
                if payload[i] == candidate[i]:
                    score += 1
            if score > best_score:
                best_score, best_user, ties = score, user_id, 1
            elif score == best_score:
                ties += 1
        return best_user if ties == 1 else None


class LegacyHiDyPaPureTracer(Tracer):
    """
    The existing two-stage Hi-DyPa trace (G=8, U=8) without torch.

    Stage 1 finds the nearest group codeword (even parity, d=2); stage 2 scores
    the full codeword across users of every tied group. For depth 2 that is
    identical to the hierarchical decoder with beam 0, so this row isolates the
    codeword layout -- (8,8) d=(2,1) -- from the decoding machinery.
    """

    name = "hi_dypa_2layer"
    note = "existing G=8/U=8 layout, two-stage trace"

    def __init__(self, num_users: int):
        spec = load_spec("l16_8_8_legacy")
        self.index = HierarchyIndex(spec, num_users)

    def codeword_for(self, row):
        return self.index.codeword_of_row(row)

    def trace(self, payload):
        return decode_path(payload, self.index).row


class SegmentTracer(Tracer):
    """
    Segment-WM tracing: RS decode of the recovered symbols, then the inherited
    naive O(N) match of the resulting payload against every user.
    """

    kind = "symbols"

    def __init__(self, name, num_users: int, exhaustive: bool, note: str = ""):
        n, k, m = SEGMENT_RS
        self.name = name
        self.rs = ReedSolomon(n=n, k=k, m=m)
        self.symbol_bits = m
        self.num_users = num_users
        self.exhaustive = exhaustive
        self.note = note
        self.codebook = None
        if exhaustive:
            started = time.perf_counter()
            self.codebook = ReedSolomonCodebook(self.rs, num_users)
            self.setup_seconds = time.perf_counter() - started
        # the naive match stage compares against every user's binary expansion
        self.user_codewords = [format(u, f"0{L_BITS}b") for u in range(num_users)]

    def codeword_for(self, row):
        return self.rs.encode(payload_to_symbols(row, self.rs.k, self.symbol_bits))

    def _naive_match(self, payload_bits: str):
        best_score = -1
        best_user = None
        ties = 0
        for user_id, candidate in enumerate(self.user_codewords):
            score = sum(a == b for a, b in zip(payload_bits, candidate))
            if score > best_score:
                best_score, best_user, ties = score, user_id, 1
            elif score == best_score:
                ties += 1
        return best_user if ties == 1 else None

    def trace(self, symbols):
        if self.codebook is not None:
            payload, _, ties = self.codebook.decode(symbols)
            if ties != 1:
                return None
        else:
            message, corrected = self.rs.decode_safe(symbols)
            if not corrected:
                return None
            payload = 0
            for i, s in enumerate(message):
                payload |= (int(s) & ((1 << self.symbol_bits) - 1)) << (self.symbol_bits * i)
        if payload >= self.num_users:
            return None
        return self._naive_match(format(payload, f"0{L_BITS}b"))


# ------------------------------------------------------------------------ build

def build_tracers(selected, users_file, num_users, include_asis=False, quiet=True):
    tracers = []

    def wants(name):
        return not selected or name in selected

    if wants("naive"):
        tracers.append(NaivePureTracer(num_users))
    if wants("hi_dypa_2layer"):
        tracers.append(LegacyHiDyPaPureTracer(num_users))
    if wants("hier_2layer_optC"):
        tracers.append(HierTracer("hier_2layer_optC", load_spec("l16_8_8_optionC"),
                                  num_users, use_tables=False,
                                  note="8+8 d=(4,2), bitwise"))
    if wants("hier_3layer_scan"):
        tracers.append(HierTracer("hier_3layer_scan", load_spec("l16_8_4_4"),
                                  num_users, use_tables=False,
                                  note="8+4+4 d=(4,2,2), bitwise"))
    if wants("hier_3layer_tables"):
        tracers.append(HierTracer("hier_3layer_tables", load_spec("l16_8_4_4"),
                                  num_users, use_tables=True,
                                  note="8+4+4 d=(4,2,2), decode tables"))
    if wants("segment_rs_ml"):
        tracers.append(SegmentTracer("segment_rs_ml", num_users, exhaustive=True,
                                     note="RS(6,4,4) nearest-codeword + naive match"))
    if wants("segment_rs_synd"):
        tracers.append(SegmentTracer("segment_rs_synd", num_users, exhaustive=False,
                                     note="RS(6,4,4) syndrome decode + naive match"))

    if include_asis:
        tracers.extend(_build_asis_tracers(users_file, num_users, quiet))
    return tracers


def _build_asis_tracers(users_file, num_users, quiet):
    """
    Time the shipped classes in src/watermark.py as they stand today.

    These import torch/transformers, so they are opt-in (--include-asis). They
    measure the current implementation cost, including the O(N) pandas lookup
    that get_codeword_for_user performs inside the tracing loop.
    """
    try:
        from src.watermark import (  # noqa: E402
            HiDyPaMultiUserWatermarker,
            NaiveMultiUserWatermarker,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! --include-asis skipped ({type(exc).__name__}: {exc})")
        return []

    class _StubLBit:
        """Only .L is consulted by the tracing path; no model is needed."""

        def __init__(self, L):
            self.L = L

        def keygen(self, key_length: int = 32) -> bytes:
            return bytes(key_length)

    out = []
    df = pd.read_csv(users_file).head(num_users)
    with _muted(quiet):
        muw = NaiveMultiUserWatermarker(_StubLBit(L_BITS))
        muw._initialize_metadata(df)
        naive = NaiveTracer(muw)
        naive.name = "naive_asis"
        naive.note = "src/watermark.py NaiveMultiUserWatermarker as shipped"
        out.append(naive)

        legacy = HiDyPaMultiUserWatermarker(
            _StubLBit(L_BITS), group_bits=8, user_bits=8, min_distance=2
        )
        legacy.load_users(users_file)
        legacy_tracer = LegacyHiDyPaTracer(legacy)
        legacy_tracer.name = "hi_dypa_2layer_asis"
        legacy_tracer.note = "src/watermark.py HiDyPaMultiUserWatermarker as shipped"
        out.append(legacy_tracer)
    return out


class _muted:
    """Silence the chatty load_users() prints during setup."""

    def __init__(self, active=True):
        self.active = active

    def __enter__(self):
        if self.active:
            self._stdout = sys.stdout
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self.active:
            sys.stdout.close()
            sys.stdout = self._stdout
        return False


# -------------------------------------------------------------------- measuring

def measure(tracer: Tracer, num_users: int, trials: int, repeat: int,
            erasure_rate: float, flip_rate: float, seed: int) -> dict:
    rng = random.Random(seed)
    rows = [rng.randrange(num_users) for _ in range(trials)]
    payloads = []
    for row in rows:
        clean = tracer.codeword_for(row)
        if tracer.kind == "symbols":
            payloads.append(corrupt_symbols(clean, erasure_rate, rng, tracer.symbol_bits))
        else:
            payloads.append(corrupt_bits(clean, erasure_rate, flip_rate, rng))

    # warm-up (fills caches, triggers any lazy work)
    for payload in payloads[: min(64, trials)]:
        tracer.trace(payload)

    timings = []
    correct = 0
    for run in range(repeat):
        started = time.perf_counter()
        hits = 0
        for payload in payloads:
            hits += 1 if tracer.trace(payload) is not None else 0
        elapsed = time.perf_counter() - started
        timings.append(elapsed / trials * 1e6)      # microseconds per trace
        if run == 0:
            correct = sum(
                1 for row, payload in zip(rows, payloads)
                if tracer.trace(payload) == row
            )

    return {
        "scheme": tracer.name,
        "note": tracer.note,
        "us_per_trace_median": statistics.median(timings),
        "us_per_trace_min": min(timings),
        "us_per_trace_stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "traces_per_second": 1e6 / statistics.median(timings),
        "exact_identification_rate": correct / trials,
        "setup_seconds": tracer.setup_seconds,
        "trials": trials,
        "repeat": repeat,
    }


# ------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Tracing-stage time benchmark: Hi-DyPa vs Segment-WM at L=16",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--users-file", default="assets/users.csv")
    parser.add_argument("--num-users", type=int, default=1000,
                        help="Users to load (capped by the hierarchy capacity, 1024).")
    parser.add_argument("--n-sweep", type=str, default=None,
                        help="Comma-separated user counts to sweep, e.g. 100,250,500,1000")
    parser.add_argument("--trials", type=int, default=2000,
                        help="Traces per timing run.")
    parser.add_argument("--repeat", type=int, default=5,
                        help="Timing runs; the median is reported.")
    parser.add_argument("--erasure-rates", type=str, default="0.0,0.05,0.10,0.20")
    parser.add_argument("--flip-rate", type=float, default=0.0)
    parser.add_argument("--schemes", type=str, default=None,
                        help="Comma-separated subset; default is all.")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=str, default="evaluation/tracing_time/results.json")
    parser.add_argument("--include-asis", action="store_true",
                        help="Also time the shipped watermark.py classes (needs torch).")
    parser.add_argument("--run-tag", type=str, default=None)
    args = parser.parse_args()

    selected = set(args.schemes.split(",")) if args.schemes else None
    rates = [float(v) for v in args.erasure_rates.split(",")]
    user_counts = ([int(v) for v in args.n_sweep.split(",")] if args.n_sweep
                   else [args.num_users])

    print("=" * 96)
    print("TRACING-STAGE BENCHMARK  (L = 16)")
    print("=" * 96)
    print(f"  python        : {platform.python_version()} on {platform.platform()}")
    print(f"  users file    : {args.users_file}")
    print(f"  user counts   : {user_counts}")
    print(f"  erasure rates : {rates}   flip rate: {args.flip_rate}")
    print(f"  trials/run    : {args.trials}   timing runs: {args.repeat}")
    print("  NOTE: this measures codeword -> identity only. Detection (2L zero-bit")
    print("        passes) is identical across L-bit schemes and dominates end-to-end.")
    print()

    records = []
    for num_users in user_counts:
        print(f"\n{'#' * 96}\n# N = {num_users} users\n{'#' * 96}")
        tracers = build_tracers(selected, args.users_file, num_users,
                                include_asis=args.include_asis)
        for rate in rates:
            print(f"\n--- erasure rate {rate:.2f} "
                  f"(flip {args.flip_rate:.2f}) ---")
            print(f"  {'scheme':<20} {'us/trace':>10} {'traces/s':>12} "
                  f"{'exact id':>9}  {'setup(s)':>9}  note")
            print("  " + "-" * 92)
            for tracer in tracers:
                result = measure(tracer, num_users, args.trials, args.repeat,
                                 rate, args.flip_rate, args.seed)
                result["num_users"] = num_users
                result["erasure_rate"] = rate
                result["flip_rate"] = args.flip_rate
                records.append(result)
                print(f"  {result['scheme']:<20} {result['us_per_trace_median']:>10.2f} "
                      f"{result['traces_per_second']:>12,.0f} "
                      f"{result['exact_identification_rate']:>8.1%}  "
                      f"{result['setup_seconds']:>9.3f}  {result['note']}")

    payload = {
        "config": {
            "L": L_BITS,
            "segment_rs": {"n": SEGMENT_RS[0], "k": SEGMENT_RS[1], "m": SEGMENT_RS[2]},
            "hierarchies": {
                name: load_spec(name).to_dict()
                for name in ("l16_8_4_4", "l16_8_8_optionC")
            },
            "trials": args.trials,
            "repeat": args.repeat,
            "erasure_rates": rates,
            "flip_rate": args.flip_rate,
            "user_counts": user_counts,
            "seed": args.seed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "run_tag": args.run_tag,
        },
        "results": records,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {len(records)} records to {args.output}")

    # headline speedups at the largest N, per erasure rate
    print("\n" + "=" * 96)
    print("SPEEDUP of hier_3layer_tables over each baseline")
    print("=" * 96)
    top_n = user_counts[-1]
    for rate in rates:
        rows = {r["scheme"]: r for r in records
                if r["num_users"] == top_n and r["erasure_rate"] == rate}
        ours = rows.get("hier_3layer_tables")
        if not ours:
            continue
        parts = []
        for name, row in rows.items():
            if name == "hier_3layer_tables":
                continue
            parts.append(f"{name} x{row['us_per_trace_median'] / ours['us_per_trace_median']:.1f}")
        print(f"  p={rate:.2f}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
