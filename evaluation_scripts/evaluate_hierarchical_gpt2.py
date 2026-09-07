#!/usr/bin/env python3
"""
evaluate_hierarchical_gpt2.py

End-to-end GPU experiment for the 3-layer Hi-DyPa hierarchy at L = 16.

For every (prompt, user) trial it embeds the scheme's codeword, generates text,
detects the L-bit codeword, and traces it back to an identity -- timing each
stage separately. Unlike benchmark_tracing_time.py (synthetic channel, CPU only)
this measures the REAL channel: how often the detector emits an erasure, what the
per-bit z-scores actually are, and what tracing accuracy follows.

Two modes:

  --mode calibrate   sweep L and max_new_tokens on a few prompts to find whether
                     the payload is carryable at all. Run this FIRST.
  --mode full        all schemes at a fixed operating point, many prompts.

Why calibration matters on GPT-2
--------------------------------
Per-bit z scales as sqrt(blocks) / L, because LBitLogitProcessor spreads a
permutation of the L bit positions across high-entropy blocks: bit i is biased at
only ~blocks/L positions while detect() normalises over all blocks. Doubling L
therefore needs ~4x the blocks to hold z constant.

GPT-2 has a hard 1024-token context (n_positions), so blocks are capped at
roughly 900 no matter what --max-new-tokens says. That ceiling, not the
hierarchy, is what decides whether L=16 is usable here.

Usage
-----
    # Step 1 -- is L=16 carryable on GPT-2 at all?
    python evaluation_scripts/evaluate_hierarchical_gpt2.py \
        --mode calibrate --model gpt2 \
        --l-bits-sweep 8,12,16 --token-sweep 256,512,900 --num-prompts 20 \
        --output-dir evaluation/hierarchical_gpt2

    # Step 2 -- full comparison at the operating point calibration chose
    python evaluation_scripts/evaluate_hierarchical_gpt2.py \
        --mode full --model gpt2 --l-bits 16 --max-new-tokens 900 \
        --num-prompts 100 --output-dir evaluation/hierarchical_gpt2
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
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


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.hierarchical_watermark import HierarchicalMultiUserWatermarker  # noqa: E402
from src.hierarchy import ERASURE_SYMBOLS, HierarchySpec, load_spec  # noqa: E402
from src.utils import get_model, parse_final_output  # noqa: E402
from src.watermark import (  # noqa: E402
    HiDyPaMultiUserWatermarker,
    LBitWatermarker,
    NaiveMultiUserWatermarker,
    ZeroBitWatermarker,
    derive_key,
)

GPT2_CONTEXT = 1024


# --------------------------------------------------------------------- detection

def detect_with_stats(lbw: LBitWatermarker, master_key: bytes, text: str) -> dict:
    """
    Mirror LBitWatermarker.detect but also return the per-bit z-scores and the
    block count, at the same cost (one forward pass, 2L scored scans).
    """
    tokenizer = lbw.model.tokenizer
    token_ids = tokenizer.encode(text, return_tensors="pt").to(lbw.model.device)[0]
    if len(token_ids) < 2:
        return {
            "codeword": "⊥" * lbw.L, "z0": [], "z1": [], "blocks": 0,
            "num_tokens": int(len(token_ids)),
        }

    with torch.no_grad():
        outputs = lbw.model._model(token_ids.unsqueeze(0))
    all_logits = outputs.logits.squeeze(0)

    zt = lbw.zero_bit.z_threshold
    recovered = []
    z0_all, z1_all = [], []
    blocks = 0
    for i in range(1, lbw.L + 1):
        z0, _, b0 = lbw.zero_bit.detect(derive_key(master_key, i, 0), text,
                                        cached_logits=all_logits)
        z1, _, b1 = lbw.zero_bit.detect(derive_key(master_key, i, 1), text,
                                        cached_logits=all_logits)
        blocks = max(blocks, b0, b1)
        z0_all.append(z0)
        z1_all.append(z1)
        if z0 <= zt and z1 <= zt:
            recovered.append("⊥")
        elif z0 > zt and z1 <= zt:
            recovered.append("0")
        elif z1 > zt and z0 <= zt:
            recovered.append("1")
        else:
            recovered.append("*")

    return {
        "codeword": "".join(recovered),
        "z0": z0_all,
        "z1": z1_all,
        "blocks": blocks,
        "num_tokens": int(len(token_ids)),
    }


def codeword_stats(recovered: str, truth: str | None) -> dict:
    erasures = sum(1 for c in recovered if c == "⊥")
    collisions = sum(1 for c in recovered if c == "*")
    decided = [i for i, c in enumerate(recovered) if c in ("0", "1")]
    bit_errors = None
    if truth is not None and decided:
        bit_errors = sum(1 for i in decided if recovered[i] != truth[i])
    return {
        "num_erasures": erasures,
        "num_collisions": collisions,
        "num_decided": len(decided),
        "erasure_rate": erasures / len(recovered) if recovered else 0.0,
        "collision_rate": collisions / len(recovered) if recovered else 0.0,
        "bit_errors_among_decided": bit_errors,
        "exact_codeword": (truth is not None and recovered == truth),
    }


# ------------------------------------------------------------------------ schemes

def make_capped_users_file(users_file: str, num_users: int, out_dir: str) -> str:
    """Write the first num_users rows to a temp CSV so every scheme sees the same set."""
    df = pd.read_csv(users_file).head(num_users)
    path = os.path.join(out_dir, f"_users_capped_{len(df)}.csv")
    df.to_csv(path, index=False)
    return path


def build_scheme(name: str, lbw: LBitWatermarker, users_file: str, num_users: int):
    """Return (multi-user watermarker, description dict)."""
    if name == "naive":
        muw = NaiveMultiUserWatermarker(lbit_watermarker=lbw)
        muw.load_users(users_file)
        info = {"scheme": "naive", "layout": f"flat {lbw.L}-bit"}
    elif name == "hi_dypa_2layer":
        half = lbw.L // 2
        muw = HiDyPaMultiUserWatermarker(
            lbit_watermarker=lbw, group_bits=half, user_bits=lbw.L - half,
            min_distance=2,
        )
        muw.load_users(users_file)
        info = {"scheme": "hi_dypa_2layer", "layout": f"G={half}/U={lbw.L - half} d=(2,1)"}
    elif name.startswith("hier:"):
        spec = load_spec(name.split(":", 1)[1])
        muw = HierarchicalMultiUserWatermarker(lbw, spec, use_tables=True)
        muw.load_users(users_file)
        info = {"scheme": name, "layout": spec.to_dict()}
    else:
        raise ValueError(f"Unknown scheme {name!r}")

    return muw, info


def hierarchy_accuracy(muw, recovered: str, true_user_id: int) -> dict:
    """Per-level accuracy and containment for a hierarchical scheme."""
    true_path = muw.path_for_user(true_user_id)
    started = time.perf_counter()
    result = muw.decode(recovered)
    trace_seconds = time.perf_counter() - started

    out = {
        "trace_seconds": trace_seconds,
        "true_path": list(true_path),
        "detected_path": list(result.path) if result.path else None,
        "num_ties": len(result.ties),
        "containment_path": list(result.containment_path),
        "containment_level": result.containment_level,
        "containment_size": len(muw.index.rows_under(result.containment_path)),
        "cumulative_distance": result.cumulative_distance,
        "candidates_evaluated": result.candidates_evaluated,
        "exact_identity": result.path == true_path,
    }
    # prefix_correct[k] = levels 1..k+1 all correct among the tied candidates'
    # common prefix (a partial answer still counts if it is right as far as it goes)
    prefix_flags = []
    shared = result.containment_path
    for level in range(len(true_path)):
        prefix_flags.append(level < len(shared) and shared[level] == true_path[level]
                            and tuple(shared[: level + 1]) == tuple(true_path[: level + 1]))
    out["prefix_correct"] = prefix_flags
    return out


def flat_accuracy(muw, recovered: str, true_user_id: int) -> dict:
    started = time.perf_counter()
    hits = muw.trace_from_codeword(recovered)
    trace_seconds = time.perf_counter() - started
    detected = hits[0]["user_id"] if len(hits) == 1 else None
    return {
        "trace_seconds": trace_seconds,
        "detected_user_id": detected,
        "num_ties": len(hits),
        "exact_identity": detected == true_user_id,
    }


# ----------------------------------------------------------------------- running

def run_trials(muw, info, lbw, master_key, prompts, users, model_name,
               max_new_tokens, is_hier) -> list[dict]:
    records = []
    for idx, (prompt, user_id) in enumerate(zip(prompts, users)):
        try:
            truth = muw.get_codeword_for_user(user_id)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! user {user_id}: {exc}")
            continue

        t0 = time.perf_counter()
        raw = muw.embed(master_key, user_id, prompt, max_new_tokens=max_new_tokens)
        text = parse_final_output(raw, model_name)
        gen_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        det = detect_with_stats(lbw, master_key, text)
        det_seconds = time.perf_counter() - t1

        record = {
            "trial": idx,
            "scheme": info["scheme"],
            "true_user_id": user_id,
            "ground_truth_codeword": truth,
            "recovered_codeword": det["codeword"],
            "blocks": det["blocks"],
            "blocks_per_bit": det["blocks"] / lbw.L if lbw.L else 0.0,
            "num_tokens": det["num_tokens"],
            "mean_z_max": float(np.mean([max(a, b) for a, b in zip(det["z0"], det["z1"])]))
            if det["z0"] else 0.0,
            "min_z_max": float(min((max(a, b) for a, b in zip(det["z0"], det["z1"])),
                                   default=0.0)),
            "gen_seconds": gen_seconds,
            "detect_seconds": det_seconds,
        }
        record.update(codeword_stats(det["codeword"], truth))
        if is_hier:
            record.update(hierarchy_accuracy(muw, det["codeword"], user_id))
        else:
            record.update(flat_accuracy(muw, det["codeword"], user_id))
        records.append(record)

        flag = "OK " if record["exact_identity"] else "-- "
        print(f"    [{idx + 1}/{len(prompts)}] {flag}user={user_id} "
              f"blocks={det['blocks']} b/bit={record['blocks_per_bit']:.0f} "
              f"erase={record['num_erasures']}/{lbw.L} "
              f"z̄={record['mean_z_max']:.2f} "
              f"gen={gen_seconds:.1f}s det={det_seconds:.1f}s "
              f"trace={record['trace_seconds'] * 1e6:.1f}us")
    return records


def summarise(records: list[dict], L: int) -> dict:
    if not records:
        return {}

    def mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(statistics.mean(vals)) if vals else None

    summary = {
        "trials": len(records),
        "L": L,
        "blocks_mean": mean("blocks"),
        "blocks_per_bit_mean": mean("blocks_per_bit"),
        "num_tokens_mean": mean("num_tokens"),
        "mean_z_max": mean("mean_z_max"),
        "min_z_max_mean": mean("min_z_max"),
        "erasure_rate": mean("erasure_rate"),
        "collision_rate": mean("collision_rate"),
        "exact_codeword_rate": sum(1 for r in records if r["exact_codeword"]) / len(records),
        "exact_identity_rate": sum(1 for r in records if r["exact_identity"]) / len(records),
        "gen_seconds_mean": mean("gen_seconds"),
        "detect_seconds_mean": mean("detect_seconds"),
        "trace_us_median": float(statistics.median(
            [r["trace_seconds"] * 1e6 for r in records])),
    }
    if "containment_level" in records[0]:
        depth = len(records[0]["true_path"])
        summary["containment_size_mean"] = mean("containment_size")
        summary["candidates_evaluated_mean"] = mean("candidates_evaluated")
        summary["prefix_accuracy"] = [
            sum(1 for r in records if r["prefix_correct"][k]) / len(records)
            for k in range(depth)
        ]
    return summary


# -------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GPU evaluation of the 3-layer Hi-DyPa hierarchy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=["calibrate", "full"], default="full")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompts-file", default="assets/prompts.txt")
    parser.add_argument("--users-file", default="assets/users.csv")
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--l-bits", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--l-bits-sweep", default="8,12,16",
                        help="calibrate mode: payload widths to try")
    parser.add_argument("--token-sweep", default="256,512,900",
                        help="calibrate mode: max_new_tokens to try")
    parser.add_argument("--schemes", default="hier:l16_8_4_4,hier:l16_8_8_optionC,"
                                             "hi_dypa_2layer,naive",
                        help="full mode: comma-separated schemes")
    parser.add_argument("--delta", type=float, default=3.5)
    parser.add_argument("--entropy-threshold", type=float, default=2.5)
    parser.add_argument("--hashing-context", type=int, default=5)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-dir", default="evaluation/hierarchical_gpt2")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.prompts_file, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    prompts = prompts[: args.num_prompts]

    print("=" * 88)
    print(f"HIERARCHICAL Hi-DyPa on {args.model}  (mode={args.mode}, tag={run_tag})")
    print("=" * 88)
    print(f"  prompts {len(prompts)}   users {args.num_users}   seed {args.seed}")
    print(f"  delta {args.delta}  entropy {args.entropy_threshold}  "
          f"hc {args.hashing_context}  z {args.z_threshold}")

    users_file = make_capped_users_file(args.users_file, args.num_users,
                                        args.output_dir)
    print(f"  users capped to {args.num_users} rows -> {users_file}")

    model = get_model(args.model)
    if args.model == "gpt2":
        print(f"  NOTE: GPT-2 context is {GPT2_CONTEXT} tokens; max_new_tokens is "
              f"clamped accordingly.")
    print(f"  device: {model.device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu   : {torch.cuda.get_device_name(0)} "
              f"({props.total_memory / 1e9:.0f} GB, capability {props.major}.{props.minor})")
        print(f"  torch : {torch.__version__}, cuda {torch.version.cuda}")

    def make_lbw(L):
        zbw = ZeroBitWatermarker(
            model=model, delta=args.delta,
            entropy_threshold=args.entropy_threshold,
            z_threshold=args.z_threshold,
            hashing_context=args.hashing_context,
        )
        return LBitWatermarker(zero_bit_watermarker=zbw, L=L)

    all_summaries = []
    all_records = []

    if args.mode == "calibrate":
        # One hierarchical config per payload width, so the sweep isolates L.
        by_L = {8: "4:8:2,4:8:2", 12: "l12_6_3_3", 16: "l16_8_4_4"}
        for L in [int(v) for v in args.l_bits_sweep.split(",")]:
            if L not in by_L:
                print(f"  ! no hierarchy defined for L={L}, skipping")
                continue
            for tokens in [int(v) for v in args.token_sweep.split(",")]:
                budget = min(tokens, GPT2_CONTEXT - 64) if args.model == "gpt2" else tokens
                print(f"\n--- calibrate L={L} max_new_tokens={budget} ---")
                lbw = make_lbw(L)
                muw, info = build_scheme(f"hier:{by_L[L]}", lbw, users_file,
                                         args.num_users)
                master_key = muw.keygen()
                users = [random.randrange(muw.N) for _ in prompts]
                records = run_trials(muw, info, lbw, master_key, prompts, users,
                                     args.model, budget, is_hier=True)
                summary = summarise(records, L)
                summary.update({"mode": "calibrate", "L": L, "max_new_tokens": budget,
                                "scheme": info["scheme"], "model": args.model})
                all_summaries.append(summary)
                all_records.extend(records)
                print(f"  => blocks/bit {summary['blocks_per_bit_mean']:.1f}  "
                      f"erasure {summary['erasure_rate']:.1%}  "
                      f"exact codeword {summary['exact_codeword_rate']:.1%}  "
                      f"exact identity {summary['exact_identity_rate']:.1%}")
    else:
        budget = (min(args.max_new_tokens, GPT2_CONTEXT - 64)
                  if args.model == "gpt2" else args.max_new_tokens)
        users_master = None
        for scheme in args.schemes.split(","):
            scheme = scheme.strip()
            print(f"\n--- {scheme}  (L={args.l_bits}, tokens={budget}) ---")
            lbw = make_lbw(args.l_bits)
            try:
                muw, info = build_scheme(scheme, lbw, users_file, args.num_users)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! could not build {scheme}: {exc}")
                continue
            master_key = muw.keygen()
            # identical user draw across schemes for a paired comparison
            if users_master is None:
                users_master = [random.randrange(muw.N) for _ in prompts]
            users = [u % muw.N for u in users_master]
            is_hier = scheme.startswith("hier:")
            records = run_trials(muw, info, lbw, master_key, prompts, users,
                                 args.model, budget, is_hier)
            summary = summarise(records, args.l_bits)
            summary.update({"mode": "full", "scheme": scheme, "layout": info["layout"],
                            "max_new_tokens": budget, "model": args.model})
            all_summaries.append(summary)
            all_records.extend(records)
            print(f"  => erasure {summary['erasure_rate']:.1%}  "
                  f"exact identity {summary['exact_identity_rate']:.1%}  "
                  f"trace {summary['trace_us_median']:.1f}us")

    # ------------------------------------------------------------------ output
    base = os.path.join(args.output_dir, f"{args.mode}_{args.model}_{run_tag}")
    with gzip.open(f"{base}_records.jsonl.gz", "wt", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")
    with open(f"{base}_summary.json", "w", encoding="utf-8") as handle:
        json.dump({
            "config": vars(args) | {"run_tag": run_tag, "gpt2_context": GPT2_CONTEXT},
            "summaries": all_summaries,
        }, handle, indent=2)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    header = f"  {'scheme/L':<26} {'blk/bit':>8} {'erase':>7} {'collide':>8} " \
             f"{'exact cw':>9} {'exact id':>9} {'trace us':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in all_summaries:
        label = (f"L={s['L']} tok={s['max_new_tokens']}" if args.mode == "calibrate"
                 else s["scheme"])
        print(f"  {label:<26} {s['blocks_per_bit_mean']:>8.1f} "
              f"{s['erasure_rate']:>6.1%} {s['collision_rate']:>7.1%} "
              f"{s['exact_codeword_rate']:>8.1%} {s['exact_identity_rate']:>8.1%} "
              f"{s['trace_us_median']:>9.1f}")
        if "prefix_accuracy" in s:
            print(f"  {'':<26} prefix accuracy per level: "
                  + "  ".join(f"L{i + 1}={v:.1%}" for i, v in enumerate(s["prefix_accuracy"]))
                  + f"   containment {s['containment_size_mean']:.1f} users")

    if args.mode == "calibrate":
        usable = [s for s in all_summaries if s["erasure_rate"] < 0.05]
        print("\n  Decision rule: smallest (L, tokens) with erasure rate < 5%.")
        if usable:
            best = max(usable, key=lambda s: s["L"])
            print(f"  -> L={best['L']} at max_new_tokens={best['max_new_tokens']} "
                  f"(erasure {best['erasure_rate']:.1%})")
        else:
            print("  -> No setting reached <5% erasure. Use the largest L whose "
                  "exact-identity rate is acceptable, or drop to L=12.")
    print(f"\nWrote {base}_summary.json and {base}_records.jsonl.gz")


if __name__ == "__main__":
    main()
