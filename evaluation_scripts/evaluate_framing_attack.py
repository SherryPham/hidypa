"""
evaluate_framing_attack.py

Evaluates framing attack resistance using Codeword Recovery Rate (CRR).

Threat model (Sec 2.2.3, Appendix C.2):
  - MAU baseline: adversary is Codeword-Aware -- the static codeword database is
    breached, so the victim codeword is known. The adversary picks the element of
    the feasible set F(T) that minimises Hamming distance to the victim.
  - Hi-DyPa: adversary is Codeword-Blind -- codewords are generated on-the-fly
    from a secret group seed Kg[g*] that the adversary does not possess (Theorem 6).
    The adversary can only pick uniformly at random from F(T).

Metric - CRR(k):
  k = number of colluding trials (how many times the adversary pair generates fresh
  watermarked texts and pools evidence via majority voting).
  More trials -> fewer undetermined bits -> smaller F(T) -> higher CRR for MAU.
  Hi-DyPa: CRR stays ~50% regardless of k because Kg[g*] is secret (Theorem 6).

  For each k in --k-values:
    - Run --n-trials independent experiments (fresh c1, c2, victim per experiment).
    - In each experiment the adversary makes max(k_values) colluding attempts;
      CRR at each k is evaluated by majority-voting only the first k detections.
    - CRR(k) = mean recovery rate (%) over n_trials experiments.

Usage:
  python evaluate_framing_attack.py --scheme naive --k-values 1 5 10 50 100 --n-trials 20
  python evaluate_framing_attack.py --scheme hi_dypa --group-bits 4 --user-bits 4 --k-values 1 5 10 50 100 --n-trials 20
"""

import argparse
import gzip
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime

from tqdm import tqdm
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import (
    ZeroBitWatermarker,
    LBitWatermarker,
    NaiveMultiUserWatermarker,
    HiDyPaMultiUserWatermarker,
)
from src.utils import get_model, parse_final_output


def json_default_encoder(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def hamming_distance(cw1, cw2):
    return sum(b1 != b2 for b1, b2 in zip(cw1, cw2))


def majority_vote(codewords):
    """
    Combine k extracted codewords via majority voting.
    Uncertain bits (not 0 or 1) are ignored in the vote.
    Position stays undetermined (bot) only if ALL k extractions were uncertain.
    """
    L = len(codewords[0])
    result = []
    for pos in range(L):
        bits = [cw[pos] for cw in codewords if cw[pos] in ("0", "1")]
        if not bits:
            result.append("bot")
        else:
            result.append("1" if sum(int(b) for b in bits) > len(bits) / 2 else "0")
    return "".join(result)


def compute_feasible_set(cw1, cw2):
    """
    Boneh-Shaw feasible set from two codewords.
    Undetermined positions (not 0/1) make both 0 and 1 achievable.
    """
    choices = []
    for b1, b2 in zip(cw1, cw2):
        v1 = b1 if b1 in ("0", "1") else None
        v2 = b2 if b2 in ("0", "1") else None
        if v1 is not None and v2 is not None and v1 == v2:
            choices.append([v1])
        else:
            choices.append(["0", "1"])
    return ["".join(combo) for combo in itertools.product(*choices)]


def _build_group_to_users(muw):
    if not hasattr(muw, "group_to_users"):
        return None
    if isinstance(muw.group_to_users, list):
        return {i: list(users) for i, users in enumerate(muw.group_to_users)}
    return dict(muw.group_to_users)


def select_participants_mau(muw):
    ids = random.sample(range(muw.N), 3)
    return ids[0], ids[1], ids[2]


def select_participants_hidypa(muw):
    group_to_users = _build_group_to_users(muw)
    groups = list(group_to_users.keys())
    if len(groups) >= 3:
        g1, g2, g3 = random.sample(groups, 3)
        c1     = random.choice(group_to_users[g1])
        c2     = random.choice(group_to_users[g2])
        victim = random.choice(group_to_users[g3])
    elif len(groups) == 2:
        g1, g2 = groups[0], groups[1]
        c1     = random.choice(group_to_users[g1])
        c2     = random.choice(group_to_users[g2])
        pool   = [u for u in group_to_users[g1] + group_to_users[g2]
                  if u not in (c1, c2)]
        victim = random.choice(pool)
    else:
        ids            = random.sample(group_to_users[groups[0]], 3)
        c1, c2, victim = ids[0], ids[1], ids[2]
    return c1, c2, victim


def run_experiment(muw, master_key, prompts, k_values, scheme, max_new_tokens, model_name):
    """
    One experiment with a fixed (c1, c2, victim) triple.
    The adversary makes max(k_values) colluding attempts total, then CRR is
    evaluated at each k by majority-voting only the first k detections.
    """
    L     = muw.lbw.L
    max_k = max(k_values)

    if scheme == "naive":
        c1_id, c2_id, victim_id = select_participants_mau(muw)
    else:
        c1_id, c2_id, victim_id = select_participants_hidypa(muw)

    cw_victim = muw.get_codeword_for_user(victim_id)

    # Accumulate codeword detections over max_k colluding attempts
    cws_c1 = []
    cws_c2 = []
    for attempt_idx in range(max_k):
        prompt = prompts[attempt_idx % len(prompts)]

        raw_c1  = muw.embed(master_key, c1_id, prompt, max_new_tokens=max_new_tokens)
        text_c1 = parse_final_output(raw_c1, model_name)
        cws_c1.append(muw.lbw.detect(master_key, text_c1))

        raw_c2  = muw.embed(master_key, c2_id, prompt, max_new_tokens=max_new_tokens)
        text_c2 = parse_final_output(raw_c2, model_name)
        cws_c2.append(muw.lbw.detect(master_key, text_c2))

    # Evaluate CRR at each requested k
    crr_at_k = {}
    for k in k_values:
        mv_c1        = majority_vote(cws_c1[:k])
        mv_c2        = majority_vote(cws_c2[:k])
        feasible_set = compute_feasible_set(mv_c1, mv_c2)

        if scheme == "naive":
            target = min(feasible_set, key=lambda w: hamming_distance(w, cw_victim))
        else:
            target = random.choice(feasible_set)

        matching     = sum(t == v for t, v in zip(target, cw_victim))
        crr_at_k[k] = (matching / L) * 100.0

    return {
        "c1_id":    c1_id,
        "c2_id":    c2_id,
        "victim_id": victim_id,
        "cw_victim": cw_victim,
        "crr_at_k":  crr_at_k,
    }


def save_raw_results(records, output_path):
    if not records:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=json_default_encoder))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Framing attack resistance: CRR vs number of colluding trials k.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scheme",    type=str, required=True, choices=["naive", "hi_dypa"])
    parser.add_argument("--group-bits", type=int, default=None)
    parser.add_argument("--user-bits",  type=int, default=None)
    parser.add_argument("--l-bits",     type=int, default=8)
    parser.add_argument("--model",      type=str, default="gpt2",
                        choices=["gpt2", "gpt-oss-20b", "gpt-oss-120b",
                                 "llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b",
                                 "opt-125m", "opt-1.3b", "opt-2.7b", "opt-6.7b",
                                 "deepseek-llm-7b"])
    parser.add_argument("--delta",             type=float, default=3.5)
    parser.add_argument("--entropy-threshold", type=float, default=2.5)
    parser.add_argument("--hashing-context",   type=int,   default=5)
    parser.add_argument("--z-threshold",       type=float, default=4.0)
    parser.add_argument("--prompts-file",      type=str,   default="assets/prompts.txt")
    parser.add_argument("--n-trials",          type=int,   default=20,
                        help="Number of independent (c1,c2,victim) experiments (default: 20)")
    parser.add_argument("--k-values",          type=int,   nargs="+", default=[1, 5, 10, 50, 100],
                        help="Colluding trial counts to evaluate (default: 1 5 10 50 100)")
    parser.add_argument("--users-file",        type=str,   default="assets/users.csv")
    parser.add_argument("--max-new-tokens",    type=int,   default=400)
    parser.add_argument("--seed",              type=int,   default=None)
    parser.add_argument("--output-dir",        type=str,   default="evaluation/framing_attack")
    parser.add_argument("--run-tag",           type=str,   default=None)
    parser.add_argument("--save-raw-results",  action="store_true")
    parser.add_argument("--raw-results-file",  type=str,   default="raw_results.jsonl.gz")

    args = parser.parse_args()

    if args.scheme == "hi_dypa":
        if args.group_bits is None or args.user_bits is None:
            parser.error("--group-bits and --user-bits are required for hi_dypa")
        if args.group_bits + args.user_bits != args.l_bits:
            parser.error(
                f"--group-bits ({args.group_bits}) + --user-bits ({args.user_bits}) "
                f"must equal --l-bits ({args.l_bits})"
            )

    k_values = sorted(set(args.k_values))
    max_k    = max(k_values)

    if args.scheme == "hi_dypa":
        scheme_parts = ["hi_dypa", f"G{args.group_bits}_U{args.user_bits}"]
    else:
        scheme_parts = ["naive", f"L{args.l_bits}"]

    base_output_dir = args.output_dir
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(parent_dir, base_output_dir)

    dir_parts = [base_output_dir] + scheme_parts
    if args.run_tag:
        dir_parts.append(args.run_tag)
    scheme_output_dir = os.path.join(*dir_parts)
    os.makedirs(scheme_output_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    seed = args.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)

    adversary_label = (
        "Codeword-Aware (picks optimal from F(T); database breached)"
        if args.scheme == "naive"
        else "Codeword-Blind (picks randomly from F(T); Kg[g*] secret, Theorem 6)"
    )

    total_gens = args.n_trials * max_k * 2

    print("\n" + "=" * 80)
    print(" " * 18 + "FRAMING ATTACK RESISTANCE EVALUATION")
    print("=" * 80)
    print(f"  Scheme:     {args.scheme}")
    if args.scheme == "hi_dypa":
        print(f"  Group bits: {args.group_bits}  User bits: {args.user_bits}")
    print(f"  L-bits:     {args.l_bits}")
    print(f"  Model:      {args.model}")
    print(f"  n_trials:   {args.n_trials}  (independent experiments)")
    print(f"  k_values:   {k_values}  (max_k={max_k})")
    print(f"  Total gens: {total_gens}  ({args.n_trials} x {max_k} x 2)")
    print(f"  Seed:       {seed}")
    print(f"  Adversary:  {adversary_label}")
    print(f"  Output:     {scheme_output_dir}")
    print("=" * 80)

    print(f"\n[1/4] Loading prompts...")
    prompts_path = os.path.join(parent_dir, args.prompts_file)
    if not os.path.exists(prompts_path):
        prompts_path = args.prompts_file
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(prompts)} prompts (will cycle as needed)")

    print(f"\n[2/4] Loading model '{args.model}'...")
    model = get_model(args.model)
    print(f"  Model loaded")

    zero_bit = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        hashing_context=args.hashing_context,
        z_threshold=args.z_threshold,
    )
    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zero_bit, L=args.l_bits)

    if args.scheme == "hi_dypa":
        muw = HiDyPaMultiUserWatermarker(
            lbit_watermarker=lbit_watermarker,
            group_bits=args.group_bits,
            user_bits=args.user_bits,
            min_distance=2,
        )
    else:
        muw = NaiveMultiUserWatermarker(lbit_watermarker=lbit_watermarker)

    users_path = os.path.join(parent_dir, args.users_file)
    if not os.path.exists(users_path):
        users_path = args.users_file

    if args.scheme == "naive":
        import tempfile
        df_all = pd.read_csv(users_path)
        if len(df_all) > 128:
            df_limited = df_all.head(128)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                df_limited.to_csv(tmp.name, index=False)
                muw.load_users(tmp.name)
            os.unlink(tmp.name)
        else:
            muw.load_users(users_path)
    else:
        muw.load_users(users_path)

    print(f"  Loaded {muw.N} users")
    master_key = muw.keygen()

    print(f"\n[3/4] Running {args.n_trials} experiments x {max_k} attempts x 2 colluders = {total_gens} generations...")

    all_experiments = []
    for trial_idx in tqdm(range(args.n_trials), desc="Experiments", unit="exp"):
        try:
            result = run_experiment(
                muw, master_key, prompts, k_values,
                args.scheme, args.max_new_tokens, args.model,
            )
            result["trial_id"] = trial_idx
            all_experiments.append(result)
        except Exception as e:
            print(f"\n  Warning: experiment {trial_idx} failed: {e}")
            continue

    if not all_experiments:
        print("  No experiments completed.")
        return

    print(f"\n[4/4] Computing CRR(k) statistics over {len(all_experiments)} experiments...")
    crr_by_k = {}
    for k in k_values:
        rates = [exp["crr_at_k"][k] for exp in all_experiments if k in exp["crr_at_k"]]
        crr_by_k[k] = {
            "mean": float(np.mean(rates)),
            "std":  float(np.std(rates)),
            "n":    len(rates),
        }

    print(f"\n  {'k':>5}  {'CRR mean':>10}  {'std':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}")
    for k in k_values:
        print(f"  {k:>5}  {crr_by_k[k]['mean']:>9.1f}%  {crr_by_k[k]['std']:>7.1f}%")

    if args.save_raw_results:
        raw_path = os.path.join(scheme_output_dir, args.raw_results_file)
        save_raw_results(all_experiments, raw_path)

    summary = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "scheme":         args.scheme,
        "model":          args.model,
        "run_tag":        args.run_tag,
        "adversary":      adversary_label,
        "parameters": {
            "l_bits":            args.l_bits,
            "group_bits":        args.group_bits if args.scheme == "hi_dypa" else None,
            "user_bits":         args.user_bits  if args.scheme == "hi_dypa" else None,
            "n_trials":          args.n_trials,
            "k_values":          k_values,
            "random_seed":       seed,
            "delta":             args.delta,
            "entropy_threshold": args.entropy_threshold,
            "hashing_context":   args.hashing_context,
            "z_threshold":       args.z_threshold,
            "max_new_tokens":    args.max_new_tokens,
        },
        "crr_by_k": crr_by_k,
    }

    json_path = os.path.join(scheme_output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default_encoder)

    csv_rows = [
        {"scheme": args.scheme, "k": k,
         "CRR_mean": round(crr_by_k[k]["mean"], 2),
         "CRR_std":  round(crr_by_k[k]["std"],  2)}
        for k in k_values
    ]
    csv_path = os.path.join(scheme_output_dir, "summary.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\n  Results saved to {scheme_output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
