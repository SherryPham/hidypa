
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
    """Convert NumPy/Pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def hamming_distance(cw1: str, cw2: str) -> int:
    return sum(b1 != b2 for b1, b2 in zip(cw1, cw2))


def compute_feasible_set(cw1: str, cw2: str) -> list:
    """
    Boneh-Shaw feasible set from two extracted codewords.
    Handles uncertain bits: if either position is uncertain both 0 and 1 are achievable.
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
    """Build {group_id: [user_ids]} from the watermarker. Returns None for naive scheme."""
    if not hasattr(muw, "group_to_users"):
        return None
    if isinstance(muw.group_to_users, list):
        return {i: list(users) for i, users in enumerate(muw.group_to_users)}
    return dict(muw.group_to_users)


def select_participants_mau(muw):
    """Select c1, c2, victim as 3 distinct random users."""
    ids = random.sample(range(muw.N), 3)
    return ids[0], ids[1], ids[2]


def select_participants_hidypa(muw):
    """
    Cross-group selection:
      >=3 groups: colluders from g1, g2; victim from distinct g3.
         2 groups: colluders from g1, g2; victim from either (excluding the two colluders).
         1 group:  same-group fallback.
    """
    group_to_users = _build_group_to_users(muw)
    groups         = list(group_to_users.keys())

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


def run_framing_trial(muw, master_key, prompt, scheme, max_new_tokens, model_name):
    """
    One framing-attack trial:
      1. Select participants (c1, c2, victim).
      2. Generate watermarked texts for c1 and c2.
      3. Extract codewords via lbw.detect().
      4. Compute F(T) from extracted codewords.
      5. Adversary picks target:
           naive   (Codeword-Aware)  -> closest to victim known codeword
           hi_dypa (Codeword-Blind)  -> uniform random from F(T)
      6. CRR = fraction of victim bits matched (%).
    """
    L = muw.lbw.L

    if scheme == "naive":
        c1_id, c2_id, victim_id = select_participants_mau(muw)
    else:
        c1_id, c2_id, victim_id = select_participants_hidypa(muw)

    raw_c1  = muw.embed(master_key, c1_id, prompt, max_new_tokens=max_new_tokens)
    text_c1 = parse_final_output(raw_c1, model_name)

    raw_c2  = muw.embed(master_key, c2_id, prompt, max_new_tokens=max_new_tokens)
    text_c2 = parse_final_output(raw_c2, model_name)

    cw_c1     = muw.lbw.detect(master_key, text_c1)
    cw_c2     = muw.lbw.detect(master_key, text_c2)
    cw_victim = muw.get_codeword_for_user(victim_id)

    feasible_set = compute_feasible_set(cw_c1, cw_c2)

    if scheme == "naive":
        target = min(feasible_set, key=lambda w: hamming_distance(w, cw_victim))
    else:
        target = random.choice(feasible_set)

    matching      = sum(t == v for t, v in zip(target, cw_victim))
    recovery_rate = (matching / L) * 100.0

    return {
        "c1_id":             c1_id,
        "c2_id":             c2_id,
        "victim_id":         victim_id,
        "cw_c1":             cw_c1,
        "cw_c2":             cw_c2,
        "cw_victim":         cw_victim,
        "feasible_set_size": len(feasible_set),
        "target":            target,
        "recovery_rate":     recovery_rate,
        "hamming_c1_c2":     hamming_distance(cw_c1, cw_c2),
    }


def save_raw_results(records, output_path):
    """Persist per-trial results as gzipped JSONL."""
    if not records:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=json_default_encoder))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Framing attack resistance evaluation via Codeword Recovery Rate (CRR).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scheme", type=str, required=True, choices=["naive", "hi_dypa"])
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
    parser.add_argument("--num-prompts",       type=int,   default=100)
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

    print("\n" + "=" * 80)
    print(" " * 18 + "FRAMING ATTACK RESISTANCE EVALUATION")
    print("=" * 80)
    print(f"  Scheme:     {args.scheme}")
    if args.scheme == "hi_dypa":
        print(f"  Group bits: {args.group_bits}  User bits: {args.user_bits}")
    print(f"  L-bits:     {args.l_bits}")
    print(f"  Model:      {args.model}")
    print(f"  Trials:     {args.num_prompts}")
    print(f"  Seed:       {seed}")
    print(f"  Adversary:  {adversary_label}")
    print(f"  Output:     {scheme_output_dir}")
    print("=" * 80)

    print(f"\n[1/4] Loading prompts...")
    prompts_path = os.path.join(parent_dir, args.prompts_file)
    if not os.path.exists(prompts_path):
        prompts_path = args.prompts_file
    with open(prompts_path, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    prompts = all_prompts[:args.num_prompts]
    print(f"  Loaded {len(prompts)} prompts")

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

    print(f"\n[3/4] Running {len(prompts)} framing-attack trials ({adversary_label})...")
    all_trials = []
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Trials", unit="trial")):
        try:
            result = run_framing_trial(
                muw, master_key, prompt, args.scheme,
                args.max_new_tokens, args.model,
            )
            result["trial_id"] = prompt_idx
            result["prompt"]   = prompt
            all_trials.append(result)
        except Exception as e:
            print(f"\n  Warning: trial {prompt_idx} failed: {e}")
            continue

    if not all_trials:
        print("  No trials completed.")
        return

    print(f"\n[4/4] Computing CRR statistics...")
    k_values = [k for k in [1, 5, 10, 50, 100] if k <= len(all_trials)]
    rates    = [t["recovery_rate"] for t in all_trials]
    crr_stats = {}
    for k in k_values:
        crr_stats[k] = {
            "mean": float(np.mean(rates[:k])),
            "std":  float(np.std(rates[:k])),
        }

    avg_fs = float(np.mean([t["feasible_set_size"] for t in all_trials]))
    avg_hd = float(np.mean([t["hamming_c1_c2"]     for t in all_trials]))

    print(f"  Trials: {len(all_trials)}  Avg |F(T)|: {avg_fs:.1f}  Avg Hamming(c1,c2): {avg_hd:.1f}")
    for k in k_values:
        print(f"  CRR(k={k:>3}) = {crr_stats[k]['mean']:5.1f}% +/- {crr_stats[k]['std']:4.1f}%")

    raw_results_path = None
    if args.save_raw_results:
        raw_results_path = os.path.join(scheme_output_dir, args.raw_results_file)
        save_raw_results(all_trials, raw_results_path)

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
            "num_trials":        len(all_trials),
            "random_seed":       seed,
            "delta":             args.delta,
            "entropy_threshold": args.entropy_threshold,
            "hashing_context":   args.hashing_context,
            "z_threshold":       args.z_threshold,
            "max_new_tokens":    args.max_new_tokens,
        },
        "avg_feasible_set_size": avg_fs,
        "avg_hamming_c1_c2":     avg_hd,
        "crr_stats":             crr_stats,
        "raw_results_file":      os.path.basename(raw_results_path) if raw_results_path else None,
    }

    json_path = os.path.join(scheme_output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default_encoder)

    csv_rows = [{"scheme": args.scheme, "k": k,
                 "CRR_mean": round(crr_stats[k]["mean"], 2),
                 "CRR_std":  round(crr_stats[k]["std"],  2)} for k in k_values]
    csv_path = os.path.join(scheme_output_dir, "summary.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\n  Results saved to {scheme_output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
