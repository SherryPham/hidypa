
import argparse
import itertools
import json
import os
import random
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.watermark import NaiveMultiUserWatermarker, HiDyPaMultiUserWatermarker



# Minimal stub — only .L is accessed during codeword generation; no model needed

class _MockLBW:
    def __init__(self, L: int):
        self.L = L



# Pure codeword helpers


def hamming_distance(cw1: str, cw2: str) -> int:
    return sum(b1 != b2 for b1, b2 in zip(cw1, cw2))


def compute_feasible_set(cw1: str, cw2: str) -> list[str]:
    """
    Enumerate F(T) for two colluders.

    Boneh-Shaw principle: at position i, if cw1[i] == cw2[i] the bit is fixed;
    otherwise either '0' or '1' is achievable.  |F(T)| = 2^(# differing positions).
    """
    choices = []
    for b1, b2 in zip(cw1, cw2):
        choices.append([b1] if b1 == b2 else ['0', '1'])
    return [''.join(combo) for combo in itertools.product(*choices)]



# Scheme initialisation (no LLM, no disk model files)


def _make_users_csv(n: int, path: str) -> None:
    pd.DataFrame({'UserId': range(n), 'Username': range(n)}).to_csv(path, index=False)


def init_mau(users_file: str, L: int) -> NaiveMultiUserWatermarker:
    muw = NaiveMultiUserWatermarker(_MockLBW(L))
    muw.load_users(users_file)
    return muw


def init_hidypa(users_file: str, Lg: int, Lu: int) -> HiDyPaMultiUserWatermarker:
    muw = HiDyPaMultiUserWatermarker(_MockLBW(Lg + Lu), group_bits=Lg, user_bits=Lu)
    muw.load_users(users_file)
    return muw


# Participant selection


def _available_groups(muw: HiDyPaMultiUserWatermarker) -> list[int]:
    return [g for g, users in enumerate(muw.group_to_users) if len(users) > 0]


def select_participants_mau(muw: NaiveMultiUserWatermarker) -> tuple[int, int, int]:
    ids = random.sample(list(muw.user_lookup.keys()), 3)
    return ids[0], ids[1], ids[2]


def select_participants_hidypa(muw: HiDyPaMultiUserWatermarker) -> tuple[int, int, int]:

    groups = _available_groups(muw)

    if len(groups) >= 3:
        g1, g2, g3 = random.sample(groups, 3)
        c1     = random.choice(muw.group_to_users[g1])
        c2     = random.choice(muw.group_to_users[g2])
        victim = random.choice(muw.group_to_users[g3])

    elif len(groups) == 2:
        g1, g2 = groups[0], groups[1]
        c1 = random.choice(muw.group_to_users[g1])
        c2 = random.choice(muw.group_to_users[g2])
        # victim from whichever group has a user that is not c1 or c2
        pool = [u for u in muw.group_to_users[g1] + muw.group_to_users[g2]
                if u not in (c1, c2)]
        victim = random.choice(pool)

    else:
        # Single group — same-group attack
        ids    = random.sample(muw.group_to_users[groups[0]], 3)
        c1, c2, victim = ids[0], ids[1], ids[2]

    return c1, c2, victim



# Single trial


def run_trial(muw, scheme_type: str, L: int) -> dict:

    if scheme_type == 'mau':
        c1_id, c2_id, victim_id = select_participants_mau(muw)
    else:
        c1_id, c2_id, victim_id = select_participants_hidypa(muw)

    cw_c1    = muw.get_codeword_for_user(c1_id)
    cw_c2    = muw.get_codeword_for_user(c2_id)
    cw_victim = muw.get_codeword_for_user(victim_id)

    feasible_set = compute_feasible_set(cw_c1, cw_c2)

    if scheme_type == 'mau':
        # Codeword-Aware: database breached — pick closest element in F(T) to victim
        target = min(feasible_set, key=lambda w: hamming_distance(w, cw_victim))
    else:
        # Codeword-Blind: Kg[g*] secret — pick uniformly at random from F(T)
        target = random.choice(feasible_set)

    matching = sum(t == v for t, v in zip(target, cw_victim))
    recovery_rate = (matching / L) * 100.0

    return {
        'recovery_rate':    recovery_rate,
        'feasible_set_size': len(feasible_set),
        'hamming_c1_c2':    hamming_distance(cw_c1, cw_c2),
    }


# Scheme-level evaluation


def evaluate_scheme(muw, scheme_type: str, L: int,
                    k_values: list[int], max_k: int) -> tuple[dict, list[dict]]:
    """
    Run max_k independent trials and compute CRR(k) for each k in k_values.
    Returns (crr_stats, raw_trial_records).
    """
    trials = []
    for _ in range(max_k):
        trials.append(run_trial(muw, scheme_type, L))

    rates = [t['recovery_rate'] for t in trials]

    crr = {}
    for k in k_values:
        subset = rates[:k]
        crr[k] = {
            'mean': float(np.mean(subset)),
            'std':  float(np.std(subset)),
        }

    return crr, trials



# Results display

def print_results_table(table_rows: list[dict], k_values: list[int]) -> None:
    k_cols = [f'k={k}' for k in k_values]
    header = f"{'Scheme':<12} {'(Lg,Lu)':<10}" + ''.join(f"{c:>8}" for c in k_cols)
    sep    = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)

    hi_dypa_rows = [r for r in table_rows if r['scheme'] == 'hi_dypa']
    mau_rows     = [r for r in table_rows if r['scheme'] == 'mau']

    for row in hi_dypa_rows:
        label  = 'Hi-DyPa'
        config = f"({row['Lg']},{row['Lu']})"
        vals   = ''.join(f"{row['crr'][k]['mean']:>7.1f}%" for k in k_values)
        print(f"{label:<12} {config:<10}{vals}")

    print(sep)

    for row in mau_rows:
        label  = 'MAU'
        config = f"({row['Lg']},{row['Lu']})"
        vals   = ''.join(f"{row['crr'][k]['mean']:>7.1f}%" for k in k_values)
        print(f"{label:<12} {config:<10}{vals}")

    print(sep)
    print('CRR(k) = mean recovery rate over k independent trials.')
    print('MAU  adversary: Codeword-Aware  — picks optimal element from F(T) (database breached).')
    print('Hi-DyPa adversary: Codeword-Blind — picks randomly from F(T) (Kg[g*] secret, Thm 6).')



def main():
    parser = argparse.ArgumentParser(
        description='Framing attack resistance evaluation via CRR.'
    )
    parser.add_argument('--n_users',    type=int, default=100,
                        help='Number of users in the system (default: 100)')
    parser.add_argument('--L',          type=int, default=8,
                        help='Total codeword length in bits (default: 8)')
    parser.add_argument('--max_k',      type=int, default=100,
                        help='Number of independent trials per configuration (default: 100)')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(parent_dir, 'evaluation', 'framing_attack'),
                        help='Output directory for results')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    L        = args.L
    k_values = [1, 5, 10, 50, 100]
    # Clamp k_values to max_k so CRR(k) is never computed on fewer trials than k
    k_values = [k for k in k_values if k <= args.max_k]

    os.makedirs(args.output_dir, exist_ok=True)

    # Write users CSV once — reused by all configurations
    users_file = os.path.join(args.output_dir, 'users_eval.csv')
    _make_users_csv(args.n_users, users_file)

    # Define all configurations: MAU + all valid Hi-DyPa splits
    configs = [('mau', 0, L)]
    for Lg in range(1, L):
        Lu = L - Lg
        if Lu > 0:
            configs.append(('hi_dypa', Lg, Lu))

    table_rows  = []
    all_results = {}

    for scheme_type, Lg, Lu in tqdm(configs, desc='Configurations'):
        label = 'MAU' if scheme_type == 'mau' else 'Hi-DyPa'
        config_name = f'{label} ({Lg},{Lu})'
        print(f'\n--- {config_name} ---')

        if scheme_type == 'mau':
            muw = init_mau(users_file, L)
            num_groups = 1
        else:
            muw = init_hidypa(users_file, Lg, Lu)
            num_groups = len(_available_groups(muw))

        crr, trials = evaluate_scheme(muw, scheme_type, L, k_values, args.max_k)

        # Diagnostics
        avg_fs  = float(np.mean([t['feasible_set_size'] for t in trials]))
        avg_hd  = float(np.mean([t['hamming_c1_c2']    for t in trials]))
        print(f"  groups={num_groups}  avg |F(T)|={avg_fs:.1f}  avg Hamming(c1,c2)={avg_hd:.1f}")
        for k in k_values:
            print(f"  CRR(k={k:>3}) = {crr[k]['mean']:5.1f}% ± {crr[k]['std']:4.1f}%")

        row = {
            'scheme':     scheme_type,
            'label':      label,
            'Lg':         Lg,
            'Lu':         Lu,
            'num_groups': num_groups,
            'crr':        crr,
        }
        table_rows.append(row)
        all_results[config_name] = {
            **row,
            'avg_feasible_set_size': avg_fs,
            'avg_hamming_c1_c2':     avg_hd,
            'raw_trials': trials,
        }

    # Print summary table
    print_results_table(table_rows, k_values)


    # Save results

    summary = {
        'generated_utc': datetime.utcnow().isoformat() + 'Z',
        'parameters': {
            'n_users':  args.n_users,
            'L':        L,
            'max_k':    args.max_k,
            'seed':     args.seed,
            'k_values': k_values,
        },
        'results': all_results,
    }

    json_path = os.path.join(args.output_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Flat CSV: one row per configuration, one column per k value
    csv_rows = []
    for row in table_rows:
        r = {
            'Scheme':    row['label'],
            '(Lg,Lu)':   f"({row['Lg']},{row['Lu']})",
            'num_groups': row['num_groups'],
        }
        for k in k_values:
            r[f'CRR_k{k}_mean'] = round(row['crr'][k]['mean'], 2)
            r[f'CRR_k{k}_std']  = round(row['crr'][k]['std'],  2)
        csv_rows.append(r)

    csv_path = os.path.join(args.output_dir, 'summary.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f'\nResults saved to {args.output_dir}')
    print(f'  summary.json  — full results including per-trial records')
    print(f'  summary.csv   — flat table for easy inspection')


if __name__ == '__main__':
    main()
