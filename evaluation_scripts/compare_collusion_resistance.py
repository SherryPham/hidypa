# compare_collusion_resistance.py
# Script to evaluate naive and hierarchical watermarking schemes under controlled collusion
# Tests with 2-3 colluders in different configurations: same group, cross group, mixed

import argparse
import gzip
import json
import os
import random
import sys
import time
from datetime import datetime

from tqdm import tqdm
import pandas as pd
import numpy as np

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import (
    ZeroBitWatermarker, 
    LBitWatermarker, 
    NaiveMultiUserWatermarker, 
    HierarchicalMultiUserWatermarker
)
from src.utils import get_model, parse_final_output


def json_default_encoder(obj):
    """Convert NumPy/Pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def merge_codewords_bitwise_majority(codewords: list[str]) -> str:
    """
    Merge multiple L-bit codewords using bitwise majority voting.
    Handles uncertain bits ('⊥') and conflict bits ('*') properly.
    
    Args:
        codewords: List of codeword strings (each should be L bits)
    
    Returns:
        Merged codeword string of length L
    """
    if not codewords:
        raise ValueError("Cannot merge empty list of codewords")
    
    L = len(codewords[0])
    if not all(len(c) == L for c in codewords):
        raise ValueError(f"All codewords must have the same length (L={L})")
    
    merged = []
    for bit_pos in range(L):
        # Collect all bits at this position
        bits_at_pos = [c[bit_pos] for c in codewords]
        
        # Filter out uncertain ('⊥') and conflict ('*') bits, keep only valid bits ('0' or '1')
        valid_bits = [b for b in bits_at_pos if b in ('0', '1')]
        
        if not valid_bits:
            # All codewords are uncertain at this position, result is uncertain
            merged.append('⊥')
        else:
            # Do majority voting on valid bits only
            valid_ints = [int(b) for b in valid_bits]
            # Majority vote: if more 1s than 0s, result is 1, else 0
            # In case of tie, default to 0
            majority_bit = 1 if sum(valid_ints) > len(valid_ints) / 2 else 0
            merged.append(str(majority_bit))
    
    return "".join(merged)


def sample_same_group(muw, k: int) -> list[int]:
    """
    Sample k users from the same group (for hierarchical scheme).
    For naive scheme, just sample k random users.
    
    Args:
        muw: Multi-user watermarker instance
        k: Number of users to sample
    
    Returns:
        List of user IDs
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent
        return sorted(random.sample(range(muw.N), k))
    
    # Hierarchical scheme: pick a random group and sample k users from it
    # Get all groups and their users
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    # Find groups with at least k users
    valid_groups = [gid for gid, users in group_to_users.items() if len(users) >= k]
    
    if not valid_groups:
        raise ValueError(f"No group has at least {k} users")
    
    # Pick a random group
    selected_group = random.choice(valid_groups)
    group_users = group_to_users[selected_group]
    
    # Sample k users from this group
    return sorted(random.sample(group_users, k))


def sample_different_groups(muw, k: int) -> list[int]:
    """
    Sample k users from k different groups (for hierarchical scheme).
    For naive scheme, just sample k random users (since all are independent).
    
    Args:
        muw: Multi-user watermarker instance
        k: Number of users to sample
    
    Returns:
        List of user IDs
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent
        return sorted(random.sample(range(muw.N), k))
    
    # Hierarchical scheme: pick k different groups
    # Get all groups
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    available_groups = list(group_to_users.keys())
    
    if len(available_groups) < k:
        raise ValueError(f"Only {len(available_groups)} groups available, need {k} different groups")
    
    # Pick k different groups
    selected_groups = random.sample(available_groups, k)
    
    # Pick one user from each group
    selected_users = []
    for group_id in selected_groups:
        group_users = group_to_users[group_id]
        selected_users.append(random.choice(group_users))
    
    return sorted(selected_users)


def sample_2_same_1_diff(muw) -> list[int]:
    """
    Sample 2 users from the same group and 1 user from a different group.
    For naive scheme, just sample 3 random users (since all are independent).
    
    Args:
        muw: Multi-user watermarker instance
    
    Returns:
        List of 3 user IDs
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent
        return sorted(random.sample(range(muw.N), 3))
    
    # Hierarchical scheme: pick 2 from one group, 1 from another
    # Get all groups
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    # Find groups with at least 2 users
    valid_groups = [gid for gid, users in group_to_users.items() if len(users) >= 2]
    
    if len(valid_groups) < 2:
        raise ValueError("Need at least 2 groups, with at least one having 2+ users")
    
    # Pick a group for the 2 users
    group_with_2 = random.choice(valid_groups)
    users_from_group = random.sample(group_to_users[group_with_2], 2)
    
    # Pick a different group for the 1 user
    other_groups = [gid for gid in group_to_users.keys() if gid != group_with_2]
    other_group = random.choice(other_groups)
    user_from_other = random.choice(group_to_users[other_group])
    
    return sorted(users_from_group + [user_from_other])


def get_group_info(muw) -> dict:
    """
    Get information about available groups for the multi-user watermarker.
    
    Args:
        muw: Multi-user watermarker instance
    
    Returns:
        Dictionary with group information including:
        - num_groups: Number of distinct groups
        - group_to_users: Mapping of group_id to list of user IDs
        - can_run_cross_group_2: Whether 2-colluder cross-group case can run
        - can_run_cross_group_3: Whether 3-colluder cross-group case can run
        - can_run_mixed: Whether mixed 2same_1diff case can run
    """
    if not hasattr(muw, 'user_to_group') or muw.user_to_group is None:
        # Naive scheme: all users are independent, all cases can run
        return {
            'num_groups': None,  # Not applicable for naive
            'group_to_users': None,
            'can_run_cross_group_2': True,
            'can_run_cross_group_3': True,
            'can_run_mixed': True
        }
    
    # Hierarchical scheme: analyze groups
    group_to_users = {}
    for user_id, group_id in muw.user_to_group.items():
        if group_id not in group_to_users:
            group_to_users[group_id] = []
        group_to_users[group_id].append(user_id)
    
    num_groups = len(group_to_users)
    
    # Check which cases can run
    can_run_cross_group_2 = num_groups >= 2
    can_run_cross_group_3 = num_groups >= 3
    
    # For mixed case: need at least 2 groups, with at least one having 2+ users
    groups_with_2plus = [gid for gid, users in group_to_users.items() if len(users) >= 2]
    can_run_mixed = num_groups >= 2 and len(groups_with_2plus) >= 1
    
    return {
        'num_groups': num_groups,
        'group_to_users': group_to_users,
        'can_run_cross_group_2': can_run_cross_group_2,
        'can_run_cross_group_3': can_run_cross_group_3,
        'can_run_mixed': can_run_mixed
    }


def save_raw_results(records: list[dict], output_path: str):
    """Persist full prompt-level collusion results as gzipped JSON Lines."""
    if not records:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, default=json_default_encoder))
            f.write("\n")


def trace_collusion(muw, master_key: bytes, merged_codeword: str, original_user_ids: list[int]) -> dict:
    """
    Try to trace back to original colluding users using the merged codeword.
    For hierarchical schemes, uses the built-in trace_from_codeword method which
    first identifies suspect groups, then searches only within those groups.
    For naive schemes, uses direct codeword matching with Hamming distance.
    Checks ALL users to properly detect false positives (innocent users incorrectly accused).
    
    Args:
        muw: Multi-user watermarker instance
        master_key: Master secret key
        merged_codeword: The merged L-bit codeword (from bitwise majority)
        original_user_ids: List of original colluding user IDs
    
    Returns:
        Dictionary with tracing results including success status
    """
    try:
        # For hierarchical schemes, use the built-in trace_from_codeword method
        # This is more efficient and accurate as it first identifies suspect groups
        if hasattr(muw, 'trace_from_codeword') and hasattr(muw, 'group_codewords') and muw.group_codewords:
            try:
                # Use hierarchical tracing: identifies suspect groups first, then searches within them
                accused_users = muw.trace_from_codeword(merged_codeword)
                matches = [u['user_id'] for u in accused_users]
                
                # Calculate Hamming distances for all users for completeness
                hamming_distances = {}
                all_user_ids = list(range(muw.N))
                valid_positions = [i for i, c in enumerate(merged_codeword) if c in ('0', '1')]
                
                for user_id in all_user_ids:
                    try:
                        user_codeword = muw.get_codeword_for_user(user_id)
                        if valid_positions:
                            hamming_dist = sum(
                                merged_codeword[i] != user_codeword[i] 
                                for i in valid_positions
                            )
                        else:
                            hamming_dist = float('inf')
                        hamming_distances[user_id] = hamming_dist
                    except Exception:
                        hamming_distances[user_id] = float('inf')
                
                return {
                    'success': len(matches) > 0,
                    'accused_user_ids': matches,  # May include non-colluders (false positives)
                    'original_user_ids': original_user_ids,
                    'matches': matches,
                    'num_matches': len(matches),
                    'merged_codeword': merged_codeword,
                    'hamming_distances': hamming_distances,
                    'method': 'hierarchical_trace'
                }
            except Exception as e:
                # If hierarchical tracing fails, fall back to direct matching
                pass
        
        # For naive schemes (or if hierarchical tracing fails), use direct codeword matching
        matches = []
        hamming_distances = {}
        
        # Check ALL users (0 to N-1), not just original colluders
        # This allows us to detect false positives (innocent users incorrectly accused)
        all_user_ids = list(range(muw.N))
        
        # Calculate valid positions once (same for all users)
        valid_positions = [
            i for i, c in enumerate(merged_codeword) 
            if c in ('0', '1')
        ]
        
        # Adaptive threshold: 25% of valid bits (rounded up to nearest integer)
        # This adapts to data quality - fewer valid bits means stricter threshold
        if valid_positions:
            adaptive_threshold = max(1, int(len(valid_positions) * 0.25 + 0.5))  # Round to nearest, minimum 1
        else:
            adaptive_threshold = float('inf')
        
        for user_id in all_user_ids:
            try:
                user_codeword = muw.get_codeword_for_user(user_id)
                
                if not valid_positions:
                    # All positions are uncertain, can't match
                    hamming_dist = float('inf')
                else:
                    # Compare only at valid positions
                    hamming_dist = sum(
                        merged_codeword[i] != user_codeword[i] 
                        for i in valid_positions
                    )
                
                hamming_distances[user_id] = hamming_dist
                
                # Use adaptive threshold: allow errors up to 25% of valid bits
                # Also need to check that we have enough valid positions to make a meaningful comparison
                if len(valid_positions) >= 3 and hamming_dist <= adaptive_threshold:
                    matches.append(user_id)
            except Exception as e:
                # If we can't get codeword for a user, skip it
                continue
        
        return {
            'success': len(matches) > 0,
            'accused_user_ids': matches,  # May include non-colluders (false positives)
            'original_user_ids': original_user_ids,
            'matches': matches,
            'num_matches': len(matches),
            'merged_codeword': merged_codeword,
            'hamming_distances': hamming_distances,
            'method': 'direct_codeword_match'
        }
    except Exception as e:
        return {
            'success': False,
            'accused_user_ids': [],
            'original_user_ids': original_user_ids,
            'reason': f'Error during tracing: {str(e)}',
            'merged_codeword': merged_codeword
        }


def evaluate_collusion_case(
    muw, master_key: bytes, prompt: str, colluder_ids: list[int],
    case_name: str, max_new_tokens: int, model_name: str
) -> dict:
    """
    Evaluate a single collusion case: embed for each colluder, recover codewords, merge, trace.
    
    Args:
        muw: Multi-user watermarker instance
        master_key: Master secret key
        prompt: The prompt to use
        colluder_ids: List of colluding user IDs
        case_name: Name of the collusion case (e.g., 'same_group_2')
        max_new_tokens: Maximum tokens to generate
        model_name: Model name for parsing output
    
    Returns:
        Dictionary with evaluation results
    """
    # Embed for each colluder
    user_texts = []
    recovered_codewords = []
    
    for user_id in colluder_ids:
        # Embed watermark
        raw_text = muw.embed(master_key, user_id, prompt, max_new_tokens=max_new_tokens)
        final_text = parse_final_output(raw_text, model_name)
        user_texts.append(final_text)
        
        # Recover L-bit codeword
        recovered_codeword = muw.lbw.detect(master_key, final_text)
        recovered_codewords.append(recovered_codeword)
    
    # Merge codewords via bitwise majority
    merged_codeword = merge_codewords_bitwise_majority(recovered_codewords)
    
    # Trace merged pattern
    trace_result = trace_collusion(muw, master_key, merged_codeword, colluder_ids)
    
    return {
        'case_name': case_name,
        'colluder_ids': colluder_ids,
        'recovered_codewords': recovered_codewords,
        'merged_codeword': merged_codeword,
        'trace_result': trace_result,
        'success': trace_result['success']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate naive and hierarchical watermarking schemes under controlled collusion",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--scheme',
        type=str,
        required=True,
        choices=['naive', 'hierarchical'],
        help='Watermarking scheme to use: naive or hierarchical'
    )
    parser.add_argument(
        '--group-bits',
        type=int,
        default=None,
        help='Number of bits for group codewords (required for hierarchical scheme)'
    )
    parser.add_argument(
        '--user-bits',
        type=int,
        default=None,
        help='Number of bits for user fingerprints (required for hierarchical scheme)'
    )
    parser.add_argument(
        '--l-bits',
        type=int,
        default=8,
        help='Total number of L-bits for watermarking (default: 8)'
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        required=True,
        help='Path to prompts file'
    )
    parser.add_argument(
        '--num-prompts',
        type=int,
        default=300,
        help='Number of prompts to use (default: 300)'
    )
    parser.add_argument(
        '--users-file',
        type=str,
        default='assets/users.csv',
        help='Path to users CSV file (default: assets/users.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
        help='Model to use for generation and detection'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=3.5,
        help='Watermark strength (default: 3.5)'
    )
    parser.add_argument(
        '--entropy-threshold',
        type=float,
        default=2.5,
        help='Entropy threshold for watermarking (default: 2.5)'
    )
    parser.add_argument(
        '--hashing-context',
        type=int,
        default=5,
        help='Hashing context window (default: 5)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=4.0,
        help='Z-score threshold for detection (default: 4.0)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=400,
        help='Maximum number of tokens to generate per user (default: 400)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--run-tag',
        type=str,
        default=None,
        help='Optional identifier appended to the output directory (e.g., job id)'
    )
    parser.add_argument(
        '--save-raw-results',
        action='store_true',
        help='If set, store per-prompt collusion records as raw_results.jsonl.gz'
    )
    parser.add_argument(
        '--raw-results-file',
        type=str,
        default='raw_results.jsonl.gz',
        help='Filename for the raw results artifact'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: auto-generated or loaded from seeds file)'
    )
    parser.add_argument(
        '--seeds-file',
        type=str,
        default=None,
        help='Path to seeds.txt file to read existing seeds from (optional). If provided and seed exists for this config, it will be reused.'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scheme == 'hierarchical':
        if args.group_bits is None or args.user_bits is None:
            parser.error("--group-bits and --user-bits are required for hierarchical scheme")
        if args.group_bits + args.user_bits != args.l_bits:
            parser.error(
                f"--group-bits ({args.group_bits}) + --user-bits ({args.user_bits}) "
                f"must equal --l-bits ({args.l_bits})"
            )
    
    # Create output directory structure
    if args.scheme == 'hierarchical':
        scheme_dir_parts = ['hierarchical', f"G{args.group_bits}_U{args.user_bits}"]
    else:
        scheme_dir_parts = ['naive', f"L{args.l_bits}"]
    
    base_output_dir = args.output_dir
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(parent_dir, base_output_dir)
    
    dir_parts = [base_output_dir, *scheme_dir_parts]
    if args.run_tag:
        dir_parts.append(args.run_tag)
    scheme_output_dir = os.path.join(*dir_parts)
    os.makedirs(scheme_output_dir, exist_ok=True)
    
    # Ensure base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    seed = args.seed
    if seed is None:
        # Try to read from seeds file if provided
        if args.seeds_file:
            seeds_file_path = args.seeds_file
            if not os.path.isabs(seeds_file_path):
                seeds_file_path = os.path.join(parent_dir, seeds_file_path)
            
            if os.path.exists(seeds_file_path):
                # Generate config name to check
                if args.scheme == 'hierarchical':
                    config_name = f"hierarchical_G{args.group_bits}_U{args.user_bits}"
                else:
                    config_name = f"naive_L{args.l_bits}"
                
                # Try to read existing seed for this config
                with open(seeds_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip().startswith(f"{config_name}:"):
                            try:
                                seed = int(line.split(":")[1].strip())
                                break
                            except (IndexError, ValueError):
                                pass
        
        # Generate a random seed if still not found
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)  # Use timestamp-based seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate config name for seeds.txt
    if args.scheme == 'hierarchical':
        config_name = f"hierarchical_G{args.group_bits}_U{args.user_bits}"
    else:
        config_name = f"naive_L{args.l_bits}"
    
    # Append seed to main seeds.txt file in base output directory
    main_seeds_file = os.path.join(base_output_dir, 'seeds.txt')
    # Write header only if file doesn't exist
    if not os.path.exists(main_seeds_file):
        with open(main_seeds_file, 'w', encoding='utf-8') as f:
            f.write("# Random seeds used for user sampling in collusion evaluation\n")
            f.write(f"# Model: {args.model}\n")
            if args.run_tag:
                f.write(f"# Run tag: {args.run_tag}\n")
            f.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: config_name: seed_value\n")
            f.write("\n")
    
    # Check if config already exists in seeds.txt
    config_found = False
    if os.path.exists(main_seeds_file):
        with open(main_seeds_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith(f"{config_name}:"):
                    config_found = True
                    break
    
    # Append this configuration's seed only if not already present
    if not config_found:
        with open(main_seeds_file, 'a', encoding='utf-8') as f:
            f.write(f"{config_name}: {seed}\n")
    
    # Create subdirectories for colluder scenarios (for summaries only)
    output_2_dir = os.path.join(scheme_output_dir, '2_colluders')
    output_3_dir = os.path.join(scheme_output_dir, '3_colluders')
    os.makedirs(output_2_dir, exist_ok=True)
    os.makedirs(output_3_dir, exist_ok=True)
    
    # Print header
    print("\n" + "="*80)
    print(" " * 20 + "COLLUSION RESISTANCE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Scheme: {args.scheme}")
    if args.scheme == 'hierarchical':
        print(f"  • Group bits: {args.group_bits}")
        print(f"  • User bits: {args.user_bits}")
    print(f"  • L-bits: {args.l_bits}")
    print(f"  • Model: {args.model}")
    print(f"  • Number of prompts: {args.num_prompts}")
    print(f"  • Random seed: {seed} (saved to {os.path.join(base_output_dir, 'seeds.txt')})")
    print(f"  • Output directory: {scheme_output_dir}")
    print("="*80)
    
    # Load prompts
    print(f"\n[1/4] Loading prompts...")
    prompts_path = os.path.join(parent_dir, args.prompts_file)
    if not os.path.exists(prompts_path):
        print(f"  Error: Prompts file not found: {prompts_path}")
        return
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(all_prompts) < args.num_prompts:
        print(f"  Warning: Only {len(all_prompts)} prompts available, using all of them")
        prompts = all_prompts
    else:
        prompts = all_prompts[:args.num_prompts]
    
    print(f"  Loaded {len(prompts)} prompts")
    
    # Load model
    print(f"\n[2/4] Loading model and initializing watermarker...")
    print(f"  Loading model '{args.model}'...")
    model = get_model(args.model)
    print(f"  Model loaded successfully")
    
    # Initialize watermarker
    print(f"\n  Initializing watermarker...")
    zero_bit = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        hashing_context=args.hashing_context,
        z_threshold=args.z_threshold
    )
    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zero_bit, L=args.l_bits)
    
    if args.scheme == 'hierarchical':
        muw = HierarchicalMultiUserWatermarker(
            lbit_watermarker=lbit_watermarker,
            group_bits=args.group_bits,
            user_bits=args.user_bits,
            min_distance=2
        )
    else:
        muw = NaiveMultiUserWatermarker(lbit_watermarker=lbit_watermarker)
    
    # Load users
    users_path = os.path.join(parent_dir, args.users_file)
    if not os.path.exists(users_path):
        print(f"  Error: Users file not found: {users_path}")
        return
    
    # For naive scheme, ensure exactly 128 users for fair comparison with hierarchical
    if args.scheme == 'naive':
        import tempfile
        df_all = pd.read_csv(users_path)
        if len(df_all) > 128:
            print(f"  Limiting to 128 users for naive scheme (for fair comparison)")
            df_limited = df_all.head(128)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df_limited.to_csv(tmp_file.name, index=False)
                tmp_users_path = tmp_file.name
            muw.load_users(tmp_users_path)
            os.unlink(tmp_users_path)
        else:
            muw.load_users(users_path)
    else:
        muw.load_users(users_path)
    
    print(f"  Loaded {muw.N} users")
    
    # Get group information to determine which cases can run
    group_info = get_group_info(muw)
    
    # Log warnings once at the start if cases need to be skipped
    if args.scheme == 'hierarchical':
        print(f"\n  Group structure: {group_info['num_groups']} groups available")
        skipped_cases = []
        if not group_info['can_run_cross_group_2']:
            skipped_cases.append("2-colluder cross-group (needs 2+ groups)")
        if not group_info['can_run_cross_group_3']:
            skipped_cases.append("3-colluder cross-group (needs 3+ groups)")
        if not group_info['can_run_mixed']:
            skipped_cases.append("3-colluder mixed (needs 2+ groups with at least one having 2+ users)")
        
        if skipped_cases:
            print(f"  Warning: Skipping the following collusion cases due to insufficient groups:")
            for case in skipped_cases:
                print(f"    - {case}")
    
    # Generate master key
    master_key = muw.keygen()
    
    # Process each prompt
    print(f"\n[3/4] Processing {len(prompts)} prompts...")
    
    all_results = []
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
        prompt_results = {
            'prompt_id': prompt_idx,
            'prompt': prompt,
            'scheme': args.scheme,
            'config': {
                'l_bits': args.l_bits,
                'group_bits': args.group_bits if args.scheme == 'hierarchical' else None,
                'user_bits': args.user_bits if args.scheme == 'hierarchical' else None,
            },
            'results': {}
        }
        
        # 2 colluders cases
        # same_group_2 - always possible (just need 1 group with 2+ users)
        try:
            colluders_same_2 = sample_same_group(muw, 2)
            result_same_2 = evaluate_collusion_case(
                muw, master_key, prompt, colluders_same_2,
                'same_group_2', args.max_new_tokens, args.model
            )
            prompt_results['results']['same_group_2'] = result_same_2
        except Exception as e:
            prompt_results['results']['same_group_2'] = {'error': str(e)}
        
        # cross_group_2 - only if we have 2+ groups
        if group_info['can_run_cross_group_2']:
            try:
                colluders_cross_2 = sample_different_groups(muw, 2)
                result_cross_2 = evaluate_collusion_case(
                    muw, master_key, prompt, colluders_cross_2,
                    'cross_group_2', args.max_new_tokens, args.model
                )
                prompt_results['results']['cross_group_2'] = result_cross_2
            except Exception as e:
                prompt_results['results']['cross_group_2'] = {'error': str(e)}
        else:
            prompt_results['results']['cross_group_2'] = {'skipped': 'Insufficient groups (need 2+)'}
        
        # 3 colluders cases
        # same_group_3 - always possible (just need 1 group with 3+ users)
        try:
            colluders_same_3 = sample_same_group(muw, 3)
            result_same_3 = evaluate_collusion_case(
                muw, master_key, prompt, colluders_same_3,
                'same_group_3', args.max_new_tokens, args.model
            )
            prompt_results['results']['same_group_3'] = result_same_3
        except Exception as e:
            prompt_results['results']['same_group_3'] = {'error': str(e)}
        
        # cross_group_3 - only if we have 3+ groups
        if group_info['can_run_cross_group_3']:
            try:
                colluders_cross_3 = sample_different_groups(muw, 3)
                result_cross_3 = evaluate_collusion_case(
                    muw, master_key, prompt, colluders_cross_3,
                    'cross_group_3', args.max_new_tokens, args.model
                )
                prompt_results['results']['cross_group_3'] = result_cross_3
            except Exception as e:
                prompt_results['results']['cross_group_3'] = {'error': str(e)}
        else:
            prompt_results['results']['cross_group_3'] = {'skipped': 'Insufficient groups (need 3+)'}
        
        # mixed_2same_1diff - only if we have 2+ groups with at least one having 2+ users
        if group_info['can_run_mixed']:
            try:
                colluders_mixed = sample_2_same_1_diff(muw)
                result_mixed = evaluate_collusion_case(
                    muw, master_key, prompt, colluders_mixed,
                    'mixed_2same_1diff', args.max_new_tokens, args.model
                )
                prompt_results['results']['mixed_2same_1diff'] = result_mixed
            except Exception as e:
                prompt_results['results']['mixed_2same_1diff'] = {'error': str(e)}
        else:
            prompt_results['results']['mixed_2same_1diff'] = {'skipped': 'Insufficient groups (need 2+ groups with at least one having 2+ users)'}
        
        all_results.append(prompt_results)
    
    # Generate summary reports
    print(f"\n[4/4] Generating summary reports...")
    
    raw_results_path = None
    if args.save_raw_results and all_results:
        raw_results_path = os.path.join(scheme_output_dir, args.raw_results_file)
        save_raw_results(all_results, raw_results_path)
        print(f"  Saved raw collusion records to: {raw_results_path}")
    else:
        print("  Raw collusion records not persisted (enable --save-raw-results to store them)")
    
    # Calculate success rates and false positives for 2-colluder cases
    case_types_2 = ['same_group_2', 'cross_group_2']
    success_rates_2 = {}
    
    for case_type in case_types_2:
        case_results = [
            r['results'].get(case_type, {}) 
            for r in all_results 
            if case_type in r['results'] 
            and 'error' not in r['results'][case_type]
            and 'skipped' not in r['results'][case_type]
        ]
        if case_results:
            successful = sum(1 for r in case_results if r.get('success', False))
            total = len(case_results)
            
            # Calculate false positives: accused users who weren't original colluders
            # Count total false positive users (for reference)
            false_positives = 0
            # Count prompts with at least one false positive (for false positive rate)
            prompts_with_false_positives = 0
            for r in case_results:
                trace_result = r.get('trace_result', {})
                accused_user_ids = set(trace_result.get('accused_user_ids', []))
                original_user_ids = set(r.get('colluder_ids', []))
                
                # False positives are users in accused_user_ids but not in original_user_ids
                false_positive_users = accused_user_ids - original_user_ids
                if false_positive_users:
                    false_positives += len(false_positive_users)
                    prompts_with_false_positives += 1
            
            success_rates_2[case_type] = {
                'successful': successful,
                'total': total,
                'success_rate': (successful / total) * 100.0 if total > 0 else 0.0,
                'false_positives': false_positives,
                'false_positive_rate': (prompts_with_false_positives / total) * 100.0 if total > 0 else 0.0
            }
    
    # Calculate success rates and false positives for 3-colluder cases
    case_types_3 = ['same_group_3', 'cross_group_3', 'mixed_2same_1diff']
    success_rates_3 = {}
    
    for case_type in case_types_3:
        case_results = [
            r['results'].get(case_type, {}) 
            for r in all_results 
            if case_type in r['results'] 
            and 'error' not in r['results'][case_type]
            and 'skipped' not in r['results'][case_type]
        ]
        if case_results:
            successful = sum(1 for r in case_results if r.get('success', False))
            total = len(case_results)
            
            # Calculate false positives: accused users who weren't original colluders
            # Count total false positive users (for reference)
            false_positives = 0
            # Count prompts with at least one false positive (for false positive rate)
            prompts_with_false_positives = 0
            for r in case_results:
                trace_result = r.get('trace_result', {})
                accused_user_ids = set(trace_result.get('accused_user_ids', []))
                original_user_ids = set(r.get('colluder_ids', []))
                
                # False positives are users in accused_user_ids but not in original_user_ids
                false_positive_users = accused_user_ids - original_user_ids
                if false_positive_users:
                    false_positives += len(false_positive_users)
                    prompts_with_false_positives += 1
            
            success_rates_3[case_type] = {
                'successful': successful,
                'total': total,
                'success_rate': (successful / total) * 100.0 if total > 0 else 0.0,
                'false_positives': false_positives,
                'false_positive_rate': (prompts_with_false_positives / total) * 100.0 if total > 0 else 0.0
            }
    
    # Save summary JSON for 2 colluders
    base_summary = {
        'scheme': args.scheme,
        'model': args.model,
        'run_tag': args.run_tag,
        'l_bits': args.l_bits,
        'group_bits': args.group_bits if args.scheme == 'hierarchical' else None,
        'user_bits': args.user_bits if args.scheme == 'hierarchical' else None,
        'num_prompts': len(all_results),
        'random_seed': seed,
        'output_directory': scheme_output_dir,
        'raw_results_file': os.path.basename(raw_results_path) if raw_results_path else None,
        'generated_utc': datetime.utcnow().isoformat() + "Z"
    }
    
    summary_2 = {
        **base_summary,
        'num_colluders': 2,
        'success_rates': success_rates_2
    }
    
    summary_json_2_path = os.path.join(output_2_dir, 'summary.json')
    with open(summary_json_2_path, 'w', encoding='utf-8') as f:
        json.dump(summary_2, f, indent=2, default=json_default_encoder)
    
    # Save summary JSON for 3 colluders
    summary_3 = {
        **base_summary,
        'num_colluders': 3,
        'success_rates': success_rates_3
    }
    
    summary_json_3_path = os.path.join(output_3_dir, 'summary.json')
    with open(summary_json_3_path, 'w', encoding='utf-8') as f:
        json.dump(summary_3, f, indent=2, default=json_default_encoder)
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("="*80)
    print(f"\n2 Colluders - Success Rates:")
    print("-"*80)
    for case_type, stats in success_rates_2.items():
        print(f"  {case_type:20s}: {stats['success_rate']:6.2f}% ({stats['successful']}/{stats['total']})")
    
    print(f"\n3 Colluders - Success Rates:")
    print("-"*80)
    for case_type, stats in success_rates_3.items():
        print(f"  {case_type:20s}: {stats['success_rate']:6.2f}% ({stats['successful']}/{stats['total']})")
    
    print(f"\n2-colluder summary saved to: {summary_json_2_path}")
    print(f"3-colluder summary saved to: {summary_json_3_path}")
    if raw_results_path:
        print(f"Raw collusion records saved to: {raw_results_path}")
    else:
        print("Raw collusion records skipped (pass --save-raw-results to capture them)")
    print("\n" + "="*80)
    print(" " * 30 + "EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
