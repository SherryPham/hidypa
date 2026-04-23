# evaluate_hi_dypa_detection.py
# Script to evaluate pure detection performance (no collusion) for hi_dypa multi-user watermarking
# at L=8, across all allocations of group bits and user bits.

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
    HiDyPaMultiUserWatermarker,
    derive_key
)
from src.utils import get_model, parse_final_output
from src.fingerprinting import generate_user_fingerprint
import torch


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


def compute_z_score(lbw, master_key, text):
    """
    Compute overall z-score from L-bit detection.
    Returns the average z-score across all bits.
    """
    tokenizer = lbw.model.tokenizer
    token_ids = tokenizer.encode(text, return_tensors='pt').to(lbw.model.device)[0]
    
    if len(token_ids) < 2:
        return 0.0
    
    with torch.no_grad():
        outputs = lbw.model._model(token_ids.unsqueeze(0))
    all_logits = outputs.logits.squeeze(0)
    
    z_scores = []
    for i in range(1, lbw.L + 1):
        z_i0, _, _ = lbw.zero_bit.detect(derive_key(master_key, i, 0), text, cached_logits=all_logits)
        z_i1, _, _ = lbw.zero_bit.detect(derive_key(master_key, i, 1), text, cached_logits=all_logits)
        # Use the maximum z-score for this bit position
        z_scores.append(max(z_i0, z_i1))
    
    return np.mean(z_scores) if z_scores else 0.0


def hamming_distance(codeword1: str, codeword2: str) -> int:
    """Calculate Hamming distance between two codewords, ignoring invalid symbols."""
    if len(codeword1) != len(codeword2):
        return float('inf')
    
    distance = 0
    for i in range(len(codeword1)):
        if codeword1[i] in ('0', '1') and codeword2[i] in ('0', '1'):
            if codeword1[i] != codeword2[i]:
                distance += 1
    
    return distance


def count_invalid_symbols(codeword: str) -> int:
    """Count the number of invalid symbols (⊥, *, ?) in a codeword."""
    return sum(1 for c in codeword if c in ('⊥', '*', '?'))


def save_raw_results(results: list[dict], output_path: str):
    """
    Persist detailed per-prompt results as gzipped JSONL.
    Each line stores one record to keep files compact.
    """
    if not results:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, default=json_default_encoder))
            f.write("\n")


def decode_naive_user(muw, recovered_codeword: str) -> int | None:
    """
    Decode user ID from recovered codeword for naive scheme.
    Returns detected_user_id or None.
    """
    # Find best matching user by Hamming distance
    best_user_id = None
    best_distance = float('inf')
    
    valid_positions = [i for i, bit in enumerate(recovered_codeword) if bit in ('0', '1')]
    if not valid_positions:
        return None
    
    for user_id in range(muw.N):
        try:
            user_codeword = muw.get_codeword_for_user(user_id)
            distance = hamming_distance(recovered_codeword, user_codeword)
            if distance < best_distance:
                best_distance = distance
                best_user_id = user_id
        except:
            continue
    
    return best_user_id


def decode_hi_dypa_user(muw, recovered_codeword: str, true_user_id: int = None) -> tuple[int | None, int | None, int | None]:
    """
    Decode group ID and user ID from recovered codeword for Hi-DyPa scheme.
    Returns (detected_group_id, detected_user_id, true_group_id).
    """
    if len(recovered_codeword) != muw.lbw.L:
        return None, None, None
    
    # Get true group ID if true_user_id is provided
    true_group_id = None
    if true_user_id is not None:
        if hasattr(muw, 'user_to_group') and isinstance(muw.user_to_group, dict):
            true_group_id = muw.user_to_group.get(true_user_id)
        elif hasattr(muw, 'user_metadata') and hasattr(muw, '_users_per_group') and muw._users_per_group > 0:
            user_row = muw.user_metadata[muw.user_metadata['UserId'] == true_user_id]
            if not user_row.empty:
                true_group_id = int(user_row.index[0]) // muw._users_per_group
    
    # Split recovered bits
    recovered_group_bits = recovered_codeword[:muw.group_bits]
    recovered_user_bits = recovered_codeword[muw.group_bits:]
    
    # Group identification: find nearest group codeword by Hamming distance
    best_group_id = None
    best_group_distance = float('inf')
    valid_group_positions = [i for i, bit in enumerate(recovered_group_bits) 
                            if bit not in ('⊥', '*', '?')]
    
    if not valid_group_positions:
        return None, None, true_group_id
    
    for group_id in range(muw._num_groups):
        group_codeword = muw._get_group_codeword_str(group_id)
        distance = sum(
            recovered_group_bits[i] != group_codeword[i]
            for i in valid_group_positions
        )
        if distance < best_group_distance:
            best_group_distance = distance
            best_group_id = group_id
    
    if best_group_id is None:
        return None, None, true_group_id
    
    # User identification: find nearest user fingerprint within the identified group
    if isinstance(muw.group_to_users, dict):
        users_in_group = muw.group_to_users.get(best_group_id, [])
    else:
        users_in_group = muw.group_to_users[best_group_id] if best_group_id < len(muw.group_to_users) else []
    if not users_in_group:
        return best_group_id, None, true_group_id
    
    valid_user_positions = [i for i, bit in enumerate(recovered_user_bits) 
                           if bit not in ('⊥', '*', '?')]
    
    if not valid_user_positions:
        return best_group_id, None, true_group_id
    
    best_user_id = None
    best_user_distance = float('inf')
    
    for user_id in users_in_group:
        user_index_in_group = users_in_group.index(user_id)
        user_fingerprint = generate_user_fingerprint(user_index_in_group, muw.user_bits)
        
        distance = sum(
            recovered_user_bits[i] != user_fingerprint[i]
            for i in valid_user_positions
        )
        
        if distance < best_user_distance:
            best_user_distance = distance
            best_user_id = user_id
    
    return best_group_id, best_user_id, true_group_id


def evaluate_prompt(
    muw, master_key, prompt, true_user_id, scheme, model_name, max_new_tokens
) -> dict:
    """
    Evaluate a single prompt: embed, detect, and decode.
    
    Returns:
        Dictionary with evaluation results
    """
    # Embed watermark
    raw_text = muw.embed(master_key, true_user_id, prompt, max_new_tokens=max_new_tokens)
    final_text = parse_final_output(raw_text, model_name)
    
    # Detect L-bit codeword
    recovered_codeword = muw.lbw.detect(master_key, final_text)
    
    # Get ground truth codeword
    try:
        ground_truth_codeword = muw.get_codeword_for_user(true_user_id)
    except:
        ground_truth_codeword = None
    
    # Compute z-score
    z_score = compute_z_score(muw.lbw, master_key, final_text)
    
    # Count invalid symbols
    num_invalid_symbols = count_invalid_symbols(recovered_codeword)
    
    # Compute Hamming distance
    hamming_dist = hamming_distance(recovered_codeword, ground_truth_codeword) if ground_truth_codeword else float('inf')
    
    result = {
        'true_user_id': true_user_id,
        'recovered_codeword': recovered_codeword,
        'ground_truth_codeword': ground_truth_codeword,
        'num_invalid_symbols': num_invalid_symbols,
        'hamming_distance': hamming_dist if hamming_dist != float('inf') else None,
        'z_score': z_score,
    }
    
    if scheme == 'naive':
        detected_user_id = decode_naive_user(muw, recovered_codeword)
        result['detected_user_id'] = detected_user_id
        result['full_identity_match'] = (detected_user_id == true_user_id)
        result['lbit_accuracy'] = (recovered_codeword == ground_truth_codeword) if ground_truth_codeword else False
    else:  # hi_dypa
        detected_group_id, detected_user_id, true_group_id = decode_hi_dypa_user(muw, recovered_codeword, true_user_id)
        
        result['true_group_id'] = true_group_id
        result['detected_group_id'] = detected_group_id
        result['detected_user_id'] = detected_user_id
        result['group_match'] = (detected_group_id == true_group_id)
        result['user_match'] = (detected_user_id == true_user_id)
        result['full_identity_match'] = (detected_group_id == true_group_id and detected_user_id == true_user_id)
        result['lbit_accuracy'] = (recovered_codeword == ground_truth_codeword) if ground_truth_codeword else False
    
    return result


def compute_metrics(results: list[dict], scheme: str) -> dict:
    """
    Compute summary metrics from per-prompt results.
    
    Args:
        results: List of per-prompt result dictionaries
        scheme: 'naive' or 'hi_dypa'
    
    Returns:
        Dictionary with computed metrics
    """
    metrics = {}
    
    if scheme == 'naive':
        # L-bit accuracy
        lbit_correct = sum(1 for r in results if r.get('lbit_accuracy', False))
        metrics['lbit_accuracy'] = lbit_correct / len(results) if results else 0.0
        
        # Full identity accuracy
        identity_correct = sum(1 for r in results if r.get('full_identity_match', False))
        metrics['full_identity_accuracy'] = identity_correct / len(results) if results else 0.0
        
        # False positives: detected wrong user
        false_positives = sum(1 for r in results 
                            if r.get('detected_user_id') is not None 
                            and not r.get('full_identity_match', False))
        metrics['false_positive_rate'] = false_positives / len(results) if results else 0.0
        
        # False negatives: failed to detect anyone
        false_negatives = sum(1 for r in results 
                            if r.get('detected_user_id') is None)
        metrics['false_negative_rate'] = false_negatives / len(results) if results else 0.0
        
    else:  # hi_dypa
        # Group detection accuracy
        group_correct = sum(1 for r in results if r.get('group_match', False))
        metrics['group_accuracy'] = group_correct / len(results) if results else 0.0
        
        # User detection accuracy (given correct group)
        correct_group_results = [r for r in results if r.get('group_match', False)]
        user_correct = sum(1 for r in correct_group_results if r.get('user_match', False))
        metrics['user_accuracy'] = user_correct / len(correct_group_results) if correct_group_results else 0.0
        
        # Full identity accuracy
        identity_correct = sum(1 for r in results if r.get('full_identity_match', False))
        metrics['full_identity_accuracy'] = identity_correct / len(results) if results else 0.0
        
        # L-bit accuracy
        lbit_correct = sum(1 for r in results if r.get('lbit_accuracy', False))
        metrics['lbit_accuracy'] = lbit_correct / len(results) if results else 0.0
        
        # False positive rate: system assigns wrong user
        false_positives = sum(1 for r in results 
                            if r.get('detected_user_id') is not None 
                            and not r.get('full_identity_match', False))
        metrics['false_positive_rate'] = false_positives / len(results) if results else 0.0
        
        # False negative rate: system fails to detect anyone
        false_negatives = sum(1 for r in results 
                            if r.get('detected_user_id') is None)
        metrics['false_negative_rate'] = false_negatives / len(results) if results else 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pure detection performance for hi_dypa multi-user watermarking",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--scheme',
        type=str,
        required=True,
        choices=['naive', 'hi_dypa'],
        help='Watermarking scheme to use: naive or hi_dypa'
    )
    parser.add_argument(
        '--group-bits',
        type=int,
        default=None,
        help='Number of bits for group codewords (required for Hi-DyPa scheme)'
    )
    parser.add_argument(
        '--user-bits',
        type=int,
        default=None,
        help='Number of bits for user fingerprints (required for Hi-DyPa scheme)'
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
        default='assets/prompts.txt',
        help='Path to prompts file (default: assets/prompts.txt)'
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
        choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b',
                 'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.1-8b',
                 'opt-125m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b',
                 'deepseek-llm-7b'],
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
        default=512,
        help='Maximum number of tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation/hi_dypa_detection',
        help='Output directory for results (default: evaluation/hi_dypa_detection)'
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
        help='If set, also save detailed per-prompt records as raw_results.jsonl.gz'
    )
    parser.add_argument(
        '--raw-results-file',
        type=str,
        default='raw_results.jsonl.gz',
        help='Filename for the raw results artifact (default: raw_results.jsonl.gz)'
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
    if args.scheme == 'hi_dypa':
        if args.group_bits is None or args.user_bits is None:
            parser.error("--group-bits and --user-bits are required for Hi-DyPa scheme")
        if args.group_bits + args.user_bits != args.l_bits:
            parser.error(
                f"--group-bits ({args.group_bits}) + --user-bits ({args.user_bits}) "
                f"must equal --l-bits ({args.l_bits})"
            )
    
    # Create output directory structure
    if args.scheme == 'hi_dypa':
        scheme_dir_parts = ['hi_dypa', f"G{args.group_bits}_U{args.user_bits}"]
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
                if args.scheme == 'hi_dypa':
                    config_name = f"hi_dypa_G{args.group_bits}_U{args.user_bits}"
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
    if args.scheme == 'hi_dypa':
        config_name = f"hi_dypa_G{args.group_bits}_U{args.user_bits}"
    else:
        config_name = f"naive_L{args.l_bits}"
    
    # Append seed to main seeds.txt file in base output directory
    main_seeds_file = os.path.join(base_output_dir, 'seeds.txt')
    # Write header only if file doesn't exist
    if not os.path.exists(main_seeds_file):
        with open(main_seeds_file, 'w', encoding='utf-8') as f:
            f.write("# Random seeds used for user selection in detection evaluation\n")
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
    
    # Print header
    print("\n" + "="*80)
    print(" " * 20 + "HI-DYPA DETECTION EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Scheme: {args.scheme}")
    if args.scheme == 'hi_dypa':
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
    
    if args.scheme == 'hi_dypa':
        muw = HiDyPaMultiUserWatermarker(
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
    
    # For naive scheme, ensure exactly 128 users for fair comparison with hi_dypa
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
    
    # Generate master key
    master_key = muw.keygen()
    
    # Process each prompt
    print(f"\n[3/4] Processing {len(prompts)} prompts...")
    
    all_results = []
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts", unit="prompt")):
        # Choose a random user ID
        true_user_id = random.randint(0, muw.N - 1)
        
        # Evaluate prompt
        try:
            result = evaluate_prompt(
                muw, master_key, prompt, true_user_id, args.scheme, args.model, args.max_new_tokens
            )
            result['prompt_id'] = prompt_idx
            result['prompt'] = prompt
            all_results.append(result)
        except Exception as e:
            print(f"\n  Warning: Error processing prompt {prompt_idx}: {e}")
            continue
    
    # Compute metrics
    print(f"\n[4/4] Computing metrics...")
    metrics = compute_metrics(all_results, args.scheme)
    
    raw_results_path = None
    if args.save_raw_results and all_results:
        raw_results_path = os.path.join(scheme_output_dir, args.raw_results_file)
        save_raw_results(all_results, raw_results_path)
        print(f"  Saved raw per-prompt results to: {raw_results_path}")
    
    # Create summary
    summary = {
        'scheme': args.scheme,
        'model': args.model,
        'run_tag': args.run_tag,
        'l_bits': args.l_bits,
        'group_bits': args.group_bits if args.scheme == 'hi_dypa' else None,
        'user_bits': args.user_bits if args.scheme == 'hi_dypa' else None,
        'num_prompts': len(all_results),
        'random_seed': seed,
        'output_directory': scheme_output_dir,
        'raw_results_file': os.path.basename(raw_results_path) if raw_results_path else None,
        'generated_utc': datetime.utcnow().isoformat() + "Z",
        'metrics': metrics
    }
    
    # Save summary JSON
    summary_json_path = os.path.join(scheme_output_dir, 'summary.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=json_default_encoder)
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 25 + "RESULTS SUMMARY")
    print("="*80)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"  {metric_name:30s}: {metric_value:6.4f}")
        else:
            print(f"  {metric_name:30s}: {metric_value}")
    
    print(f"\nSummary saved to: {summary_json_path}")
    if raw_results_path:
        print(f"Raw results saved to: {raw_results_path}")
    else:
        print("Raw results skipped (pass --save-raw-results to capture them)")
    print("\n" + "="*80)
    print(" " * 30 + "EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

