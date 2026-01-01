# evaluate_multiuser_performance.py
# Script to evaluate performance metrics (memory, computation, storage) for multi-user watermarking schemes
# Compares: naive and hierarchical schemes (with different group/user bit allocations)

import argparse
import json
import os
import sys
import time
import tracemalloc
import psutil
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import ZeroBitWatermarker, LBitWatermarker, NaiveMultiUserWatermarker, HierarchicalMultiUserWatermarker


def get_model(model_name: str):
    """Function to instantiate the correct model based on its name."""
    if model_name == 'gpt2':
        return GPT2Model()
    elif model_name == 'gpt-oss-20b':
        return GptOssModel()
    elif model_name == 'gpt-oss-120b':
        return GptOss120bModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_storage_size_mb(obj):
    """Estimate storage size of an object in MB."""
    import sys
    size_bytes = sys.getsizeof(obj)
    # For numpy arrays, get actual data size
    if isinstance(obj, np.ndarray):
        size_bytes = obj.nbytes
    elif isinstance(obj, dict):
        size_bytes = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
    elif isinstance(obj, pd.DataFrame):
        size_bytes = obj.memory_usage(deep=True).sum()
    return size_bytes / 1024 / 1024


def load_prompts(prompts_file: str, max_prompts: int) -> List[str]:
    """Load prompts from a text file, returning up to max_prompts entries."""
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise ValueError(f"No valid prompts found in {prompts_file}")
    
    if max_prompts is None or max_prompts <= 0 or max_prompts >= len(prompts):
        return prompts
    
    return prompts[:max_prompts]


def aggregate_numeric_metrics(metrics_list: List[dict]) -> dict:
    """Average numeric metrics across a list of metric dictionaries."""
    if not metrics_list:
        return {}
    
    sums = {}
    counts = {}
    
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    
    return {key: (sums[key] / counts[key]) for key in sums}


def measure_initialization(muw, users_file: str, scheme_name: str) -> dict:
    """Measure initialization time, memory, and storage."""
    print(f"\n--- Measuring Initialization for {scheme_name} ---")
    
    # Start memory tracking
    tracemalloc.start()
    memory_before = get_memory_mb()
    
    # Measure initialization time
    start_time = time.perf_counter()
    
    # For fair comparison, limit all schemes to 128 users
    import tempfile
    df_all = pd.read_csv(users_file)
    if len(df_all) > 128:
        print(f"  Limiting to 128 users for {scheme_name} scheme (for fair comparison)")
        df_limited = df_all.head(128)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df_limited.to_csv(tmp_file.name, index=False)
            tmp_users_path = tmp_file.name
        muw.load_users(tmp_users_path)
        os.unlink(tmp_users_path)
    else:
        muw.load_users(users_file)
    
    init_time = time.perf_counter() - start_time
    
    # Measure memory after
    memory_after = get_memory_mb()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate storage
    storage_mb = 0
    num_groups = 0
    users_per_group = 0
    
    # Hierarchical scheme: store group codewords
    if hasattr(muw, 'group_codewords') and muw.group_codewords:
        storage_mb = get_storage_size_mb(muw.group_codewords)
        num_groups = len(muw.group_codewords)
        if hasattr(muw, 'group_to_users') and muw.group_to_users:
            # Get average users per group
            users_per_group = sum(len(users) for users in muw.group_to_users.values()) / len(muw.group_to_users) if muw.group_to_users else 0
    # Naive scheme: no storage (computed on-the-fly)
    
    return {
        'init_time_sec': init_time,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_peak_mb': peak / 1024 / 1024,  # tracemalloc returns bytes
        'memory_delta_mb': memory_after - memory_before,
        'storage_mb': storage_mb,
        'num_groups': num_groups,
        'users_per_group': users_per_group,
        'num_users': muw.N
    }


def measure_embedding(muw, master_key: bytes, user_id: int, prompt: str, max_new_tokens: int, scheme_name: str) -> Tuple[dict, str]:
    """Measure embedding time, memory, and overhead. Returns metrics dict and watermarked text."""
    print(f"\n--- Measuring Embedding for {scheme_name} ---")
    
    # Get baseline (no watermarking) - just model generation
    model = muw.lbw.model
    tokenizer = model.tokenizer
    device = model.device
    
    # Baseline generation time
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    attention_mask = torch.ones_like(input_ids)
    
    # Baseline time
    memory_before = get_memory_mb()
    start_time = time.perf_counter()
    with torch.no_grad():
        baseline_output = model._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True, top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
    baseline_time = time.perf_counter() - start_time
    memory_after_baseline = get_memory_mb()
    
    # Watermarked generation time
    memory_before_watermark = get_memory_mb()
    start_time = time.perf_counter()
    watermarked_text = muw.embed(master_key, user_id, prompt, max_new_tokens=max_new_tokens)
    embed_time = time.perf_counter() - start_time
    memory_after_watermark = get_memory_mb()
    
    # Count tokens
    baseline_tokens = len(baseline_output[0]) - len(input_ids[0])
    watermarked_tokens = len(tokenizer.encode(watermarked_text))
    
    metrics = {
        'embed_time_sec': embed_time,
        'baseline_time_sec': baseline_time,
        'overhead_time_sec': embed_time - baseline_time,
        'overhead_percent': ((embed_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0,
        'time_per_token_ms': (embed_time / watermarked_tokens * 1000) if watermarked_tokens > 0 else 0,
        'memory_before_mb': memory_before_watermark,
        'memory_after_mb': memory_after_watermark,
        'memory_delta_mb': memory_after_watermark - memory_before_watermark,
        'num_tokens': watermarked_tokens
    }
    
    return metrics, watermarked_text


def measure_detection(muw, master_key: bytes, text: str, scheme_name: str) -> tuple[dict, str]:
    """
    Measure detection time, HMAC operations, and memory.
    
    Returns:
        tuple: (metrics_dict, recovered_codeword) - The recovered codeword can be reused for tracing.
    """
    print(f"\n--- Measuring Detection for {scheme_name} ---")
    
    # Count HMAC operations: 2L (one for each bit position, each with 0 and 1)
    L = muw.lbw.L
    expected_hmac_ops = 2 * L
    
    memory_before = get_memory_mb()
    start_time = time.perf_counter()
    
    # Detection: model forward pass + 2L zero-bit detections
    recovered_codeword = muw.lbw.detect(master_key, text)
    
    detect_time = time.perf_counter() - start_time
    memory_after = get_memory_mb()
    
    metrics = {
        'detect_time_sec': detect_time,
        'hmac_operations': expected_hmac_ops,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_delta_mb': memory_after - memory_before,
        'recovered_codeword': recovered_codeword
    }
    
    return metrics, recovered_codeword


def measure_tracing(muw, master_key: bytes, text: str, recovered_codeword: str = None, scheme_name: str = None) -> dict:
    """
    Measure pure tracing time (comparisons only, without detection).
    
    Args:
        muw: Multi-user watermarker instance
        master_key: Master secret key (not used if recovered_codeword provided)
        text: Watermarked text (not used if recovered_codeword provided)
        recovered_codeword: Pre-computed recovered codeword from detection
        scheme_name: Name of the scheme for logging
    """
    if scheme_name:
        print(f"\n--- Measuring Tracing (comparisons only) for {scheme_name} ---")
    else:
        print(f"\n--- Measuring Tracing (comparisons only) ---")
    
    # Count comparisons: N for naive, hierarchical uses group-based tracing
    N = muw.N
    expected_comparisons = N
    
    if hasattr(muw, 'group_codewords') and muw.group_codewords:
        num_groups = len(muw.group_codewords)
        # Hierarchical scheme: first compare groups, then users within suspect groups
        # Typical: G group comparisons + users_in_suspect_groups user comparisons
        potential_optimized_comparisons = num_groups
    else:
        num_groups = None
        potential_optimized_comparisons = None
    
    # If recovered_codeword not provided, we need to get it (but this shouldn't happen in normal flow)
    if recovered_codeword is None:
        print("Warning: No recovered_codeword provided, running detection first...")
        recovered_codeword = muw.lbw.detect(master_key, text)
    
    memory_before = get_memory_mb()
    start_time = time.perf_counter()
    
    # Pure tracing: just comparisons, no detection
    accused_users = muw.trace_from_codeword(recovered_codeword)
    
    trace_time = time.perf_counter() - start_time
    memory_after = get_memory_mb()
    
    # Count actual comparisons made
    actual_comparisons = N  # Naive: N user comparisons
    if hasattr(muw, 'group_codewords') and muw.group_codewords:
        # Hierarchical: G group comparisons + users in suspect groups
        num_groups_compared = len(muw.group_codewords)
        # Estimate users compared (typically 1-2 suspect groups * users_per_group)
        if hasattr(muw, 'group_to_users') and muw.group_to_users:
            avg_users_per_group = sum(len(users) for users in muw.group_to_users.values()) / len(muw.group_to_users) if muw.group_to_users else 0
            # Assume 1-2 suspect groups on average
            estimated_users_compared = min(2 * avg_users_per_group, N)
        else:
            estimated_users_compared = 0
        actual_comparisons = num_groups_compared + estimated_users_compared
    
    return {
        'trace_time_sec': trace_time,
        'comparisons_count': expected_comparisons,
        'actual_comparisons_estimate': actual_comparisons,
        'potential_optimized_comparisons': potential_optimized_comparisons,
        'num_groups': num_groups,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_delta_mb': memory_after - memory_before,
        'accused_users_count': len(accused_users)
    }


def measure_scalability(users_file: str, L: int, group_bits: int = None, user_bits: int = None) -> dict:
    """Measure scalability metrics: max users, groups, etc."""
    df = pd.read_csv(users_file)
    N = len(df)
    
    max_users_naive = 2 ** L
    
    if group_bits is not None and user_bits is not None:
        # Hierarchical scheme: max_groups = 2^group_bits, max_users_per_group = 2^user_bits
        max_groups = 2 ** group_bits
        max_users_per_group = 2 ** user_bits
        max_users_hierarchical = max_groups * max_users_per_group
    else:
        max_groups = None
        max_users_per_group = None
        max_users_hierarchical = None
    
    return {
        'current_users': N,
        'max_users_naive': max_users_naive,
        'max_users_hierarchical': max_users_hierarchical,
        'max_groups': max_groups,
        'max_users_per_group': max_users_per_group,
        'l_bits': L,
        'group_bits': group_bits,
        'user_bits': user_bits
    }


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
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate performance metrics (memory, computation, storage) for multi-user watermarking schemes",
        formatter_class=argparse.RawTextHelpFormatter
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
        '--l-bits',
        type=int,
        default=8,
        help='Number of L-bits for watermarking (default: 8)'
    )
    parser.add_argument(
        '--group-bits',
        type=int,
        default=None,
        help='Number of bits for group codewords (for hierarchical scheme, required if testing hierarchical)'
    )
    parser.add_argument(
        '--user-bits',
        type=int,
        default=None,
        help='Number of bits for user fingerprints (for hierarchical scheme, required if testing hierarchical)'
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
        '--prompt',
        type=str,
        default=None,
        help='Optional single prompt for generation; overrides prompts file when provided'
    )
    parser.add_argument(
        '--user-id',
        type=int,
        default=64,
        help='User ID to use for embedding test (default: 64)'
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        default='assets/prompts.txt',
        help='Path to prompts file (default: assets/prompts.txt)'
    )
    parser.add_argument(
        '--max-prompts',
        type=int,
        default=100,
        help='Number of prompts to evaluate from prompts file when --prompt is not set (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation/multiuser_performance',
        help='Output directory for results (default: evaluation/multiuser_performance)'
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
    parser.add_argument(
        '--hierarchical-only',
        action='store_true',
        help='If set, only evaluate hierarchical scheme (skip naive baseline)'
    )
    parser.add_argument(
        '--unified-summary-path',
        type=str,
        default=None,
        help='Path to unified summary CSV file to append results to (for aggregating all configs)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to repo root
    users_file_path = args.users_file
    if not os.path.isabs(users_file_path):
        users_file_path = os.path.join(parent_dir, users_file_path)
    
    prompts_file_path = args.prompts_file
    if not os.path.isabs(prompts_file_path):
        prompts_file_path = os.path.join(parent_dir, prompts_file_path)
    
    output_dir_path = args.output_dir
    if not os.path.isabs(output_dir_path):
        output_dir_path = os.path.join(parent_dir, output_dir_path)
    
    # Create output directory
    os.makedirs(output_dir_path, exist_ok=True)
    
    print("=" * 80)
    print("Multi-User Watermarking Performance Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"L-bits: {args.l_bits}")
    if args.hierarchical_only:
        print("Mode: Hierarchical-only (naive skipped)")
    if args.group_bits is not None and args.user_bits is not None:
        print(f"Hierarchical: G={args.group_bits}, U={args.user_bits}")
    print(f"User ID: {args.user_id}")
    print(f"Users file: {users_file_path}")
    print(f"Output directory: {output_dir_path}")
    if args.unified_summary_path:
        print(f"Unified summary: {args.unified_summary_path}")
    print("=" * 80)
    
    if args.prompt:
        prompts_to_use = [args.prompt.strip()]
    else:
        prompts_to_use = load_prompts(prompts_file_path, args.max_prompts)
    
    if not prompts_to_use:
        raise ValueError("No prompts available for evaluation.")
    
    print(f"Total prompts to evaluate: {len(prompts_to_use)}")
    
    # Load model
    print(f"\nLoading model '{args.model}'...")
    model = get_model(args.model)
    
    # Setup watermarking stack
    zbw = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        z_threshold=args.z_threshold,
        hashing_context=args.hashing_context
    )
    lbw = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)
    
    # Test schemes: naive and hierarchical (if group_bits/user_bits provided)
    schemes = []
    
    if not args.hierarchical_only:
        schemes.append(('naive', None, None, None))
    
    if args.group_bits is not None and args.user_bits is not None:
        if args.group_bits + args.user_bits != args.l_bits:
            raise ValueError(
                f"--group-bits ({args.group_bits}) + --user-bits ({args.user_bits}) "
                f"must equal --l-bits ({args.l_bits})"
            )
        schemes.append(('hierarchical', args.group_bits, args.user_bits, 2))  # min_distance=2 for hierarchical
    
    if not schemes:
        raise ValueError("No schemes to evaluate. Either provide --group-bits/--user-bits or omit --hierarchical-only")
    
    all_results = {}
    
    for scheme_name, group_bits, user_bits, min_distance in schemes:
        print(f"\n{'=' * 80}")
        if scheme_name == 'hierarchical':
            print(f"Evaluating Scheme: {scheme_name} (G={group_bits}, U={user_bits})")
        else:
            print(f"Evaluating Scheme: {scheme_name}")
        print(f"{'=' * 80}")
        
        # Create watermarker
        if scheme_name == 'naive':
            muw = NaiveMultiUserWatermarker(lbit_watermarker=lbw)
        else:  # hierarchical
            muw = HierarchicalMultiUserWatermarker(
                lbit_watermarker=lbw,
                group_bits=group_bits,
                user_bits=user_bits,
                min_distance=min_distance
            )
        
        scheme_results = {}
        
        # 1. Initialization metrics
        try:
            init_metrics = measure_initialization(muw, users_file_path, scheme_name)
            scheme_results['initialization'] = init_metrics
            print(f"Initialization completed: {init_metrics['init_time_sec']:.4f}s")
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            scheme_results['initialization'] = {'error': str(e)}
            continue
        
        # 2. Scalability metrics
        scalability_metrics = measure_scalability(users_file_path, args.l_bits, group_bits, user_bits)
        scheme_results['scalability'] = scalability_metrics
        
        # 3. Generate master key
        master_key = muw.keygen()
        
        # 4-6. Embedding, detection, and tracing metrics across prompts
        embedding_metrics_list = []
        detection_metrics_list = []
        tracing_metrics_list = []
        
        for prompt_idx, prompt in enumerate(prompts_to_use):
            watermarked_text = None
            try:
                embed_metrics, watermarked_text = measure_embedding(
                    muw, master_key, args.user_id, prompt, args.max_new_tokens, scheme_name
                )
                embed_metrics['prompt'] = prompt
                embed_metrics['prompt_index'] = prompt_idx
                embedding_metrics_list.append(embed_metrics)
            except Exception as e:
                print(f"✗ Embedding failed for prompt #{prompt_idx}: {e}")
                continue
            
            if watermarked_text:
                recovered_codeword = None
                try:
                    detect_metrics, recovered_codeword = measure_detection(muw, master_key, watermarked_text, scheme_name)
                    detect_metrics['prompt'] = prompt
                    detect_metrics['prompt_index'] = prompt_idx
                    detection_metrics_list.append(detect_metrics)
                except Exception as e:
                    print(f"✗ Detection failed for prompt #{prompt_idx}: {e}")
                
                try:
                    # Pass recovered_codeword to avoid re-running detection
                    trace_metrics = measure_tracing(muw, master_key, watermarked_text, recovered_codeword, scheme_name)
                    trace_metrics['prompt'] = prompt
                    trace_metrics['prompt_index'] = prompt_idx
                    tracing_metrics_list.append(trace_metrics)
                except Exception as e:
                    print(f"✗ Tracing failed for prompt #{prompt_idx}: {e}")
        
        if embedding_metrics_list:
            avg_embedding = aggregate_numeric_metrics(embedding_metrics_list)
            scheme_results['embedding'] = {
                'average': avg_embedding,
                'per_prompt': embedding_metrics_list
            }
            print(f"Embedding completed for {len(embedding_metrics_list)} prompt(s); average time {avg_embedding.get('embed_time_sec', 0):.4f}s")
        else:
            scheme_results['embedding'] = {'error': 'No successful embeddings'}
        
        if detection_metrics_list:
            avg_detection = aggregate_numeric_metrics(detection_metrics_list)
            scheme_results['detection'] = {
                'average': avg_detection,
                'per_prompt': detection_metrics_list
            }
            print(f"Detection completed for {len(detection_metrics_list)} prompt(s); average time {avg_detection.get('detect_time_sec', 0):.4f}s")
        else:
            scheme_results['detection'] = {'error': 'No detection results available'}
        
        if tracing_metrics_list:
            avg_tracing = aggregate_numeric_metrics(tracing_metrics_list)
            scheme_results['tracing'] = {
                'average': avg_tracing,
                'per_prompt': tracing_metrics_list
            }
            print(f"Tracing completed for {len(tracing_metrics_list)} prompt(s); average time {avg_tracing.get('trace_time_sec', 0):.4f}s")
        else:
            scheme_results['tracing'] = {'error': 'No tracing results available'}
        
        all_results[scheme_name] = scheme_results
    
    # Save results
    results_file = os.path.join(output_dir_path, 'performance_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=json_default_encoder)
    print(f"\nResults saved to: {results_file}")
    
    # Create summary CSV
    summary_data = []
    for scheme_tuple in schemes:
        scheme_name = scheme_tuple[0]
        if scheme_name not in all_results:
            continue
        results = all_results[scheme_name]
        
        # Create scheme label
        if scheme_name == 'hierarchical':
            scheme_label = f"hierarchical_G{scheme_tuple[1]}_U{scheme_tuple[2]}"
        else:
            scheme_label = scheme_name
        
        row = {'scheme': scheme_label}
        
        # Initialization
        if 'initialization' in results and 'error' not in results['initialization']:
            init = results['initialization']
            row['init_time_sec'] = init.get('init_time_sec', 0)
            row['init_memory_mb'] = init.get('memory_delta_mb', 0)
            row['storage_mb'] = init.get('storage_mb', 0)
            row['num_groups'] = init.get('num_groups', 0)
            row['users_per_group'] = init.get('users_per_group', 0)
        
        # Embedding
        if 'embedding' in results and 'error' not in results['embedding']:
            embed_avg = results['embedding'].get('average', {})
            row['embed_time_sec'] = embed_avg.get('embed_time_sec', 0)
            row['embed_overhead_percent'] = embed_avg.get('overhead_percent', 0)
            row['time_per_token_ms'] = embed_avg.get('time_per_token_ms', 0)
        
        # Detection
        if 'detection' in results and 'error' not in results['detection']:
            detect_avg = results['detection'].get('average', {})
            row['detect_time_sec'] = detect_avg.get('detect_time_sec', 0)
            row['hmac_operations'] = detect_avg.get('hmac_operations', 0)
        
        # Tracing
        if 'tracing' in results and 'error' not in results['tracing']:
            trace_avg = results['tracing'].get('average', {})
            row['trace_time_sec'] = trace_avg.get('trace_time_sec', 0)
            row['trace_comparisons'] = trace_avg.get('comparisons_count', 0)
        
        # Scalability
        if 'scalability' in results:
            scale = results['scalability']
            if scheme_name == 'naive':
                row['max_users'] = scale.get('max_users_naive', 0)
            else:  # hierarchical
                row['max_users'] = scale.get('max_users_hierarchical', 0)
                row['max_groups'] = scale.get('max_groups', 0)
                row['max_users_per_group'] = scale.get('max_users_per_group', 0)
                row['group_bits'] = scale.get('group_bits', 0)
                row['user_bits'] = scale.get('user_bits', 0)
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save local summary (in output directory)
    summary_file = os.path.join(output_dir_path, 'performance_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")
    
    # If unified summary path provided, append to it
    if args.unified_summary_path:
        unified_path = args.unified_summary_path
        if not os.path.isabs(unified_path):
            unified_path = os.path.join(parent_dir, unified_path)
        
        # Ensure parent directory exists
        unified_dir = os.path.dirname(unified_path)
        if unified_dir:  # Only create if there's a directory component
            os.makedirs(unified_dir, exist_ok=True)
        
        # Append to existing CSV or create new one
        if os.path.exists(unified_path):
            existing_df = pd.read_csv(unified_path)
            # Combine, avoiding duplicates based on scheme name
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            # Remove duplicates if any (keep last occurrence)
            combined_df = combined_df.drop_duplicates(subset=['scheme'], keep='last')
        else:
            combined_df = summary_df
        
        combined_df.to_csv(unified_path, index=False)
        print(f"Unified summary updated: {unified_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir_path}")


if __name__ == '__main__':
    main()

