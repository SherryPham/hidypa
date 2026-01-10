# run_lbit_parameter_sweep.py
# Script to run L-bit embedding and detection for L = 8 (fixed)
# Sweeps through different delta and entropy_threshold values
# Uses prompts.txt and outputs JSONs and CSV summary

import argparse
import json
import os
import random
import sys
from tqdm import tqdm
import pandas as pd

# Add the parent directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import LBitWatermarker, ZeroBitWatermarker
from src.utils import get_model, parse_final_output


def generate_random_message(L: int) -> str:
    """Generate a random L-bit binary message."""
    return ''.join(random.choice(['0', '1']) for _ in range(L))


def calculate_bit_accuracy(original: str, recovered: str) -> dict:
    """
    Calculate bit accuracy metrics.
    Returns dict with: total_bits, correct_bits, undecided_bits, accuracy_percent
    """
    if len(original) != len(recovered):
        return {
            'total_bits': len(original),
            'correct_bits': 0,
            'undecided_bits': 0,
            'accuracy_percent': 0.0,
            'exact_match': False
        }
    
    correct = 0
    undecided = 0
    
    for orig_bit, rec_bit in zip(original, recovered):
        if rec_bit == '⊥' or rec_bit == '*':
            undecided += 1
        elif orig_bit == rec_bit:
            correct += 1
    
    total_decided = len(original) - undecided
    accuracy = (correct / len(original)) * 100.0 if len(original) > 0 else 0.0
    
    return {
        'total_bits': len(original),
        'correct_bits': correct,
        'undecided_bits': undecided,
        'decided_bits': total_decided,
        'accuracy_percent': accuracy,
        'exact_match': original == recovered
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run L-bit parameter sweep for L = 8 (fixed) with different delta and entropy_threshold values",
        formatter_class=argparse.RawTextHelpFormatter
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
        help='Cap on number of prompts to process (default: 100; set <=0 for all prompts)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
        help='Model to use for generation and detection'
    )
    parser.add_argument(
        '--deltas',
        type=float,
        nargs='+',
        default=[2.0, 2.5, 3.0, 3.5, 4.0],
        help='List of delta values to test (default: 2.0 2.5 3.0 3.5 4.0)'
    )
    parser.add_argument(
        '--entropy-thresholds',
        type=float,
        nargs='+',
        default=[1.5, 2.0, 2.5, 3.0, 3.5],
        help='List of entropy threshold values to test (default: 1.5 2.0 2.5 3.0 3.5)'
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
        default='evaluation/lbit_parameter_sweep',
        help='Output directory for results (default: evaluation/lbit_parameter_sweep)'
    )
    parser.add_argument(
        '--min-l',
        type=int,
        default=8,
        help='Minimum L value (default: 8)'
    )
    parser.add_argument(
        '--max-l',
        type=int,
        default=8,
        help='Maximum L value (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Validate L range
    if args.min_l < 1 or args.max_l > 64 or args.min_l > args.max_l:
        print("Error: Invalid L range. min_l must be >= 1, max_l must be <= 64, and min_l <= max_l")
        return
    
    # Create output directory (relative to project root)
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(parent_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompts
    prompts_path = os.path.join(parent_dir, args.prompts_file)
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file not found: {prompts_path}")
        return
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    total_prompts = len(prompts)
    if args.max_prompts and args.max_prompts > 0 and args.max_prompts < total_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Loaded {total_prompts} prompts from {prompts_path}. Limiting run to first {len(prompts)} prompts.")
    else:
        print(f"Loaded {total_prompts} prompts from {prompts_path}")
    print(f"Testing {len(args.deltas)} delta values: {args.deltas}")
    print(f"Testing {len(args.entropy_thresholds)} entropy threshold values: {args.entropy_thresholds}")
    print(f"Testing L values: {list(range(args.min_l, args.max_l + 1))}")
    print(f"Total parameter combinations: {len(args.deltas) * len(args.entropy_thresholds) * (args.max_l - args.min_l + 1)}")
    
    # Load model
    print(f"\nLoading model '{args.model}'...")
    model = get_model(args.model)
    
    # Summary data for CSV
    summary_data = []
    
    # Calculate total iterations for progress tracking
    total_combinations = len(args.deltas) * len(args.entropy_thresholds) * (args.max_l - args.min_l + 1) * len(prompts)
    current_iteration = 0
    
    # Loop through all parameter combinations
    for L in range(args.min_l, args.max_l + 1):
        for delta in args.deltas:
            for entropy_threshold in args.entropy_thresholds:
                print(f"\n{'='*60}")
                print(f"Processing L={L}, delta={delta}, entropy_threshold={entropy_threshold}")
                print(f"{'='*60}")
                
                # Initialize watermarker for this parameter combination
                zero_bit = ZeroBitWatermarker(
                    model=model,
                    delta=delta,
                    entropy_threshold=entropy_threshold,
                    hashing_context=args.hashing_context,
                    z_threshold=args.z_threshold
                )
                lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zero_bit, L=L)
                
                # Results for this parameter combination
                param_results = []
                
                # Process each prompt
                for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"L={L}, δ={delta}, ε={entropy_threshold}")):
                    try:
                        # Generate random message
                        message = generate_random_message(L)
                        
                        # Generate master key
                        master_key = lbit_watermarker.keygen()
                        
                        # Embed watermark
                        raw_text = lbit_watermarker.embed(
                            master_key,
                            message,
                            prompt,
                            args.max_new_tokens
                        )
                        
                        # Parse output
                        watermarked_text = parse_final_output(raw_text, args.model)
                        
                        # Detect watermark
                        recovered_message = lbit_watermarker.detect(master_key, watermarked_text)
                        
                        # Calculate accuracy
                        accuracy_metrics = calculate_bit_accuracy(message, recovered_message)
                        
                        # Store result
                        result = {
                            'prompt_id': prompt_idx,
                            'prompt': prompt,
                            'L': L,
                            'delta': delta,
                            'entropy_threshold': entropy_threshold,
                            'original_message': message,
                            'recovered_message': recovered_message,
                            'accuracy_metrics': accuracy_metrics,
                            'parameters': {
                                'delta': delta,
                                'entropy_threshold': entropy_threshold,
                                'hashing_context': args.hashing_context,
                                'z_threshold': args.z_threshold,
                                'max_new_tokens': args.max_new_tokens
                            }
                        }
                        
                        param_results.append(result)
                        
                        # Add to summary
                        summary_data.append({
                            'L': L,
                            'Delta': delta,
                            'Entropy_Threshold': entropy_threshold,
                            'Prompt_ID': prompt_idx,
                            'Original_Message': message,
                            'Recovered_Message': recovered_message,
                            'Total_Bits': accuracy_metrics['total_bits'],
                            'Correct_Bits': accuracy_metrics['correct_bits'],
                            'Undecided_Bits': accuracy_metrics['undecided_bits'],
                            'Decided_Bits': accuracy_metrics['decided_bits'],
                            'Accuracy_Percent': accuracy_metrics['accuracy_percent'],
                            'Exact_Match': accuracy_metrics['exact_match']
                        })
                        
                        current_iteration += 1
                        
                    except Exception as e:
                        print(f"\nError processing prompt {prompt_idx} for L={L}, delta={delta}, entropy={entropy_threshold}: {e}")
                        result = {
                            'prompt_id': prompt_idx,
                            'prompt': prompt,
                            'L': L,
                            'delta': delta,
                            'entropy_threshold': entropy_threshold,
                            'error': str(e)
                        }
                        param_results.append(result)
                        
                        # Add error to summary
                        summary_data.append({
                            'L': L,
                            'Delta': delta,
                            'Entropy_Threshold': entropy_threshold,
                            'Prompt_ID': prompt_idx,
                            'Original_Message': None,
                            'Recovered_Message': None,
                            'Total_Bits': None,
                            'Correct_Bits': None,
                            'Undecided_Bits': None,
                            'Decided_Bits': None,
                            'Accuracy_Percent': None,
                            'Exact_Match': False,
                            'Error': str(e)
                        })
                        current_iteration += 1
                        continue
                
                # Save JSON for this parameter combination
                json_filename = f'lbit_L{L}_delta{delta}_entropy{entropy_threshold}_results.json'
                json_path = os.path.join(output_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(param_results, f, indent=2)
                print(f"\nSaved results to {json_path}")
                
                # Calculate summary statistics for this parameter combination
                successful_results = [r for r in param_results if 'error' not in r]
                if successful_results:
                    accuracies = [r['accuracy_metrics']['accuracy_percent'] for r in successful_results]
                    exact_matches = sum(1 for r in successful_results if r['accuracy_metrics']['exact_match'])
                    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                    exact_match_rate = (exact_matches / len(successful_results)) * 100.0 if successful_results else 0.0
                    
                    print(f"Summary for L={L}, δ={delta}, ε={entropy_threshold}:")
                    print(f"  Total prompts: {len(param_results)}")
                    print(f"  Successful: {len(successful_results)}")
                    print(f"  Average accuracy: {avg_accuracy:.2f}%")
                    print(f"  Exact match rate: {exact_match_rate:.2f}%")
    
    # Create CSV summary
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Save detailed CSV file
        csv_path = os.path.join(output_dir, 'lbit_parameter_sweep_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Detailed CSV summary saved to: {csv_path}")
        
        # Create summary statistics by parameter combination
        summary_stats = []
        for L in range(args.min_l, args.max_l + 1):
            for delta in args.deltas:
                for entropy_threshold in args.entropy_thresholds:
                    param_data = df[(df['L'] == L) & (df['Delta'] == delta) & (df['Entropy_Threshold'] == entropy_threshold)]
                    if len(param_data) > 0:
                        # Filter out any rows with errors
                        successful = param_data.dropna(subset=['Accuracy_Percent'])
                        if len(successful) > 0 and 'Accuracy_Percent' in successful.columns:
                            summary_stats.append({
                                'L': L,
                                'Delta': delta,
                                'Entropy_Threshold': entropy_threshold,
                                'Total_Prompts': len(param_data),
                                'Successful': len(successful),
                                'Avg_Accuracy_Percent': successful['Accuracy_Percent'].mean(),
                                'Median_Accuracy_Percent': successful['Accuracy_Percent'].median(),
                                'Min_Accuracy_Percent': successful['Accuracy_Percent'].min(),
                                'Max_Accuracy_Percent': successful['Accuracy_Percent'].max(),
                                'Exact_Matches': successful['Exact_Match'].sum() if 'Exact_Match' in successful.columns else 0,
                                'Exact_Match_Rate_Percent': (successful['Exact_Match'].sum() / len(successful)) * 100.0 if 'Exact_Match' in successful.columns else 0.0,
                                'Avg_Undecided_Bits': successful['Undecided_Bits'].mean() if 'Undecided_Bits' in successful.columns else 0.0,
                                'Avg_Decided_Bits': successful['Decided_Bits'].mean() if 'Decided_Bits' in successful.columns else 0.0
                            })
        
        # Save summary statistics CSV
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_csv_path = os.path.join(output_dir, 'lbit_parameter_sweep_summary_by_params.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Summary statistics CSV saved to: {summary_csv_path}")
            print(f"{'='*60}")
            
            # Print summary table
            print("\nSummary Statistics by Parameter Combination:")
            print(summary_df.to_string(index=False))
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

