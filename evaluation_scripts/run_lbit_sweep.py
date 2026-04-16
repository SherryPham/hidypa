# run_lbit_sweep.py
# Script to run L-bit embedding and detection for specified L values
# Uses prompts.txt and outputs JSONs and Excel summary

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


def load_existing_results(output_dir: str, exclude_l_values: list = None) -> list:
    """
    Load existing JSON results from previous runs.
    Returns a list of summary data entries from existing results.
    """
    if exclude_l_values is None:
        exclude_l_values = []
    
    existing_summary_data = []
    
    # Look for existing JSON files
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('lbit_L') and filename.endswith('_results.json'):
                # Extract L value from filename
                try:
                    L = int(filename.replace('lbit_L', '').replace('_results.json', ''))
                    # Skip if this L is in the exclude list (will be regenerated)
                    if L in exclude_l_values:
                        print(f"Skipping existing L={L} (will be regenerated)")
                        continue
                    
                    json_path = os.path.join(output_dir, filename)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        l_results = json.load(f)
                    
                    # Convert to summary format
                    for result in l_results:
                        if 'error' not in result:
                            accuracy_metrics = result.get('accuracy_metrics', {})
                            existing_summary_data.append({
                                'L': L,
                                'Prompt_ID': result.get('prompt_id', 0),
                                'Original_Message': result.get('original_message', ''),
                                'Recovered_Message': result.get('recovered_message', ''),
                                'Total_Bits': accuracy_metrics.get('total_bits', 0),
                                'Correct_Bits': accuracy_metrics.get('correct_bits', 0),
                                'Undecided_Bits': accuracy_metrics.get('undecided_bits', 0),
                                'Decided_Bits': accuracy_metrics.get('decided_bits', 0),
                                'Accuracy_Percent': accuracy_metrics.get('accuracy_percent', 0.0),
                                'Exact_Match': accuracy_metrics.get('exact_match', False)
                            })
                        else:
                            existing_summary_data.append({
                                'L': L,
                                'Prompt_ID': result.get('prompt_id', 0),
                                'Original_Message': None,
                                'Recovered_Message': None,
                                'Total_Bits': None,
                                'Correct_Bits': None,
                                'Undecided_Bits': None,
                                'Decided_Bits': None,
                                'Accuracy_Percent': None,
                                'Exact_Match': False,
                                'Error': result.get('error', 'Unknown error')
                            })
                    
                    print(f"Loaded existing results for L={L} from {filename}")
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {filename}: {e}")
                    continue
    
    return existing_summary_data


def main():
    parser = argparse.ArgumentParser(
        description="Run L-bit embedding and detection sweep for specified L values",
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
        help='Limit on number of prompts to process (default: 100; set <=0 for all prompts)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b',
                 'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.1-8b',
                 'opt-125m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b'],
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
        default='evaluation/lbit_sweep',
        help='Output directory for results (default: evaluation/lbit_sweep)'
    )
    parser.add_argument(
        '--min-l',
        type=int,
        default=None,
        help='Minimum L value (used if --l-values not specified, default: 4)'
    )
    parser.add_argument(
        '--max-l',
        type=int,
        default=None,
        help='Maximum L value (used if --l-values not specified, default: 20)'
    )
    parser.add_argument(
        '--l-values',
        type=int,
        nargs='+',
        default=None,
        help='Specific L values to test (e.g., --l-values 4 5 6 7). If specified, overrides --min-l and --max-l'
    )
    parser.add_argument(
        '--csv-only',
        action='store_true',
        help='Only create CSV/Excel summary from existing JSON files, do not run new evaluations'
    )
    
    args = parser.parse_args()
    
    # Create output directory (relative to project root)
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(parent_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # If csv-only mode, just load all existing results and create summary
    if args.csv_only:
        print("="*60)
        print("CSV-Only Mode: Creating summary from existing JSON files")
        print("="*60)
        
        # Load ALL existing results (no exclusions)
        print(f"\nLoading all existing results from {output_dir}...")
        existing_summary_data = load_existing_results(output_dir, exclude_l_values=[])
        
        if not existing_summary_data:
            print(f"No existing JSON files found in {output_dir}")
            return
        
        print(f"Loaded {len(existing_summary_data)} existing result entries")
        
        # Create summary from existing data only
        all_summary_data = existing_summary_data
        df = pd.DataFrame(all_summary_data)
        
        # Get all unique L values from the data
        all_l_values = sorted(df['L'].unique())
        print(f"\nFound results for L values: {all_l_values}")
        
        # Create summary statistics by L
        summary_stats = []
        for L in all_l_values:
            l_data = df[df['L'] == L]
            if len(l_data) > 0:
                # Filter out any rows with errors (they won't have accuracy metrics)
                successful = l_data.dropna(subset=['Accuracy_Percent'])
                if len(successful) > 0 and 'Accuracy_Percent' in successful.columns:
                    summary_stats.append({
                        'L': L,
                        'Total_Prompts': len(l_data),
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
        
        # Save as CSV files (not Excel)
        csv_path = os.path.join(output_dir, 'lbit_sweep_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"CSV summary saved to: {csv_path}")
        
        if summary_stats:
            summary_csv_path = os.path.join(output_dir, 'lbit_sweep_summary_by_L.csv')
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Summary by L saved to: {summary_csv_path}")
        
        print(f"{'='*60}")
        
        # Print summary table
        if summary_stats:
            print("\nSummary Statistics by L:")
            summary_df = pd.DataFrame(summary_stats)
            print(summary_df.to_string(index=False))
        
        print(f"\nAll results saved to: {output_dir}")
        return
    
    # Determine which L values to use
    if args.l_values is not None:
        # Use specified L values
        l_values = sorted(set(args.l_values))  # Remove duplicates and sort
        # Validate L values
        if any(L < 1 or L > 64 for L in l_values):
            print("Error: All L values must be between 1 and 64")
            return
    else:
        # Use range from min-l to max-l
        min_l = args.min_l if args.min_l is not None else 4
        max_l = args.max_l if args.max_l is not None else 20
        
        # Validate L range
        if min_l < 1 or max_l > 64 or min_l > max_l:
            print("Error: Invalid L range. min_l must be >= 1, max_l must be <= 64, and min_l <= max_l")
            return
        
        l_values = list(range(min_l, max_l + 1))
    
    print(f"Will test L values: {l_values}")
    
    # Load existing results (excluding L values that will be regenerated)
    print(f"\nLoading existing results from {output_dir}...")
    existing_summary_data = load_existing_results(output_dir, exclude_l_values=l_values)
    if existing_summary_data:
        print(f"Loaded {len(existing_summary_data)} existing result entries")
    
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
    
    # Load model
    print(f"Loading model '{args.model}'...")
    model = get_model(args.model)
    
    # Summary data for Excel
    summary_data = []
    
    # Loop through L values
    for L in l_values:
        print(f"\n{'='*60}")
        print(f"Processing L = {L}")
        print(f"{'='*60}")
        
        # Initialize watermarker for this L
        zero_bit = ZeroBitWatermarker(
            model=model,
            delta=args.delta,
            entropy_threshold=args.entropy_threshold,
            hashing_context=args.hashing_context,
            z_threshold=args.z_threshold
        )
        lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zero_bit, L=L)
        
        # Results for this L
        l_results = []
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"L={L}")):
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
                    'original_message': message,
                    'recovered_message': recovered_message,
                    'accuracy_metrics': accuracy_metrics,
                    'parameters': {
                        'delta': args.delta,
                        'entropy_threshold': args.entropy_threshold,
                        'hashing_context': args.hashing_context,
                        'z_threshold': args.z_threshold,
                        'max_new_tokens': args.max_new_tokens
                    }
                }
                
                l_results.append(result)
                
                # Add to summary
                summary_data.append({
                    'L': L,
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
                
            except Exception as e:
                print(f"\nError processing prompt {prompt_idx} for L={L}: {e}")
                result = {
                    'prompt_id': prompt_idx,
                    'prompt': prompt,
                    'L': L,
                    'error': str(e)
                }
                l_results.append(result)
                
                # Add error to summary
                summary_data.append({
                    'L': L,
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
                continue
        
        # Save JSON for this L
        json_path = os.path.join(output_dir, f'lbit_L{L}_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(l_results, f, indent=2)
        print(f"\nSaved results for L={L} to {json_path}")
        
        # Calculate summary statistics for this L
        successful_results = [r for r in l_results if 'error' not in r]
        if successful_results:
            accuracies = [r['accuracy_metrics']['accuracy_percent'] for r in successful_results]
            exact_matches = sum(1 for r in successful_results if r['accuracy_metrics']['exact_match'])
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            exact_match_rate = (exact_matches / len(successful_results)) * 100.0 if successful_results else 0.0
            
            print(f"L={L} Summary:")
            print(f"  Total prompts: {len(l_results)}")
            print(f"  Successful: {len(successful_results)}")
            print(f"  Average accuracy: {avg_accuracy:.2f}%")
            print(f"  Exact match rate: {exact_match_rate:.2f}%")
    
    # Create CSV summary (merge existing and new results)
    all_summary_data = existing_summary_data + summary_data
    if not summary_data and existing_summary_data:
        print(f"\nNo new results generated, but found {len(existing_summary_data)} existing result entries")
    if all_summary_data:
        df = pd.DataFrame(all_summary_data)
        
        # Get all unique L values from the merged data
        all_l_values = sorted(df['L'].unique())
        print(f"\nCreating CSV summary with L values: {all_l_values}")
        
        # Create summary statistics by L
        summary_stats = []
        for L in all_l_values:
            l_data = df[df['L'] == L]
            if len(l_data) > 0:
                # Filter out any rows with errors (they won't have accuracy metrics)
                successful = l_data.dropna(subset=['Accuracy_Percent'])
                if len(successful) > 0 and 'Accuracy_Percent' in successful.columns:
                    summary_stats.append({
                        'L': L,
                        'Total_Prompts': len(l_data),
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
        
        # Save CSV files
        csv_path = os.path.join(output_dir, 'lbit_sweep_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"CSV summary saved to: {csv_path}")
        
        if summary_stats:
            summary_csv_path = os.path.join(output_dir, 'lbit_sweep_summary_by_L.csv')
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Summary by L saved to: {summary_csv_path}")
        print(f"{'='*60}")
        
        # Print summary table
        if summary_stats:
            print("\nSummary Statistics by L:")
            summary_df = pd.DataFrame(summary_stats)
            print(summary_df.to_string(index=False))
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

