# main_multiuser.py

import argparse
import json
import os
import sys

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import GPT2Model, GptOssModel, GptOss120bModel
from src.watermark import (
    ZeroBitWatermarker,
    LBitWatermarker,
    NaiveMultiUserWatermarker,
    GroupedMultiUserWatermarker,
    HiDyPaMultiUserWatermarker,
)

def parse_final_output(raw_text: str, model_name: str) -> str:
    """
    Extracts the content from the 'final' channel of a gpt-oss output.
    If the model is not gpt-oss or the separator isn't found, it returns the original text.
    """
    if 'gpt-oss' in model_name:
        separator = "assistantfinal"
        if separator in raw_text:
            # Return the part of the string that comes after the last occurrence of the separator
            return raw_text.split(separator)[-1].strip()
    
    # For non-gpt-oss models or if separator is missing, return the full text
    return raw_text

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

def main():
    parser = argparse.ArgumentParser(description="Multi-User Watermarking Tool (naive + grouped schemes).")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Base arguments for both commands ---
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--users-file', type=str, default='assets/users.csv', help="Path to the user metadata CSV file.")
    base_parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'])
    base_parser.add_argument('--key-file', '-k', type=str, default='demonstration/multiuser_master.key', help="Path to the master secret key.")
    # L-bit parameters
    base_parser.add_argument('--delta', type=float, default=3.5)
    base_parser.add_argument('--entropy-threshold', type=float, default=2.5)
    base_parser.add_argument('--hashing-context', type=int, default=5)
    base_parser.add_argument('--z-threshold', type=float, default=4.0)
    base_parser.add_argument('--l-bits', type=int, default=10)
    base_parser.add_argument(
        '--scheme',
        type=str,
        default='grouped',
        choices=['naive', 'grouped', 'hi_dypa'],
        help="Multi-user scheme to use (default: grouped).",
    )
    base_parser.add_argument('--min-distance', type=int, default=2, choices=[2, 3],
                            help="Minimum Hamming distance between codewords for collusion resistance (default: 2).")
    base_parser.add_argument('--max-groups', type=int, default=None,
                            help="Maximum number of groups allowed (default: auto-calculated based on min-distance).")
    base_parser.add_argument('--users-per-group', type=int, default=None,
                            help="Number of users per group (default: auto-calculated based on min-distance).")
    base_parser.add_argument('--group-bits', type=int, default=None,
                            help="Number of bits for group codewords (required for Hi-DyPa scheme).")
    base_parser.add_argument('--user-bits', type=int, default=None,
                            help="Number of bits for user fingerprints within groups (required for Hi-DyPa scheme).")

    # --- Generate Command ---
    gen_parser = subparsers.add_parser('generate', help='Generate text watermarked for a specific user.', parents=[base_parser])
    gen_parser.add_argument('prompt', type=str, nargs='?', help="Direct prompt text (optional if --prompt-file is used).")
    gen_parser.add_argument('--prompt-file', type=str, help="Path to a text file containing the prompt (e.g., previous user's output).")
    gen_parser.add_argument('--prompt-suffix', type=str, help="Additional text to append after the prompt file content (e.g., 'Rewrite the paragraph above and add...').")
    gen_parser.add_argument('--user-id', type=int, required=True, help="Integer ID of the user to watermark for (e.g., 2).")
    gen_parser.add_argument('--max-new-tokens', type=int, default=512)
    gen_parser.add_argument('--output-file', '-o', type=str, default='demonstration/multiuser_output.txt')

    # --- Trace Command ---
    trace_parser = subparsers.add_parser('trace', help='Trace a watermarked text back to a user ID.', parents=[base_parser])
    trace_parser.add_argument('input_file', type=str)
    
    args = parser.parse_args()

    # --- Setup Watermarking Stack ---
    print(f"Loading model '{args.model}'...")
    model = get_model(args.model)

    zbw = ZeroBitWatermarker(
        model=model, 
        delta=args.delta, 
        entropy_threshold=args.entropy_threshold, 
        z_threshold=args.z_threshold,
        hashing_context=args.hashing_context
    )
    lbw = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)
    
    if args.scheme == 'hi_dypa':
        if args.group_bits is None or args.user_bits is None:
            print("Error: --group-bits and --user-bits are required for Hi-DyPa scheme.")
            return
        if args.group_bits + args.user_bits != args.l_bits:
            print(f"Error: --group-bits ({args.group_bits}) + --user-bits ({args.user_bits}) must equal --l-bits ({args.l_bits}).")
            return
        muw = HiDyPaMultiUserWatermarker(
            lbit_watermarker=lbw,
            group_bits=args.group_bits,
            user_bits=args.user_bits,
            min_distance=args.min_distance,
            max_groups=args.max_groups,
            users_per_group=args.users_per_group
        )
    elif args.scheme == 'grouped':
        muw = GroupedMultiUserWatermarker(
            lbit_watermarker=lbw, 
            min_distance=args.min_distance,
            max_groups=args.max_groups,
            users_per_group=args.users_per_group
        )
    else:
        muw = NaiveMultiUserWatermarker(lbit_watermarker=lbw)

    # --- Command Logic ---
    try:
        muw.load_users(args.users_file)
        print(f"Loaded {muw.N} users from {args.users_file} using '{args.scheme}' scheme")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error initializing multi-user scheme: {e}")
        return

    if args.command == 'generate':
        # Build the prompt from file and/or direct text
        prompt_parts = []
        
        if args.prompt_file:
            if not os.path.exists(args.prompt_file):
                print(f"Error: Prompt file '{args.prompt_file}' not found.")
                return
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                prompt_parts.append(file_content)
        
        if args.prompt_suffix:
            prompt_parts.append(args.prompt_suffix)
        
        if args.prompt:
            # If both prompt file and direct prompt are provided, direct prompt takes precedence
            if args.prompt_file:
                print("Warning: Both --prompt-file and direct prompt provided. Using direct prompt.")
            prompt_parts = [args.prompt]
        
        if not prompt_parts:
            print("Error: Must provide either 'prompt' argument, --prompt-file, or both --prompt-file and --prompt-suffix.")
            return
        
        # Join prompt parts with a space (or newline if file + suffix)
        if len(prompt_parts) > 1 and args.prompt_file and args.prompt_suffix:
            # File content + suffix: join with newline for better readability
            final_prompt = "\n\n".join(prompt_parts)
        else:
            final_prompt = prompt_parts[0]
        
        if args.prompt_file:
            print(f"Loaded prompt from file: {args.prompt_file}")
        if args.prompt_suffix:
            print(f"Appended suffix to prompt")
        
        # Load existing master key if it exists, otherwise generate a new one
        if os.path.exists(args.key_file):
            with open(args.key_file, 'r') as f:
                master_key_hex = f.read().strip()
                master_key = bytes.fromhex(master_key_hex)
            print(f"Loaded existing master key from {args.key_file}")
        else:
            master_key = muw.keygen()
            with open(args.key_file, 'w') as f:
                f.write(master_key.hex())
            print(f"Generated new master key and saved to {args.key_file}")
        
        raw_text = muw.embed(master_key, args.user_id, final_prompt, max_new_tokens=args.max_new_tokens)
        final_text = parse_final_output(raw_text, args.model.__class__.__name__.lower())
        
        print("\n--- Final Watermarked Response ---")
        print(final_text)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
            
        print(f"\nOutput for User ID {args.user_id} saved to {args.output_file}")

    elif args.command == 'trace':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text_to_trace = f.read()
        with open(args.key_file, 'r') as f:
            master_key_hex = f.read()
        master_key = bytes.fromhex(master_key_hex)
        
        accused_users = muw.trace(master_key, text_to_trace)
        
        print("\n--- Trace Results ---")
        if accused_users:
            total_users = len(accused_users)
            group_ids = [u.get('group_id') for u in accused_users if u.get('group_id') is not None]
            unique_groups = len(set(group_ids)) if group_ids else 0
            
            print(f"  Users detected: {total_users}")
            print(f"  Groups detected: {unique_groups}")
            print(f"  Details:")
            
            has_group_info = unique_groups > 0
            if has_group_info:
                grouped_users = {}
                for user in accused_users:
                    gid = user.get('group_id')
                    grouped_users.setdefault(gid, []).append(user)
                
                for gid in sorted(grouped_users.keys(), key=lambda x: (-1 if x is None else x)):
                    members = grouped_users[gid]
                    username_label = f"Group {gid}" if gid is not None else "Ungrouped users"
                    print(f"     {username_label}: {len(members)} user(s)")
                    
                    for entry in members[:20]:
                        username = entry.get('username') or 'N/A'
                        print(f"        - User ID: {entry['user_id']}, Username: {username}, "
                              f"Match: {entry['match_score_percent']:.2f}%")
                    
                    if gid is not None and len(members) > 20:
                        remaining = len(members) - 20
                        print(f"        ... all {len(members)} users belong to Group {gid} "
                              f"(showing first 20)")
                    
                    if gid is not None and len(members) > 1:
                        print("        Note: Users in the same group share an identical codeword, "
                              "so they tie when traced.")
            else:
                for user in accused_users:
                    username = user.get('username') or 'N/A'
                    print(f"     - User ID: {user['user_id']}, Username: {username}, "
                          f"Match: {user['match_score_percent']:.2f}%")
        else:
            print("  Could not confidently trace text to any user.")
        print("---------------------")

if __name__ == "__main__":
    main()