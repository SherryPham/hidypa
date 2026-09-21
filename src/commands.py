
import os
import json
from tqdm import tqdm

from src.watermark import ZeroBitWatermarker, LBitWatermarker, derive_key
from src.utils import (
    get_model, parse_final_output, generate_unwatermarked,
    get_paraphraser, perturb_text, parse_filename
)


def cmd_generate(args):
    """Handle the 'generate' command."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.key_file) if os.path.dirname(args.key_file) else '.', exist_ok=True)
    
    print(f"Loading model '{args.model}'...")
    model = get_model(args.model)
    
    print(f"\nPrompt: '{args.prompt}'")
    
    if args.no_watermark:
        final_text = generate_unwatermarked(model, args.prompt, args.max_new_tokens, args.model)
        print("\n--- Generated Text ---")
        print(final_text)
        print("------------------------")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"\nOutput saved to {args.output_file}")
        
    else:
        watermarker = ZeroBitWatermarker(
            model=model, 
            delta=args.delta, 
            entropy_threshold=args.entropy_threshold, 
            hashing_context=args.hashing_context
        )
        secret_key = watermarker.keygen()
        
        print("\nEmbedding watermark...")
        # Unpack the text and the final parameters used
        raw_watermarked_text, final_params = watermarker.embed(secret_key, args.prompt, args.max_new_tokens)
        
        # Parse the raw output to get the final answer
        final_text = parse_final_output(raw_watermarked_text, args.model)
        
        print("\n--- Final Watermarked Response ---")
        print(final_text)
        print("----------------------------------")
        print(f"(Final parameters used: {final_params})")
        
        # Save the parsed output and key
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"\nWatermarked text saved to {args.output_file}")
        
        with open(args.key_file, 'wb') as f:
            f.write(secret_key)
        print(f"Secret key saved to {args.key_file}")


def cmd_detect(args):
    """Handle the 'detect' command."""
    print(f"Loading model '{args.model}' for tokenizer...")
    model = get_model(args.model)
    
    try:
        with open(args.key_file, 'rb') as f:
            secret_key = f.read()
        print(f"Loaded secret key from {args.key_file}")

        with open(args.input_file, 'r', encoding='utf-8') as f:
            text_to_check = f.read()
        print(f"Loaded text from {args.input_file}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return
        
    print("\nRunning detection algorithm...")

    pass_1_params = {
        'delta': 3.5,
        'hashing_context': args.hashing_context,
        'entropy_threshold': args.entropy_threshold
    }
    watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **pass_1_params)
    z_score, is_detected, block_count = watermarker_pass1.detect(secret_key, text_to_check)
    
    final_params = pass_1_params

    if block_count < 75 and not is_detected:
        print(f"Initial block count ({block_count}) is low. Running a more aggressive second pass...")
        
        pass_2_params = pass_1_params.copy()
        pass_2_params['entropy_threshold'] -= 2.0

        # Check if the new parameters are within the valid ranges
        if pass_2_params['entropy_threshold'] >= 1.0:
            watermarker_pass2 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **pass_2_params)
            # Overwrite the results with the second pass
            z_score, is_detected, block_count = watermarker_pass2.detect(secret_key, text_to_check)
            final_params = pass_2_params

    print("\n--- Detection Results ---")
    print(f"  Z-Score: {z_score:.4f}")
    print(f"  Threshold: {args.z_threshold}")
    print(f"  Detected: {'Yes' if is_detected else 'No'}")
    print(f"  Blocks Found: {block_count}")
    print(f"  Final Params Used: {final_params}")
    print("-------------------------")


def cmd_generate_lbit(args):
    """Handle the 'generate_lbit' command."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.key_file) if os.path.dirname(args.key_file) else '.', exist_ok=True)
    
    model = get_model(args.model)

    zbw = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        hashing_context=args.hashing_context,
        z_threshold=args.z_threshold
    )
    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)

    print(f"\nGenerating key for an {args.l_bits}-bit message...")
    master_secret_key = lbit_watermarker.keygen()  # Single key
    print(f"Message to embed: {args.message}")

    print(f"\nPrompt: '{args.prompt}'")
    print("\nEmbedding message into generated text...")
    watermarked_text = lbit_watermarker.embed(master_secret_key, args.message, args.prompt, args.max_new_tokens)

    print("\n--- Watermarked Text ---")
    print(watermarked_text)
    print("------------------------")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(watermarked_text)

    with open(args.key_file, 'w') as f:
        json.dump(master_secret_key.hex(), f)  # Save single key as hex
    print(f"Watermarked L-bit text saved to {args.output_file}")
    print(f"Secret key saved to {args.key_file}")


def cmd_detect_lbit(args):
    """Handle the 'detect_lbit' command."""
    model = get_model(args.model)
    zbw = ZeroBitWatermarker(
        model=model,
        delta=args.delta,
        entropy_threshold=args.entropy_threshold,
        hashing_context=args.hashing_context,
        z_threshold=args.z_threshold
    )
    lbit_watermarker = LBitWatermarker(zero_bit_watermarker=zbw, L=args.l_bits)

    try:
        with open(args.key_file, 'r') as f:
            secret_key_hex = json.load(f)
        print(f"Loaded secret key from {args.key_file}")

        master_secret_key = bytes.fromhex(secret_key_hex)  # Single key

        with open(args.input_file, 'r', encoding='utf-8') as f:
            text_to_check = f.read()
        print(f"Loaded text from {args.input_file}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return
    except (json.JSONDecodeError, ValueError):
        print(f"Error: Could not parse the key file '{args.key_file}'. It may be corrupted or in the wrong format.")
        return

    print(f"\nExtracting {args.l_bits}-bit message...")
    recovered_message = lbit_watermarker.detect(master_secret_key, text_to_check)

    print("\n--- Extraction Results ---")
    print(f"  Recovered L-bit message: {recovered_message}")
    print("--------------------------")


def cmd_evaluate(args):
    """Handle the 'evaluate' command."""
    print("--- Starting End-to-End Evaluation ---")
    # Ensure output directory exists (already creates it, but ensure parent exists too)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Stage 1: Setup ---
    model = get_model(args.model)
    paraphraser_model, paraphraser_tokenizer = get_paraphraser(model.device)
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    total_prompts = len(prompts)
    if args.max_prompts and args.max_prompts > 0 and args.max_prompts < total_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Found {total_prompts} prompts. Limiting evaluation to first {len(prompts)}.")
    else:
        print(f"Found {total_prompts} prompts. Using all for evaluation.")

    if args.deltas:
        param_to_sweep, sweep_values = 'delta', args.deltas
    elif args.hashing_contexts:
        param_to_sweep, sweep_values = 'hashing_context', args.hashing_contexts
    else:
        param_to_sweep, sweep_values = 'entropy_threshold', args.entropy_thresholds
    
    print(f"Sweeping parameter: '{param_to_sweep}' with values: {sweep_values}")
    key_map = {}

    # --- Stage 2: Generation and Perturbation ---
    for i, prompt in enumerate(tqdm(prompts, desc="Generating Texts")):
        unwatermarked_text = generate_unwatermarked(model, prompt, args.max_new_tokens, args.model)
        with open(os.path.join(args.output_dir, f"{i}_unwatermarked.txt"), 'w', encoding='utf-8') as f:
            f.write(unwatermarked_text)
        
        for value in sweep_values:
            params = {
                'delta': value if param_to_sweep == 'delta' else args.delta,
                'hashing_context': value if param_to_sweep == 'hashing_context' else args.hashing_context,
                'entropy_threshold': value if param_to_sweep == 'entropy_threshold' else args.entropy_threshold
            }
            watermarker = ZeroBitWatermarker(model=model, **params)
            secret_key = watermarker.keygen()
            
            # The embed function now returns the text AND the final params used
            raw_watermarked_text, final_params = watermarker.embed(secret_key, prompt, args.max_new_tokens)
            
            clean_final_text = parse_final_output(raw_watermarked_text, args.model)
            perturbed_texts = perturb_text(clean_final_text, paraphraser_model, paraphraser_tokenizer, model.device)
            
            # Filename now uses the initial parameters from the embedding stage
            base_filename = f"{i}_wm_delta_{value if param_to_sweep == 'delta' else args.delta:.1f}_hc_{value if param_to_sweep == 'hashing_context' else args.hashing_context}_et_{value if param_to_sweep == 'entropy_threshold' else args.entropy_threshold:.1f}"

            for p_name, p_text in [('clean', clean_final_text)] + list(perturbed_texts.items()):
                filename = f"{base_filename}_{p_name.replace(' ', '_')}.txt"
                key_map[filename] = secret_key.hex()
                with open(os.path.join(args.output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(p_text)
            
            if args.l_bit_message:
                if len(args.l_bit_message) != args.l_bits or any(c not in {'0', '1'} for c in args.l_bit_message):
                    print(f"Error: L-bit message must be a {args.l_bits}-bit string")
                    continue
                lbit_watermarker = LBitWatermarker(zero_bit_watermarker=watermarker, L=args.l_bits)
                master_secret_key = lbit_watermarker.keygen()
                lbit_text = lbit_watermarker.embed(master_secret_key, args.l_bit_message, prompt, args.max_new_tokens)
                lbit_clean_text = parse_final_output(lbit_text, args.model)
                lbit_perturbed_texts = perturb_text(lbit_clean_text, paraphraser_model, paraphraser_tokenizer, model.device)
                lbit_base_filename = f"{i}_lbit_delta_{params['delta']:.1f}_hc_{params['hashing_context']}_et_{params['entropy_threshold']:.1f}"
                for p_name, p_text in [('clean', lbit_clean_text)] + list(lbit_perturbed_texts.items()):
                    filename = f"{lbit_base_filename}_{p_name.replace(' ', '_')}.txt"
                    key_map[filename] = master_secret_key.hex()
                    with open(os.path.join(args.output_dir, filename), 'w', encoding='utf-8') as f:
                        f.write(p_text)

    key_map_path = os.path.join(args.output_dir, 'key_map.json')
    with open(key_map_path, 'w') as f:
        json.dump(key_map, f, indent=2)
        print(f"\nGeneration complete. Key map saved to '{key_map_path}'")

    # --- Stage 3: Detection and Analysis ---
    print("\n--- Starting Detection & Analysis Stage ---")
    all_files = [f for f in os.listdir(args.output_dir) if f.endswith('.txt')]
    results = []

    for filename in tqdm(all_files, desc="Detecting Watermarks"):
        parsed_info = parse_filename(filename)
        if not parsed_info:
            print(f"\nWarning: Could not parse filename '{filename}'. Skipping.")
            continue

        with open(os.path.join(args.output_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
        
        result_entry = {'filename': filename, **parsed_info}
        
        if parsed_info['type'] == 'unwatermarked':
            watermarker = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold)
            random_key = watermarker.keygen()

            z_score, _, block_count = watermarker.detect(random_key, text)
            result_entry['z_score'] = z_score
            result_entry['final_block_count'] = block_count
            result_entry['final_entropy_threshold'] = watermarker.entropy_threshold

        elif parsed_info['type'] == 'watermarked': # Watermarked file
            secret_key_hex = key_map.get(filename)
            final_et = parsed_info['entropy_threshold']
            if not secret_key_hex:
                print(f"Error: Key not found {filename}")
                z_score, block_count = float('nan'), 0
            else:
                secret_key = bytes.fromhex(secret_key_hex)
                # --- NEW TWO-PASS DETECTION LOGIC (Mirrors 'detect' command) ---
                # 1. First pass with original parameters from filename
                wm_params_pass1 = {k: v for k, v in parsed_info.items() if k in ['delta', 'hashing_context', 'entropy_threshold']}
                watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **wm_params_pass1)
                z_score, is_detected, block_count = watermarker_pass1.detect(secret_key, text)
                
                final_et = wm_params_pass1['entropy_threshold']

                # 2. If block count is low AND it wasn't detected, try a more aggressive pass
                if block_count < 75 and not is_detected:
                    pass_2_params = wm_params_pass1.copy()
                    pass_2_params['entropy_threshold'] -= 2.0

                    # Check if the new parameters are within the valid range
                    if pass_2_params['entropy_threshold'] >= 1.0:
                        watermarker_pass2 = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **pass_2_params)
                        # Overwrite results with the second pass
                        z_score, is_detected, block_count = watermarker_pass2.detect(secret_key, text)
                        final_et = pass_2_params['entropy_threshold']
            result_entry['z_score'] = z_score
            result_entry['final_block_count'] = block_count
            result_entry['final_entropy_threshold'] = final_et
        elif parsed_info['type'] == 'lbit':
            secret_key_hex = key_map.get(filename)
            if not secret_key_hex:
                print(f"Error: Key not found for {filename}")
                result_entry['recovered_message'] = ""
                result_entry['recovery_accuracy'] = 0.0
                result_entry['z_score'] = None
                result_entry['final_block_count'] = None
                result_entry['final_entropy_threshold'] = parsed_info['entropy_threshold']
            else:
                master_secret_key = bytes.fromhex(secret_key_hex)
                lbit_params = {k: v for k, v in parsed_info.items() if k in ['delta', 'hashing_context', 'entropy_threshold']}
                watermarker = ZeroBitWatermarker(model=model, z_threshold=args.z_threshold, **lbit_params)
                lbit_watermarker = LBitWatermarker(model=model, L=args.l_bits, z_threshold=args.z_threshold, **lbit_params)
                recovered_message = lbit_watermarker.detect(master_secret_key, text)
                result_entry['recovered_message'] = recovered_message
                result_entry['final_entropy_threshold'] = lbit_params['entropy_threshold']
                z_scores = []
                block_counts = []
                for i in range(1, args.l_bits + 1):
                    z_i0, _, bc0 = lbit_watermarker.zero_bit.detect(derive_key(master_secret_key, i, 0), text)
                    z_i1, _, bc1 = lbit_watermarker.zero_bit.detect(derive_key(master_secret_key, i, 1), text)
                    z_scores.append((z_i0, z_i1))
                    block_counts.append(max(bc0, bc1))
                if block_counts and min(block_counts) < 75:
                    pass_2_params = lbit_params.copy()
                    pass_2_params['entropy_threshold'] -= 2.0
                    if pass_2_params['entropy_threshold'] >= 1.0:
                        lbit_watermarker_pass2 = LBitWatermarker(model=model, L=args.l_bits, z_threshold=args.z_threshold, **pass_2_params)
                        recovered_message = lbit_watermarker_pass2.detect(master_secret_key, text)
                        result_entry['recovered_message'] = recovered_message
                        result_entry['final_entropy_threshold'] = pass_2_params['entropy_threshold']
                        z_scores = []
                        block_counts = []
                        for i in range(1, args.l_bits + 1):
                            z_i0, _, bc0 = lbit_watermarker_pass2.zero_bit.detect(derive_key(master_secret_key, i, 0), text)
                            z_i1, _, bc1 = lbit_watermarker_pass2.zero_bit.detect(derive_key(master_secret_key, i, 1), text)
                            z_scores.append((z_i0, z_i1))
                            block_counts.append(max(bc0, bc1))
                result_entry['z_score'] = z_scores
                result_entry['final_block_count'] = block_counts
                if args.l_bit_message:
                    correct_bits = sum(a == b for a, b in zip(args.l_bit_message, recovered_message) if b not in {str(None), '?'})
                    total_bits = len(args.l_bit_message)
                    result_entry['recovery_accuracy'] = (correct_bits / total_bits) * 100 if total_bits > 0 else 0.0
                else:
                    result_entry['recovery_accuracy'] = 0.0

        results.append(result_entry)

    results_path = os.path.join(args.output_dir, 'analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

        print(f"\nEvaluation complete. Final analysis summary saved to '{results_path}'.")
    print("You can now run 'analyse.py' on this directory to generate plots.")


# Command dispatch dictionary
COMMAND_HANDLERS = {
    'generate': cmd_generate,
    'detect': cmd_detect,
    'generate_lbit': cmd_generate_lbit,
    'detect_lbit': cmd_detect_lbit,
    'evaluate': cmd_evaluate,
}


def dispatch_command(args):
    """Dispatch to the appropriate command handler based on args.command."""
    handler = COMMAND_HANDLERS.get(args.command)
    if handler:
        handler(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

