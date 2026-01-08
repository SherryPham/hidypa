# cli_parser.py: Argument parsing and validators for the CLI

import argparse
import nltk


# --- Argparse Type Checkers for Parameter Constraints ---
def check_delta_range(value):
    """Checks if a single delta value is between 1.0 and 5.0."""
    try:
        f_value = float(value)
        if 1.0 <= f_value <= 5.0:
            return f_value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"'{value}' is an invalid delta. Must be a float between 1.0 and 5.0.")


def check_entropy_range(value):
    """Checks if a single entropy_threshold is between 1.0 and 6.0."""
    try:
        f_value = float(value)
        if 1.0 <= f_value <= 6.0:
            return f_value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"'{value}' is an invalid entropy threshold. Must be a float between 1.0 and 6.0.")


def check_list_of(checker_func):
    """Factory to create a checker for comma-separated lists."""
    def _check_list(value_str):
        try:
            # Split the string by comma and apply the single value checker to each part
            return [checker_func(v.strip()) for v in value_str.split(',')]
        except argparse.ArgumentTypeError as e:
            # Re-raise the error from the checker function for a clear message
            raise argparse.ArgumentTypeError(f"Invalid value in list '{value_str}': {e}")
        except Exception:
            raise argparse.ArgumentTypeError(f"Could not parse '{value_str}' as a comma-separated list of numbers.")

    return _check_list


def setup_nltk():
    """Checks for the NLTK 'punkt' tokenizer model"""
    # Try to load NLTK data path from config file
    nltk_data_path = None
    try:
        from config.paths import NLTK_DATA_PATH
        nltk_data_path = NLTK_DATA_PATH
    except ImportError:
        # Config file not found, use default NLTK paths
        pass
    
    # Add custom path to NLTK's search paths if configured
    if nltk_data_path and nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    # Now, verify that the package can be found without triggering a download
    try:
        nltk.data.find('tokenizers/punkt')
        
    except LookupError:
        print("ERROR: NLTK 'punkt' package not found in the specified path.")
        print(f"Please run 'download_models_hpc.py' on a login node first.")
        # Exit if the required data is not found
        exit(1)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
        
    except LookupError:
        print("ERROR: NLTK 'punkt_tab' package not found in the specified path.")
        print(f"Please run 'download_models_hpc.py' on a login node first.")
        # Exit if the required data is not found
        exit(1)


def build_parser():
    """Builds and returns the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(description="Zero-Bit Watermarking Tool for LLMs.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Generate Command ---
    parser_gen = subparsers.add_parser('generate', help='Generate and watermark text.')
    parser_gen.add_argument('prompt', type=str, help='The prompt to send to the model.')
    parser_gen.add_argument('--model', type=str, default='gpt2',
                            choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                            help='The model to use for generation.')
    parser_gen.add_argument('--max-new-tokens', type=int, default=2048, help='Maximum number of new tokens to generate.')
    parser_gen.add_argument('--output-file', '-o', type=str, default='demonstration/watermarked_output.txt',
                            help='File to save the generated text.')
    parser_gen.add_argument('--key-file', '-k', type=str, default='demonstration/secret.key',
                            help='File to save the secret key.')
    parser_gen.add_argument('--no-watermark', action='store_true', help='Generate plain text without a watermark.')
    parser_gen.add_argument('--hashing-context', type=int, default=5, help='The number of previous tokens to use for the PRF context.')
    parser_gen.add_argument('--delta', type=check_delta_range, default=3.5, help='Watermark bias strength (1.0 to 5.0).')
    parser_gen.add_argument('--entropy-threshold', type=check_entropy_range, default=2.5, help='Entropy threshold to define a block (1.0 to 6.0).')

    # --- Detect Command ---
    parser_detect = subparsers.add_parser('detect', help='Detect a watermark in a text file.')
    parser_detect.add_argument('input_file', type=str, help='Path to the text file to check.')
    parser_detect.add_argument('--model', type=str, required=True,
                                choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                                help='The model that was used to generate the text (for tokenizer matching).')
    parser_detect.add_argument('--key-file', '-k', type=str, default='demonstration/secret.key',
                                help='Path to the secret key file.')
    parser_detect.add_argument('--z-threshold', type=float, default=4.0,
                                help='The z-score threshold for detection.')
    parser_detect.add_argument('--entropy-threshold', type=check_entropy_range, default=2.5, help='Entropy threshold used during generation (1.0 to 6.0).')
    parser_detect.add_argument('--hashing-context', type=int, default=5, help='The hashing context used during generation.')
    
    # --- Evaluate Command ---
    parser_eval = subparsers.add_parser('evaluate', help='Run a full evaluation of the watermarking framework.')
    parser_eval.add_argument('--prompts-file', type=str, required=True, help='Path to a .txt file with one prompt per line.')
    parser_eval.add_argument('--max-prompts', type=int, default=100,
                             help='How many prompts to evaluate (default: 100; set <=0 to use all).')
    parser_eval.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'], help='Model to use for generation.')
    parser_eval.add_argument('--output-dir', type=str, default='evaluation/evaluation_results', help='Directory to save all generated texts and results.')
    parser_eval.add_argument('--max-new-tokens', type=int, default=2048, help='Tokens to generate for each prompt.')
    parser_eval.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for detection.')
    parser_eval.add_argument('--l-bit-message', type=str, help='L-bit binary message to embed (e.g., "0101...").')
    parser_eval.add_argument('--l-bits', type=int, default=32, help='Number of bits for the L-bit message.')

    # --- Generate L-Bit Command ---
    parser_lgen = subparsers.add_parser('generate_lbit', help='Generate and watermark with L-bit message.')
    parser_lgen.add_argument('prompt', type=str, help='The prompt to send to the model.')
    parser_lgen.add_argument('--message', type=str, required=True, help='L-bit binary message (e.g., "0101...").')
    parser_lgen.add_argument('--l-bits', type=int, default=32, help='Number of bits for the message.')
    parser_lgen.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                            help='The model to use for generation.')
    parser_lgen.add_argument('--max-new-tokens', type=int, default=512, help='Maximum number of new tokens to generate.')
    parser_lgen.add_argument('--output-file', '-o', type=str, default='demonstration/watermarked_lbit.txt', help='File to save the generated text.')
    parser_lgen.add_argument('--key-file', '-k', type=str, default='demonstration/secret_lbit.key', help='File to save the secret key.')
    parser_lgen.add_argument('--delta', type=check_delta_range, default=3.5, help='Watermark bias strength (1.0 to 5.0).')
    parser_lgen.add_argument('--entropy-threshold', type=check_entropy_range, default=2.5, help='Entropy threshold to define a block (1.0 to 6.0).')
    parser_lgen.add_argument('--hashing-context', type=int, default=5, help='The number of previous tokens to use for the PRF context.')
    parser_lgen.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for detection.')

    # --- Detect L-Bit Command ---
    parser_ldet = subparsers.add_parser('detect_lbit', help='Detect L-bit message in text.')
    parser_ldet.add_argument('input_file', type=str, help='Path to the text file to check.')
    parser_ldet.add_argument('--l-bits', type=int, default=32, help='Number of bits for the message.')
    parser_ldet.add_argument('--model', type=str, required=True, choices=['gpt2', 'gpt-oss-20b', 'gpt-oss-120b'],
                            help='The model that was used to generate the text (for tokenizer matching).')
    parser_ldet.add_argument('--key-file', type=str, default='demonstration/secret_lbit.key', help='Path to the secret key file.')
    parser_ldet.add_argument('--delta', type=check_delta_range, default=3.5, help='Watermark bias strength used during generation (1.0 to 5.0).')
    parser_ldet.add_argument('--entropy-threshold', type=check_entropy_range, default=2.5, help='Entropy threshold used during generation (1.0 to 6.0).')
    parser_ldet.add_argument('--hashing-context', type=int, default=5, help='The hashing context used during generation.')
    parser_ldet.add_argument('--z-threshold', type=float, default=4.0, help='The z-score threshold for detection.')

    # Arguments for parameter sweeps
    parser_eval.add_argument('--delta', type=check_delta_range, default=3.5, help='A single delta value (1.0 to 5.0).')
    parser_eval.add_argument('--entropy-threshold', type=check_entropy_range, default=2.5, help='A single entropy threshold value (1.0 to 6.0).')
    parser_eval.add_argument('--hashing-context', type=int, default=5, help='A single hashing context value.')

    # Mutually exclusive group for list-style arguments
    sweep_group = parser_eval.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument('--deltas', type=check_list_of(check_delta_range), help='Comma-separated list of delta values to sweep (each between 1.0-5.0).')
    sweep_group.add_argument('--hashing-contexts', type=lambda s: [int(v) for v in s.split(',')], help='Comma-separated list of hashing contexts to sweep.')
    sweep_group.add_argument('--entropy-thresholds', type=check_list_of(check_entropy_range), help='Comma-separated list of entropy thresholds to sweep (each between 1.0-6.0).')

    return parser

