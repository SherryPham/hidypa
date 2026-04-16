import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download Llama 1B model to HPC cache")
    parser.add_argument("--token", type=str, help="HuggingFace token (or set HF_TOKEN environment variable)")
    args = parser.parse_args()

    # Use HF_HOME as the download directory
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        raise ValueError("Please set HF_HOME before running this script.")

    print(f"Using Hugging Face cache directory: {cache_dir}")

    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN provided. Llama 3.2 1B is a gated model.")
        print("If you haven't authenticated via huggingface-cli, this may fail.")
        print("Please accept the terms at: https://huggingface.co/meta-llama/Llama-3.2-1B")
    
    model_id = "meta-llama/Llama-3.2-1B"
    print(f"\nDownloading and caching {model_id}...")
    
    try:
        AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
        print("✅ Tokenizer cached successfully.")
        
        print("Downloading model weights... (This may take a few minutes)")
        AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, token=hf_token)
        print("✅ Model cached successfully.")
    except Exception as e:
        print(f"\n❌ Failed to download model: {e}")
        print("\nMake sure you have accepted the Meta license agreement on the Hugging Face website and are providing a valid access token.")

if __name__ == "__main__":
    main()
