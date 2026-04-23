# cli_utils.py: Utility functions for the CLI

import torch
import random
import re
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models import GPT2Model, GptOssModel, GptOss120bModel, LlamaModel, LLAMA_MODEL_IDS, OPTModel, OPT_MODEL_IDS, DeepSeekModel, DEEPSEEK_MODEL_IDS


def get_model(model_name: str):
    """Function to instantiate the correct model based on its name."""
    if model_name == 'gpt2':
        return GPT2Model()
    elif model_name == 'gpt-oss-20b':
        return GptOssModel()
    elif model_name == 'gpt-oss-120b':
        return GptOss120bModel()
    elif model_name in LLAMA_MODEL_IDS or model_name.startswith('meta-llama/'):
        return LlamaModel(model_name)
    elif model_name in OPT_MODEL_IDS or model_name.startswith('facebook/opt'):
        return OPTModel(model_name)
    elif model_name in DEEPSEEK_MODEL_IDS or model_name.startswith('deepseek-ai/'):
        return DeepSeekModel(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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


def generate_unwatermarked(model_wrapper, prompt: str, max_new_tokens: int, model_name: str):
    """Helper to generate text without a watermark for comparison."""
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device
    
    # Conditional text formatting based on base vs chat model
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to(device)
    else:
        token_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Create explicit attention mask to avoid warning when pad_token_id == eos_token_id
    attention_mask = torch.ones_like(token_ids)
    
    with torch.no_grad():
        output_ids = model_wrapper._model.generate(
            token_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
    
    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Parse the output before returning
    return parse_final_output(raw_output, model_name)


def get_paraphraser(device):
    """Loads a T5 model for the substitution/paraphrasing attack."""
    model_id = "Vamsi/T5_Paraphrase_Paws"
    print(f"Loading paraphrasing model ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    return model, tokenizer


def perturb_text(text: str, paraphraser_model, paraphraser_tokenizer, device) -> dict:
    """
    Applies stronger, more realistic perturbations to a text to test robustness.
    This version is more robust and guarantees all attack types are returned.
    """
    perturbed = {}
    
    # Use NLTK for robust sentence splitting
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK sentence tokenization failed. Error: {e}. Falling back to simple split.")
        sentences = text.split('.') # Fallback
        
    num_sentences = len(sentences)

    # --- 1. Deletion Attacks ---
    num_to_delete = max(1, int(num_sentences * 0.20))
    
    # a) Delete from the start
    if num_sentences > num_to_delete:
        perturbed['delete_start_20_percent'] = " ".join(sentences[num_to_delete:])
    else:
        perturbed['delete_start_20_percent'] = text # Not enough sentences to delete, return original

    # b) Delete from the end
    if num_sentences > num_to_delete:
        perturbed['delete_end_20_percent'] = " ".join(sentences[:-num_to_delete])
    else:
        perturbed['delete_end_20_percent'] = text

    # c) Delete from the middle
    if num_sentences > num_to_delete * 2: # Need enough sentences to see a middle
        mid_index = num_sentences // 2
        start_del = max(0, mid_index - num_to_delete // 2)
        end_del = start_del + num_to_delete
        middle_deleted_sentences = sentences[:start_del] + sentences[end_del:]
        perturbed['delete_middle_20_percent'] = " ".join(middle_deleted_sentences)
    else:
        perturbed['delete_middle_20_percent'] = text

    # --- 2. Substitution Attack ---
    if num_sentences > 0:
        num_to_paraphrase = max(1, int(num_sentences * 0.30))
        indices_to_paraphrase = random.sample(range(num_sentences), min(num_to_paraphrase, num_sentences))
        
        substituted_sentences = list(sentences)
        for i in indices_to_paraphrase:
            sentence_to_paraphrase = sentences[i].strip()
            if not sentence_to_paraphrase: continue
            
            if len(sentence_to_paraphrase.split()) > 400:
                print(f"\nWarning: Skipping paraphrase for a sentence with >400 words.")
                continue
            
            try:
                # Use the format required by the fine-tuned paraphrasing model
                input_text = f"paraphrase: {sentence_to_paraphrase} </s>"
                input_ids = paraphraser_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

                with torch.no_grad():
                    outputs = paraphraser_model.generate(
                        input_ids, 
                        max_length=int(len(input_ids[0]) * 2.0), # Allow longer paraphrases
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=1.2, # Increase creativity
                        num_return_sequences=3 # Generate 3 different options
                    )
                
                decoded_outputs = [paraphraser_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                paraphrased_sentence = random.choice(decoded_outputs)
                substituted_sentences[i] = paraphrased_sentence

            except Exception as e:
                print(f"\nWarning: Paraphrasing sentence failed. Skipping. Error: {e}")
                
        perturbed['substitute_30_percent'] = " ".join(substituted_sentences)
    else:
        print("Warning: No sentences found")
        perturbed['substitute_30_percent'] = text

    return perturbed


def parse_filename(filename: str):
    """Parses the complex evaluation filename to extract parameters."""
    if "unwatermarked" in filename:
        try:
            prompt_id = int(filename.split('_')[0])
            return {'type': 'unwatermarked', 'prompt_id': prompt_id}
        except (ValueError, IndexError):
            return None

    match = re.match(
        r"(\d+)_wm_delta_([\d.]+)_hc_(\d+)_et_([\d.]+)_([\w_]+)\.txt",
        filename
    )
    if match:
        prompt_id, delta, hc, et, perturbation_full = match.groups()
        # The perturbation name might have multiple parts, join them back
        perturbation = perturbation_full.replace('_', ' ')
        
        return {
            'type': 'watermarked',
            'prompt_id': int(prompt_id),
            'delta': float(delta),
            'hashing_context': int(hc),
            'entropy_threshold': float(et),
            'perturbation': perturbation
        }
    match = re.match(
        r"(\d+)_lbit_delta_([\d.]+)_hc_(\d+)_et_([\d.]+)_([\w_]+)\.txt",
        filename
    )
    if match:
        prompt_id, delta, hc, et, perturbation_full = match.groups()
        perturbation = perturbation_full.replace('_', ' ')
        return {
            'type': 'lbit',
            'prompt_id': int(prompt_id),
            'delta': float(delta),
            'hashing_context': int(hc),
            'entropy_threshold': float(et),
            'perturbation': perturbation
        }
    
    return None

