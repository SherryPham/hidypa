# watermark.py, contains functions relating to the watermark framework

import os
import hmac
import hashlib
import random
import torch
import numpy as np
import pandas as pd
from .models import LanguageModel
from transformers import LogitsProcessor

from tqdm import tqdm
from .fingerprinting import FingerprintingCode, generate_user_fingerprint

def _calculate_entropy(logits):
    """Calculates the Shannon entropy of a logits distribution."""
    probabilities = torch.softmax(logits, dim=-1)
    # Add a small epsilon to prevent log(0)
    p_log_p = probabilities * torch.log2(probabilities + 1e-9)
    # Sum over the vocabulary dimension
    return -torch.sum(p_log_p, dim=-1)

def derive_key(master_secret_key: bytes, index: int, bit: int) -> bytes:
    """
    derives a 32-byte key using binary encoding
    """
    message = f"{index}_{bit}".encode("utf-8")
    new_message = hmac.new(key=master_secret_key, msg=message, digestmod=hashlib.sha256)
    return new_message.digest()[:32]

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor to apply the CGZ23 soft-scoring watermark bias.
    """
    def __init__(self, watermarker_instance, params: dict, sk: bytes):
        """
        Initializes the processor.
        
        Args:
            watermarker_instance: The main ZeroBitWatermarker object to access its methods.
            params (dict): A dictionary of dynamic parameters for this specific generation pass.
            sk (bytes): The secret key.
        """
        self.watermarker = watermarker_instance
        self.params = params
        self.sk = sk

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the watermark bias to the logits 'scores' only if entropy is high.
        """
        # Calculate entropy of the original, unbiased logits
        entropy = _calculate_entropy(scores)

        # Use the dynamic entropy_threshold from the params dictionary
        if entropy >= self.params['entropy_threshold']:
            # Call the method from the main watermarker instance
            score_vector = self.watermarker._get_score_vector(self.sk, input_ids[0])
            
            # Use the dynamic delta from the params dictionary
            scores += self.params['delta'] * score_vector
        
        return scores


class ZeroBitWatermarker:
    """
    Implements the block-by-block zero-bit watermarking scheme.
    """

    def __init__(self, model: LanguageModel, delta: float = 2.5, z_threshold: float = 4.0, entropy_threshold: float = 4.0, hashing_context: int = 5):
        """
        Initializes the watermarker.

        Args:
            model (LanguageModel): The model wrapper.
            delta (float): The bias strength (scaling factor for the score vector).
            z_threshold (float): The threshold for detecting the watermark.
            entropy_threshold (float): The entropy value above which a watermark is embedded.
            hashing_context (int): The number of previous tokens to use for the PRF context.
        """
        self.model = model
        self.delta = delta
        self.z_threshold = z_threshold
        self.entropy_threshold = entropy_threshold
        self.hashing_context = hashing_context
        
    def keygen(self, key_length: int = 32) -> bytes:
        """KeyGen: Generates a random secret key."""
        return os.urandom(key_length)

    def _get_score_vector(self, sk: bytes, prefix_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generates a pseudorandom score vector for a given prefix.
        The vector is drawn from a standard normal distribution N(0, 1).

        Args:
            sk (bytes): The secret key.
            prefix_tokens (torch.Tensor): The tokenized prefix.

        Returns:
            torch.Tensor: A tensor of scores, one for each token in the vocabulary.
        """
        if len(prefix_tokens) > self.hashing_context:
            context_tokens = prefix_tokens[-self.hashing_context:]
        else:
            context_tokens = prefix_tokens

        prefix_str = str(context_tokens.tolist())
        h = hmac.new(sk, prefix_str.encode('utf-8'), hashlib.sha256)

        # The full 32-byte (256-bit) hash digest is too large for torch's 64-bit seed.
        # We truncate the hash to the first 8 bytes (64 bits) to create a valid seed.
        seed_bytes = h.digest()[:8]
        seed = int.from_bytes(seed_bytes, 'big', signed=True)
        
        # Use the seed to create a deterministic torch random number generator
        rng_generator = torch.Generator(device=self.model.device)
        rng_generator.manual_seed(seed)
        
        # Generate scores from N(0, 1)
        score_vector = torch.randn(
            self.model.vocab_size, 
            generator=rng_generator, 
            device=self.model.device
        )
        return score_vector
    
    def _generate_with_params(self, sk, prompt, max_new_tokens, params, **kwargs):
        """A helper to generate text with a specific set of watermarking parameters."""
        tokenizer = self.model.tokenizer
        
        # The LogitsProcessor needs the specific params for this run
        logits_processor = WatermarkLogitsProcessor(self, params, sk)

        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            ).to(self.model.device)
        else:
            token_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        
        # Create explicit attention mask to avoid warning when pad_token_id == eos_token_id
        attention_mask = torch.ones_like(token_ids)
        
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            logits_processor=[logits_processor],
            do_sample=True, top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            **kwargs
        )

        with torch.no_grad():
            output_ids = self.model._model.generate(token_ids, **gen_kwargs)
            
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def _count_blocks(self, text, params):
        """A helper to count the number of blocks in a text for a given entropy threshold."""
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, return_tensors='pt').to(self.model.device)[0]
        
        if len(token_ids) < 2: return 0
        
        with torch.no_grad():
            outputs = self.model._model(token_ids.unsqueeze(0))
        all_logits = outputs.logits.squeeze(0)

        block_count = 0
        for i in range(len(token_ids) - 1):
            entropy = _calculate_entropy(all_logits[i, :])
            if entropy >= params['entropy_threshold']:
                block_count += 1
        return block_count

    def embed(self, sk: bytes, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Embeds a watermark using a two-pass technique to ensure sufficient block count.
        Returns the raw watermarked text and the final parameters used.
        """
        # Pass 1: Generate with the initial parameters
        pass_1_params = {
            'delta': self.delta, 
            'entropy_threshold': self.entropy_threshold,
            'hashing_context': self.hashing_context
        }
        raw_text_pass1 = self._generate_with_params(sk, prompt, max_new_tokens, pass_1_params, **kwargs)
        block_count_pass1 = self._count_blocks(raw_text_pass1, pass_1_params)

        # If the first pass has enough blocks, we're done.
        if block_count_pass1 >= 75:
            # Return the text and the parameters that were used to generate it
            return raw_text_pass1, pass_1_params

        # Pass 2: If not enough blocks, try one more aggressive pass
        pass_2_params = {
            'delta': self.delta + 1.0,
            'entropy_threshold': self.entropy_threshold - 2.0,
            'hashing_context': self.hashing_context
        }
        # Ensure the new entropy threshold is valid
        if pass_2_params['entropy_threshold'] < 1.0:
            return raw_text_pass1, pass_1_params

        raw_text_pass2 = self._generate_with_params(sk, prompt, max_new_tokens, pass_2_params, **kwargs)
        
        # Return the result of the second pass, regardless of its block count
        return raw_text_pass2, pass_2_params

    def detect(self, sk: bytes, text: str, cached_logits: torch.Tensor = None) -> tuple[float, bool, int]:
        """
        Detects a watermark by checking for the signature only at high-entropy positions (blocks).
        """
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, return_tensors='pt').to(self.model.device)[0]
        
        num_tokens = len(token_ids)
        if num_tokens < 2:
            return 0.0, False, 0

        # Get the logits for every token position in one go.
        if cached_logits is None:
            with torch.no_grad():
            # The model returns logits for predicting the *next* token at each position.
            # So, outputs.logits[0, i, :] are the logits for predicting token i+1.
                outputs = self.model._model(token_ids.unsqueeze(0))
            all_logits = outputs.logits.squeeze(0) # Remove batch dimension
        else:
            all_logits = cached_logits

            
        total_score = 0.0
        block_count = 0 # Count of high-entropy positions found
        
        # We check each token from the first potential hash context up to the end.
        for i in range(self.hashing_context, num_tokens - 1):
            # The logits for predicting the token at position i
            current_logits = all_logits[i-1, :]
            entropy = _calculate_entropy(current_logits)

            if entropy >= self.entropy_threshold:
                block_count += 1
                
                # The token that was actually chosen at this position
                current_token_id = token_ids[i].item()
                
                # The prefix used for hashing ends at the previous token
                prefix_for_hash = token_ids[:i]
                score_vector = self._get_score_vector(sk, prefix_for_hash)
                
                token_score = score_vector[current_token_id].item()
                total_score += token_score
        
        if block_count == 0:
            return 0.0, False, 0

        z_score = total_score / np.sqrt(block_count)
        
        return z_score, z_score > self.z_threshold, block_count
    
    def parse_first_block(self, text: str, prefix: str, params: dict) -> str:
        """Extracts the first high-entropy block from text, or None if None"""
        combined_text = prefix + text
        tokenizer = self.model.tokenizer
        device = self.model.device
        tokens = tokenizer.encode(combined_text, return_tensors='pt').to(device)[0]

        if tokens.size(0) < 2:
            return None
        
        with torch.no_grad():
            model_ouput = self.model._model(tokens.unsqueeze(0))
        logits = model_ouput.logits.squeeze(0)
        prefix_len = len(tokenizer.encode(prefix, return_tensor='pt')[0])
        block_end_index = None

        # Scan from prefix end to find first high-entropy position (start of block)
        for pos in range(prefix_len, len(tokens) - 1):
            entropy = _calculate_entropy(logits[pos])
            if entropy >= params["entropy_threshold"]:
                block_end_index = pos + 1
                break
            
        if block_end_index is None:
            return None
        
        # Extract block text (from text start to first_block_end - prefix_len)
        block_tokens = tokens[prefix_len:block_end_index]

        return tokenizer.decode(block_tokens, skip_special_tokens=True)
    
class LBitWatermarker:
    def __init__(self, zero_bit_watermarker: ZeroBitWatermarker = None, model: LanguageModel = None, L: int = 32, delta: float = 2.5, z_threshold: float = 4.0, entropy_threshold: float = 4.0, hashing_context: int = 5):
        if zero_bit_watermarker:
            self.zero_bit = zero_bit_watermarker
            self.model = zero_bit_watermarker.model
        else:
            if model is None:
                raise ValueError("Must provide either zero_bit_watermarker or model")
            self.zero_bit = ZeroBitWatermarker(model, delta, z_threshold, entropy_threshold, hashing_context)
            self.model = model
        self.L = L

    def keygen(self, key_length: int = 32) -> bytes:
        return os.urandom(key_length)

    def embed(self, master_secret_key: bytes, message: str, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if len(message) != self.L or any(c not in {'0', '1'} for c in message):
            raise ValueError(f"Message must be a {self.L}-bit binary string")
        tokenizer = self.model.tokenizer
        device = self.model.device

        pass_1_params = {
            'delta': self.zero_bit.delta,
            'entropy_threshold': self.zero_bit.entropy_threshold,
            'hashing_context': self.zero_bit.hashing_context
        }
        logit_processor = LBitLogitProcessor(self.zero_bit, master_secret_key, message)

        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            ).to(device)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Create explicit attention mask to avoid warning when pad_token_id == eos_token_id
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            logits_processor=[logit_processor],
            do_sample=True, top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            **kwargs
        )

        with torch.no_grad():
            output_ids = self.model._model.generate(input_ids, **gen_kwargs)
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        block_count = self.zero_bit._count_blocks(raw_text, pass_1_params)

        if block_count >= 75:
            return raw_text

        pass_2_params = {
            'delta': self.zero_bit.delta + 1.0,
            'entropy_threshold': max(1.0, self.zero_bit.entropy_threshold - 2.0),
            'hashing_context': self.zero_bit.hashing_context
        }
        self.zero_bit.delta = pass_2_params['delta']
        self.zero_bit.entropy_threshold = pass_2_params['entropy_threshold']
        with torch.no_grad():
            output_ids = self.model._model.generate(input_ids, **gen_kwargs)
        self.zero_bit.delta = pass_1_params['delta']
        self.zero_bit.entropy_threshold = pass_1_params['entropy_threshold']
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def detect(self, master_secret_key: bytes, text: str) -> str:
        """Recovers L-bit message or None per bit if neither detects"""
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, return_tensors='pt').to(self.model.device)[0]
        if len(token_ids) < 2:
            return str(None) * self.L

        with torch.no_grad():
            outputs = self.model._model(token_ids.unsqueeze(0))
        all_logits = outputs.logits.squeeze(0)

        recovered_message = ""
        for i in range(1, self.L + 1):
            z_i0, detected0, _ = self.zero_bit.detect(derive_key(master_secret_key, i, 0), text, cached_logits=all_logits)
            z_i1, detected1, _ = self.zero_bit.detect(derive_key(master_secret_key, i, 1), text, cached_logits=all_logits)
            if z_i0 <= self.zero_bit.z_threshold and z_i1 <= self.zero_bit.z_threshold:
                recovered_message += "⊥"
            elif z_i0 > self.zero_bit.z_threshold and z_i1 <= self.zero_bit.z_threshold:
                recovered_message += "0"
            elif z_i1 > self.zero_bit.z_threshold and z_i0 <= self.zero_bit.z_threshold:
                recovered_message += "1"
            else:
                recovered_message += "*"
        return recovered_message
    
class LBitLogitProcessor(LogitsProcessor):
    def __init__(self, watermarker, master_secret_key: bytes, message: str):
        self.watermarker = watermarker
        self.master_secret_key = master_secret_key
        self.message = message
        self.L = len(message)
        self.index_bit = 0
        # Initialize with a random permutation of [1, 2, ..., L]
        self.current_permutation = list(range(1, self.L + 1))
        random.shuffle(self.current_permutation)
        self.permutation_index = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Modifies logits to encode the L-Bit message during generation"""
        if len(input_ids[0]) < 2:
            return scores
        
        entropy = _calculate_entropy(scores)
        if entropy < self.watermarker.entropy_threshold:
            return scores
        
        # Check if we've used all positions in current permutation
        if self.permutation_index >= self.L:
            # Generate a new random permutation for the next cycle
            self.current_permutation = list(range(1, self.L + 1))
            random.shuffle(self.current_permutation)
            self.permutation_index = 0
        
        # Get the bit position from the current permutation
        bit_position = self.current_permutation[self.permutation_index]
        bit_value = int(self.message[bit_position - 1])  # Either 0 or 1
        self.permutation_index += 1
        self.index_bit += 1

        key = derive_key(self.master_secret_key, bit_position, bit_value)
        context_ids = input_ids[0][-self.watermarker.hashing_context:]
        score_vector = self.watermarker._get_score_vector(key, context_ids)
        scores = scores + self.watermarker.delta * score_vector

        return scores

class NaiveMultiUserWatermarker:
    """
    Implements the original per-user binary multi-user scheme without grouping.
    Each user receives the binary expansion of their user ID (padded/truncated to L bits).
    """
    def __init__(self, lbit_watermarker: LBitWatermarker):
        self.lbw = lbit_watermarker
        self.user_metadata: pd.DataFrame | None = None
        self.user_lookup: dict[int, pd.Series] = {}
        self.N: int = 0
    
    def keygen(self, key_length: int = 32) -> bytes:
        return self.lbw.keygen(key_length)
    
    def load_users(self, users_file: str) -> pd.DataFrame:
        """Loads user metadata and prepares lookup tables."""
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"User metadata file {users_file} not found")
        
        df = pd.read_csv(users_file)
        self._initialize_metadata(df)
        return self.user_metadata
    
    def _initialize_metadata(self, df: pd.DataFrame):
        if "UserId" not in df.columns:
            raise ValueError("users_file must contain a 'UserId' column.")
        
        df = df.copy()
        df["UserId"] = df["UserId"].astype(int)
        if df["UserId"].duplicated().any():
            raise ValueError("Duplicate UserId entries detected in users file.")
        
        df = df.sort_values("UserId").reset_index(drop=True)
        max_users = 2 ** self.lbw.L
        if len(df) > max_users:
            print(
                f"Warning: users file contains {len(df)} entries but L={self.lbw.L} "
                f"only supports {max_users}. Using first {max_users} users."
            )
            df = df.head(max_users)
        
        self.user_metadata = df
        self.N = len(df)
        self.user_lookup = {int(row["UserId"]): row for _, row in df.iterrows()}
    
    def _require_metadata(self):
        if self.user_metadata is None or not self.user_lookup:
            raise ValueError("User metadata not loaded. Call load_users(...) first.")
    
    def _validate_user_id(self, user_id: int):
        if user_id not in self.user_lookup:
            if self.N == 0:
                raise ValueError("User metadata not loaded.")
            raise ValueError(f"User ID {user_id} not found in loaded metadata.")
    
    def get_codeword_for_user(self, user_id: int) -> str:
        """Returns the naive binary codeword for a user."""
        max_users = 2 ** self.lbw.L
        if user_id >= max_users or user_id < 0:
            raise ValueError(
                f"User ID {user_id} cannot be represented with {self.lbw.L} bits (max {max_users - 1})."
            )
        return format(user_id, f'0{self.lbw.L}b')
    
    def _log_embed(self, user_id: int, codeword: str):
        print(f"Embedding codeword '{codeword}' for User ID {user_id} (naive scheme)...")
    
    def embed(self, master_key: bytes, user_id: int, prompt: str, **kwargs) -> str:
        self._require_metadata()
        self._validate_user_id(user_id)
        codeword = self.get_codeword_for_user(user_id)
        self._log_embed(user_id, codeword)
        return self.lbw.embed(master_key, codeword, prompt, **kwargs)
    
    def trace(self, master_key: bytes, text: str, **kwargs) -> list[dict]:
        self._require_metadata()
        print("Extracting L-bit codeword from text...")
        recovered_codeword = self.lbw.detect(master_key, text, **kwargs)
        print(f"  - Recovered Codeword: {recovered_codeword}")
        return self.trace_from_codeword(recovered_codeword)
    
    def trace_from_codeword(self, recovered_codeword: str) -> list[dict]:
        """
        Traces users from a recovered codeword (without running detection).
        This allows separation of detection and tracing for performance measurement.
        
        Args:
            recovered_codeword (str): The recovered L-bit codeword from detection.
        
        Returns:
            list[dict]: List of accused users with match scores.
        """
        self._require_metadata()
        return self._match_users_from_codeword(recovered_codeword)
    
    def _match_users_from_codeword(self, recovered_codeword: str) -> list[dict]:
        valid_positions = [idx for idx, bit in enumerate(recovered_codeword) if bit not in ('⊥', '*')]
        if not valid_positions:
            print("No decided bits recovered; cannot trace.")
            return []
        
        best_score = -1
        candidates: list[dict] = []
        
        for user_id, row in self.user_lookup.items():
            codeword = self.get_codeword_for_user(user_id)
            matches, total = self._score_codewords(recovered_codeword, codeword, valid_positions)
            if total == 0:
                continue
            if matches > best_score:
                best_score = matches
                candidates = [self._format_trace_entry(user_id, row, matches, total)]
            elif matches == best_score:
                candidates.append(self._format_trace_entry(user_id, row, matches, total))
        
        return candidates
    
    def _score_codewords(self, recovered: str, candidate: str, valid_positions: list[int]) -> tuple[int, int]:
        matches = sum(recovered[i] == candidate[i] for i in valid_positions)
        total = len(valid_positions)
        return matches, total
    
    def _format_trace_entry(self, user_id: int, row: pd.Series, matches: int, total: int) -> dict:
        username = row.get("Username") if row is not None else None
        return {
            "user_id": user_id,
            "username": username,
            "match_score_percent": (matches / total) * 100 if total else 0.0,
            "group_id": None
        }


class GroupedMultiUserWatermarker(NaiveMultiUserWatermarker):
    """
    Enhanced grouped multi-user scheme that layers fingerprinting codes on top of
    the naive scheme to provide collusion resistance and group awareness.
    """
    def __init__(self, lbit_watermarker: LBitWatermarker, min_distance: int = 2, 
                 max_groups: int = None, users_per_group: int = None):
        super().__init__(lbit_watermarker=lbit_watermarker)
        self.fingerprinter = FingerprintingCode(
            L=self.lbw.L, 
            min_distance=min_distance,
            max_groups=max_groups,
            users_per_group=users_per_group
        )
    
    def load_users(self, users_file: str) -> pd.DataFrame:
        """Generates BCH-based codewords and stores metadata from the users file."""
        self.fingerprinter.gen(users_file=users_file)
        self.user_metadata = self.fingerprinter.user_metadata
        self.N = self.fingerprinter.N
        self.user_lookup = {int(row["UserId"]): row for _, row in self.user_metadata.iterrows()}
        return self.user_metadata
    
    def get_codeword_for_user(self, user_id: int) -> str:
        if self.fingerprinter.codewords is None or self.fingerprinter.codewords.size == 0:
            raise ValueError("Fingerprinter codes not generated. Call load_users(...) first.")
        if user_id < 0 or user_id >= self.fingerprinter.N:
            raise ValueError(f"User ID {user_id} is out of range for {self.fingerprinter.N} users.")
        codeword_array = self.fingerprinter.codewords[user_id]
        return "".join(map(str, codeword_array))
    
    def _log_embed(self, user_id: int, codeword: str):
        group_id = self.fingerprinter.user_to_group.get(user_id, None)
        if group_id is not None:
            print(f"User ID {user_id} belongs to Group {group_id}")
            print(f"Embedding codeword '{codeword}' for User ID {user_id} (Group {group_id})...")
        else:
            print(f"Embedding codeword '{codeword}' for User ID {user_id} (grouped scheme)...")
    
    def trace(self, master_key: bytes, text: str, **kwargs) -> list[dict]:
        if self.fingerprinter.codewords is None or self.fingerprinter.user_metadata is None:
            raise ValueError("Fingerprinter codes or metadata not loaded. Call load_users(...) first.")
        
        print("Extracting L-bit codeword from text...")
        noisy_codeword = self.lbw.detect(master_key, text, **kwargs)
        print(f"  - Recovered Codeword: {noisy_codeword}")
        
        return self.trace_from_codeword(noisy_codeword)
    
    def trace_from_codeword(self, noisy_codeword: str) -> list[dict]:
        """
        Traces users from a recovered codeword (without running detection).
        This allows separation of detection and tracing for performance measurement.
        
        Args:
            noisy_codeword (str): The recovered L-bit codeword from detection.
        
        Returns:
            list[dict]: List of accused users with match scores.
        """
        if self.fingerprinter.codewords is None or self.fingerprinter.user_metadata is None:
            raise ValueError("Fingerprinter codes or metadata not loaded. Call load_users(...) first.")
        
        print("Tracing codeword to find user(s)...")
        accused_users = self.fingerprinter.trace(noisy_codeword)
        return accused_users


class HierarchicalMultiUserWatermarker(NaiveMultiUserWatermarker):
    """
    Hierarchical multi-user watermarking scheme that combines group codewords
    (with minimum distance for cross-group collusion resistance) with per-user
    fingerprints within each group.
    
    The combined codeword is: group_code[group_bits] + user_code[user_bits] = L bits
    """
    def __init__(self, lbit_watermarker: LBitWatermarker, group_bits: int, 
                 user_bits: int, min_distance: int = 2, max_groups: int = None,
                 users_per_group: int = None):
        """
        Initializes the hierarchical multi-user watermarker.
        
        Args:
            lbit_watermarker (LBitWatermarker): The L-bit watermarker to use for embedding.
            group_bits (int): Number of bits for group codewords.
            user_bits (int): Number of bits for user fingerprints within groups.
            min_distance (int): Minimum Hamming distance between group codewords.
            max_groups (int, optional): Maximum number of groups allowed. If None, calculated automatically.
            users_per_group (int, optional): Number of users per group. If None, calculated automatically.
        
        Raises:
            ValueError: If group_bits + user_bits != L, or if constraints are violated.
        """
        super().__init__(lbit_watermarker=lbit_watermarker)
        
        if group_bits + user_bits != self.lbw.L:
            raise ValueError(
                f"group_bits ({group_bits}) + user_bits ({user_bits}) must equal L ({self.lbw.L})"
            )
        
        self.group_bits = group_bits
        self.user_bits = user_bits
        self.min_distance = min_distance
        self.max_groups = max_groups
        self.users_per_group = users_per_group
        
        # Validate constraints
        # For min_distance=2, theoretical maximum is 2^(group_bits-1)
        # For other min_distance values, use 2^group_bits as a conservative upper bound
        if min_distance == 2:
            max_possible_groups = 2 ** (group_bits - 1)
        else:
            max_possible_groups = 2 ** group_bits
        max_possible_users_per_group = 2 ** user_bits
        
        if max_groups is not None and max_groups > max_possible_groups:
            raise ValueError(
                f"max_groups ({max_groups}) exceeds maximum allowed by group_bits={group_bits} "
                f"and min_distance={min_distance} (max {max_possible_groups})"
            )
        
        if users_per_group is not None and users_per_group > max_possible_users_per_group:
            raise ValueError(
                f"users_per_group ({users_per_group}) exceeds maximum allowed by user_bits={user_bits} "
                f"(max {max_possible_users_per_group})"
            )
        
        # FingerprintingCode for group codewords (length = group_bits)
        self.group_fingerprinter = FingerprintingCode(
            L=group_bits,
            min_distance=min_distance,
            max_groups=max_groups,
            users_per_group=None  # Not used for group codeword generation
        )
        
        # Store group codewords and user-to-group mapping
        self.group_codewords: dict[int, str] = {}  # group_id -> codeword string
        self.user_to_group: dict[int, int] = {}  # user_id -> group_id
        self.group_to_users: dict[int, list[int]] = {}  # group_id -> list of user_ids
    
    def _generate_group_codewords_direct(self, num_groups: int) -> dict[int, str]:
        """
        Generate group codewords directly without temp file (optimized).
        For min_distance=2, generates even-parity codewords efficiently.
        
        Args:
            num_groups: Number of group codewords to generate
            
        Returns:
            dict mapping group_id (0 to num_groups-1) to codeword string
        """
        if self.min_distance == 2:
            # Optimized even-parity codeword generation (50% fewer iterations)
            # Instead of iterating 0 to 2^L and checking parity, we iterate 0 to 2^(L-1)
            # and append a parity bit to make total parity even. This generates the same
            # codewords in the same order as the original method.
            L = self.group_bits
            max_codewords = 2 ** (L - 1)
            
            if num_groups > max_codewords:
                raise ValueError(
                    f"Requested {num_groups} groups but min_distance=2 with {L} bits "
                    f"only supports maximum {max_codewords} groups"
                )
            
            # Handle edge case: when L=1, we can only have 1 group with codeword "0"
            if L == 1:
                if num_groups > 1:
                    raise ValueError(
                        f"Requested {num_groups} groups but min_distance=2 with 1 bit "
                        f"only supports 1 group"
                    )
                return {0: "0"}
            
            codewords = {}
            count = 0
            
            # Pre-compute format templates (optimization #3)
            format_template = f'0{L}b'
            format_base_template = f'0{L-1}b'
            
            # Optimized even-parity generation: iterate only 2^(L-1) instead of 2^L (50% faster)
            # For each base number, append a parity bit to make total parity even
            # This generates the same codewords in the same order as the original method
            try:
                # Python 3.10+ has efficient bit_count()
                for base in range(2 ** (L - 1)):
                    base_parity = base.bit_count() % 2
                    # Append bit to make total parity even: if base is even parity, append 0; if odd, append 1
                    parity_bit = '0' if base_parity == 0 else '1'
                    codeword_str = format(base, format_base_template) + parity_bit
                    codewords[count] = codeword_str
                    count += 1
                    if count == num_groups:
                        break
            except AttributeError:
                # Fallback for Python < 3.10: use bin().count('1')
                for base in range(2 ** (L - 1)):
                    base_parity = bin(base).count('1') % 2
                    # Append bit to make total parity even: if base is even parity, append 0; if odd, append 1
                    parity_bit = '0' if base_parity == 0 else '1'
                    codeword_str = format(base, format_base_template) + parity_bit
                    codewords[count] = codeword_str
                    count += 1
                    if count == num_groups:
                        break
            
            print(f"Generated {len(codewords)} group codewords directly "
                  f"(min_distance=2, {L} bits, max {max_codewords} possible)")
            
            return codewords
        else:
            # For min_distance > 2, we still need to use FingerprintingCode
            # This is less common, so we'll keep the original approach for now
            raise NotImplementedError(
                f"Direct codeword generation not yet implemented for min_distance={self.min_distance}. "
                "Please use min_distance=2 for optimized performance."
            )
    
    def load_users(self, users_file: str) -> pd.DataFrame:
        """
        Loads user metadata and generates hierarchical codeword structure.
        Uses FingerprintingCode to generate group codewords, then assigns
        simple binary fingerprints to users within each group.
        """
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"User metadata file {users_file} not found")
        
        # Load user metadata
        df = pd.read_csv(users_file)
        if "UserId" not in df.columns:
            raise ValueError("users_file must contain a 'UserId' column.")
        
        df = df.copy()
        df["UserId"] = df["UserId"].astype(int)
        if df["UserId"].duplicated().any():
            raise ValueError("Duplicate UserId entries detected in users file.")
        
        df = df.sort_values("UserId").reset_index(drop=True)
        
        # Enforce hierarchical capacity rules early (before computing num_groups)
        # For min_distance=2, max_groups = 2^(group_bits-1)
        if self.min_distance == 2:
            max_groups_allowed = 2 ** (self.group_bits - 1)
        else:
            max_groups_allowed = 2 ** self.group_bits
        
        # Handle group-only mode (user_bits == 0) and hierarchical mode (user_bits > 0)
        if self.user_bits == 0:
            # Group-only mode: one user per group
            max_users_allowed = max_groups_allowed
        else:
            # Hierarchical mode: max_groups * users_per_group
            users_per_group_auto = 2 ** self.user_bits
            max_users_allowed = max_groups_allowed * users_per_group_auto
        
        if len(df) > max_users_allowed:
            print(
                f"Warning: users file contains {len(df)} entries but hierarchical config "
                f"(G={self.group_bits}, U={self.user_bits}, min_distance={self.min_distance}) "
                f"only supports {max_users_allowed} users. Truncating to {max_users_allowed}."
            )
            df = df.head(max_users_allowed)
        
        # Enforce overall L-bit capacity (additional safety check)
        max_supported_users = 2 ** self.lbw.L
        if len(df) > max_supported_users:
            print(
                f"Warning: users file contains {len(df)} entries but L={self.lbw.L} "
                f"only supports {max_supported_users}. Using first {max_supported_users} users."
            )
            df = df.head(max_supported_users)
        
        # Determine users_per_group and max_groups
        if self.users_per_group is not None:
            users_per_group = self.users_per_group
        else:
            # Default: use maximum capacity based on user_bits
            users_per_group = 2 ** self.user_bits
        
        # Calculate theoretical maximum groups based on min_distance
        # For min_distance=2, theoretical maximum is 2^(group_bits-1)
        # For other min_distance values, use 2^group_bits as a conservative upper bound
        if self.min_distance == 2:
            theoretical_max_groups = 2 ** (self.group_bits - 1)
        else:
            theoretical_max_groups = 2 ** self.group_bits
        
        max_groups_allowed = self.max_groups if self.max_groups is not None else theoretical_max_groups
        max_total_users_allowed = max_groups_allowed * users_per_group
        
        if len(df) > max_total_users_allowed:
            if self.max_groups is not None:
                constraint_str = (
                    f"max_groups={self.max_groups} and users_per_group={users_per_group}"
                )
            else:
                constraint_str = (
                    f"group_bits={self.group_bits} (max {theoretical_max_groups} groups) "
                    f"and users_per_group={users_per_group}"
                )
            print(
                f"Warning: limiting users to {max_total_users_allowed} to satisfy "
                f"{constraint_str}."
            )
            df = df.head(max_total_users_allowed)
        
        self.user_metadata = df
        self.N = len(df)
        self.user_lookup = {int(row["UserId"]): row for _, row in df.iterrows()}
        
        # Truncate users to hierarchical capacity based on min_distance
        # This ensures users don't get assigned to non-existent groups
        max_supported = theoretical_max_groups * users_per_group
        
        if self.N > max_supported:
            print(f"Warning: CSV contains {self.N} users, but hierarchical config (G={self.group_bits}, U={self.user_bits}, min_distance={self.min_distance}) only supports {max_supported} users. Truncating to {max_supported}.")
            self.user_metadata = self.user_metadata.head(max_supported)
            self.N = len(self.user_metadata)
            self.user_lookup = {int(row["UserId"]): row for _, row in self.user_metadata.iterrows()}
        
        # Calculate number of groups needed
        num_groups = (self.N + users_per_group - 1) // users_per_group
        
        # Limit users if they exceed capacity (if max_groups is set)
        if self.max_groups is not None:
            max_users_allowed = self.max_groups * users_per_group
            if self.N > max_users_allowed:
                print(f"Warning: CSV contains {self.N} users, but max_groups={self.max_groups} and "
                      f"users_per_group={users_per_group} only allows {max_users_allowed} users. "
                      f"Using only the first {max_users_allowed} users.")
                self.user_metadata = self.user_metadata.head(max_users_allowed)
                self.N = max_users_allowed
                self.user_lookup = {int(row["UserId"]): row for _, row in self.user_metadata.iterrows()}
                # Recalculate num_groups with limited users
                num_groups = (self.N + users_per_group - 1) // users_per_group
        
        # Check against max_groups constraint if provided
        if self.max_groups is not None:
            if num_groups > self.max_groups:
                raise ValueError(
                    f"Number of groups needed ({num_groups}) exceeds max_groups ({self.max_groups}). "
                    f"Consider increasing max_groups or users_per_group."
                )
        
        # Check against theoretical maximum
        # For min_distance=2, theoretical maximum is 2^(group_bits-1)
        # For other min_distance values, use 2^group_bits as a conservative upper bound
        if self.min_distance == 2:
            max_possible_groups = 2 ** (self.group_bits - 1)
        else:
            max_possible_groups = 2 ** self.group_bits
        if num_groups > max_possible_groups:
            raise ValueError(
                f"Number of groups needed ({num_groups}) exceeds maximum allowed by "
                f"group_bits={self.group_bits} and min_distance={self.min_distance} (max {max_possible_groups})"
            )
        
        # Generate group codewords directly (optimized - no temp file needed)
        # For min_distance=2, use optimized even-parity generation
        # For other min_distance values, fall back to FingerprintingCode
        if self.min_distance == 2:
            # Optimized path: generate codewords directly
            self.group_codewords = self._generate_group_codewords_direct(num_groups)
        else:
            # Fallback: use FingerprintingCode for min_distance > 2
            # This is less common, so we keep the original approach
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                tmp_file.write("UserId,Username\n")
                for gid in range(num_groups):
                    tmp_file.write(f"{gid},{gid}\n")
                tmp_file_path = tmp_file.name
            
            try:
                # Temporarily set users_per_group to 1 to ensure each "user" gets its own group
                original_users_per_group = self.group_fingerprinter.users_per_group
                original_max_groups = self.group_fingerprinter.max_groups
                self.group_fingerprinter.users_per_group = 1
                self.group_fingerprinter.max_groups = num_groups
                self.group_fingerprinter.gen(users_file=tmp_file_path)
                # Restore original values
                self.group_fingerprinter.users_per_group = original_users_per_group
                self.group_fingerprinter.max_groups = original_max_groups
            finally:
                os.unlink(tmp_file_path)
            
            # Extract group codewords
            for group_id in range(num_groups):
                group_codeword_array = self.group_fingerprinter.codewords[group_id]
                self.group_codewords[group_id] = "".join(map(str, group_codeword_array))
        
        # Assign users to groups and build mappings
        # Pre-allocate lists for all groups (optimization #1: avoids repeated dict lookups)
        self.user_to_group = {}
        self.group_to_users = {gid: [] for gid in range(num_groups)}
        
        for index, row in self.user_metadata.iterrows():
            user_id = int(row["UserId"])
            # Use index instead of user_id to ensure groups are 0, 1, 2, ..., num_groups-1
            # This matches the group codewords that were generated for groups 0 to num_groups-1
            group_id = index // users_per_group
            self.user_to_group[user_id] = group_id
            self.group_to_users[group_id].append(user_id)
        
        print(f"Loaded {self.N} users into {num_groups} groups ({users_per_group} users per group)")
        print(f"Group codewords: {self.group_bits} bits, User fingerprints: {self.user_bits} bits")
        if self.max_groups is not None:
            print(f"Maximum groups constraint: {self.max_groups}")
        
        return self.user_metadata
    
    def get_codeword_for_user(self, user_id: int) -> str:
        """
        Returns the combined L-bit codeword for a user: group_code + user_code.
        
        Args:
            user_id (int): The user ID.
        
        Returns:
            str: Combined codeword of length L (group_bits + user_bits).
        """
        self._require_metadata()
        self._validate_user_id(user_id)
        
        # Get group ID
        group_id = self.user_to_group.get(user_id)
        if group_id is None:
            raise ValueError(f"User ID {user_id} not assigned to any group.")
        
        # Get group codeword
        group_code = self.group_codewords.get(group_id)
        if group_code is None:
            raise ValueError(f"Group {group_id} codeword not found.")
        
        # Get user index within group
        users_in_group = self.group_to_users.get(group_id, [])
        try:
            user_index_in_group = users_in_group.index(user_id)
        except ValueError:
            raise ValueError(f"User ID {user_id} not found in group {group_id}.")
        
        # Generate user fingerprint (handle group-only mode where user_bits=0)
        if self.user_bits == 0:
            user_code = ""  # Group-only mode: no user fingerprint
        else:
            user_code = generate_user_fingerprint(user_index_in_group, self.user_bits)
        
        # Combine: group_code + user_code
        combined_code = group_code + user_code
        
        if len(combined_code) != self.lbw.L:
            raise ValueError(
                f"Combined codeword length mismatch: expected {self.lbw.L}, got {len(combined_code)}"
            )
        
        return combined_code
    
    def _log_embed(self, user_id: int, codeword: str):
        group_id = self.user_to_group.get(user_id, None)
        if group_id is not None:
            group_code = codeword[:self.group_bits]
            user_code = codeword[self.group_bits:] if self.user_bits > 0 else ""
            print(f"User ID {user_id} belongs to Group {group_id}")
            if self.user_bits == 0:
                print(f"Embedding group-only codeword '{codeword}' for User ID {user_id} "
                      f"(Group {group_id}: '{group_code}')...")
            else:
                print(f"Embedding hierarchical codeword '{codeword}' for User ID {user_id} "
                      f"(Group {group_id}: '{group_code}' + User: '{user_code}')...")
        else:
            print(f"Embedding codeword '{codeword}' for User ID {user_id} (hierarchical scheme)...")
    
    def embed(self, master_key: bytes, user_id: int, prompt: str, **kwargs) -> str:
        """Embeds the hierarchical codeword for a user."""
        self._require_metadata()
        self._validate_user_id(user_id)
        combined_code = self.get_codeword_for_user(user_id)
        self._log_embed(user_id, combined_code)
        return self.lbw.embed(master_key, combined_code, prompt, **kwargs)
    
    def trace(self, master_key: bytes, text: str, **kwargs) -> list[dict]:
        """
        Traces watermarked text using staged tracing for better performance:
        1. Recovering L bits using LBitWatermarker.detect
        2. Splitting into group_bits and user_bits
        3. Identifying suspect groups by comparing group bits only
        4. Searching users only within suspect groups by comparing full codeword
        """
        self._require_metadata()
        
        print("Extracting L-bit codeword from text...")
        recovered_bits = self.lbw.detect(master_key, text, **kwargs)
        print(f"  - Recovered Codeword: {recovered_bits}")
        
        return self.trace_from_codeword(recovered_bits)
    
    def trace_from_codeword(self, recovered_bits: str) -> list[dict]:
        """
        Traces users from a recovered codeword (without running detection).
        This allows separation of detection and tracing for performance measurement.
        
        Args:
            recovered_bits (str): The recovered L-bit codeword from detection.
        
        Returns:
            list[dict]: List of accused users with match scores.
        """
        self._require_metadata()
        
        if len(recovered_bits) != self.lbw.L:
            print(f"Warning: Recovered codeword length ({len(recovered_bits)}) != L ({self.lbw.L})")
            return []
        
        # Split recovered bits
        recovered_group_bits = recovered_bits[:self.group_bits]
        recovered_user_bits = recovered_bits[self.group_bits:]
        
        print(f"  - Group part: {recovered_group_bits} ({self.group_bits} bits)")
        print(f"  - User part: {recovered_user_bits} ({self.user_bits} bits)")
        
        # Stage 1: Find suspect groups by comparing group bits only
        valid_group_positions = [i for i, bit in enumerate(recovered_group_bits) 
                                if bit not in ('⊥', '*', '?')]
        
        if not valid_group_positions:
            print("No valid bits in group part; cannot identify groups.")
            return []
        
        # Calculate distances for all groups
        group_distances = []
        for group_id, group_codeword in self.group_codewords.items():
            distance = sum(
                recovered_group_bits[i] != group_codeword[i]
                for i in valid_group_positions
            )
            group_distances.append((group_id, distance))
        
        if not group_distances:
            print("Could not identify any groups.")
            return []
        
        # Find minimum distance and collect all groups with that distance (suspect groups)
        min_group_distance = min(dist for _, dist in group_distances)
        suspect_groups = [group_id for group_id, dist in group_distances if dist == min_group_distance]
        
        print(f"Identified {len(suspect_groups)} suspect group(s) with Hamming distance {min_group_distance}: {suspect_groups}")
        
        # Stage 2: Search users only within suspect groups, comparing full codeword
        valid_full_positions = [i for i, bit in enumerate(recovered_bits) 
                               if bit not in ('⊥', '*', '?')]
        
        if not valid_full_positions:
            print("No valid bits in recovered codeword; cannot identify users.")
            return []
        
        # Handle group-only mode (user_bits=0): return all users in suspect groups
        if self.user_bits == 0:
            accused = []
            for group_id in suspect_groups:
                users_in_group = self.group_to_users.get(group_id, [])
                for user_id in users_in_group:
                    row = self.user_lookup.get(user_id)
                    accused.append({
                        "user_id": user_id,
                        "username": row.get("Username") if row is not None else None,
                        "match_score_percent": 100.0,  # Perfect match in group-only mode
                        "group_id": group_id
                    })
            print(f"Identified User(s) {[u['user_id'] for u in accused]} in suspect groups (group-only mode)")
            return accused
        
        # Compare full codeword against users in suspect groups
        all_candidates = []
        best_full_distance = float('inf')
        
        for group_id in suspect_groups:
            users_in_group = self.group_to_users.get(group_id, [])
            if not users_in_group:
                continue
            
            for user_id in users_in_group:
                # Get the full codeword for this user
                user_codeword = self.get_codeword_for_user(user_id)
                
                # Calculate Hamming distance on full codeword (only valid positions)
                distance = sum(
                    recovered_bits[i] != user_codeword[i]
                    for i in valid_full_positions
                )
                
                if distance < best_full_distance:
                    best_full_distance = distance
                    all_candidates = [(user_id, group_id, distance)]
                elif distance == best_full_distance:
                    all_candidates.append((user_id, group_id, distance))
        
        if not all_candidates:
            print("No users found in suspect groups.")
            return []
        
        # Return all users tied for best match
        accused = []
        total_valid = len(valid_full_positions)
        for user_id, group_id, distance in all_candidates:
            row = self.user_lookup.get(user_id)
            match_score = ((total_valid - distance) / total_valid * 100) if total_valid > 0 else 0.0
            accused.append({
                "user_id": user_id,
                "username": row.get("Username") if row is not None else None,
                "match_score_percent": match_score,
                "group_id": group_id
            })
        
        print(f"Identified User(s) {[u['user_id'] for u in accused]} "
              f"in suspect groups (full codeword Hamming distance: {best_full_distance})")
        
        return accused