# segment_watermark.py: Segment-assignment multi-bit watermarking baseline.
#
# Re-implementation of the scheme from "Provably Robust Multi-bit Watermarking
# for AI-generated Text" (referred to here as Segment-WM) inside the Hi-DyPa
# framework, so it can be evaluated as one extra row of Tables II-XIII using the
# same models, prompts, users, attacks and metrics as MAU and Hi-DyPa.
#
# Scheme summary (reference: segment-wm/wm/generator.py, segment-wm/wm/detector.py):
#   * The b-bit payload is split into k symbols of m bits (b = k * m) and
#     Reed-Solomon encoded into n symbols over GF(2^m).
#   * At every generated position the context n-gram seeds a PRNG that draws a
#     green list of gamma * V tokens; the green bias vector is cyclically
#     shifted by the value of the symbol assigned to that position and delta is
#     added to the shifted green tokens.
#   * The position -> segment assignment is either drawn from the same PRNG
#     ('rs') or read from a frequency-balanced hash of the previous token
#     ('rsbh', the paper's headline variant).
#   * Detection recomputes the green list per position, accumulates a COUNT
#     array of size 2^m per segment, takes the arg-max per segment, and RS
#     decodes the resulting n symbols back to the payload.
#
# The public API mirrors LBitWatermarker so that NaiveMultiUserWatermarker can
# drive it unchanged: keygen / embed(master_key, bitstring, prompt) /
# detect(master_key, text) -> L-character string.

import functools
import hashlib
import hmac
import math
import os

import torch
from transformers import LogitsProcessor

from .models import LanguageModel
from .watermark import NaiveMultiUserWatermarker
from .reedsolomon import (
    ReedSolomon,
    ReedSolomonCodebook,
    payload_to_symbols,
    symbols_to_payload,
)

# Payload sizes up to this many codewords use exhaustive nearest-codeword
# decoding instead of bounded-distance syndrome decoding.
MAX_EXHAUSTIVE_CODEBOOK = 1 << 16

# Default RS parameters per payload width. For L = 8 the rate constraints used
# by the Segment-WM authors' rs_search.py (k/n >= 0.6) cannot be met with any
# (n, k, m) since k * m = 8 forces k <= 2, so we use the smallest even-redundancy
# code that still corrects one symbol error: RS(4, 2) over GF(16).
DEFAULT_RS_PARAMS = {
    8: (4, 2, 4),      # n, k, m  -> t = 1
    12: (6, 3, 4),     # t = 1 (rs_search.py reports (6, 3, 4) for b = 12)
    16: (6, 4, 4),     # t = 1
    32: (6, 4, 8),     # matches the reference README example
}


def derive_segment_seed(master_key: bytes) -> int:
    """Derive the Segment-WM integer PRNG seed from a Hi-DyPa master key."""
    digest = hmac.new(master_key, b"segment-wm-seed", hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


@functools.lru_cache(maxsize=8192)
def max_count_moments(num_tokens: int, num_candidates: int, gamma: float) -> tuple[float, float]:
    """
    Mean and variance of max(C_1..C_M) where C_v ~ Binomial(num_tokens, gamma)
    are the per-shift green-token counts of one segment under the null
    hypothesis (unwatermarked text).

    The reference Segment-WM z-score compares the sum of per-segment maxima
    against gamma * T, which ignores the upward bias of taking a maximum over
    2^m candidates. That bias is large: with T = 124 and M = 16, unwatermarked
    GPT-2 text scores z > 4. Centring on the true null moments instead makes the
    statistic approximately N(0, 1) under H0, so the same z-threshold used for
    the L-bit schemes becomes meaningful for the segment scheme too.
    """
    if num_tokens <= 0:
        return 0.0, 0.0

    log_gamma = math.log(gamma) if gamma > 0 else float("-inf")
    log_one_minus = math.log1p(-gamma) if gamma < 1 else float("-inf")
    log_fact_n = math.lgamma(num_tokens + 1)

    # Survival function of the maximum: P(max > x) = 1 - F(x)^M
    cdf = 0.0
    mean = 0.0
    second = 0.0
    for x in range(num_tokens + 1):
        log_pmf = (
            log_fact_n
            - math.lgamma(x + 1)
            - math.lgamma(num_tokens - x + 1)
            + x * log_gamma
            + (num_tokens - x) * log_one_minus
        )
        cdf += math.exp(log_pmf)
        cdf = min(cdf, 1.0)
        survival = 1.0 - cdf ** num_candidates
        # E[X] = sum_{x>=0} P(X > x); E[X^2] = sum_{x>=0} (2x + 1) P(X > x)
        mean += survival
        second += (2 * x + 1) * survival
        if survival < 1e-15 and x > num_tokens * gamma:
            break

    variance = max(second - mean * mean, 0.0)
    return mean, variance


class SegmentLogitsProcessor(LogitsProcessor):
    """Applies the shifted green-list bias for one RS-encoded payload."""

    def __init__(self, watermarker: "SegmentWatermarker", base_seed: int,
                 gf_symbols: list[int]):
        self.wm = watermarker
        self.base_seed = base_seed
        self.gf_symbols = gf_symbols
        self.positions_watermarked = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        wm = self.wm
        if input_ids.shape[1] < wm.ngram:
            return scores

        width = min(wm.vocab_size, scores.shape[-1])
        context = input_ids[0, -wm.ngram:]
        seed = wm.seed_from_context(self.base_seed, context)

        green, segment_index = wm.draw_green_and_segment(seed, context)
        symbol = self.gf_symbols[segment_index]

        # bias.roll(-symbol)[j] != 0  <=>  (j + symbol) mod V in green
        shifted = torch.remainder(green - symbol, wm.vocab_size)
        shifted = shifted[shifted < width].to(scores.device)

        scores[0].index_add_(
            0, shifted, torch.full_like(shifted, wm.delta, dtype=scores.dtype)
        )

        if wm.suppress_eos and wm.eos_token_id is not None and wm.eos_token_id < width:
            scores[0, wm.eos_token_id] = -65000.0

        self.positions_watermarked += 1
        return scores


class SegmentWatermarker:
    """
    Segment-assignment multi-bit watermarker with a Reed-Solomon outer code.

    Drop-in replacement for LBitWatermarker inside NaiveMultiUserWatermarker.
    """

    def __init__(
        self,
        model: LanguageModel,
        L: int = 8,
        n_segments: int | None = None,
        k_segments: int | None = None,
        segment_bit: int | None = None,
        gamma: float = 0.5,
        delta: float = 6.0,
        ngram: int = 1,
        hash_key: int = 35317,
        z_threshold: float = 4.0,
        gate_on_z: bool = True,
        z_mode: str = "calibrated",
        scoring_method: str = "none",
        assignment: str = "rs",
        bh_map: dict | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        suppress_eos: bool = False,
        exhaustive_decode: bool = True,
    ):
        self.model = model
        self.L = L

        if n_segments is None or k_segments is None or segment_bit is None:
            if L not in DEFAULT_RS_PARAMS:
                raise ValueError(
                    f"No default RS parameters for L={L}; pass n_segments/k_segments/segment_bit"
                )
            n_segments, k_segments, segment_bit = DEFAULT_RS_PARAMS[L]

        if k_segments * segment_bit != L:
            raise ValueError(
                f"k_segments ({k_segments}) * segment_bit ({segment_bit}) must equal L ({L})"
            )

        self.n_segments = n_segments
        self.k_segments = k_segments
        self.segment_bit = segment_bit
        self.symbol_space = 1 << segment_bit

        self.rs = ReedSolomon(n=n_segments, k=k_segments, m=segment_bit)
        self.exhaustive_decode = exhaustive_decode
        self.num_payloads = 1 << L
        self.codebook = None
        if exhaustive_decode and self.num_payloads <= MAX_EXHAUSTIVE_CODEBOOK:
            self.codebook = ReedSolomonCodebook(self.rs, self.num_payloads)

        self.gamma = gamma
        self.delta = delta
        self.ngram = ngram
        self.hash_key = hash_key
        self.z_threshold = z_threshold
        self.gate_on_z = gate_on_z
        if z_mode not in ("calibrated", "raw"):
            raise ValueError(f"z_mode must be 'calibrated' or 'raw', got {z_mode!r}")
        self.z_mode = z_mode
        if scoring_method not in ("none", "v1", "v2"):
            raise ValueError(
                f"scoring_method must be 'none', 'v1' or 'v2', got {scoring_method!r}"
            )
        self.scoring_method = scoring_method

        if assignment not in ("rs", "rsbh"):
            raise ValueError(f"assignment must be 'rs' or 'rsbh', got {assignment!r}")
        if assignment == "rsbh" and not bh_map:
            raise ValueError(
                "assignment='rsbh' requires a balanced-hash map "
                "(build one with helper_scripts/build_segment_bh_map.py)"
            )
        self.assignment = assignment
        self.bh_map = bh_map

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.suppress_eos = suppress_eos

        self.vocab_size = int(model.vocab_size)
        self.green_size = int(self.gamma * self.vocab_size)
        self.eos_token_id = getattr(model.tokenizer, "eos_token_id", None)

        # CPU generator, matching the reference implementation.
        self._rng = torch.Generator()
        self._offsets = torch.arange(self.symbol_space, dtype=torch.long)

    # ------------------------------------------------------------------ keys

    def keygen(self, key_length: int = 32) -> bytes:
        return os.urandom(key_length)

    def seed_from_context(self, base_seed: int, context_ids) -> int:
        """Reference 'hash' seeding: seed = seed * salt + token, folded to 64 bits."""
        seed = base_seed
        for token in context_ids:
            seed = (seed * self.hash_key + int(token)) % (2 ** 64 - 1)
        return seed

    def draw_green_and_segment(self, seed: int, context_ids) -> tuple[torch.Tensor, int]:
        """
        Draw the green list and the segment index for one position.

        The draw order (permutation first, then the segment index) matches the
        reference implementation so that generator and detector stay in sync.
        """
        self._rng.manual_seed(seed)
        permutation = torch.randperm(self.vocab_size, generator=self._rng)
        green = permutation[: self.green_size]

        if self.assignment == "rsbh":
            key = int(context_ids[0])
            segment_index = self.bh_map.get(key, key % self.n_segments)
        else:
            segment_index = int(
                torch.randint(
                    low=0, high=self.n_segments, size=(1,), generator=self._rng
                ).item()
            )
        return green, segment_index

    # -------------------------------------------------------------- encoding

    def set_payload_space(self, num_payloads: int) -> None:
        """
        Restrict nearest-codeword decoding to the payloads actually in use.

        The other schemes match a recovered codeword only against real users
        (`decode_naive_user` iterates `range(muw.N)`), so the segment decoder
        searches the same set rather than all 2^L payloads.
        """
        num_payloads = max(1, min(int(num_payloads), 1 << self.L))
        if num_payloads == self.num_payloads:
            return
        self.num_payloads = num_payloads
        if self.exhaustive_decode and num_payloads <= MAX_EXHAUSTIVE_CODEBOOK:
            self.codebook = ReedSolomonCodebook(self.rs, num_payloads)

    def encode_payload(self, payload: int) -> list[int]:
        symbols = payload_to_symbols(payload, self.k_segments, self.segment_bit)
        return self.rs.encode(symbols)

    # -------------------------------------------------------------- generate

    def embed(self, master_secret_key: bytes, message: str, prompt: str,
              max_new_tokens: int = 512, **kwargs) -> str:
        if len(message) != self.L or any(c not in {"0", "1"} for c in message):
            raise ValueError(f"Message must be a {self.L}-bit binary string")

        payload = int(message, 2)
        gf_symbols = self.encode_payload(payload)
        base_seed = derive_segment_seed(master_secret_key)

        tokenizer = self.model.tokenizer
        device = self.model.device

        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        processor = SegmentLogitsProcessor(self, base_seed, gf_symbols)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            logits_processor=[processor],
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids),
            **kwargs,
        )
        if self.top_k:
            gen_kwargs["top_k"] = self.top_k

        with torch.no_grad():
            output_ids = self.model._model.generate(input_ids, **gen_kwargs)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ---------------------------------------------------------------- detect

    def score_text(self, master_secret_key: bytes, text: str) -> dict:
        """
        Accumulate the per-segment COUNT arrays and decode.

        Two accumulators are maintained in a single pass:

        * `counts` uses every scored position (or the subset selected by
          `scoring_method`) and is what the payload is decoded from, so the
          decoder keeps the full watermark signal.
        * `unique_counts` uses only the first occurrence of each
          (context, token) pair. Repeated n-grams reuse the same green list, so
          their contributions are perfectly correlated and inflate the aggregate
          statistic; dropping them makes the null distribution of the
          null-centred z-score approximately N(0, 1) on unwatermarked text,
          which is what allows the same --z-threshold used by the L-bit schemes
          to gate the segment detector.
        """
        tokenizer = self.model.tokenizer
        base_seed = derive_segment_seed(master_secret_key)

        token_ids = tokenizer.encode(text)
        start_pos = self.ngram + 1

        counts = torch.zeros(self.n_segments, self.symbol_space)
        unique_counts = torch.zeros(self.n_segments, self.symbol_space)
        tokens_per_segment = [0] * self.n_segments
        unique_tokens_per_segment = [0] * self.n_segments
        scored_tokens = 0
        unique_tokens = 0

        seen_decode = set()
        seen_unique = set()

        for cur_pos in range(start_pos, len(token_ids)):
            context = token_ids[cur_pos - self.ngram: cur_pos]
            token_id = int(token_ids[cur_pos]) % self.vocab_size

            pair_key = tuple(context) + (token_ids[cur_pos],)
            is_unique = pair_key not in seen_unique
            if is_unique:
                seen_unique.add(pair_key)

            use_for_decode = True
            if self.scoring_method == "v1":
                context_key = tuple(context)
                use_for_decode = context_key not in seen_decode
                seen_decode.add(context_key)
            elif self.scoring_method == "v2":
                use_for_decode = is_unique

            if not (use_for_decode or is_unique):
                continue

            seed = self.seed_from_context(base_seed, context)
            green, segment_index = self.draw_green_and_segment(seed, context)

            # score[v] == 1  <=>  (token_id + v) mod V is in the green list
            offsets = torch.remainder(token_id + self._offsets, self.vocab_size)
            mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            mask[green] = True
            hits = mask[offsets].float()

            if use_for_decode:
                counts[segment_index] += hits
                tokens_per_segment[segment_index] += 1
                scored_tokens += 1
            if is_unique:
                unique_counts[segment_index] += hits
                unique_tokens_per_segment[segment_index] += 1
                unique_tokens += 1

        if scored_tokens == 0:
            return {
                "payload": None,
                "symbols": [],
                "z_score": 0.0,
                "z_score_raw": 0.0,
                "z_score_calibrated": 0.0,
                "num_tokens": 0,
                "num_unique_tokens": unique_tokens,
                "tokens_per_segment": tokens_per_segment,
                "symbol_distance": None,
                "rs_corrected": False,
            }

        symbols = [int(counts[i].argmax().item()) for i in range(self.n_segments)]

        # Reference statistic (segment-wm/wm/detector.py::get_pvalue_segment_based).
        max_sum = float(counts.max(dim=1).values.sum().item())
        raw_variance = self.gamma * (1.0 - self.gamma) * scored_tokens
        z_raw = (
            (max_sum - self.gamma * scored_tokens) / math.sqrt(raw_variance)
            if raw_variance > 0 else 0.0
        )

        # Null-centred statistic over de-duplicated positions.
        unique_max_sum = float(unique_counts.max(dim=1).values.sum().item())
        null_mean = 0.0
        null_var = 0.0
        for num_tokens in unique_tokens_per_segment:
            mean, variance = max_count_moments(num_tokens, self.symbol_space, self.gamma)
            null_mean += mean
            null_var += variance
        z_calibrated = (
            (unique_max_sum - null_mean) / math.sqrt(null_var) if null_var > 0 else 0.0
        )

        if self.codebook is not None:
            payload, distance, _ = self.codebook.decode(symbols)
            corrected = True
        else:
            message_symbols, corrected = self.rs.decode_safe(symbols)
            payload = symbols_to_payload(message_symbols, self.segment_bit)
            distance = None

        return {
            "payload": payload,
            "symbols": symbols,
            "z_score": z_calibrated if self.z_mode == "calibrated" else z_raw,
            "z_score_raw": z_raw,
            "z_score_calibrated": z_calibrated,
            "num_tokens": scored_tokens,
            "num_unique_tokens": unique_tokens,
            "tokens_per_segment": tokens_per_segment,
            "symbol_distance": distance,
            "rs_corrected": corrected,
        }

    def detect(self, master_secret_key: bytes, text: str, **kwargs) -> str:
        """Recover the L-bit payload, or a fully undecided codeword when the
        aggregate z-score does not clear the detection threshold."""
        result = self.score_text(master_secret_key, text)
        if result["payload"] is None:
            return "⊥" * self.L
        if self.gate_on_z and result["z_score"] < self.z_threshold:
            return "⊥" * self.L
        return format(result["payload"], f"0{self.L}b")

    def compute_z_score(self, master_secret_key: bytes, text: str) -> float:
        """Segment-based z-score, comparable in role to the L-bit average z-score."""
        return self.score_text(master_secret_key, text)["z_score"]

    def describe(self) -> dict:
        return {
            "scheme": "segment",
            "L": self.L,
            "rs": {"n": self.n_segments, "k": self.k_segments, "m": self.segment_bit,
                   "t": self.rs.t},
            "gamma": self.gamma,
            "delta": self.delta,
            "ngram": self.ngram,
            "assignment": self.assignment,
            "decoder": "exhaustive-nearest" if self.codebook is not None else "syndrome",
            "gate_on_z": self.gate_on_z,
            "z_mode": self.z_mode,
            "z_threshold": self.z_threshold,
            "scoring_method": self.scoring_method,
            "sampling": {"temperature": self.temperature, "top_p": self.top_p,
                         "top_k": self.top_k},
            "suppress_eos": self.suppress_eos,
        }


class SegmentMultiUserWatermarker(NaiveMultiUserWatermarker):
    """
    Flat multi-user tracing on top of the Segment-WM embedding.

    Like the MAU baseline, a user's codeword is the binary expansion of their
    user ID, so this appears in the paper tables as (Lg, Lu) = (0, 8). All the
    tracing / merging / feasible-set machinery in the evaluation scripts is
    inherited from NaiveMultiUserWatermarker unchanged.
    """

    def __init__(self, segment_watermarker: SegmentWatermarker):
        super().__init__(lbit_watermarker=segment_watermarker)

    def _initialize_metadata(self, df):
        super()._initialize_metadata(df)
        # Decode against the loaded users only, matching decode_naive_user.
        self.lbw.set_payload_space(self.N)

    def _log_embed(self, user_id: int, codeword: str):
        print(f"Embedding payload '{codeword}' for User ID {user_id} (segment scheme)...")

