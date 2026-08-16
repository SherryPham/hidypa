#!/usr/bin/env python3
"""
Build the balanced-hash (BH) token -> segment map used by the 'rsbh' variant of
the Segment-WM baseline.

This reproduces segment-wm/balance_hash/helper.py without its compiled C++
helper: the vocabulary is shuffled with a secret key, and the shuffled sequence
of token frequencies is split into n contiguous buckets whose frequency mass is
as equal as possible. The minimax contiguous partition is solved exactly by
binary search over the bucket capacity plus a greedy sweep, which matches the
optimum found by the reference dynamic program.

Usage
-----
    python helper_scripts/build_segment_bh_map.py \
        --model gpt2 \
        --segments 4 \
        --corpus assets/prompts.txt \
        --output assets/segment_bh_map_gpt2_n4.json

The map must be identical at generation and detection time, so build it once
per (model, n) pair and pass the same file to every evaluation script via
--segment-bh-map.
"""

import argparse
import hashlib
import json
import os
import pickle
import struct
import sys

import numpy as np

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.utils import get_model  # noqa: E402


def key_to_seed(key: int) -> int:
    """Reference helper: SHA-256 of the key, first 4 bytes as a uint32 seed."""
    digest = hashlib.sha256(key.to_bytes(16, "little", signed=False)).digest()
    return struct.unpack("I", digest[0:4])[0]


def min_max_partition(weights: np.ndarray, buckets: int) -> list[int]:
    """
    Split `weights` into `buckets` contiguous runs minimising the largest run
    sum. Returns the run lengths.
    """
    total = float(weights.sum())
    low, high = float(weights.max()), total

    def runs_needed(capacity: float) -> int:
        count, current = 1, 0.0
        for w in weights:
            if current + w > capacity:
                count += 1
                current = float(w)
            else:
                current += float(w)
        return count

    for _ in range(200):
        mid = (low + high) / 2.0
        if runs_needed(mid) <= buckets:
            high = mid
        else:
            low = mid

    capacity = high
    lengths, current, length = [], 0.0, 0
    for w in weights:
        if current + w > capacity and lengths.__len__() < buckets - 1 and length > 0:
            lengths.append(length)
            current, length = float(w), 1
        else:
            current += float(w)
            length += 1
    lengths.append(length)

    # Pad in case the sweep produced fewer runs than requested.
    while len(lengths) < buckets:
        biggest = int(np.argmax(lengths))
        if lengths[biggest] < 2:
            lengths.append(0)
            continue
        half = lengths[biggest] // 2
        lengths[biggest] -= half
        lengths.append(half)
    return lengths


def token_frequencies(model_name: str, corpus_paths: list[str], vocab_size: int) -> np.ndarray:
    model = get_model(model_name)
    tokenizer = model.tokenizer
    vocab_size = vocab_size or int(model.vocab_size)

    freq = np.zeros(vocab_size, dtype=np.float64)
    total = 0
    for path in corpus_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for token_id in tokenizer.encode(line):
                    if 0 <= token_id < vocab_size:
                        freq[token_id] += 1
                        total += 1
    print(f"  Counted {total} tokens over {len(corpus_paths)} corpus file(s)")
    # Reference behaviour: tokens never observed get a frequency of 1.
    freq[freq == 0] = 1.0
    return freq


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model whose tokenizer defines the vocabulary")
    parser.add_argument("--segments", type=int, required=True,
                        help="Number of buckets (n, the RS-encoded segment count)")
    parser.add_argument("--corpus", type=str, nargs="+", default=["assets/prompts.txt"],
                        help="Text files used to estimate token frequencies")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path (.json or .pkl)")
    parser.add_argument("--key", type=int, default=426371835,
                        help="Shuffle key (default: the reference value)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Override the vocabulary size (defaults to the model's)")
    args = parser.parse_args()

    corpus_paths = []
    for path in args.corpus:
        resolved = path if os.path.isabs(path) else os.path.join(parent_dir, path)
        if not os.path.exists(resolved):
            parser.error(f"Corpus file not found: {resolved}")
        corpus_paths.append(resolved)

    print(f"[1/3] Counting token frequencies with the {args.model} tokenizer...")
    freq = token_frequencies(args.model, corpus_paths, args.vocab_size)
    vocab_size = len(freq)

    print(f"[2/3] Balancing {vocab_size} tokens into {args.segments} buckets...")
    indices = np.arange(vocab_size)
    rng = np.random.RandomState(key_to_seed(args.key))
    rng.shuffle(indices)
    shuffled = freq[indices]

    lengths = min_max_partition(shuffled, args.segments)

    mapping = {}
    base = 0
    for segment_index, length in enumerate(lengths):
        for offset in range(length):
            mapping[int(indices[base + offset])] = segment_index
        base += length

    masses = []
    base = 0
    for length in lengths:
        masses.append(float(shuffled[base:base + length].sum()))
        base += length
    spread = (max(masses) - min(masses)) / max(1.0, float(np.mean(masses)))
    print(f"  Bucket sizes:  {lengths}")
    print(f"  Bucket masses: {[round(x) for x in masses]}  (spread {spread:.4%})")

    print(f"[3/3] Writing {len(mapping)} entries to {args.output}...")
    output = args.output if os.path.isabs(args.output) else os.path.join(parent_dir, args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if output.endswith(".json"):
        with open(output, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in mapping.items()}, f)
    else:
        with open(output, "wb") as f:
            pickle.dump(mapping, f)
    print("Done.")


if __name__ == "__main__":
    main()
