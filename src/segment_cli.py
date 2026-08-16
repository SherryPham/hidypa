# segment_cli.py: shared argparse / construction helpers for the Segment-WM
# baseline row, so the seven evaluation scripts stay in sync.

import json
import os
import pickle

from .segment_watermark import SegmentMultiUserWatermarker, SegmentWatermarker

SEGMENT_SCHEME = "segment"

# Schemes that use a flat codeword (binary expansion of the user ID) and are
# therefore handled by the 'naive' code paths in the evaluation scripts.
FLAT_SCHEMES = ("naive", SEGMENT_SCHEME)


def is_flat_scheme(scheme: str) -> bool:
    return scheme in FLAT_SCHEMES


def add_segment_args(parser):
    """Register the --segment-* options on an evaluation script's parser."""
    group = parser.add_argument_group(
        "Segment-WM baseline (only used when --scheme segment)"
    )
    group.add_argument(
        "--segment-rs", type=int, nargs=3, metavar=("N", "K", "M"), default=None,
        help="Reed-Solomon parameters: n encoded symbols, k payload symbols, "
             "m bits per symbol (k*m must equal --l-bits). Default for L=8: 4 2 4",
    )
    group.add_argument(
        "--segment-gamma", type=float, default=0.5,
        help="Green-list fraction (default: 0.5, the Segment-WM paper value)",
    )
    group.add_argument(
        "--segment-delta", type=float, default=6.0,
        help="Green-list logit boost (default: 6.0, the Segment-WM paper value). "
             "Not comparable to Hi-DyPa's --delta, which scales a Gaussian score vector.",
    )
    group.add_argument(
        "--segment-ngram", type=int, default=1,
        help="Context width used to seed the green list (default: 1)",
    )
    group.add_argument(
        "--segment-hash-key", type=int, default=35317,
        help="Salt for the context hash (default: 35317, the reference value)",
    )
    group.add_argument(
        "--segment-assign", type=str, default="rs", choices=["rs", "rsbh"],
        help="Position -> segment assignment: 'rs' draws it from the context PRNG, "
             "'rsbh' reads a frequency-balanced hash map (needs --segment-bh-map)",
    )
    group.add_argument(
        "--segment-bh-map", type=str, default=None,
        help="Path to a balanced-hash map produced by helper_scripts/build_segment_bh_map.py",
    )
    group.add_argument(
        "--segment-temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7, matching the Hi-DyPa pipeline; "
             "the Segment-WM paper uses 1.0)",
    )
    group.add_argument(
        "--segment-top-p", type=float, default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
    )
    group.add_argument(
        "--segment-top-k", type=int, default=50,
        help="Top-k sampling cutoff (default: 50, matching the Hi-DyPa pipeline; "
             "use 0 for the Segment-WM setting)",
    )
    group.add_argument(
        "--segment-suppress-eos", action="store_true",
        help="Force generation to run for the full length by suppressing EOS "
             "(the reference implementation does this; off by default so that "
             "generation stops exactly like the other schemes)",
    )
    group.add_argument(
        "--segment-z-threshold", type=float, default=-1.0,
        help="Aggregate z-score below which the segment detector abstains. The "
             "default (-1.0) never abstains, matching the reference implementation, "
             "so FP = 1 - Acc for this row. Pass a positive value (e.g. the same 4.0 "
             "used by --z-threshold) for the abstaining variant; see "
             "docs/SEGMENT_WM_BASELINE.md for why that costs accuracy on repetitive text.",
    )
    group.add_argument(
        "--segment-z-mode", type=str, default="calibrated", choices=["calibrated", "raw"],
        help="Statistic used for abstention: 'calibrated' centres the sum of "
             "per-segment maxima on its null mean (approximately N(0,1) on "
             "unwatermarked text, so --z-threshold means the same thing as for the "
             "other schemes); 'raw' is the uncentred formula from the Segment-WM "
             "reference code, which reads above 4 even on unwatermarked text",
    )
    group.add_argument(
        "--segment-syndrome-decode", action="store_true",
        help="Use bounded-distance RS syndrome decoding instead of the default "
             "exhaustive nearest-codeword decoding",
    )
    return parser


def load_bh_map(path: str) -> dict:
    """Load a balanced-hash map (pickle or JSON) as {token_id: segment_index}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Balanced-hash map not found: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def build_segment_watermarker(args, model) -> SegmentMultiUserWatermarker:
    """Construct the Segment-WM multi-user watermarker from parsed CLI args."""
    n = k = m = None
    if getattr(args, "segment_rs", None):
        n, k, m = args.segment_rs

    bh_map = None
    if args.segment_assign == "rsbh":
        if not args.segment_bh_map:
            raise ValueError(
                "--segment-assign rsbh requires --segment-bh-map "
                "(build one with helper_scripts/build_segment_bh_map.py)"
            )
        bh_map = load_bh_map(args.segment_bh_map)

    z_threshold = args.segment_z_threshold

    watermarker = SegmentWatermarker(
        model=model,
        L=args.l_bits,
        n_segments=n,
        k_segments=k,
        segment_bit=m,
        gamma=args.segment_gamma,
        delta=args.segment_delta,
        ngram=args.segment_ngram,
        hash_key=args.segment_hash_key,
        z_threshold=z_threshold,
        gate_on_z=z_threshold >= 0,
        z_mode=args.segment_z_mode,
        assignment=args.segment_assign,
        bh_map=bh_map,
        temperature=args.segment_temperature,
        top_p=args.segment_top_p,
        top_k=args.segment_top_k,
        suppress_eos=args.segment_suppress_eos,
        exhaustive_decode=not args.segment_syndrome_decode,
    )
    print("  Segment-WM configuration:")
    for key, value in watermarker.describe().items():
        print(f"    - {key}: {value}")
    if not watermarker.gate_on_z:
        print("    NOTE: the segment decoder never abstains, so this row's "
              "false positive rate equals 1 - accuracy.")
    return SegmentMultiUserWatermarker(watermarker)


def scheme_dir_parts(args) -> list[str]:
    """Output-directory components for the scheme under evaluation."""
    if args.scheme == "hi_dypa":
        return ["hi_dypa", f"G{args.group_bits}_U{args.user_bits}"]
    if args.scheme == SEGMENT_SCHEME:
        return ["segment", f"L{args.l_bits}"]
    return ["naive", f"L{args.l_bits}"]


def config_name(args) -> str:
    """Key used in the shared seeds.txt file."""
    if args.scheme == "hi_dypa":
        return f"hi_dypa_G{args.group_bits}_U{args.user_bits}"
    if args.scheme == SEGMENT_SCHEME:
        return f"segment_L{args.l_bits}"
    return f"naive_L{args.l_bits}"
