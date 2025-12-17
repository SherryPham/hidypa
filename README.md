# Cryptographic Watermarking for Large Language Models (LLMs)

A comprehensive framework for embedding and detecting statistical watermarks in LLM-generated text. This implementation supports three watermarking schemes: **zero-bit** (binary detection), **L-bit** (message embedding), and **multi-user fingerprinting** (user tracing).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Zero-Bit Watermarking](#zero-bit-watermarking)
  - [L-Bit Watermarking](#l-bit-watermarking)
  - [Multi-User Fingerprinting](#multi-user-fingerprinting)
  - [Batch Evaluation](#batch-evaluation)
- [File-by-File Usage Guide](#file-by-file-usage-guide)
- [Parameters and Tuning](#parameters-and-tuning)
- [Expected Outputs](#expected-outputs)
- [Model Information](#model-information)
- [HPC Cluster Usage](#hpc-cluster-usage)
- [Troubleshooting](#troubleshooting)
- [License and Citation](#license-and-citation)

---

## Overview

This repository implements cryptographic watermarking techniques for LLM text generation. The watermarking is:
- **Statistical**: Based on PRF-derived bias in token selection
- **Unobtrusive**: Only applied at high-entropy (uncertain) positions
- **Cryptographically secure**: Uses HMAC-SHA256 for key derivation
- **Robust**: Resistant to common text perturbation attacks

### How It Works

**Zero-Bit Watermarking:**
1. During generation, when the model's next-token entropy exceeds a threshold, add a pseudorandom score vector (derived from secret key + context) to the logits
2. During detection, recompute the same score vectors and calculate a z-score over all watermarked positions ("blocks")
3. If z-score > threshold → text is watermarked

**L-Bit Watermarking:**
1. Derive per-bit keys from a single master key using HMAC(master_key, "i_b") where i is bit position, b ∈ {0,1}
2. Cycle through bits at each high-entropy block, embedding the target bitstring
3. During detection, test both hypotheses (0 and 1) for each bit position and recover the message

**Multi-User Fingerprinting:**
- **Grouped Scheme (BCH-Based):**
  1. Generate BCH codewords with guaranteed minimum Hamming distance (2, 3, or 4)
  2. Assign users to groups sequentially (all users in a group share the same group codeword)
  3. Embed the user's group codeword using L-bit watermarking
  4. During tracing, match recovered codeword to group(s) and identify accused users

- **Hierarchical Scheme:**
  1. Generate group codewords with minimum Hamming distance (for cross-group collusion resistance)
  2. Assign simple binary fingerprints to users within each group
  3. Combine group codeword + user fingerprint to create L-bit message
  4. Embed the combined codeword using L-bit watermarking
  5. During tracing, first identify the group, then identify the user within that group

---

## Features

**Three watermarking modes**: Zero-bit detection, L-bit message embedding, multi-user tracing
**Multiple model support**: GPT-2 (local), GPT-OSS-20B, GPT-OSS-120B
**Multiple interfaces**: CLI, SLURM batch scripts
**Robustness testing**: Built-in perturbation attacks (deletion, paraphrasing)
**Comprehensive evaluation**: Parameter sweeps, automated plotting, statistical analysis
**HPC-ready**: Offline model caching, SLURM job templates
**Well-documented**: Copy-paste commands, parameter guides, usage examples

---

## Repository Structure

```
Cryptographic-Watermarking-for-LLM/
│
├── main.py                          # Main CLI entry point (zero-bit, L-bit, evaluation)
├── requirements.txt                 # Python dependencies
├── COMMANDS.md                      # Copy-paste ready command examples
│
├── src/                             # Core source code
│   ├── watermark.py                 # Watermarking implementations
│   │                                  - ZeroBitWatermarker
│   │                                  - LBitWatermarker
│   │                                  - NaiveMultiUserWatermarker
│   │                                  - GroupedMultiUserWatermarker
│   │                                  - HierarchicalMultiUserWatermarker
│   ├── models.py                    # Model abstractions (GPT-2, GPT-OSS variants)
│   ├── fingerprinting.py            # Multi-user codeword generation & tracing
│   ├── commands.py                  # CLI command handlers
│   ├── parser.py                    # Argument parsing & validation
│   ├── utils.py                     # Helper utilities (parsing, perturbations)
│   └── main_multiuser.py            # Multi-user CLI (generate, trace)
│
├── evaluation_scripts/              # Evaluation and experiment scripts
│   ├── compare_collusion_resistance.py  # Compare naive vs fingerprinting approaches
│   ├── evaluate_multiuser_performance.py  # Multi-user performance evaluation
│   ├── evaluate_hierarchical_detection.py  # Pure detection performance for hierarchical schemes
│   ├── evaluate_hierarchical_robustness.py  # Robustness to deletion attacks for hierarchical schemes
│   ├── evaluate_paraphrasing_attack.py  # Robustness to paraphrasing (T5-small) attacks
│   ├── evaluate_synonym_attack.py  # Robustness to synonym substitution attacks
│   ├── evaluate_rewrite_attack.py  # Robustness to LLM rewrite attacks
│   ├── run_lbit_sweep.py            # L-bit parameter sweep
│   ├── run_lbit_parameter_sweep.py  # L-bit parameter sweep (alternative)
│   ├── run_detection_only.py        # Standalone detection script
│   ├── redo_paraphrase_attack.py    # Re-run perturbation attacks
│   └── test_undetectability.py      # Statistical undetectability tests
│
├── evaluation_scripts_local/        # Local convenience wrappers for evaluation scripts
│   ├── run_hierarchical_detection_local.py  # Run hierarchical detection evaluation locally
│   ├── run_hierarchical_performance_local.py  # Run hierarchical performance evaluation locally
│   ├── run_hierarchical_robustness_local.py  # Run hierarchical robustness evaluation locally
│   ├── run_paraphrasing_attack_local.py  # Run paraphrasing attack evaluation locally
│   ├── run_rewrite_attack_local.py  # Run rewrite attack evaluation locally
│   ├── run_synonym_attack_local.py  # Run synonym attack evaluation locally
│   └── run_collusion_resistance_local.py  # Run collusion resistance evaluation locally
│
├── helper_scripts/                  # Analysis and utility scripts
│   ├── analyse.py                   # Generate plots from evaluation results
│   ├── generate_users.py            # Create user database CSV
│   ├── compute_code_capacity.py     # Compute code capacity for fingerprinting
│   ├── visualise_blocks.py          # Visualize watermark blocks
│   ├── visualise_lbit_blocks.py     # Visualize L-bit blocks
│   ├── visualize_groups.py         # Visualize multi-user groups
│   ├── create_collusion_scenario.py # Create collusion test scenarios
│   └── download_models_hpc.py       # Pre-download models for HPC
│
├── slurm_scripts/                   # HPC cluster batch job scripts
│   ├── run_collusion_eval_hpc.sh    # Collusion resistance evaluation
│   ├── run_multiuser_performance_eval_hpc.sh  # Multi-user performance evaluation
│   ├── run_lbit_sweep_hpc.sh        # L-bit parameter sweep
│   ├── run_hierarchical_detection_hpc.sh  # Hierarchical detection evaluation
│   ├── run_hierarchical_robustness_hpc.sh  # Hierarchical robustness evaluation
│   ├── run_paraphrasing_attack_hpc.sh  # Paraphrasing attack evaluation (T5-small)
│   ├── run_synonym_attack_hpc.sh  # Synonym substitution attack evaluation
│   └── run_rewrite_attack_hpc.sh  # Rewrite attack evaluation
│
├── assets/                          # Data files
│   ├── users.csv                    # 1000 users (UserIds 0-999)
│   └── prompts.txt                  # Evaluation prompts (typically 300+)
│
├── evaluation/                      # Evaluation results (auto-created)
│   ├── evaluation_results/          # Main evaluation outputs
│   ├── evaluation_results_lbit/      # L-bit evaluation results
│   ├── lbit_sweep/                  # L-bit parameter sweep results
│   ├── collusion_resistance_2_colluders/  # Collusion resistance evaluation (2 colluders)
│   ├── local_multiuser_perf_user500/  # Multi-user performance evaluation results
│   ├── hierarchical_detection/     # Hierarchical detection evaluation results
│   └── robustness/                 # Hierarchical robustness evaluation results
│
└── demonstration/                   # Example outputs
    ├── multiuser_user0.txt          # Multi-user example (user 0)
    ├── hierarchical_user0.txt      # Hierarchical scheme example (user 0)
    ├── hierarchical_user30.txt     # Hierarchical scheme example (user 30)
    ├── hierarchical_user100.txt    # Hierarchical scheme example (user 100)
    └── hierarchical_user239.txt    # Hierarchical scheme example (user 239)
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) CUDA-capable GPU for large models

### Setup

**Windows (cmd):**
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Dependencies

Core dependencies from `requirements.txt`:
- **torch**: PyTorch for model inference
- **transformers**: HuggingFace models (GPT-2, T5)
- **numpy**: Numerical operations
- **accelerate**: Multi-GPU support
- **sentencepiece**: Tokenizer support
- **pandas**: Data manipulation
- **matplotlib, seaborn**: Visualization
- **nltk**: Text processing
- **protobuf**: Model serialization

---

## Quick Start

### Zero-Bit Watermarking (5 minutes)

**Generate watermarked text:**
```bat
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 512 -o output.txt
```

**Output:**
- `output.txt`: Generated watermarked text
- `secret.key`: Secret key (DO NOT SHARE)

**Detect watermark:**
```bat
python main.py detect output.txt --model gpt2 --key-file secret.key
```

**Expected output:**
```
=== Detection Results ===
Z-score: 12.45
Blocks detected: 87
Decision: WATERMARKED ✓
```

---

## Detailed Usage

### Zero-Bit Watermarking

Zero-bit watermarking provides **binary detection**: is the text watermarked (yes/no)?

#### Generate

```bat
python main.py generate "Your prompt here" ^
  --model gpt2 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  -o output.txt ^
  --key-file secret.key
```

**Parameters:**
- `--model`: Model to use (`gpt2`, `gpt-oss-20b`, `gpt-oss-120b`)
- `--delta`: Watermark strength (1.0-5.0, default 2.5)
- `--entropy-threshold`: Minimum entropy for watermarking (1.0-6.0, default 4.0)
- `--max-new-tokens`: Number of tokens to generate
- `-o, --output-file`: Output file path
- `--key-file`: Where to save the secret key

**Expected outputs:**
1. **output.txt**: Contains the generated text
2. **secret.key**: Binary file containing the secret key (32 bytes)

**Example output.txt:**
```
The future of AI is rapidly evolving, with new breakthroughs happening every year.
Machine learning models are becoming more sophisticated, capable of understanding
complex patterns in data and making predictions with unprecedented accuracy. From
natural language processing to computer vision, AI systems are transforming
industries and reshaping how we interact with technology.
```

#### Detect

```bat
python main.py detect output.txt ^
  --model gpt2 ^
  --z-threshold 4.0 ^
  --entropy-threshold 4.0 ^
  --key-file secret.key
```

**Parameters:**
- Text file to check (positional argument)
- `--model`: Same model used for generation
- `--z-threshold`: Detection threshold (default 4.0)
- `--entropy-threshold`: Must match generation (default 4.0)
- `--key-file`: Secret key from generation

**Expected output:**
```
=== Detection Results ===
Model: gpt2
Entropy threshold: 4.0
Z-threshold: 4.0
Z-score: 12.45
Blocks detected: 87
Decision: WATERMARKED ✓

Note: Text successfully detected as watermarked with high confidence.
```

**Two-pass logic:** If block count < 75, automatically retries with `entropy_threshold - 2.0`

---

### L-Bit Watermarking

L-bit watermarking embeds and recovers a **binary message** of length L.

#### Generate

```bat
python main.py generate_lbit "The future of AI is" ^
  --model gpt2 ^
  --message 01010101 ^
  --l-bits 8 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  -o output_lbit.txt ^
  --key-file secret_lbit.key
```

**Important:** `--l-bits` must equal the length of `--message`

**Expected outputs:**
1. **output_lbit.txt**: Generated text with embedded message
2. **secret_lbit.key**: Master secret key (all per-bit keys derived from this)

#### Detect

```bat
python main.py detect_lbit output_lbit.txt ^
  --model gpt2 ^
  --l-bits 8 ^
  --z-threshold 4.0 ^
  --entropy-threshold 4.0 ^
  --key-file secret_lbit.key
```

**Expected output:**
```
=== L-bit Detection Results ===
Model: gpt2
L-bits: 8
Target message: 01010101
Recovered message: 01010101
Bit accuracy: 8/8 (100%)
Undecided bits: 0

Decision: MESSAGE RECOVERED SUCCESSFULLY ✓
```

**Possible outcomes:**
- **Exact match**: `01010101` (all bits recovered)
- **Partial recovery**: `0101⊥1⊥1` (⊥ = undecided, insufficient signal)
- **Failed recovery**: `⊥⊥⊥⊥⊥⊥⊥⊥` (no watermark detected)

**Tips for reducing ⊥ (undecided bits):**
- Increase `--delta` (stronger watermark)
- Lower `--entropy-threshold` during generation (more blocks)
- Increase `--max-new-tokens` (longer text)
- Lower `--z-threshold` during detection (more sensitive)

---

### Multi-User Fingerprinting (BCH-Based)

Trace generated text back to specific users using BCH error-correcting codes with guaranteed minimum Hamming distance for improved collusion resistance.

#### How It Works

- **BCH Codes**: Codewords are generated with guaranteed minimum Hamming distance (2, 3, or 4)
- **Group Assignment**: Users are assigned to groups sequentially:
  - `group_id = user_id // users_per_group`
  - All users in the same group share the same group codeword
  - This prevents collusion attacks where users combine codewords to frame others
- **Minimum Distance Options**:
  - `--min-distance 2`: Up to 100 groups, 10 users per group (default)
  - `--min-distance 3`: Up to 50 groups, 20 users per group

#### User Database

The `assets/users.csv` file contains:
```csv
UserId,Username
0,0
1,1
2,2
...
999,999
```

You can customize usernames or add more users. The number of groups is limited by the minimum distance setting.

#### Generate for User

```bat
python -m src.main_multiuser generate ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --min-distance 2 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  --key-file demonstration\multiuser_master.key ^
  -o demonstration\multiuser_user0.txt ^
  "The future of AI is"
```

**Key points:**
- L=10 supports up to 2¹⁰ = 1024 users
- `--min-distance 2` (default) assigns users to groups of 10
- User 0 belongs to Group 0 (users 0-19)
- User 20 belongs to Group 1 (users 20-39)
- All users in the same group share the same codeword

**Expected outputs:**
1. **multiuser_user0.txt**: Text watermarked with user 0's group codeword
2. **multiuser_master.key**: Master key (shared across all users)
3. Console output showing: "User ID 0 belongs to Group 0"

#### Trace User

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --min-distance 2 ^
  --key-file demonstration\multiuser_master.key ^
  demonstration\multiuser_user0.txt
```

**Expected output:**
```
--- Trace Results ---
  Text traced back to user(s):
     - User ID: 0, Username: 0, Group: 0, Match: 100.00%
       User ID 0 belongs to Group 0
```

**Collusion detection:** 
- Returns up to 16 best matches if multiple users' codewords match closely
- Shows group membership for each accused user
- Detects collusion when recovered codeword contains `*` symbols (conflicting bits)

#### Visualize Groups

View group assignments, codewords, and verify minimum distance:

```bat
python helper_scripts/visualize_groups.py ^
  --users-file assets/users.csv ^
  --l-bits 10 ^
  --min-distance 2
```

**Output includes:**
- Group assignments with codewords and user ranges
- Minimum distance verification between all group codewords
- Statistics (average users per group, codeword distribution)
- Any distance violations (if minimum distance is not satisfied)

Use `--detailed` flag to see all user IDs in each group.

---

### Batch Evaluation

Run parameter sweeps and perturbation attacks across multiple prompts.

#### Run Evaluation

```bat
python main.py evaluate ^
  --prompts-file assets/prompts.txt ^
  --model gpt2 ^
  --delta "2.0, 2.5, 3.0" ^
  --entropy-thresholds "3.0, 3.5, 4.0" ^
  --max-new-tokens 512 ^
  --output-dir evaluation/evaluation_results
```

**What it does:**
1. Generates clean text for each prompt
2. Creates perturbed variants:
   - Delete first 20% of sentences
   - Delete last 20% of sentences
   - Delete middle 20% of sentences
   - Paraphrase 30% of sentences (T5 model)
3. Runs detection on all variants
4. Saves results to `evaluation/evaluation_results/analysis_results.json`
5. By default only the first 100 prompts are evaluated. Pass `--max-prompts 300` (or any number ≤ total prompts) when you want the full sweep.

**Expected output files:**
```
evaluation/evaluation_results/
├── analysis_results.json        # Detailed results (z-scores, block counts, decisions)
├── generated_text_*.txt         # Generated text files
└── keys/                        # Secret keys
```

#### Analyze Results

```bat
python helper_scripts\analyse.py evaluation/evaluation_results --z-threshold 4.0
```

**Generated plots:**
- `completeness_soundness_distribution.png`: Detection accuracy distribution
- `robustness_boxplot.png`: Robustness across perturbation types
- `parameter_sweep_*.png`: Parameter impact visualizations

**Summary statistics:**
```
=== Evaluation Summary ===
Total prompts: 75
Clean text detection rate: 98.7%
Average z-score (clean): 15.23
False positive rate: 0.0%

Robustness (perturbed text):
  Delete start 20%: 87.3% detected
  Delete end 20%: 89.1% detected
  Delete middle 20%: 85.7% detected
  Paraphrase 30%: 76.4% detected
```

---

## File-by-File Usage Guide

### Core Python Files

#### `main.py` (17 lines)
**Purpose:** CLI entry point
**Usage:** Dispatches to subcommands (generate, detect, evaluate, etc.)
**Run:** `python main.py <command> [args]`

#### `src/watermark.py` (460 lines)
**Purpose:** Core watermarking algorithms
**Classes:**
- `ZeroBitWatermarker`: Binary detection
- `LBitWatermarker`: Message embedding
- `NaiveMultiUserWatermarker`: Legacy per-user fingerprinting
- `GroupedMultiUserWatermarker`: Fingerprinting with grouped codes
- `WatermarkLogitsProcessor`: Transformers integration

**Key functions:**
- `derive_key(secret_key, context, suffix)`: HMAC-SHA256 key derivation
- `calculate_entropy(logits)`: Shannon entropy calculation
- `generate(...)`: Watermarked text generation
- `detect(...)`: Watermark detection

**Not called directly** (used via CLI)

#### `src/models.py` (157 lines)
**Purpose:** Model abstraction layer
**Classes:**
- `LanguageModel`: Abstract base
- `GPT2Model`: GPT-2 (local, < 2GB VRAM)
- `GptOssModel`: 20B parameter model (16GB+ VRAM)
- `GptOss120bModel`: 120B parameter model (80GB+ VRAM)

**Methods:**
- `get_logits(input_ids)`: Compute next-token logits
- `tokenizer`: Access tokenizer
- `vocab_size`, `device`: Model properties

**Usage:** Automatically instantiated by CLI based on `--model` flag

#### `src/fingerprinting.py` (296 lines)
**Purpose:** Multi-user codeword management using BCH error-correcting codes
**Class:** `FingerprintingCode`

**Features:**
- BCH-based codeword generation with guaranteed minimum Hamming distance
- Group-based user assignment for improved collusion resistance
- Sequential group assignment: `group_id = user_id // users_per_group`

**Methods:**
- `gen(users_file)`: Load users and generate BCH codewords with minimum distance
- `trace(recovered_message)`: Find users matching noisy codeword (includes group info)

**Parameters:**
- `L` (int): Codeword length (default: 10)
- `min_distance` (int): Minimum Hamming distance between codewords (2 or 3, default: 2)
- `c` (int): Maximum number of colluders (default: 16)

**Example:**
```python
from src.fingerprinting import FingerprintingCode

# Initialize with minimum distance 2 (default)
code = FingerprintingCode(L=10, min_distance=2)

# Load users and generate codewords
code.gen(users_file='assets/users.csv')

# Users are assigned to groups:
# - Users 0-19 → Group 0
# - Users 20-39 → Group 1
# - etc.

# Trace noisy recovery
recovered = "0000000010"  # 1 bit flipped
matches = code.trace(recovered)
# Returns: [{"user_id": 0, "username": "0", "group_id": 0, "match_score_percent": 90.0, ...}]
```

#### `src/commands.py` (398 lines)
**Purpose:** CLI command implementations
**Functions:**
- `cmd_generate(args)`: Zero-bit generation
- `cmd_detect(args)`: Zero-bit detection
- `cmd_generate_lbit(args)`: L-bit generation
- `cmd_detect_lbit(args)`: L-bit detection
- `cmd_evaluate(args)`: Batch evaluation

**Not called directly** (invoked by `main.py` based on subcommand)

#### `src/parser.py` (157 lines)
**Purpose:** Argument parsing and validation
**Features:**
- Subcommand routing
- Parameter range validation
- NLTK setup and verification
- Default value handling

**Not called directly** (used by `main.py`)

#### `src/utils.py` (209 lines)
**Purpose:** Helper utilities
**Functions:**
- `instantiate_model(model_name)`: Create model instance
- `parse_output(text, model_name)`: Clean model output
- `delete_sentences(text, portion)`: Deletion attack
- `paraphrase_sentences(text, portion)`: Paraphrasing attack
- `parse_filename(filename)`: Extract metadata from filenames

**Usage in scripts:**
```python
from src.utils import instantiate_model, delete_sentences

model = instantiate_model('gpt2')
perturbed = delete_sentences(text, 'start', 0.2)  # Remove first 20%
```

#### `src/main_multiuser.py` (128 lines)
**Purpose:** Multi-user CLI
**Commands:**
- `generate`: Watermark text for user
- `trace`: Identify user from text

**Run:**
```bat
python -m src.main_multiuser generate [args]
python -m src.main_multiuser trace [args]
```

### Helper Scripts

#### `evaluation_scripts/compare_collusion_resistance.py`
**Purpose:** Compare naive vs fingerprinting multi-user watermarking approaches for collusion resistance
**Usage:**
```bat
python evaluation_scripts\compare_collusion_resistance.py ^
  --prompts-file assets/prompts.txt ^
  --max-prompts 100 ^
  --num-colluders 2 ^
  --model gpt2
```

**Features:**
- Tests three approaches: naive, min-distance-2, min-distance-3
- Uses same colluding users across all approaches per prompt (fair comparison)
- Two combination methods: normal and with deletion
- **Smart case skipping**: Automatically skips collusion cases that require more groups than available (e.g., skips cross-group tests for G=1, skips 3-colluder cross-group for G=2)
- **Reduced verbosity**: Logs warnings once per configuration instead of per prompt
- Generates comparison table, JSON results, per-prompt JSONs, and CSV summary
- Organized output structure by approach and prompt
- Supports `--csv-only` mode to rebuild summaries without regenerating text

#### `evaluation_scripts/evaluate_hierarchical_detection.py`
**Purpose:** Evaluate pure detection performance (no collusion) for hierarchical multi-user watermarking at L=8, across all allocations of group bits and user bits
**Usage:**
```bat
python evaluation_scripts/evaluate_hierarchical_detection.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/hierarchical_detection
```

**Features:**
- Evaluates 9 configurations: naive (L=8) and hierarchical (G=1,U=7 through G=8,U=0)
- For each prompt: chooses random user, embeds watermark, detects codeword, decodes IDs
- Logs per-prompt: true/detected IDs, codewords, Hamming distance, z-scores, match statuses
- Computes metrics: group accuracy, user accuracy, full identity accuracy, L-bit accuracy, false positive/negative rates
- Saves per-prompt JSON files and summary JSON
- Supports both naive and hierarchical schemes

#### `evaluation_scripts/evaluate_hierarchical_robustness.py`
**Purpose:** Evaluate robustness to deletion attacks for hierarchical multi-user watermarking at L=8, across all allocations of group bits and user bits
**Usage:**
```bat
python evaluation_scripts/evaluate_hierarchical_robustness.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/robustness
```

**Features:**
- Same structure as `evaluate_hierarchical_detection.py` but tests robustness to deletion attacks
- Tests 16 attack variants per prompt: 4 deletion percents (0.05, 0.10, 0.15, 0.20) × 4 deletion modes (start, middle, end, random)
- For each attack variant: applies deletion, detects codeword, decodes IDs, computes metrics
- Logs per-attack: deletion_percent, deletion_mode, recovered codeword, Hamming distance, z-scores, match statuses
- Computes metrics: group accuracy, user accuracy, full identity accuracy, average invalid symbols, average Hamming distance, average z-score, false positive/negative rates
- Saves all attack results to a single `results.json` file and summary JSON
- Supports both naive and hierarchical schemes

#### `evaluation_scripts/evaluate_paraphrasing_attack.py`
**Purpose:** Evaluate robustness against paraphrasing attacks (single-pass T5-small) for both naive and hierarchical schemes at L=8
**Usage:**
```bat
python evaluation_scripts/evaluate_paraphrasing_attack.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/paraphrasing_attack
```

**Features:**
- Tests 16 attack variants per prompt: 4 paraphrase ratios (0.05, 0.10, 0.15, 0.20) × 4 modes (start, middle, end, random)
- Mirrors the robustness workflow but uses T5-small paraphrasing instead of deletion attacks
- Evaluates naive plus all eight hierarchical splits (G=1,U=7 … G=8,U=0) in a single run
- Records per-attack variant: paraphrase_ratio, paraphrase_mode, recovered codeword, invalid symbols, Hamming distance, z-score, group/user matches
- Computes the same metrics as robustness (group/user/full accuracy, L-bit accuracy, false positive/negative rates, averages)
- Saves all attack results to a single `raw_results.jsonl.gz` file (if enabled) and summary JSON/CSV per configuration under `evaluation/paraphrasing_attack`

#### `evaluation_scripts/evaluate_synonym_attack.py`
**Purpose:** Evaluate robustness against synonym substitution attacks (WordNet, 10% of tokens) for both naive and hierarchical schemes at L=8
**Usage:**
```bat
python evaluation_scripts/evaluate_synonym_attack.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/synonym_attack
```

**Features:**
- Tests 16 attack variants per prompt: 4 synonym ratios (0.05, 0.10, 0.15, 0.20) × 4 modes (start, middle, end, random)
- Mirrors the robustness workflow but uses WordNet synonym substitution instead of deletion attacks
- Evaluates naive plus all eight hierarchical splits (G=1,U=7 … G=8,U=0) automatically
- Records per-attack variant: synonym_ratio, synonym_mode, recovered codeword, invalid symbols, Hamming distance, z-score, group/user matches
- Computes the same aggregate metrics as robustness and saves all attack results to `raw_results.jsonl.gz` (if enabled) and summary JSON/CSV under `evaluation/synonym_attack/<scheme_dir>`

#### `evaluation_scripts/evaluate_rewrite_attack.py`
**Purpose:** Evaluate robustness against full-text rewrites generated deterministically by the same base LLM used for watermarking (GPT-2 / GPT-OSS variants)
**Usage:**
```bat
python evaluation_scripts/evaluate_rewrite_attack.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/rewrite_attack
```

**Features:**
- Tests 16 attack variants per prompt: 4 rewrite ratios (0.05, 0.10, 0.15, 0.20) × 4 modes (start, middle, end, random)
- Mirrors the robustness workflow but uses LLM-based rewriting instead of deletion attacks
- Uses `apply_llm_rewrite` to prompt the same LM to rewrite selected sentences deterministically (no sampling) before detection
- Evaluates naive plus all hierarchical splits, mirroring other attack scripts
- Records per-attack variant: rewrite_ratio, rewrite_mode, recovered codeword, invalid symbols, Hamming distance, z-score, group/user matches
- Outputs all attack results to `raw_results.jsonl.gz` (if enabled) and summary JSON/CSV under `evaluation/rewrite_attack/<scheme_dir>`

#### `helper_scripts/analyse.py` (261 lines)
**Purpose:** Generate plots and statistics from evaluation results
**Usage:**
```bat
python helper_scripts\analyse.py evaluation/evaluation_results --z-threshold 4.0
```

**Outputs:**
- `completeness_soundness_distribution.png`
- `robustness_boxplot.png`
- `summary_analysis.txt`
- Console output with quantitative metrics

#### `helper_scripts/generate_users.py`
**Purpose:** Create custom user databases
**Usage:**
```bat
python helper_scripts\generate_users.py --num-users 1000 -o my_users.csv
```

**Output:** CSV file with UserId, Username columns

#### `helper_scripts/visualise_blocks.py`
**Purpose:** Visualize watermark block positions in text
**Usage:**
```bat
python helper_scripts\visualise_blocks.py output.txt --key-file secret.key --model gpt2
```

**Output:** Text with highlighted watermarked blocks (terminal colors or HTML)

#### `helper_scripts/visualise_lbit_blocks.py`
**Purpose:** Visualize L-bit embedding pattern
**Usage:**
```bat
python helper_scripts\visualise_lbit_blocks.py output_lbit.txt --key-file secret_lbit.key --model gpt2 --l-bits 8
```

**Output:** Shows which bit is embedded at each block position

#### `evaluation_scripts/test_undetectability.py`
**Purpose:** Statistical tests for undetectability
**Usage:**
```bat
python evaluation_scripts\test_undetectability.py --model gpt2 --num-samples 100
```

**Output:** Chi-square test results, KL divergence metrics

#### `helper_scripts/download_models_hpc.py`
**Purpose:** Pre-cache models for offline HPC environments
**Usage:**
```bat
python helper_scripts\download_models_hpc.py --model gpt-oss-20b --cache-dir /shared/models
```

### Local Evaluation Scripts

Convenience wrappers for running evaluation scripts locally (without SLURM). These scripts sequentially invoke the corresponding evaluation scripts for all configurations.

#### `evaluation_scripts_local/run_hierarchical_detection_local.py`
**Purpose:** Run all hierarchical detection configurations locally (naive + 8 hierarchical splits)
**Usage:**
```bat
python evaluation_scripts_local\run_hierarchical_detection_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/hierarchical_detection
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hierarchical splits: G=1,U=7 through G=8,U=0)
- Forwards shared arguments to `evaluate_hierarchical_detection.py`
- Generates consolidated summary CSV across all configurations
- Designed for single workstation use (mirrors HPC SLURM script functionality)

#### `evaluation_scripts_local/run_hierarchical_performance_local.py`
**Purpose:** Run all hierarchical performance evaluations locally (memory, computation, storage metrics)
**Usage:**
```bat
python evaluation_scripts_local\run_hierarchical_performance_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 100 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 500 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/local_multiuser_perf_user500
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hierarchical splits)
- Measures performance metrics: memory usage, computation time, storage size
- Forwards shared arguments to `evaluate_multiuser_performance.py`
- Generates consolidated performance summary

#### `evaluation_scripts_local/run_hierarchical_robustness_local.py`
**Purpose:** Run all hierarchical robustness evaluations locally (deletion attack resistance)
**Usage:**
```bat
python evaluation_scripts_local\run_hierarchical_robustness_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/robustness
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hierarchical splits)
- Tests 16 deletion attack variants per prompt (4 deletion percents: 5%, 10%, 15%, 20% × 4 deletion modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_hierarchical_robustness.py`
- Generates consolidated summary CSV across all configurations with `num_attack_variants_per_prompt` and `total_attack_results` fields

#### `evaluation_scripts_local/run_paraphrasing_attack_local.py`
**Purpose:** Run all paraphrasing attack evaluations locally (T5-small paraphrasing)
**Usage:**
```bat
python evaluation_scripts_local\run_paraphrasing_attack_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/paraphrasing_attack
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hierarchical splits)
- Tests 16 paraphrasing attack variants per prompt (4 ratios: 5%, 10%, 15%, 20% × 4 modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_paraphrasing_attack.py`
- Generates consolidated summary CSV across all configurations with `num_attack_variants_per_prompt` and `total_attack_results` fields

#### `evaluation_scripts_local/run_rewrite_attack_local.py`
**Purpose:** Run all rewrite attack evaluations locally (LLM-based rewriting)
**Usage:**
```bat
python evaluation_scripts_local\run_rewrite_attack_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/rewrite_attack
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hierarchical splits)
- Tests 16 rewrite attack variants per prompt (4 ratios: 5%, 10%, 15%, 20% × 4 modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_rewrite_attack.py`
- Generates consolidated summary CSV across all configurations with `num_attack_variants_per_prompt` and `total_attack_results` fields

#### `evaluation_scripts_local/run_synonym_attack_local.py`
**Purpose:** Run all synonym substitution attack evaluations locally (WordNet synonym substitution)
**Usage:**
```bat
python evaluation_scripts_local\run_synonym_attack_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/synonym_attack
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hierarchical splits)
- Tests 16 synonym substitution attack variants per prompt (4 ratios: 5%, 10%, 15%, 20% × 4 modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_synonym_attack.py`
- Generates consolidated summary CSV across all configurations with `num_attack_variants_per_prompt` and `total_attack_results` fields

### SLURM Scripts

All scripts in `slurm_scripts/` are HPC cluster batch job scripts for running evaluations.

**Available scripts:**
- `run_collusion_eval_hpc.sh`: Collusion resistance evaluation (300 prompts, 64-hour limit)
- `run_multiuser_performance_eval_hpc.sh`: Multi-user performance evaluation
- `run_lbit_sweep_hpc.sh`: L-bit parameter sweep
- `run_hierarchical_detection_hpc.sh`: Hierarchical detection evaluation (300 prompts, 64-hour limit)
- `run_hierarchical_robustness_hpc.sh`: Hierarchical robustness evaluation (300 prompts, 16 variants per prompt, 64-hour limit)
- `run_paraphrasing_attack_hpc.sh`: Paraphrasing attack evaluation (300 prompts, 16 variants per prompt, 64-hour limit)
- `run_rewrite_attack_hpc.sh`: Rewrite attack evaluation (300 prompts, 16 variants per prompt, 64-hour limit)
- `run_synonym_attack_hpc.sh`: Synonym substitution attack evaluation (300 prompts, 16 variants per prompt, 64-hour limit)

**Configuration:**
- All scripts use **300 prompts** per configuration
- All scripts have **64-hour time limit**
- Attack scripts (robustness, paraphrasing, rewrite, synonym) test **16 variants per prompt** (4 intensities × 4 modes)
- All scripts use **Apptainer** containers (no host Python/venv setup needed)
- All scripts run on **skylake-gpu** partition

**Usage:**
```bash
# Run hierarchical detection evaluation (9 configurations, 300 prompts each)
sbatch slurm_scripts/run_hierarchical_detection_hpc.sh

# Run hierarchical robustness evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_hierarchical_robustness_hpc.sh

# Run paraphrasing attack evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_paraphrasing_attack_hpc.sh

# Run rewrite attack evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_rewrite_attack_hpc.sh

# Run synonym attack evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_synonym_attack_hpc.sh

# Run collusion resistance evaluation (9 configurations, 300 prompts each)
sbatch slurm_scripts/run_collusion_eval_hpc.sh
```

### Asset Files

#### `assets/users.csv` (1000 rows)
**Format:**
```csv
UserId,Username
0,0
1,1
...
```

**Usage:** Required for multi-user fingerprinting

#### `assets/prompts.txt` (~300 prompts, configurable)
**Format:** One prompt per line
```text
The future of artificial intelligence is
Write a Python function to calculate fibonacci numbers
Explain quantum computing in simple terms
```

**Usage:** Batch evaluation input (default runs first 100 prompts; override with `--max-prompts N`)

---

## Parameters and Tuning

Key parameters:

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--delta` | 1.0-5.0 | 2.5 | Watermark strength (higher = stronger signal, lower fluency) |
| `--entropy-threshold` | 1.0-6.0 | 4.0 | Minimum entropy to watermark (higher = fewer, cleaner blocks) |
| `--hashing-context` | 1-10 | 5 | Number of previous tokens for PRF context |
| `--max-new-tokens` | 50-2048 | 512 | Generation length |

### Detection Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--z-threshold` | 1.0-6.0 | 4.0 | Detection decision threshold (lower = more sensitive) |
| `--entropy-threshold` | 1.0-6.0 | 4.0 | Must match generation for accurate block counting |

### Preset Configurations

**Balanced (default):**
```bat
--delta 2.5 --entropy-threshold 4.0 --z-threshold 4.0
```
Good all-around performance

**High Fluency:**
```bat
--delta 2.0 --entropy-threshold 4.5 --z-threshold 4.0
```
Minimal impact on text quality, slightly weaker signal

**Strong Detection:**
```bat
--delta 3.5 --entropy-threshold 3.5 --z-threshold 3.5
```
Maximum robustness, may slightly reduce fluency

---

## Expected Outputs

### Zero-Bit Example

**Generated text** (`my_watermark.txt`):
```
The future of AI is incredibly promising and multifaceted. As machine learning
algorithms become more sophisticated, we're seeing breakthrough applications in
healthcare diagnostics, autonomous vehicles, and natural language understanding.
The integration of AI into everyday tools is accelerating, making technology more
intuitive and accessible. However, this progress also brings important ethical
considerations about privacy, bias, and transparency that we must address as a
society.
```

**Detection output:**
```
=== Detection Results ===
Z-score: 14.89
Blocks detected: 92
Decision: WATERMARKED ✓
```

### L-Bit Example

**Input message:** `11001010` (8 bits)

**Generated text** (`my_lbit.txt`):
```
The future of AI is transforming how we interact with technology. From voice
assistants that understand context to recommendation systems that predict our
preferences, artificial intelligence has become deeply integrated into daily life.
```

**Detection output:**
```
=== L-bit Detection Results ===
Target message: 11001010
Recovered message: 11001010
Bit accuracy: 8/8 (100%)
Undecided bits: 0
Decision: MESSAGE RECOVERED SUCCESSFULLY ✓
```

### Multi-User Example

**User 0** (codeword: `0000000000`):
```
=== Tracing Results ===
Recovered codeword: 0000000000
Top matching users:
  1. User ID: 0, Username: 0, Bit matches: 10/10 (100%)
Decision: Text traced to User 0 ✓
```

**User 888** (codeword: `1101111000`):
```
=== Tracing Results ===
Recovered codeword: 1101111000
Top matching users:
  1. User ID: 888, Username: 888, Bit matches: 10/10 (100%)
Decision: Text traced to User 888 ✓
```

### Evaluation Summary

After running batch evaluation:
```
=== Evaluation Summary ===
Total prompts tested: 100 (subset of ~300 available prompts)
Model: gpt2
Parameter sweep: delta=[2.0, 2.5, 3.0], entropy=[3.0, 3.5, 4.0]

Clean text results:
  Detection rate: 98.7% (97/100 prompts)
  Average z-score: 15.23
  Average blocks: 94.5
  False positive rate: 0.0% (0/100 control texts)

Perturbation robustness:
  Delete start 20%: 87.3% (87/100)
  Delete end 20%: 89.1% (89/100)
  Delete middle 20%: 85.7% (86/100)
  Paraphrase 30%: 76.4% (76/100)

Best parameter combination:
  delta=2.5, entropy_threshold=3.5
  Clean detection: 100%, Avg robustness: 86.2%
```

---

## Model Information

### GPT-2 (Local)

- **Parameters:** 124M
- **Context length:** 1024 tokens
- **VRAM:** < 2GB (CPU compatible)
- **Speed:** ~10 tokens/sec (CPU), ~50 tokens/sec (GPU)
- **Use case:** Development, testing, quick experiments

**Important:** Keep `prompt_length + max_new_tokens ≤ 1024`

### GPT-OSS-20B

- **Parameters:** 20B
- **Context length:** 2048 tokens
- **VRAM:** 16GB+ (recommend A100 40GB)
- **Speed:** ~2-5 tokens/sec
- **Use case:** Production, high-quality generation

**Setup:**
```python
from src.models import GptOssModel
model = GptOssModel()  # Auto device_map="auto"
```

### GPT-OSS-120B

- **Parameters:** 120B
- **Context length:** 2048 tokens
- **VRAM:** 80GB+ (recommend A100 80GB or multi-GPU)
- **Speed:** ~0.5-2 tokens/sec
- **Use case:** Research, maximum quality

**Setup:**
```python
from src.models import Gpt Oss120bModel
model = GptOss120bModel()  # Requires substantial resources
```

---

## HPC Cluster Usage

### Pre-download Models

On login node (with internet):
```bash
export HF_HOME=/shared/models
export NLTK_DATA=/shared/nltk_data
python helper_scripts/download_models_hpc.py --model gpt2
python -c "import nltk; nltk.download('punkt', download_dir='/shared/nltk_data')"
```

### Submit Job

All SLURM scripts use **Apptainer** containers (no host Python/venv setup needed) and run on the **skylake-gpu** partition with **64-hour time limits**. Pick the SLURM wrapper that matches your evaluation:

```bash
# Hierarchical detection evaluation (9 configurations, 300 prompts each)
sbatch slurm_scripts/run_hierarchical_detection_hpc.sh

# Hierarchical robustness evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_hierarchical_robustness_hpc.sh

# Paraphrasing attack evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_paraphrasing_attack_hpc.sh

# Synonym substitution attack evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_synonym_attack_hpc.sh

# Rewrite attack evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_rewrite_attack_hpc.sh

# Collusion resistance evaluation (9 configurations, 300 prompts each)
sbatch slurm_scripts/run_collusion_eval_hpc.sh

# Multi-user performance evaluation
sbatch slurm_scripts/run_multiuser_performance_eval_hpc.sh

# L-bit parameter sweep
sbatch slurm_scripts/run_lbit_sweep_hpc.sh
```

**Note:** Attack scripts (robustness, paraphrasing, rewrite, synonym) test 16 variants per prompt (4 intensities × 4 modes: start, middle, end, random), so they take significantly longer than detection-only evaluations.
```

### Monitor Job

```bash
squeue -u $USER
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS
```

### Retrieve Results

```bash
cat slurm-<job_id>.out
ls -lh demonstration/
```

---

## Troubleshooting

### Common Issues

#### 1. NLTK Error: `punkt` or `punkt_tab` not found

**Solution:**
```bat
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Or set NLTK_DATA environment variable:
```bat
set NLTK_DATA=C:\path\to\nltk_data
```

#### 2. GPT-2 IndexError: position embeddings

**Error:**
```
IndexError: index out of range in self
```

**Cause:** Prompt + generation exceeds 1024 tokens

**Solution:** Reduce `--max-new-tokens` or shorten prompt
```bat
python main.py generate "Short prompt" --max-new-tokens 512 --model gpt2
```

#### 3. Low Block Count or Borderline Detection

**Symptoms:**
- Blocks < 75
- Z-score near threshold
- Frequent "NOT WATERMARKED" on watermarked text

**Solutions:**
1. Lower `--entropy-threshold` during generation (more blocks):
   ```bat
   --entropy-threshold 3.5  # instead of 4.0
   ```

2. Increase `--delta` (stronger signal):
   ```bat
   --delta 3.0  # instead of 2.5
   ```

3. Generate longer text:
   ```bat
   --max-new-tokens 512  # instead of 256
   ```

4. Lower `--z-threshold` during detection (more sensitive):
   ```bat
   --z-threshold 3.5  # instead of 4.0
   ```

#### 4. Many Undecided Bits (⊥) in L-Bit Recovery

**Symptoms:**
```
Recovered message: 01⊥1⊥⊥01
```

**Solutions:**
1. Slightly lower `--z-threshold` during detection:
   ```bat
   --z-threshold 3.5
   ```

2. Increase watermark strength during generation:
   ```bat
   --delta 3.0
   ```

3. Generate longer text (more blocks):
   ```bat
   --max-new-tokens 512
   ```

4. Lower `--entropy-threshold` during generation:
   ```bat
   --entropy-threshold 3.5
   ```

#### 5. CUDA Out of Memory (Large Models)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Use smaller model:
   ```bat
   --model gpt2  # instead of gpt-oss-20b
   ```

2. Enable device sharding (automatic in `models.py`):
   ```python
   device_map="auto"  # Already configured
   ```

3. Reduce batch size (if applicable)

4. Use multi-GPU setup

#### 7. Paraphrase Attack Timeout

**Symptoms:** T5 paraphraser hangs during evaluation

**Solutions:**
1. Skip paraphrase attack:
   Edit `src/commands.py`, comment out paraphrase perturbation

2. Use smaller T5 model (already using `t5-small`)

3. Increase timeout in evaluation script

---

## Advanced Usage

### Custom Perturbation Attacks

Add custom attacks in `src/utils.py`:

```python
def custom_attack(text: str) -> str:
    """Your custom perturbation logic."""
    # Example: replace all numbers with words
    import re
    replacements = {'0': 'zero', '1': 'one', '2': 'two'}
    for digit, word in replacements.items():
        text = text.replace(digit, word)
    return text
```

### Programmatic API Usage

```python
from src.watermark import ZeroBitWatermarker
from src.models import GPT2Model
import secrets

# Initialize
model = GPT2Model()
secret_key = secrets.token_bytes(32)
watermarker = ZeroBitWatermarker(model, secret_key, delta=2.5, entropy_threshold=4.0)

# Generate
watermarked_text = watermarker.generate(
    prompt="The future of AI is",
    max_new_tokens=512
)

# Detect
z_score, blocks = watermarker.detect(
    watermarked_text,
    z_threshold=4.0
)

print(f"Z-score: {z_score:.2f}, Blocks: {blocks}")
print(f"Detected: {z_score > 4.0}")
```

### Collusion Resistance Comparison

Compare naive vs fingerprinting multi-user watermarking approaches for collusion resistance.

#### Run Comparison

```bat
python evaluation_scripts/compare_collusion_resistance.py ^
  --prompts-file assets/prompts.txt ^
  --max-prompts 100 ^
  --model gpt2 ^
  --users-file assets/users.csv ^
  --num-colluders 2 ^
  --l-bits 10 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 400 ^
  --deletion-percentage 0.05 ^
  --output-dir evaluation/collusion_resistance
```

**What it does:**
1. Tests three approaches:
   - **Naive**: Binary user ID-based fingerprinting
   - **Min-distance-2**: Fingerprinting with minimum Hamming distance 2
   - **Min-distance-3**: Fingerprinting with minimum Hamming distance 3
2. For each prompt:
   - Selects colluding users from different groups (ensures fair comparison)
   - Uses the same users for all three approaches
   - Generates watermarked text for each colluding user
3. Two combination methods:
   - **Normal**: Concatenates texts directly
   - **With deletion**: Deletes 5% of each user's text before combining
4. Attempts to trace back to original colluding users
5. Calculates success rates for each approach and combination method

**Output structure:**
```
evaluation/collusion_resistance_<N>/
├── naive/
│   ├── prompt_0/
│   │   ├── master_key.key
│   │   ├── user_<ID>_text.txt
│   │   ├── combined_normal.txt
│   │   └── combined_with_deletion.txt
│   └── prompt_1/, prompt_2/, ...
├── min-distance-2/
│   └── prompt_0/, prompt_1/, ...
├── min-distance-3/
│   └── prompt_0/, prompt_1/, ...
├── prompt_results/
│   └── prompt_0_results.json, prompt_1_results.json, ...
├── collusion_resistance_results_<N>users.json
└── collusion_resistance_summary_<N>users.csv
```

**Expected output:**
- Console: Comparison table showing success rates
- JSON: Detailed results with trace information for each test case
- CSV: Summary statistics for easy analysis
- Per-prompt JSON files (`prompt_results/prompt_<ID>_results.json`) for quick post-processing

**Parameters:**
- `--num-colluders`: Number of colluding users (2 or 3, default: 2)
- `--deletion-percentage`: Percentage of text to delete per user (default: 0.05 for 5%)
- `--max-prompts`: Number of prompts to test (default: 100)
- Output directory automatically appends `_<num_colluders>` (e.g., `collusion_resistance_2`)
- `--csv-only`: Skip generation and rebuild JSON/CSV summaries from existing per-prompt results

---

### Hierarchical Detection Performance Evaluation

Evaluate pure detection performance (no collusion) for hierarchical multi-user watermarking at L=8, across all allocations of group bits and user bits.

#### Run Evaluation

**Naive scheme (L=8, no hierarchy):**
```bat
python evaluation_scripts/evaluate_hierarchical_detection.py ^
  --scheme naive ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/hierarchical_detection
```

**Hierarchical scheme (G=4, U=4):**
```bat
python evaluation_scripts/evaluate_hierarchical_detection.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/hierarchical_detection
```

**What it does:**
1. Evaluates 9 configurations:
   - **Naive**: L=8, no hierarchy, every user gets a flat L-bit codeword
   - **Hierarchical G=1, U=7**: 1 group, 128 users per group
   - **Hierarchical G=2, U=6**: 2 groups, 64 users per group
   - **Hierarchical G=3, U=5**: 4 groups, 32 users per group
   - **Hierarchical G=4, U=4**: 8 groups, 16 users per group
   - **Hierarchical G=5, U=3**: 16 groups, 8 users per group
   - **Hierarchical G=6, U=2**: 32 groups, 4 users per group
   - **Hierarchical G=7, U=1**: 64 groups, 2 users per group
   - **Hierarchical G=8, U=0**: 128 groups, 1 user per group (group-only mode)
2. For each prompt:
   - Chooses a random user ID
   - Embeds watermark
   - Detects L-bit codeword
   - Decodes group ID and user ID (for hierarchical)
   - Logs: true/detected IDs, codewords, Hamming distance, z-scores, match statuses
3. Computes metrics:
   - **For naive**: L-bit accuracy, full identity accuracy, false positive/negative rates
   - **For hierarchical**: group accuracy, user accuracy (given correct group), full identity accuracy, L-bit accuracy, false positive/negative rates

**Output structure:**
```
evaluation/hierarchical_detection/
├── naive_L8/
│   ├── prompt_0.json
│   ├── prompt_1.json
│   ├── ...
│   └── summary.json
├── hierarchical_G1_U7/
│   ├── prompt_0.json
│   ├── prompt_1.json
│   ├── ...
│   └── summary.json
├── hierarchical_G2_U6/
│   └── ...
└── ... (other configurations)
```

**Per-prompt JSON contains:**
- `true_user_id`, `detected_user_id`
- `true_group_id`, `detected_group_id` (for hierarchical)
- `recovered_codeword`, `ground_truth_codeword`
- `num_invalid_symbols`, `hamming_distance`, `z_score`
- `group_match`, `user_match`, `full_identity_match`, `lbit_accuracy`

**Summary JSON contains:**
- Configuration (scheme, l_bits, group_bits, user_bits)
- Number of prompts
- Computed metrics (accuracy rates, false positive/negative rates)

**HPC Usage:**
Run all 8 configurations on HPC:
```bash
sbatch slurm_scripts/run_hierarchical_detection_hpc.sh
```

---

### Hierarchical Robustness Evaluation

Evaluate robustness to deletion attacks for hierarchical multi-user watermarking at L=8, across all allocations of group bits and user bits.

#### Run Evaluation

**Naive scheme (L=8, no hierarchy):**
```bat
python evaluation_scripts/evaluate_hierarchical_robustness.py ^
  --scheme naive ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/robustness
```

**Hierarchical scheme (G=4, U=4):**
```bat
python evaluation_scripts/evaluate_hierarchical_robustness.py ^
  --scheme hierarchical ^
  --group-bits 4 ^
  --user-bits 4 ^
  --l-bits 8 ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/robustness
```

**What it does:**
1. Same as hierarchical detection evaluation, but applies deletion attacks before detection
2. For each prompt:
   - Generates watermarked text (same as detection evaluation)
   - Applies 16 deletion attack variants:
     - 4 deletion percents: 5%, 10%, 15%, 20%
     - 4 deletion modes: start, middle, end, random
   - For each attacked text: detects codeword, decodes IDs, computes metrics
3. Deletion attack details:
   - **start**: Remove first k tokens (k = percent × total tokens)
   - **end**: Remove last k tokens
   - **middle**: Remove k tokens centered at middle position
   - **random**: Remove k randomly sampled tokens
4. Computes metrics:
   - **For naive**: Full identity accuracy, false positive/negative rates, average invalid symbols, average Hamming distance, average z-score
   - **For hierarchical**: Group accuracy, user accuracy, full identity accuracy, false positive/negative rates, average invalid symbols, average Hamming distance, average z-score

**Output structure:**
```
evaluation/robustness/
├── naive/naive_L8/
│   ├── results.json          # All attack results (one entry per attack variant)
│   └── summary.json          # Summary metrics
├── hierarchical/G4_U4/
│   ├── results.json          # All attack results (one entry per attack variant)
│   └── summary.json          # Summary metrics
└── ... (other configurations)
```

**Results JSON contains (per entry):**
- `prompt_id`, `prompt`, `true_user_id`
- `deletion_percent`, `deletion_mode`
- `recovered_codeword`, `ground_truth_codeword`
- `num_invalid_symbols`, `hamming_distance`, `z_score`
- `detected_user_id`, `detected_group_id` (for hierarchical)
- `group_match`, `user_match`, `full_identity_match`

**Summary JSON contains:**
- Configuration (scheme, l_bits, group_bits, user_bits)
- Number of prompts, number of attack variants per prompt, total attack results
- Deletion percents and modes tested
- Computed metrics (accuracy rates, false positive/negative rates, averages)

**HPC Usage:**
Run all 9 configurations (naive + 8 hierarchical variants) on HPC:
```bash
sbatch slurm_scripts/run_hierarchical_robustness_hpc.sh
```

**Note:** This evaluation takes significantly longer than detection evaluation (~16× longer) since each prompt generates 16 attack variants instead of 1 clean result.

---

### Multi-User Collusion Testing

Test robustness against multiple users combining outputs:

```python
from src.watermark import GroupedMultiUserWatermarker

# Initialize grouped scheme (min distance 2, default)
grouped = GroupedMultiUserWatermarker(lbit_watermarker=lbw, min_distance=2)
grouped.load_users('assets/users.csv')

# Generate from multiple users
user_texts = []
for user_id in [0, 5, 10]:
    # ... generate watermarked text for each user ...
    user_texts.append(text)

# Combine texts (e.g., interleave sentences)
combined = combine_texts(user_texts)  # Your logic

# Trace (includes group information)
matches = grouped.trace(master_key, combined)
for match in matches:
    print(f"User {match['user_id']} (Group {match['group_id']}): "
          f"{match['match_score_percent']:.2f}% match")
    if match.get('collusion_detected'):
        print(f"  Collusion detected at positions: {match['collusion_positions']}")
```

---

## Performance Benchmarks

Approximate generation speeds (256 tokens):

| Model | Hardware | Speed | VRAM | Time (256 tokens) |
|-------|----------|-------|------|-------------------|
| GPT-2 | CPU (8 cores) | ~10 tok/s | ~1GB RAM | ~25 seconds |
| GPT-2 | GPU (RTX 3090) | ~50 tok/s | ~1.5GB | ~5 seconds |
| GPT-OSS-20B | A100 40GB | ~3 tok/s | ~18GB | ~85 seconds |
| GPT-OSS-120B | A100 80GB | ~1 tok/s | ~75GB | ~4 minutes |

Detection is typically 2-3x faster than generation.

---

## License and Citation

### License

[Specify your license here, e.g., MIT, Apache 2.0, GPL-3.0]

### Citation

If you use this codebase in your research, please cite:

```bibtex
@software{cryptographic_watermarking_llm,
  title={Cryptographic Watermarking for Large Language Models},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/Cryptographic-Watermarking-for-LLM}
}
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

### Issues

Report bugs and request features at: [GitHub Issues](https://github.com/yourusername/Cryptographic-Watermarking-for-LLM/issues)

---

## Acknowledgments

This implementation builds upon research in statistical watermarking and cryptographic fingerprinting for LLMs. Special thanks to:
- HuggingFace Transformers team
- PyTorch contributors
- Research community in AI safety and provenance

---

## Quick Reference

### Essential Commands

```bat
# Setup
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Zero-bit
python main.py generate "Prompt" --model gpt2 -o out.txt
python main.py detect out.txt --model gpt2 --key-file secret.key

# L-bit
python main.py generate_lbit "Prompt" --message 01010101 --l-bits 8 --model gpt2 -o out.txt
python main.py detect_lbit out.txt --l-bits 8 --model gpt2 --key-file secret.key

# Multi-user
python -m src.main_multiuser generate --user-id 0 --l-bits 10 --model gpt2 -o out.txt "Prompt"
python -m src.main_multiuser trace out.txt --l-bits 10 --model gpt2

# Evaluate
python main.py evaluate --prompts-file assets/prompts.txt --model gpt2
python helper_scripts\analyse.py evaluation/evaluation_results
```

### Parameter Cheatsheet

| Scenario | delta | entropy_threshold | z_threshold | max_new_tokens |
|----------|-------|-------------------|-------------|----------------|
| Default | 2.5 | 4.0 | 4.0 | 256 |
| High fluency | 2.0 | 4.5 | 4.0 | 256 |
| Strong detection | 3.5 | 3.5 | 3.5 | 512 |
| Short text | 3.0 | 3.5 | 3.5 | 128 |
| Long text | 2.5 | 4.0 | 4.0 | 1024 |

---

## Frequently Asked Questions

**Q: Can I use this with other models like LLaMA or Claude?**
A: Yes! Extend the `LanguageModel` base class in `src/models.py` with your model's implementation.

**Q: Is the watermark detectable by humans?**
A: No. The watermark operates at the statistical level and doesn't introduce visible patterns or artifacts.

**Q: Can the watermark survive translation?**
A: Partially. Translation may disrupt token-level watermarks, but semantic-preserving perturbations (like paraphrasing) show good robustness.

**Q: How secure is the key derivation?**
A: Uses HMAC-SHA256, a cryptographically secure PRF. Keep the master key secret and use adequate entropy (32 bytes recommended).

**Q: Can I watermark existing text?**
A: No. Watermarking must occur during generation (modifies logits before sampling). This is a generative watermark, not a post-hoc embedding.

**Q: What's the maximum message length for L-bit?**
A: Depends on text length and parameters. Longer text → more blocks → can embed more bits. Typical: 32-bit message in 512 tokens (GPT-2, default params).

---

**For more details, see:**
- `COMMANDS.md` - Copy-paste command examples
- `src/watermark.py` - Implementation details
- GitHub Issues - Community support

**Happy watermarking! 🔐**
