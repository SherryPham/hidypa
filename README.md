# Hi-DyPa: Practical Multi-User Watermarking for Detection and Tracing in LLM System

A comprehensive framework for embedding and detecting statistical watermarks in LLM-generated text. This implementation supports five watermarking schemes: **zero-bit** (binary detection), **L-bit** (message embedding), **naive multi-user** (baseline fingerprinting), **grouped multi-user** (BCH-based fingerprinting), and **Hi-DyPa** (hierarchical multi-user fingerprinting).

## About This Repository

This codebase provides an open-source reimplementation of the multi-user watermarking schemes proposed by Cohen et al. in *Watermarking Language Models for Many Adaptive Users*. In particular, we implement the three theoretical schemes described in the paper as a unified, executable framework.

Building on this reimplementation, we develop **Hi-DyPa**, a hierarchical multi-user watermarking scheme that extends the baseline framework with structured attribution and improved tracing efficiency. All baseline schemes and Hi-DyPa share the same underlying embedding pipeline, enabling controlled and fair comparisons.

### Implemented Schemes

This repository implements the following watermarking schemes:

1. **Zero-Bit Watermarking** (`ZeroBitWatermarker`)
   - Binary detection: determines whether text is watermarked (yes/no)
   - Foundation for all other schemes
   - Uses statistical z-score analysis over watermarked positions

2. **L-Bit Watermarking** (`LBitWatermarker`)
   - Embeds and recovers arbitrary binary messages of length L
   - Enables message encoding in generated text
   - Uses per-bit key derivation via HMAC-SHA256

3. **Naive Multi-User Watermarking** (`NaiveMultiUserWatermarker`)
   - Baseline multi-user scheme from Cohen et al.
   - Assigns each user a unique L-bit codeword (binary expansion of user ID)
   - Simple but vulnerable to collusion attacks

4. **Grouped Multi-User Watermarking** (`GroupedMultiUserWatermarker`)
   - BCH-based scheme from Cohen et al. with guaranteed minimum Hamming distance
   - Groups users and assigns group codewords with error-correcting properties
   - Improved collusion resistance compared to naive scheme
   - Uses BCH even-parity construction with minimum Hamming distance of 2

5. **Hi-DyPa Multi-User Watermarking** (`HiDyPaMultiUserWatermarker`)
   - **Our proposed hierarchical scheme** that extends the baseline framework
   - Combines group-level codewords (with minimum Hamming distance) with per-user fingerprints
   - Two-stage tracing: first identifies the group, then the user within the group
   - Flexible allocation of bits between group and user identifiers
   - Improved scalability and tracing efficiency compared to baseline schemes

All schemes share the same underlying L-bit embedding mechanism, ensuring fair and controlled comparisons across different multi-user approaches.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Zero-Bit Watermarking](#zero-bit-watermarking)
  - [L-Bit Watermarking](#l-bit-watermarking)
  - [Naive Multi-User Watermarking](#naive-multi-user-watermarking)
  - [Grouped Multi-User Watermarking](#grouped-multi-user-watermarking)
  - [Hi-DyPa Multi-User Watermarking](#hi-dypa-multi-user-watermarking)
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
- **Naive Scheme:**
  1. Assign each user a unique L-bit codeword (binary expansion of user ID)
  2. Embed the user's codeword using L-bit watermarking
  3. During tracing, match recovered codeword directly to user IDs
  4. Simple but vulnerable to collusion attacks

- **Grouped Scheme (BCH-Based):**
  1. Generate BCH codewords with guaranteed minimum Hamming distance of 2 (even-parity construction)
  2. Assign users to groups sequentially (all users in a group share the same group codeword)
  3. Embed the user's group codeword using L-bit watermarking
  4. During tracing, match recovered codeword to group(s) and identify accused users
  5. Improved collusion resistance compared to naive scheme

- **Hi-DyPa Scheme:**
  1. Generate group codewords with minimum Hamming distance (for cross-group collusion resistance)
  2. Assign simple binary fingerprints to users within each group
  3. Combine group codeword + user fingerprint to create L-bit message
  4. Embed the combined codeword using L-bit watermarking
  5. During tracing, first identify the group, then identify the user within that group
  6. Improved scalability and tracing efficiency compared to baseline schemes

---

## Features

**Five watermarking schemes**: Zero-bit detection, L-bit message embedding, naive multi-user tracing, grouped multi-user tracing (BCH-based), and Hi-DyPa hierarchical multi-user tracing
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
│   │                                  - HiDyPaMultiUserWatermarker
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
│   ├── evaluate_hi_dypa_detection.py  # Pure detection performance for Hi-DyPa schemes
│   ├── evaluate_hi_dypa_robustness.py  # Robustness to deletion attacks for Hi-DyPa schemes
│   ├── evaluate_paraphrasing_attack.py  # Robustness to paraphrasing (T5-small) attacks
│   ├── evaluate_synonym_attack.py  # Robustness to synonym substitution attacks
│   ├── evaluate_rewrite_attack.py  # Robustness to LLM rewrite attacks
│   ├── run_lbit_sweep.py            # L-bit parameter sweep
│   └── run_lbit_parameter_sweep.py  # L-bit parameter sweep (alternative)
│
├── evaluation_scripts_local/        # Local convenience wrappers for evaluation scripts
│   ├── run_hi_dypa_detection_local.py  # Run hi_dypa detection evaluation locally
│   ├── run_hi_dypa_performance_local.py  # Run hi_dypa performance evaluation locally
│   ├── run_hi_dypa_robustness_local.py  # Run hi_dypa robustness evaluation locally
│   ├── run_paraphrasing_attack_local.py  # Run paraphrasing attack evaluation locally
│   ├── run_rewrite_attack_local.py  # Run rewrite attack evaluation locally
│   ├── run_synonym_attack_local.py  # Run synonym attack evaluation locally
│   └── run_collusion_resistance_local.py  # Run collusion resistance evaluation locally
│
├── helper_scripts/                  # Analysis and utility scripts
│   ├── analyse.py                   # Generate plots from evaluation results
│   ├── anonymize_evaluation_paths.py # Anonymize paths in evaluation results
│   ├── compute_code_capacity.py     # Compute code capacity for fingerprinting
│   ├── create_collusion_scenario.py # Create collusion test scenarios
│   ├── download_flan_prompts.py     # Download FLAN prompts for evaluation
│   ├── download_models_hpc.py       # Pre-download models for HPC
│   ├── generate_users.py            # Create user database CSV
│   ├── rename_hierarchical_to_hidypa.py # Rename hierarchical to hi_dypa naming
│   ├── visualise_blocks.py          # Visualize watermark blocks
│   ├── visualise_lbit_blocks.py     # Visualize L-bit blocks
│   └── visualize_groups.py          # Visualize multi-user groups
│
├── slurm_scripts/                   # HPC cluster batch job scripts
│   ├── run_collusion_eval_hpc.sh    # Collusion resistance evaluation
│   ├── run_multiuser_performance_eval_hpc.sh  # Multi-user performance evaluation
│   ├── run_lbit_sweep_hpc.sh        # L-bit parameter sweep
│   ├── run_hi_dypa_detection_hpc.sh  # Hi-DyPa detection evaluation
│   ├── run_hi_dypa_robustness_hpc.sh  # Hi-DyPa robustness evaluation
│   ├── run_paraphrasing_attack_hpc.sh  # Paraphrasing attack evaluation (T5-small)
│   ├── run_synonym_attack_hpc.sh    # Synonym substitution attack evaluation
│   └── run_rewrite_attack_hpc.sh    # Rewrite attack evaluation
│
├── assets/                          # Data files
│   ├── users.csv                    # 1000 users (UserIds 0-999)
│   └── prompts.txt                  # Evaluation prompts (typically 300+)
│
├── evaluation/                      # Evaluation results (auto-created by HPC jobs)
│   ├── hi_dypa_detection/      # Detection performance results
│   │   ├── naive/L8/job_*/          # Naive L=8 results
│   │   ├── hi_dypa/G*_U*/job_*/ # Hi-DyPa results per config
│   │   ├── seeds.txt                # Random seeds used
│   │   └── summary_all_configs.csv  # Aggregated summary across all configs
│   ├── robustness/                  # Deletion attack robustness results
│   │   ├── naive/L8/job_*/          # Naive L=8 results
│   │   ├── hi_dypa/G*_U*/job_*/ # Hi-DyPa results per config
│   │   ├── seeds.txt
│   │   ├── summary_all_configs.csv
│   │   └── summary_all_configs_concise.csv
│   ├── paraphrasing_attack/         # T5-small paraphrasing attack results
│   │   ├── naive/L8/job_*/
│   │   ├── hi_dypa/G*_U*/job_*/
│   │   ├── seeds.txt
│   │   ├── summary_all_configs.csv
│   │   └── summary_all_configs_concise.csv
│   ├── synonym_attack/              # WordNet synonym substitution attack results
│   │   ├── naive/L8/job_*/
│   │   ├── hi_dypa/G*_U*/job_*/
│   │   ├── seeds.txt
│   │   ├── summary_all_configs.csv
│   │   └── summary_all_configs_concise.csv
│   └── rewrite_attack/              # LLM rewrite attack results
│       ├── naive/L8/job_*/
│       ├── hi_dypa/G*_U*/job_*/
│       ├── seeds.txt
│       ├── summary_all_configs.csv
│       └── summary_all_configs_concise.csv
│
├── evaluation_2_backup/             # Backup of multiuser performance results
│   └── multiuser_performance/
│       ├── naive/L8/
│       ├── hi_dypa/G*_U*/
│       └── performance_summary.csv
│
└── demonstration/                   # Example outputs
    └── hi_dypa_user0.txt       # Hi-DyPa scheme example (user 0)
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
- **pyside6**: GUI components (optional, for visualization tools)
- **openpyxl**: Excel file support
- **psutil**: System/process utilities
- **datasets**: HuggingFace datasets (for prompt downloading)

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
- `--delta`: Watermark strength (1.0-5.0, default 3.5)
- `--entropy-threshold`: Minimum entropy for watermarking (1.0-6.0, default 2.5)
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

### Naive Multi-User Watermarking

The baseline multi-user scheme from Cohen et al. that assigns each user a unique L-bit codeword (binary expansion of user ID). This scheme is simple but vulnerable to collusion attacks where multiple users combine their outputs.

#### How It Works

- Each user receives a unique L-bit codeword based on their user ID
- The codeword is the binary representation of the user ID (padded/truncated to L bits)
- Uses L-bit watermarking to embed the user's codeword
- During tracing, matches recovered codeword to user IDs

#### Generate for User

```bat
python -m src.main_multiuser generate ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 42 ^
  --l-bits 10 ^
  --scheme naive ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  --key-file demonstration\naive_master.key ^
  -o demonstration\naive_user42.txt ^
  "The future of AI is"
```

**Key points:**
- L=10 supports up to 2¹⁰ = 1024 users
- Each user gets a unique codeword (no grouping)
- User 42's codeword is the binary representation of 42

**Expected outputs:**
1. **naive_user42.txt**: Text watermarked with user 42's codeword
2. **naive_master.key**: Master key (shared across all users)
3. Console output showing the user's codeword

#### Trace User

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --scheme naive ^
  --key-file demonstration\naive_master.key ^
  demonstration\naive_user42.txt
```

**Expected output:**
```
--- Trace Results ---
  Text traced back to user(s):
     - User ID: 42, Username: 42, Match: 100.00%
```

**Limitations:**
- Vulnerable to collusion: multiple users can combine codewords to frame others
- No error correction: single bit errors can misidentify users
- Consider using Grouped or Hi-DyPa schemes for better collusion resistance

---

### Grouped Multi-User Watermarking

Trace generated text back to specific users using BCH error-correcting codes with guaranteed minimum Hamming distance for improved collusion resistance. This is the BCH-based scheme from Cohen et al.

#### How It Works

- **BCH Codes**: Codewords are generated with guaranteed minimum Hamming distance of 2 (even-parity construction)
- **Group Assignment**: Users are assigned to groups sequentially:
  - `group_id = user_id // users_per_group`
  - All users in the same group share the same group codeword
  - This prevents collusion attacks where users combine codewords to frame others
- **Group Count**: Number of groups = 2^(group_bits - 1) due to BCH even-parity codeword construction
  - Example: With L=10 bits, max groups = 2^(10-1) = 512 groups

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

You can customize usernames or add more users. The number of groups is limited to 2^(L-1) due to BCH even-parity construction.

#### Generate for User

```bat
python -m src.main_multiuser generate ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 2 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  --key-file demonstration\grouped_master.key ^
  -o demonstration\grouped_user0.txt ^
  "The future of AI is"
```

**Key points:**
- L=10 supports up to 2^(10-1) = 512 groups (due to BCH even-parity construction)
- Users are assigned to groups sequentially based on `--users-per-group`
- All users in the same group share the same codeword

**Expected outputs:**
1. **grouped_user0.txt**: Text watermarked with user 0's group codeword
2. **grouped_master.key**: Master key (shared across all users)
3. Console output showing: "User ID 0 belongs to Group 0"

#### Trace User

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 2 ^
  --key-file demonstration\grouped_master.key ^
  demonstration\grouped_user0.txt
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

### Hi-DyPa Multi-User Watermarking

**Our proposed hierarchical scheme** that extends the baseline framework with structured attribution and improved tracing efficiency. Hi-DyPa combines group-level codewords (with minimum Hamming distance for cross-group collusion resistance) with per-user fingerprints within each group.

#### How It Works

- **Hierarchical Structure**: Two-stage codeword design
  - **Group codewords**: Generated using BCH even-parity codes with guaranteed minimum Hamming distance of 2
  - **User fingerprints**: Simple binary representations of user index within each group
  - **Combined codeword**: `group_code[group_bits] + user_code[user_bits] = L bits`
- **Two-Stage Tracing**: 
  1. First identifies the group from the group codeword
  2. Then identifies the user within that group from the user fingerprint
- **Flexible Bit Allocation**: You can allocate L bits between group and user identifiers
  - More group bits → more groups, fewer users per group
  - More user bits → fewer groups, more users per group
- **Improved Efficiency**: Hierarchical structure enables faster tracing and better scalability

#### Generate for User

**Basic usage (G=4, U=4, L=8):**
```bat
python -m src.main_multiuser generate ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 8 ^
  --scheme hi_dypa ^
  --group-bits 4 ^
  --user-bits 4 ^
  --min-distance 2 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  --key-file demonstration\hi_dypa_master.key ^
  -o demonstration\hi_dypa_user0.txt ^
  "The future of AI is"
```

**With explicit group/user control:**
```bat
python -m src.main_multiuser generate ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 8 ^
  --scheme hi_dypa ^
  --group-bits 4 ^
  --user-bits 4 ^
  --min-distance 2 ^
  --max-groups 8 ^
  --users-per-group 16 ^
  --delta 2.5 ^
  --entropy-threshold 4.0 ^
  --max-new-tokens 512 ^
  --key-file demonstration\hi_dypa_master.key ^
  -o demonstration\hi_dypa_user0.txt ^
  "The future of AI is"
```

**Key points:**
- `--group-bits` + `--user-bits` must equal `--l-bits`
- L=8 with G=4, U=4: up to 2^(4-1) = 8 groups, 2⁴ = 16 users per group (128 total users)
- User 0 belongs to Group 0, User 0 within that group
- Each user's codeword = group codeword (4 bits) + user fingerprint (4 bits)

**Parameters:**
- `--group-bits`: Number of bits for group codewords (must satisfy `group-bits + user-bits == l-bits`)
- `--user-bits`: Number of bits for user fingerprints within groups
- `--min-distance`: Minimum Hamming distance between group codewords (fixed at 2)
- `--max-groups` (optional): Maximum number of groups allowed (default: auto-calculated)
- `--users-per-group` (optional): Number of users per group (default: auto-calculated, max = 2^user_bits)

**Constraints:**
- `--max-groups` must be ≤ 2^(group_bits-1) due to BCH even-parity construction (e.g., with group_bits=4, max 8 groups)
- `--users-per-group` must be ≤ 2^user_bits (e.g., with user_bits=4, max 16 users per group)
- If CSV contains more users than `max_groups × users_per_group`, only the first N users are used

**Expected outputs:**
1. **hi_dypa_user0.txt**: Text watermarked with user 0's combined codeword
2. **hi_dypa_master.key**: Master key (shared across all users)
3. Console output showing: "User ID 0 belongs to Group 0, User 0 within group"

#### Trace User

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 8 ^
  --scheme hi_dypa ^
  --group-bits 4 ^
  --user-bits 4 ^
  --min-distance 2 ^
  --max-groups 8 ^
  --users-per-group 16 ^
  --key-file demonstration\hi_dypa_master.key ^
  demonstration\hi_dypa_user0.txt
```

**Important:** Use the same `--group-bits`, `--user-bits`, `--min-distance`, `--max-groups`, and `--users-per-group` values that were used during generation.

**Expected output:**
```
--- Trace Results ---
  Text traced back to user(s):
     - User ID: 0, Username: 0, Group: 0, Match: 100.00%
       User ID 0 belongs to Group 0, User 0 within group
```

**Two-stage tracing:**
- First stage: Identifies the group from the group codeword portion
- Second stage: Identifies the user within that group from the user fingerprint portion
- More efficient than flat codeword matching

**Advantages over baseline schemes:**
- **Improved scalability**: Hierarchical structure supports larger user bases
- **Faster tracing**: Two-stage process reduces search space
- **Flexible allocation**: Can optimize group/user bit allocation for specific use cases
- **Collusion resistance**: Group codewords maintain minimum Hamming distance

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

#### `src/watermark.py` (1247 lines)
**Purpose:** Core watermarking algorithms
**Classes:**
- `ZeroBitWatermarker`: Binary detection
- `LBitWatermarker`: Message embedding
- `NaiveMultiUserWatermarker`: Legacy per-user fingerprinting
- `GroupedMultiUserWatermarker`: Fingerprinting with grouped codes
- `HiDyPaMultiUserWatermarker`: Hierarchical multi-user fingerprinting
- `WatermarkLogitsProcessor`: Transformers integration (zero-bit)
- `LBitLogitProcessor`: Transformers integration (L-bit)

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

#### `src/fingerprinting.py` (376 lines)
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
- `min_distance` (int): Minimum Hamming distance between codewords (fixed at 2)
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

#### `src/commands.py` (404 lines)
**Purpose:** CLI command implementations
**Functions:**
- `cmd_generate(args)`: Zero-bit generation
- `cmd_detect(args)`: Zero-bit detection
- `cmd_generate_lbit(args)`: L-bit generation
- `cmd_detect_lbit(args)`: L-bit detection
- `cmd_evaluate(args)`: Batch evaluation

**Not called directly** (invoked by `main.py` based on subcommand)

#### `src/parser.py` (166 lines)
**Purpose:** Argument parsing and validation
**Features:**
- Subcommand routing
- Parameter range validation
- NLTK setup and verification
- Default value handling

**Not called directly** (used by `main.py`)

#### `src/utils.py` (208 lines)
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

#### `src/main_multiuser.py` (254 lines)
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
- Evaluates 9 configurations: naive (L=8) and hi_dypa (G=1,U=7 through G=8,U=0)
- Uses the same sampled colluding users across schemes per prompt (fair comparison)
- Supports multiple collusion patterns:
  - 2 colluders: `same_group_2`, `cross_group_2`
  - 3 colluders: `same_group_3`, `cross_group_3`, `mixed_2same_1diff`
- **Smart case skipping**: Automatically skips collusion cases that require more groups than available (e.g., skips cross-group tests for G=1, skips 3-colluder cross-group for G=2)
- **Reduced verbosity**: Logs warnings once per configuration instead of per prompt
- For each configuration and colluder count, saves:
  - Per‑prompt raw results (optional gzipped JSONL if `--save-raw-results` is enabled)
  - A `summary.json` in `2_colluders/` and `3_colluders/` with:
    - `success_rates[case_type].successful` / `total` / `success_rate`
    - `success_rates[case_type].false_positives` / `false_positive_rate` (accusations of users outside the colluding set)
- The local wrapper `run_collusion_resistance_local.py` aggregates all `summary.json` files into `evaluation/collusion_resistance/summary_all_configs.csv`, which includes:
  - Per‑case success counts/rates (same/cross/mixed, 2‑ and 3‑colluder)
  - Aggregate success rates: `overall_success_rate`, `success_rate_2_colluders`, `success_rate_3_colluders`
  - Aggregate false‑positive rates: `overall_false_positive_rate`, `false_positive_rate_2_colluders`, `false_positive_rate_3_colluders`
  - One row per configuration × colluder count (18 rows for naive + 8 Hi-DyPa configs × {2,3} colluders)

#### `evaluation_scripts/evaluate_hi_dypa_detection.py`
**Purpose:** Evaluate pure detection performance (no collusion) for hi_dypa multi-user watermarking at L=8, across all allocations of group bits and user bits
**Usage:**
```bat
python evaluation_scripts/evaluate_hi_dypa_detection.py ^
  --scheme hi_dypa ^
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
  --output-dir evaluation/hi_dypa_detection
```

**Features:**
- Evaluates 9 configurations: naive (L=8) and hi_dypa (G=1,U=7 through G=8,U=0)
- For each prompt: chooses random user, embeds watermark, detects codeword, decodes IDs
- Logs per-prompt: true/detected IDs, codewords, Hamming distance, z-scores, match statuses
- Computes metrics: group accuracy, user accuracy, full identity accuracy, L-bit accuracy, false positive/negative rates
- Saves per-prompt JSON files and summary JSON
- Supports both naive and Hi-DyPa schemes

#### `evaluation_scripts/evaluate_hi_dypa_robustness.py`
**Purpose:** Evaluate robustness to deletion attacks for hi_dypa multi-user watermarking at L=8, across all allocations of group bits and user bits
**Usage:**
```bat
python evaluation_scripts/evaluate_hi_dypa_robustness.py ^
  --scheme hi_dypa ^
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
- Same structure as `evaluate_hi_dypa_detection.py` but tests robustness to deletion attacks
- Tests 16 attack variants per prompt: 4 deletion percents (0.05, 0.10, 0.15, 0.20) × 4 deletion modes (start, middle, end, random)
- For each attack variant: applies deletion, detects codeword, decodes IDs, computes metrics
- Logs per-attack: `deletion_percent`, `deletion_mode`, recovered codeword, Hamming distance, z-scores, match statuses
- Computes metrics: group accuracy, user accuracy, full identity accuracy, average invalid symbols, average Hamming distance, average z-score, false positive/negative rates
- Saves all attack results to a single `raw_results.jsonl.gz` file (if enabled) plus per‑configuration `summary.json` and `summary.csv`
  - `summary.json` now contains:
    - `metrics`: aggregate metrics over all 16 variants
    - `metrics_by_variant`: metrics for each specific (percent, mode) variant
  - `summary.csv` stores the aggregate `metrics` as a simple metric/value table for that configuration
- Supports both naive and Hi-DyPa schemes

#### `evaluation_scripts/evaluate_paraphrasing_attack.py`
**Purpose:** Evaluate robustness against paraphrasing attacks (single-pass T5-small) for both naive and Hi-DyPa schemes at L=8
**Usage:**
```bat
python evaluation_scripts/evaluate_paraphrasing_attack.py ^
  --scheme hi_dypa ^
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
- Evaluates naive plus all eight hi_dypa splits (G=1,U=7 … G=8,U=0) in a single run
- Records per-attack variant: `paraphrase_ratio`, `paraphrase_mode`, recovered codeword, invalid symbols, Hamming distance, z-score, group/user matches
- Computes the same metrics as robustness (group/user/full accuracy, L-bit accuracy, false positive/negative rates, averages)
- Saves all attack results to a single `raw_results.jsonl.gz` file (if enabled) and per‑configuration `summary.json` / `summary.csv` under `evaluation/paraphrasing_attack`
  - `summary.json` contains both aggregate `metrics` and `metrics_by_variant` (metrics for each `(paraphrase_ratio, paraphrase_mode)` pair)
  - Local aggregation (`run_paraphrasing_attack_local.py`) produces `evaluation/paraphrasing_attack/summary_all_configs.csv` with one row **per variant per configuration** (16 rows per config), including variant columns and all metrics

#### `evaluation_scripts/evaluate_synonym_attack.py`
**Purpose:** Evaluate robustness against synonym substitution attacks (WordNet, 10% of tokens) for both naive and Hi-DyPa schemes at L=8
**Usage:**
```bat
python evaluation_scripts/evaluate_synonym_attack.py ^
  --scheme hi_dypa ^
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
- Evaluates naive plus all eight hi_dypa splits (G=1,U=7 … G=8,U=0) automatically
- Records per-attack variant: `synonym_ratio`, `synonym_mode`, recovered codeword, invalid symbols, Hamming distance, z-score, group/user matches
- Computes the same aggregate metrics as robustness and saves all attack results to `raw_results.jsonl.gz` (if enabled) and per‑configuration `summary.json` / `summary.csv` under `evaluation/synonym_attack/<scheme_dir>`
  - `summary.json` contains both aggregate `metrics` and `metrics_by_variant` (metrics for each `(synonym_ratio, synonym_mode)` pair)
  - Local aggregation (`run_synonym_attack_local.py`) produces `evaluation/synonym_attack/summary_all_configs.csv` with one row **per variant per configuration** (16 rows per config)

#### `evaluation_scripts/evaluate_rewrite_attack.py`
**Purpose:** Evaluate robustness against full-text rewrites generated deterministically by the same base LLM used for watermarking (GPT-2 / GPT-OSS variants)
**Usage:**
```bat
python evaluation_scripts/evaluate_rewrite_attack.py ^
  --scheme hi_dypa ^
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
- Evaluates naive plus all hi_dypa splits, mirroring other attack scripts
- Records per-attack variant: `rewrite_ratio`, `rewrite_mode`, recovered codeword, invalid symbols, Hamming distance, z-score, group/user matches
- Outputs all attack results to `raw_results.jsonl.gz` (if enabled) and per‑configuration `summary.json` / `summary.csv` under `evaluation/rewrite_attack/<scheme_dir>`
  - `summary.json` contains both aggregate `metrics` and `metrics_by_variant` (metrics for each `(rewrite_ratio, rewrite_mode)` pair)
  - Local aggregation (`run_rewrite_attack_local.py`) produces `evaluation/rewrite_attack/summary_all_configs.csv` with one row **per variant per configuration** (16 rows per config)

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

#### `helper_scripts/download_flan_prompts.py`
**Purpose:** Download FLAN prompts for evaluation
**Usage:**
```bat
python helper_scripts\download_flan_prompts.py --output-file assets/prompts.txt --num-prompts 300
```

**Output:** Text file with one prompt per line

#### `helper_scripts/download_models_hpc.py`
**Purpose:** Pre-cache models for offline HPC environments
**Usage:**
```bat
python helper_scripts\download_models_hpc.py --model gpt-oss-20b --cache-dir /shared/models
```

### Local Evaluation Scripts

Convenience wrappers for running evaluation scripts locally (without SLURM). These scripts sequentially invoke the corresponding evaluation scripts for all configurations.

#### `evaluation_scripts_local/run_hi_dypa_detection_local.py`
**Purpose:** Run all hi_dypa detection configurations locally (naive + 8 hi_dypa splits)
**Usage:**
```bat
python evaluation_scripts_local\run_hi_dypa_detection_local.py ^
  --prompts-file assets/prompts.txt ^
  --num-prompts 300 ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --delta 3.5 ^
  --entropy-threshold 2.5 ^
  --hashing-context 5 ^
  --z-threshold 4.0 ^
  --max-new-tokens 512 ^
  --output-dir evaluation/hi_dypa_detection
```

**Features:**
- Runs all 9 configurations sequentially (naive L=8 + 8 hi_dypa splits: G=1,U=7 through G=8,U=0)
- Forwards shared arguments to `evaluate_hi_dypa_detection.py`
- Generates consolidated summary CSV across all configurations
- Designed for single workstation use (mirrors HPC SLURM script functionality)

#### `evaluation_scripts_local/run_hi_dypa_performance_local.py`
**Purpose:** Run all hi_dypa performance evaluations locally (memory, computation, storage metrics)
**Usage:**
```bat
python evaluation_scripts_local\run_hi_dypa_performance_local.py ^
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
- Runs all 9 configurations sequentially (naive L=8 + 8 hi_dypa splits)
- Measures performance metrics: memory usage, computation time, storage size
- Forwards shared arguments to `evaluate_multiuser_performance.py`
- Generates consolidated performance summary

#### `evaluation_scripts_local/run_hi_dypa_robustness_local.py`
**Purpose:** Run all hi_dypa robustness evaluations locally (deletion attack resistance)
**Usage:**
```bat
python evaluation_scripts_local\run_hi_dypa_robustness_local.py ^
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
- Runs all 9 configurations sequentially (naive L=8 + 8 hi_dypa splits)
- Tests 16 deletion attack variants per prompt (4 deletion percents: 5%, 10%, 15%, 20% × 4 deletion modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_hi_dypa_robustness.py`
- Generates consolidated `evaluation/robustness/summary_all_configs.csv` across all configurations, with:
  - One row **per variant per configuration** (16 rows per config)
  - Variant columns (`deletion_percent`, `deletion_mode`) plus all robustness metrics
  - `num_attack_variants_per_prompt` and `total_attack_results` fields per configuration

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
- Runs all 9 configurations sequentially (naive L=8 + 8 hi_dypa splits)
- Tests 16 paraphrasing attack variants per prompt (4 ratios: 5%, 10%, 15%, 20% × 4 modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_paraphrasing_attack.py`
- Generates consolidated `evaluation/paraphrasing_attack/summary_all_configs.csv` across all configurations, with:
  - One row **per variant per configuration** (16 rows per config)
  - Variant columns (`paraphrase_ratio`, `paraphrase_mode`) plus all robustness metrics
  - `num_attack_variants_per_prompt` and `total_attack_results` fields per configuration

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
- Runs all 9 configurations sequentially (naive L=8 + 8 hi_dypa splits)
- Tests 16 rewrite attack variants per prompt (4 ratios: 5%, 10%, 15%, 20% × 4 modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_rewrite_attack.py`
- Generates consolidated `evaluation/rewrite_attack/summary_all_configs.csv` across all configurations, with:
  - One row **per variant per configuration** (16 rows per config)
  - Variant columns (`rewrite_ratio`, `rewrite_mode`) plus all robustness metrics
  - `num_attack_variants_per_prompt` and `total_attack_results` fields per configuration

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
- Runs all 9 configurations sequentially (naive L=8 + 8 hi_dypa splits)
- Tests 16 synonym substitution attack variants per prompt (4 ratios: 5%, 10%, 15%, 20% × 4 modes: start, middle, end, random)
- Forwards shared arguments to `evaluate_synonym_attack.py`
- Generates consolidated `evaluation/synonym_attack/summary_all_configs.csv` across all configurations, with:
  - One row **per variant per configuration** (16 rows per config)
  - Variant columns (`synonym_ratio`, `synonym_mode`) plus all robustness metrics
  - `num_attack_variants_per_prompt` and `total_attack_results` fields per configuration

### SLURM Scripts

All scripts in `slurm_scripts/` are HPC cluster batch job scripts for running evaluations.

**Available scripts:**
- `run_collusion_eval_hpc.sh`: Collusion resistance evaluation (300 prompts, 64-hour limit)
- `run_multiuser_performance_eval_hpc.sh`: Multi-user performance evaluation
- `run_lbit_sweep_hpc.sh`: L-bit parameter sweep
- `run_hi_dypa_detection_hpc.sh`: Hi-DyPa detection evaluation (300 prompts, 64-hour limit)
- `run_hi_dypa_robustness_hpc.sh`: Hi-DyPa robustness evaluation (300 prompts, 16 variants per prompt, 64-hour limit)
- `run_paraphrasing_attack_hpc.sh`: Paraphrasing attack evaluation (300 prompts, 16 variants per prompt, 64-hour limit)
- `run_rewrite_attack_hpc.sh`: Rewrite attack evaluation (300 prompts, 16 variants per prompt, 64-hour limit)
- `run_synonym_attack_hpc.sh`: Synonym substitution attack evaluation (300 prompts, 16 variants per prompt, 64-hour limit)

**Configuration:**
- All scripts use **300 prompts** per configuration
- All scripts have **64-hour time limit**
- Attack scripts (robustness, paraphrasing, rewrite, synonym) test **16 variants per prompt** (4 intensities × 4 modes)
- All scripts use **Apptainer** containers (no host Python/venv setup needed)
- Partition and account settings are configured in `config/hpc_paths.sh` (see [HPC Cluster Usage](#hpc-cluster-usage))

**Usage:**
```bash
# Run hi_dypa detection evaluation (9 configurations, 300 prompts each)
sbatch slurm_scripts/run_hi_dypa_detection_hpc.sh

# Run hi_dypa robustness evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_hi_dypa_robustness_hpc.sh

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
| `--delta` | 1.0-5.0 | 3.5 | Watermark strength (higher = stronger signal, lower fluency) |
| `--entropy-threshold` | 1.0-6.0 | 2.5 | Minimum entropy to watermark (higher = fewer, cleaner blocks) |
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
--delta 3.5 --entropy-threshold 2.5 --z-threshold 4.0
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
from src.models import GptOss120bModel
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

All SLURM scripts use **Apptainer** containers (no host Python/venv setup needed). Configure your partition and account in `config/hpc_paths.sh` (see [HPC Cluster Usage](#hpc-cluster-usage)). Pick the SLURM wrapper that matches your evaluation:

```bash
# Hi-DyPa detection evaluation (9 configurations, 300 prompts each)
sbatch slurm_scripts/run_hi_dypa_detection_hpc.sh

# Hi-DyPa robustness evaluation (9 configurations, 300 prompts, 16 variants each)
sbatch slurm_scripts/run_hi_dypa_robustness_hpc.sh

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
1. Tests two approaches:
   - **Naive**: Binary user ID-based fingerprinting (no grouping)
   - **Hi-DyPa**: Hierarchical fingerprinting with minimum Hamming distance 2
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
├── hi_dypa/
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

### Hi-DyPa Detection Performance Evaluation

Evaluate pure detection performance (no collusion) for hi_dypa multi-user watermarking at L=8, across all allocations of group bits and user bits.

#### Run Evaluation

**Naive scheme (L=8, no hierarchy):**
```bat
python evaluation_scripts/evaluate_hi_dypa_detection.py ^
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
  --output-dir evaluation/hi_dypa_detection
```

**Hi-DyPa scheme (G=4, U=4):**
```bat
python evaluation_scripts/evaluate_hi_dypa_detection.py ^
  --scheme hi_dypa ^
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
  --output-dir evaluation/hi_dypa_detection
```

**What it does:**
1. Evaluates 9 configurations:
   - **Naive**: L=8, no hierarchy, every user gets a flat L-bit codeword
   - **Hi-DyPa G=1, U=7**: 1 group, 128 users per group
   - **Hi-DyPa G=2, U=6**: 2 groups, 64 users per group
   - **Hi-DyPa G=3, U=5**: 4 groups, 32 users per group
   - **Hi-DyPa G=4, U=4**: 8 groups, 16 users per group
   - **Hi-DyPa G=5, U=3**: 16 groups, 8 users per group
   - **Hi-DyPa G=6, U=2**: 32 groups, 4 users per group
   - **Hi-DyPa G=7, U=1**: 64 groups, 2 users per group
   - **Hi-DyPa G=8, U=0**: 128 groups, 1 user per group (group-only mode)
2. For each prompt:
   - Chooses a random user ID
   - Embeds watermark
   - Detects L-bit codeword
   - Decodes group ID and user ID (for hi_dypa)
   - Logs: true/detected IDs, codewords, Hamming distance, z-scores, match statuses
3. Computes metrics:
   - **For naive**: L-bit accuracy, full identity accuracy, false positive/negative rates
   - **For hi_dypa**: group accuracy, user accuracy (given correct group), full identity accuracy, L-bit accuracy, false positive/negative rates

**Output structure:**
```
evaluation/hi_dypa_detection/
├── naive_L8/
│   ├── prompt_0.json
│   ├── prompt_1.json
│   ├── ...
│   └── summary.json
├── hi_dypa_G1_U7/
│   ├── prompt_0.json
│   ├── prompt_1.json
│   ├── ...
│   └── summary.json
├── hi_dypa_G2_U6/
│   └── ...
└── ... (other configurations)
```

**Per-prompt JSON contains:**
- `true_user_id`, `detected_user_id`
- `true_group_id`, `detected_group_id` (for hi_dypa)
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
sbatch slurm_scripts/run_hi_dypa_detection_hpc.sh
```

---

### Hi-DyPa Robustness Evaluation

Evaluate robustness to deletion attacks for hi_dypa multi-user watermarking at L=8, across all allocations of group bits and user bits.

#### Run Evaluation

**Naive scheme (L=8, no hierarchy):**
```bat
python evaluation_scripts/evaluate_hi_dypa_robustness.py ^
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

**Hi-DyPa scheme (G=4, U=4):**
```bat
python evaluation_scripts/evaluate_hi_dypa_robustness.py ^
  --scheme hi_dypa ^
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
1. Same as hi_dypa detection evaluation, but applies deletion attacks before detection
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
   - **For hi_dypa**: Group accuracy, user accuracy, full identity accuracy, false positive/negative rates, average invalid symbols, average Hamming distance, average z-score

**Output structure:**
```
evaluation/robustness/
├── naive/naive_L8/
│   ├── results.json          # All attack results (one entry per attack variant)
│   └── summary.json          # Summary metrics
├── hi_dypa/G4_U4/
│   ├── results.json          # All attack results (one entry per attack variant)
│   └── summary.json          # Summary metrics
└── ... (other configurations)
```

**Results JSON contains (per entry):**
- `prompt_id`, `prompt`, `true_user_id`
- `deletion_percent`, `deletion_mode`
- `recovered_codeword`, `ground_truth_codeword`
- `num_invalid_symbols`, `hamming_distance`, `z_score`
- `detected_user_id`, `detected_group_id` (for hi_dypa)
- `group_match`, `user_match`, `full_identity_match`

**Summary JSON contains:**
- Configuration (scheme, l_bits, group_bits, user_bits)
- Number of prompts, number of attack variants per prompt, total attack results
- Deletion percents and modes tested
- Computed metrics (accuracy rates, false positive/negative rates, averages)

**HPC Usage:**
Run all 9 configurations (naive + 8 hi_dypa variants) on HPC:
```bash
sbatch slurm_scripts/run_hi_dypa_robustness_hpc.sh
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
  title={Hi-DyPa: Practical Multi-User Watermarking for Detection and Tracing in LLM System},
  year={2025}
}
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

### Issues

For issues and feature requests, please refer to the anonymous repository.

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
| Default | 3.5 | 2.5 | 4.0 | 512 |
| High fluency | 2.0 | 4.5 | 4.0 | 256 |
| Strong detection | 3.5 | 2.5 | 3.5 | 512 |
| Short text | 3.0 | 3.5 | 3.5 | 128 |
| Long text | 3.5 | 2.5 | 4.0 | 1024 |

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

**Happy watermarking! 🔐**
