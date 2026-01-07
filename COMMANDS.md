> **Note:** This file contains commands for running the watermarking schemes (zero-bit, L-bit, multi-user) only. For evaluation scripts (e.g., `evaluate_hi_dypa_detection.py`, `compare_collusion_resistance.py`, `evaluate_synonym_attack.py`), see the README.md file.

## Local quickstart (Windows cmd)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Zero-Bit Watermarking

### Generate Watermarked Text

Embed a zero-bit watermark (binary detection only, no message):

```bat
python main.py generate "The future of AI is" --model gpt2 --max-new-tokens 512 ^
  -o demonstration/watermarked_output.txt ^
  -k demonstration/secret.key
```

**Output:**
- `demonstration/watermarked_output.txt`: Generated text with embedded watermark
- `demonstration/secret.key`: Secret key (required for detection)

**Generate plain text (no watermark):**
```bat
python main.py generate "The future of AI is" --model gpt2 --no-watermark --max-new-tokens 512 ^
  -o demonstration/output_plain.txt
```

### Detect Watermark

Detect if text contains a zero-bit watermark:

```bat
python main.py detect demonstration/watermarked_output.txt --model gpt2 ^
  --key-file demonstration/secret.key
```

**Output:**
- Z-score (statistical measure of watermark presence)
- Detection decision (WATERMARKED / NOT WATERMARKED)
- Block count (number of watermarked positions detected)

---

## L-Bit Watermarking

### Generate Watermarked Text with Embedded Message

Embed a specific bitstring message (L bits) into the generated text:

```bat
python main.py generate_lbit "The future of AI is" --model gpt2 ^
  --message 01010101 --l-bits 8 ^
  --max-new-tokens 512 ^
  -o demonstration/watermarked_lbit.txt ^
  --key-file demonstration/secret_lbit.key
```

**Parameters:**
- `--message`: Binary string to embed (must be exactly L bits)
- `--l-bits`: Length of the message (must match message length)

**Output:**
- `demonstration/watermarked_lbit.txt`: Generated text with embedded message
- `demonstration/secret_lbit.key`: Secret key (required for recovery)

### Recover Embedded Message

Recover the embedded bitstring from watermarked text:

```bat
python main.py detect_lbit demonstration/watermarked_lbit.txt --model gpt2 ^
  --l-bits 8 ^
  --key-file demonstration/secret_lbit.key
```

**Output:**
- Recovered message (may contain `⊥` for undecided bits)
- Bit-by-bit recovery results

---

## Multi-User Fingerprinting

### Setup

The `assets/users.csv` file contains user metadata:
```text
UserId,Username
0,0
1,1
2,2
...
999,999
```

### Generate Watermarked Text for a User

Embed a user-specific fingerprint (group codeword) into generated text:

**Using default grouping:**
```bat
python -m src.main_multiuser generate "The future of AI is" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 2 ^
  --max-new-tokens 512 ^
  -o demonstration/multiuser_user0.txt
```

**Using custom grouping:**
```bat
python -m src.main_multiuser generate "The future of AI is" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 2 ^
  --max-groups 25 ^
  --users-per-group 40 ^
  --max-new-tokens 512 ^
  -o demonstration/multiuser_user0.txt
```

**Group Configuration (for grouped scheme):**
- `--min-distance 2`: Defaults to 100 groups, 10 users per group
- `--min-distance 3`: Defaults to 50 groups, 20 users per group (stronger collusion resistance)
- `--max-groups <N>`: Maximum number of groups allowed (overrides default)
- `--users-per-group <N>`: Number of users per group (overrides default)

**Note:** If the CSV contains more users than can fit within the specified constraints, only the first N users will be used (where N = max_groups × users_per_group).

**Output:**
- `demonstration/multiuser_user0.txt`: Generated text with user's group codeword embedded
- `demonstration/multiuser_master.key`: Master key (shared across all users, created automatically)

### Trace Watermarked Text to User(s)

Trace watermarked text back to the originating user(s):

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --scheme grouped ^
  --min-distance 2 ^
  demonstration/multiuser_user0.txt
```

**Important:** Use the same `--min-distance`, `--max-groups`, and `--users-per-group` values that were used during generation.

**Output:**
- Accused user(s) with match scores
- Group membership information
- Collusion detection (if multiple users' codewords are detected)

**Naive Scheme (no grouping):**
```bat
python -m src.main_multiuser generate "Binary fingerprint" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 42 ^
  --l-bits 10 ^
  --scheme naive ^
  --max-new-tokens 256 ^
  -o demonstration/naive_user42.txt
```

```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 10 ^
  --scheme naive ^
  demonstration/naive_user42.txt
```

**Hi-DyPa Scheme (group codewords + per-user fingerprints):**

**Basic usage (automatic group/user allocation):**
```bat
python -m src.main_multiuser generate "The future of AI is" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 8 ^
  --scheme hi_dypa ^
  --group-bits 6 ^
  --user-bits 2 ^
  --min-distance 2 ^
  --max-new-tokens 512 ^
  -o demonstration/hi_dypa_user0.txt
```

**With explicit group/user control:**
```bat
python -m src.main_multiuser generate "The future of AI is" ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --user-id 0 ^
  --l-bits 8 ^
  --scheme hi_dypa ^
  --group-bits 6 ^
  --user-bits 2 ^
  --min-distance 2 ^
  --max-groups 32 ^
  --users-per-group 4 ^
  --max-new-tokens 512 ^
  -o demonstration/hi_dypa_user0.txt
```

**Parameters:**
- `--group-bits`: Number of bits for group codewords (must satisfy `group-bits + user-bits == l-bits`)
- `--user-bits`: Number of bits for user fingerprints within groups
- `--min-distance`: Minimum Hamming distance between group codewords (2 or 3)
- `--max-groups` (optional): Maximum number of groups allowed (default: auto-calculated)
- `--users-per-group` (optional): Number of users per group (default: auto-calculated, max = 2^user_bits)

**Trace hi_dypa watermarked text:**
```bat
python -m src.main_multiuser trace ^
  --users-file assets/users.csv ^
  --model gpt2 ^
  --l-bits 8 ^
  --scheme hi_dypa ^
  --group-bits 6 ^
  --user-bits 2 ^
  --min-distance 2 ^
  --max-groups 32 ^
  --users-per-group 4 ^
  demonstration/hi_dypa_user0.txt
```

**Important:** Use the same `--group-bits`, `--user-bits`, `--min-distance`, `--max-groups`, and `--users-per-group` values that were used during generation.

**Constraints:**
- `--max-groups` must be ≤ 2^group_bits (e.g., with group_bits=6, max 64 groups)
- `--users-per-group` must be ≤ 2^user_bits (e.g., with user_bits=2, max 4 users per group)
- If CSV contains more users than `max_groups × users_per_group`, only the first N users are used

**How Hi-DyPa Scheme Works:**
- Combines group codewords (with minimum distance for cross-group collusion resistance) with per-user fingerprints
- Each user's codeword = `group_code[group_bits] + user_code[user_bits]`
- Group codewords are generated using BCH codes with guaranteed minimum Hamming distance
- User fingerprints are simple binary representations of user index within the group
- During tracing, first identifies the group, then identifies the user within that group

---

## Key Parameters

### Watermark Strength
- `--delta`: Bias strength (default: 3.5). Higher = easier detection, may lower fluency. Range: 2.0-3.5
- `--entropy-threshold`: Where to apply watermark (default: 2.5). Higher = fewer blocks, more selective. Range: 3.5-4.5
- `--z-threshold`: Detection threshold (default: 4.0). Lower = more sensitive, risks false positives

### Model Constraints
- GPT-2: Keep `prompt_length + max_new_tokens ≤ 1024` (context limit)
- Recommended: `--max-new-tokens 512` for GPT-2

### L-Bit Specific
- `--l-bits`: Length of message to embed (must match `--message` length)
- If you see `⊥` in recovered bits, increase `--delta` or decrease `--entropy-threshold` during generation

---

## Notes

- First run downloads models; ensure internet connection for GPT-2
- The attention_mask warning for GPT-2 can be ignored
- Secret keys are cryptographically secure (HMAC-SHA256); keep them private
- Watermarking must occur during generation (cannot watermark existing text)
