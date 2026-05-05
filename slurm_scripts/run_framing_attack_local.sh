#!/bin/bash
# =============================================================================
# Local run — Framing attack resistance (DeepSeek-7B)
# Run from the hidypa/ directory with the venv activated:
#   bash slurm_scripts/run_framing_attack_local.sh
# =============================================================================

set -e

RUN_TAG=${RUN_TAG:-local_$(date +%Y%m%d_%H%M%S)}
echo "Using run tag: ${RUN_TAG}"
echo "Running framing attack resistance for all 8 configurations (DeepSeek-LLM-7B)"
echo "n_trials=20, max_k=10 — total generations per config: 400"
echo ""

mkdir -p evaluation/framing_attack

# =============================================================================
# Configuration 1: Naive / MAU baseline (Codeword-Aware adversary)
# =============================================================================
echo "=========================================="
echo "Configuration 1: Naive / MAU (L=8)"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme naive \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 2: Hi-DyPa G=1, U=7
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 3: Hi-DyPa G=2, U=6
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 4: Hi-DyPa G=3, U=5
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 5: Hi-DyPa G=4, U=4 (paper main config)
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4 (paper main config)"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 6: Hi-DyPa G=5, U=3
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 7: Hi-DyPa G=6, U=2
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 8: Hi-DyPa G=7, U=1
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
python evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --n-trials 20 \
    --k-values 1 5 10 50 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

echo ""
echo "=========================================="
echo "All framing attack evaluations complete!"
echo "Results saved to evaluation/framing_attack/"
echo "=========================================="
