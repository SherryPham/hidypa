#!/bin/bash
# =============================================================================
# Local Windows run — Table 4: Robustness to deletion attacks (DeepSeek-7B)
# Run from the hidypa/ directory with the venv activated:
#   bash slurm_scripts/run_hi_dypa_robustness_local.sh
# =============================================================================

set -e

RUN_TAG=${RUN_TAG:-local_$(date +%Y%m%d_%H%M%S)}
echo "Using run tag: ${RUN_TAG}"
echo "Running Table 4: Robustness to deletion attacks (9 configurations) with DeepSeek-LLM-7B"
echo "Deletion percents: 5%, 10%, 15%, 20% × modes: start, middle, end, random"
echo ""

mkdir -p evaluation/robustness

# =============================================================================
# Configuration 1: Naive (L=8, no hierarchy)
# =============================================================================
echo "=========================================="
echo "Configuration 1: Naive (L=8)"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme naive \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 2: Hi-DyPa G=1, U=7
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 3: Hi-DyPa G=2, U=6
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 4: Hi-DyPa G=3, U=5
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 5: Hi-DyPa G=4, U=4 (paper main config)
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4 (paper main config)"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 6: Hi-DyPa G=5, U=3
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 7: Hi-DyPa G=6, U=2
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 8: Hi-DyPa G=7, U=1
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 9: Group-only G=8, U=0
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 9: Group-only G=8, U=0"
echo "=========================================="
python evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 8 \
    --user-bits 0 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model deepseek-llm-7b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

echo ""
echo "=========================================="
echo "All 9 robustness configurations complete!"
echo "Results saved to evaluation/robustness/"
echo "=========================================="
