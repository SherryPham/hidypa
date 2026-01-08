#!/bin/bash

# =============================================================================
# SLURM Job Configuration
# NOTE: Update account and partition for your HPC environment
# =============================================================================
#SBATCH --job-name=hier_robustness
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --output=slurm_out/slurm-%j.out

# =============================================================================
# Load HPC-specific paths from config file
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config/hpc_paths.sh"

module --force purge
module load apptainer

PROJECT=${HPC_PROJECT}
SIF=${HPC_SIF}
HF=${HPC_HF_CACHE}

export HF_HOME=$HF
export HF_HUB_CACHE=$HF
export HF_DATASETS_CACHE=$HF
export TRANSFORMERS_CACHE=$HF
export NLTK_DATA=$HF/nltk_data
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p slurm_out
cd $PROJECT

run_py () {
  apptainer exec --nv \
    -B ${HPC_BIND_PATH} \
    --env HF_HOME=$HF_HOME \
    --env HF_HUB_CACHE=$HF_HUB_CACHE \
    --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
    --env TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
    --env NLTK_DATA=$NLTK_DATA \
    --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE \
    --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
    "$SIF" python3 "$@"
}

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
echo "Using run tag: ${RUN_TAG}"

echo "Running hi_dypa robustness evaluation for all 9 configurations..."
echo "L = 8 for all configurations"
echo "Testing deletion attacks: 4 percents × 4 modes = 16 variants per prompt"
echo ""

# Configuration 1: Naive (L=8, no hierarchy)
echo "=========================================="
echo "Configuration 1: Naive (L=8)"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme naive \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 2: Hi-DyPa G=1, U=7 → 1 group, 128 users per group
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 3: Hi-DyPa G=2, U=6 → 2 groups, 64 users per group
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 4: Hi-DyPa G=3, U=5 → 4 groups, 32 users per group
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 5: Hi-DyPa G=4, U=4 → 8 groups, 16 users per group
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 6: Hi-DyPa G=5, U=3 → 16 groups, 8 users per group
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 7: Hi-DyPa G=6, U=2 → 32 groups, 4 users per group
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 8: Hi-DyPa G=7, U=1 → 64 groups, 2 users per group
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

# Configuration 9: Group-only G=8, U=0 → 128 groups, 1 user per group
echo ""
echo "=========================================="
echo "Configuration 9: Group-only G=8, U=0"
echo "=========================================="
run_py evaluation_scripts/evaluate_hi_dypa_robustness.py \
    --scheme hi_dypa \
    --group-bits 8 \
    --user-bits 0 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/robustness \
    --run-tag ${RUN_TAG}

echo ""
echo "=========================================="
echo "All robustness evaluations complete!"
echo "=========================================="
