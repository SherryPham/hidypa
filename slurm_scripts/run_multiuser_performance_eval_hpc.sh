#!/bin/bash

# =============================================================================
# SLURM Job Configuration
# NOTE: Update account and partition for your HPC environment
# =============================================================================
#SBATCH --job-name=multiuser_perf_eval
#SBATCH --account=oz411
#SBATCH -p skylake-gpu
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

UNIFIED_SUMMARY="${PROJECT}/evaluation/multiuser_performance/performance_summary.csv"

# Step 1: Run naive baseline once
echo "=========================================="
echo "Running Naive Baseline (L=8)"
echo "=========================================="
run_py evaluation_scripts/evaluate_multiuser_performance.py \
    --users-file assets/users.csv \
    --model gpt2 \
    --l-bits 8 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --prompts-file assets/prompts.txt \
    --max-prompts 300 \
    --user-start 1 \
    --user-end 10 \
    --output-dir evaluation/multiuser_performance/naive/L8 \
    --unified-summary-path "${UNIFIED_SUMMARY}"

# Step 2: Run all Hi-DyPa configs (hi_dypa-only)
GU_CONFIGS=(
    "1 7"
    "2 6"
    "3 5"
    "4 4"
    "5 3"
    "6 2"
    "7 1"
    "8 0"
)

for config in "${GU_CONFIGS[@]}"; do
    read -r group_bits user_bits <<< "$config"
    echo ""
    echo "=========================================="
    echo "Running Hi-DyPa G=${group_bits}, U=${user_bits}"
    echo "=========================================="
    
    run_py evaluation_scripts/evaluate_multiuser_performance.py \
        --users-file assets/users.csv \
        --model gpt2 \
        --l-bits 8 \
        --group-bits "${group_bits}" \
        --user-bits "${user_bits}" \
        --delta 3.5 \
        --entropy-threshold 2.5 \
        --hashing-context 5 \
        --z-threshold 4.0 \
        --max-new-tokens 512 \
        --prompts-file assets/prompts.txt \
        --max-prompts 300 \
        --user-start 1 \
        --user-end 10 \
        --hi_dypa-only \
        --output-dir "evaluation/multiuser_performance/hi_dypa/G${group_bits}_U${user_bits}" \
        --unified-summary-path "${UNIFIED_SUMMARY}"
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Unified summary: ${UNIFIED_SUMMARY}"
echo "=========================================="
