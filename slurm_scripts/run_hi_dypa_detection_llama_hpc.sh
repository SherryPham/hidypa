#!/bin/bash

# =============================================================================
# SLURM Job Configuration for OzSTAR HPC (Llama-3.2-1B)
# =============================================================================
#SBATCH --job-name=hier_detect_llama
#SBATCH --account=YOUR_ACCOUNT_ID          # Replace with your OzSTAR project code (e.g., oz000)
#SBATCH --partition=gpu                    # OzSTAR GPU partition
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                  # Good number of CPU cores for data loading processing
#SBATCH --mem=32G                          # 32GB RAM is plenty for 1B model
#SBATCH --time=48:00:00
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

# Change to the hidypa project directory before running
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

RUN_TAG=${RUN_TAG:-job_llama_1b_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
echo "Using run tag: ${RUN_TAG}"

echo "Running hi_dypa detection evaluation for Llama 1B"
echo "=========================================="

run_py evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 300 \
    --users-file assets/users.csv \
    --model llama-3.2-1b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/hi_dypa_detection_llama \
    --run-tag ${RUN_TAG}

echo "Evaluation complete."
