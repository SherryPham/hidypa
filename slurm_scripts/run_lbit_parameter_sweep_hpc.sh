#!/bin/bash

# =============================================================================
# SLURM Job Configuration
# NOTE: Update account and partition for your HPC environment
# =============================================================================
#SBATCH --job-name=lbit_param_sweep
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

echo "=========================================="
echo "Running L-bit Parameter Sweep"
echo "=========================================="
echo "L = 8 (fixed)"
echo "Delta values: [2.0, 2.5, 3.0, 3.5, 4.0]"
echo "Entropy threshold values: [1.5, 2.0, 2.5, 3.0, 3.5]"
echo "Total combinations: 25"
echo "Prompts: 100 (default)"
echo ""

run_py evaluation_scripts/run_lbit_parameter_sweep.py \
    --prompts-file assets/prompts.txt \
    --max-prompts 100 \
    --model gpt2 \
    --min-l 8 \
    --max-l 8 \
    --deltas 2.0 2.5 3.0 3.5 4.0 \
    --entropy-thresholds 1.5 2.0 2.5 3.0 3.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir evaluation/lbit_parameter_sweep

echo ""
echo "=========================================="
echo "Parameter sweep complete!"
echo "Results saved to: evaluation/lbit_parameter_sweep"
echo "=========================================="
