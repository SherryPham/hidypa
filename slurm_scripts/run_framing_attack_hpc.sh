#!/bin/bash

# =============================================================================
# SLURM Job Configuration
# NOTE: Update account and partition for your HPC environment
# =============================================================================
#SBATCH --job-name=framing_attack
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

echo "Running framing attack resistance evaluation for all 8 configurations..."
echo "L = 8 for all configurations, model = gpt2"
echo ""

# Configuration 1: Naive / MAU baseline (Codeword-Aware adversary)
echo "=========================================="
echo "Configuration 1: Naive / MAU (L=8)"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme naive \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 2: Hi-DyPa G=1, U=7
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 3: Hi-DyPa G=2, U=6
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 4: Hi-DyPa G=3, U=5
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 5: Hi-DyPa G=4, U=4
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 6: Hi-DyPa G=5, U=3
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 7: Hi-DyPa G=6, U=2
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
    --users-file assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# Configuration 8: Hi-DyPa G=7, U=1
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
run_py evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file assets/prompts.txt \
    --num-prompts 100 \
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
