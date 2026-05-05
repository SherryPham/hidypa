#!/bin/bash

# =============================================================================
# SLURM Job Configuration — Framing attack resistance (GPT-2, OzSTAR)
# =============================================================================
#SBATCH --job-name=framing_attack
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/trpham/hidypa/slurm_out/slurm-%j.out

# =============================================================================
# Environment setup
# =============================================================================
module --force purge
module load apptainer

CODE_DIR="/home/trpham/hidypa"
SIF="/fred/oz411/trpham/hidypa.sif"
HF_CACHE="/fred/oz411/trpham/hf_cache"
HF_HUB="/fred/oz411/trpham/hf_cache/hub"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=${HF_HUB}
export TRANSFORMERS_CACHE=${HF_HUB}
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd ${CODE_DIR}
mkdir -p slurm_out
mkdir -p evaluation/framing_attack

APPTAINER_RUN="apptainer exec --nv \
    --bind ${CODE_DIR}:/workspace \
    --bind ${HF_CACHE}:${HF_CACHE} \
    --env HF_HOME=${HF_CACHE} \
    --env HF_HUB_CACHE=${HF_HUB} \
    --env TRANSFORMERS_CACHE=${HF_HUB} \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    ${SIF}"

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
echo "Using run tag: ${RUN_TAG}"
echo "Running framing attack resistance evaluation for all 8 configurations (GPT-2)..."
echo "L = 8 for all configurations"
echo ""

# =============================================================================
# Configuration 1: Naive / MAU baseline (Codeword-Aware adversary)
# =============================================================================
echo "=========================================="
echo "Configuration 1: Naive / MAU (L=8)"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme naive \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 2: Hi-DyPa G=1, U=7
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 3: Hi-DyPa G=2, U=6
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 4: Hi-DyPa G=3, U=5
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 5: Hi-DyPa G=4, U=4 (paper main config)
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4 (paper main config)"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 6: Hi-DyPa G=5, U=3
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 7: Hi-DyPa G=6, U=2
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

# =============================================================================
# Configuration 8: Hi-DyPa G=7, U=1
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_framing_attack.py \
    --scheme hi_dypa \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --prompts-file /workspace/assets/prompts.txt \
    --n-trials 20 \
    --max-k 10 \
    --users-file /workspace/assets/users.csv \
    --max-new-tokens 400 \
    --seed 42 \
    --output-dir /workspace/evaluation/framing_attack \
    --run-tag ${RUN_TAG} \
    --save-raw-results

echo ""
echo "=========================================="
echo "All framing attack evaluations complete!"
echo "Results saved to evaluation/framing_attack/"
echo "=========================================="
