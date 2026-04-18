#!/bin/bash

# =============================================================================
# SLURM Job Configuration — Table 2: Clean-text detection (OPT model, OzSTAR)
# Uses Apptainer container instead of venv
# =============================================================================
#SBATCH --job-name=hidypa_detection_opt
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --output=/home/trpham/hidypa/slurm_out/slurm-%j.out

# =============================================================================
# Environment setup
# =============================================================================
module --force purge
module load apptainer

CODE_DIR="/home/trpham/hidypa"
SIF="/fred/oz411/trpham/hidypa.sif"
HF_CACHE="/fred/oz411/trpham/hf_cache"

export HF_HOME=${HF_CACHE}
export HF_HUB_CACHE=${HF_CACHE}
export TRANSFORMERS_CACHE=${HF_CACHE}
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd ${CODE_DIR}
mkdir -p slurm_out
mkdir -p evaluation/hi_dypa_detection

# Apptainer run command with GPU passthrough and bind mounts
APPTAINER_RUN="apptainer exec --nv \
    --bind ${CODE_DIR}:/workspace \
    --bind ${HF_CACHE}:${HF_CACHE} \
    --env HF_HOME=${HF_CACHE} \
    --env HF_HUB_CACHE=${HF_CACHE} \
    --env TRANSFORMERS_CACHE=${HF_CACHE} \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    ${SIF}"

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
echo "Using run tag: ${RUN_TAG}"
echo "Running Table 2: Clean-text detection (9 configurations) with OPT-1.3B"
echo ""

# =============================================================================
# Configuration 1: Naive (L=8, no hierarchy)
# =============================================================================
echo "=========================================="
echo "Configuration 1: Naive (L=8)"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme naive \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 2: Hi-DyPa G=1, U=7 → 2 groups, 128 users per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 3: Hi-DyPa G=2, U=6 → 4 groups, 64 users per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 4: Hi-DyPa G=3, U=5 → 8 groups, 32 users per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 5: Hi-DyPa G=4, U=4 → 16 groups, 16 users per group (paper main)
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4 (paper main config)"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 6: Hi-DyPa G=5, U=3 → 32 groups, 8 users per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 7: Hi-DyPa G=6, U=2 → 64 groups, 4 users per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 8: Hi-DyPa G=7, U=1 → 128 groups, 2 users per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

# =============================================================================
# Configuration 9: Group-only G=8, U=0 → 256 groups, 1 user per group
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 9: Group-only G=8, U=0"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hi_dypa_detection.py \
    --scheme hi_dypa \
    --group-bits 8 \
    --user-bits 0 \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --num-prompts 300 \
    --users-file /workspace/assets/users.csv \
    --model opt-1.3b \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --output-dir /workspace/evaluation/hi_dypa_detection \
    --run-tag ${RUN_TAG}

echo ""
echo "=========================================="
echo "All 9 configurations complete!"
echo "Results saved to evaluation/hi_dypa_detection/"
echo "=========================================="
