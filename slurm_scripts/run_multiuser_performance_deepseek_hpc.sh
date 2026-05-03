#!/bin/bash

# =============================================================================
# SLURM Job Configuration — Table 8: Computational overhead (DeepSeek-7B, OzSTAR)
# Uses Apptainer container instead of venv
# =============================================================================
#SBATCH --job-name=hidypa_perf_deepseek
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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
mkdir -p evaluation/multiuser_performance

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
UNIFIED_SUMMARY="/workspace/evaluation/multiuser_performance/unified_summary_${RUN_TAG}.csv"

echo "Using run tag: ${RUN_TAG}"
echo "Running Table 8: Computational overhead (9 configurations) with DeepSeek-LLM-7B"
echo "Unified summary: ${UNIFIED_SUMMARY}"
echo ""

# Common args for all runs
COMMON_ARGS="
    --model deepseek-llm-7b \
    --l-bits 8 \
    --prompts-file /workspace/assets/prompts.txt \
    --max-prompts 20 \
    --users-file /workspace/assets/users.csv \
    --user-start 1 \
    --user-end 5 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --unified-summary-path ${UNIFIED_SUMMARY}"

# =============================================================================
# Configuration 1: Naive (L=8, no hierarchy)
# =============================================================================
echo "=========================================="
echo "Configuration 1: Naive (L=8)"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --output-dir /workspace/evaluation/multiuser_performance/naive_L8

# =============================================================================
# Configuration 2: Hi-DyPa G=1, U=7
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 2: Hi-DyPa G=1, U=7"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 1 \
    --user-bits 7 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G1_U7

# =============================================================================
# Configuration 3: Hi-DyPa G=2, U=6
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 3: Hi-DyPa G=2, U=6"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 2 \
    --user-bits 6 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G2_U6

# =============================================================================
# Configuration 4: Hi-DyPa G=3, U=5
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 4: Hi-DyPa G=3, U=5"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 3 \
    --user-bits 5 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G3_U5

# =============================================================================
# Configuration 5: Hi-DyPa G=4, U=4 (paper main config)
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 5: Hi-DyPa G=4, U=4 (paper main config)"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 4 \
    --user-bits 4 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G4_U4

# =============================================================================
# Configuration 6: Hi-DyPa G=5, U=3
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 6: Hi-DyPa G=5, U=3"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 5 \
    --user-bits 3 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G5_U3

# =============================================================================
# Configuration 7: Hi-DyPa G=6, U=2
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 7: Hi-DyPa G=6, U=2"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 6 \
    --user-bits 2 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G6_U2

# =============================================================================
# Configuration 8: Hi-DyPa G=7, U=1
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 8: Hi-DyPa G=7, U=1"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 7 \
    --user-bits 1 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G7_U1

# =============================================================================
# Configuration 9: Group-only G=8, U=0
# =============================================================================
echo ""
echo "=========================================="
echo "Configuration 9: Group-only G=8, U=0"
echo "=========================================="
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_multiuser_performance.py \
    ${COMMON_ARGS} \
    --hi_dypa-only \
    --group-bits 8 \
    --user-bits 0 \
    --output-dir /workspace/evaluation/multiuser_performance/hi_dypa_G8_U0

echo ""
echo "=========================================="
echo "All 9 computational overhead configurations complete!"
echo "Results saved to evaluation/multiuser_performance/"
echo "Unified summary: ${UNIFIED_SUMMARY}"
echo "=========================================="
