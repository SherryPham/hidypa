#!/bin/bash

#SBATCH --job-name=collusion_eval
#SBATCH --account=oz411
#SBATCH -p skylake-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_out/slurm-%j.out
module --force purge
module load apptainer

PROJECT=/fred/oz411/kpham/crypto-watermark
SIF=/fred/oz411/kpham/containers/crypto-watermark.sif
HF=/fred/oz411/kpham/huggingface

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
    -B /fred \
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

echo "Running collusion resistance evaluation for all 9 configurations..."
echo "L = 8 for all configurations"
echo ""

# Configuration 1: Naive (L=8, no hierarchy)
echo "=========================================="
echo "Configuration 1: Naive (L=8)"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme naive \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 2: Hierarchical G=1, U=7 → 1 group, 128 users per group
echo ""
echo "=========================================="
echo "Configuration 2: Hierarchical G=1, U=7"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 1 \
    --user-bits 7 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 3: Hierarchical G=2, U=6 → 2 groups, 64 users per group
echo ""
echo "=========================================="
echo "Configuration 3: Hierarchical G=2, U=6"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 2 \
    --user-bits 6 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 4: Hierarchical G=3, U=5 → 4 groups, 32 users per group
echo ""
echo "=========================================="
echo "Configuration 4: Hierarchical G=3, U=5"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 3 \
    --user-bits 5 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 5: Hierarchical G=4, U=4 → 8 groups, 16 users per group
echo ""
echo "=========================================="
echo "Configuration 5: Hierarchical G=4, U=4"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 4 \
    --user-bits 4 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 6: Hierarchical G=5, U=3 → 16 groups, 8 users per group
echo ""
echo "=========================================="
echo "Configuration 6: Hierarchical G=5, U=3"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 5 \
    --user-bits 3 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 7: Hierarchical G=6, U=2 → 32 groups, 4 users per group
echo ""
echo "=========================================="
echo "Configuration 7: Hierarchical G=6, U=2"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 6 \
    --user-bits 2 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 8: Hierarchical G=7, U=1 → 64 groups, 2 users per group
echo ""
echo "=========================================="
echo "Configuration 8: Hierarchical G=7, U=1"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 7 \
    --user-bits 1 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

# Configuration 9: Group-only G=8, U=0 → 128 groups, 1 user per group
echo ""
echo "=========================================="
echo "Configuration 9: Group-only G=8, U=0"
echo "=========================================="
run_py evaluation_scripts/compare_collusion_resistance.py \
    --scheme hierarchical \
    --group-bits 8 \
    --user-bits 0 \
    --l-bits 8 \
    --prompts-file assets/prompts.txt \
    --num-prompts 200 \
    --users-file assets/users.csv \
    --model gpt2 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 400 \
    --output-dir evaluation/collusion_resistance \
    --run-tag ${RUN_TAG}

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
