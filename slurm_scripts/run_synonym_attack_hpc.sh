#!/bin/bash

#SBATCH --job-name=hier_synonym
#SBATCH --account=oz411
#SBATCH -p skylake-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=64:00:00
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

echo "Running synonym attack evaluation for all 9 configurations..."
echo "L = 8 for all configurations"
echo "Single-pass WordNet synonym substitution at 10% of tokens"
echo ""

run_eval () {
    local scheme=$1
    local group_bits=$2
    local user_bits=$3
    local label=$4

    echo ""
    echo "=========================================="
    echo "Configuration: ${label}"
    echo "=========================================="

    if [ "${scheme}" = "naive" ]; then
        run_py evaluation_scripts/evaluate_synonym_attack.py \
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
            --output-dir evaluation/synonym_attack \
            --run-tag ${RUN_TAG}
    else
        run_py evaluation_scripts/evaluate_synonym_attack.py \
            --scheme hi_dypa \
            --group-bits "${group_bits}" \
            --user-bits "${user_bits}" \
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
            --output-dir evaluation/synonym_attack \
            --run-tag ${RUN_TAG}
    fi
}

# Configuration 1: Naive (L=8, no hierarchy)
run_eval "naive" 0 0 "Naive (L=8)"

# Configuration 2: Hi-DyPa G=1, U=7
run_eval "hi_dypa" 1 7 "Hi-DyPa G=1, U=7"

# Configuration 3: Hi-DyPa G=2, U=6
run_eval "hi_dypa" 2 6 "Hi-DyPa G=2, U=6"

# Configuration 4: Hi-DyPa G=3, U=5
run_eval "hi_dypa" 3 5 "Hi-DyPa G=3, U=5"

# Configuration 5: Hi-DyPa G=4, U=4
run_eval "hi_dypa" 4 4 "Hi-DyPa G=4, U=4"

# Configuration 6: Hi-DyPa G=5, U=3
run_eval "hi_dypa" 5 3 "Hi-DyPa G=5, U=3"

# Configuration 7: Hi-DyPa G=6, U=2
run_eval "hi_dypa" 6 2 "Hi-DyPa G=6, U=2"

# Configuration 8: Hi-DyPa G=7, U=1
run_eval "hi_dypa" 7 1 "Hi-DyPa G=7, U=1"

# Configuration 9: Group-only G=8, U=0
run_eval "hi_dypa" 8 0 "Group-only G=8, U=0"

echo ""
echo "=========================================="
echo "All synonym attack evaluations complete!"
echo "=========================================="

