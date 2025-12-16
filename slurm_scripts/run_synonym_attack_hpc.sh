#!/bin/bash

#SBATCH --job-name=hier_synonym
#SBATCH --account=oz411
#SBATCH -p volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
    $SIF python3 "$@"
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
            --num-prompts 200 \
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
            --scheme hierarchical \
            --group-bits "${group_bits}" \
            --user-bits "${user_bits}" \
            --l-bits 8 \
            --prompts-file assets/prompts.txt \
            --num-prompts 200 \
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

# Configuration 2: Hierarchical G=1, U=7
run_eval "hierarchical" 1 7 "Hierarchical G=1, U=7"

# Configuration 3: Hierarchical G=2, U=6
run_eval "hierarchical" 2 6 "Hierarchical G=2, U=6"

# Configuration 4: Hierarchical G=3, U=5
run_eval "hierarchical" 3 5 "Hierarchical G=3, U=5"

# Configuration 5: Hierarchical G=4, U=4
run_eval "hierarchical" 4 4 "Hierarchical G=4, U=4"

# Configuration 6: Hierarchical G=5, U=3
run_eval "hierarchical" 5 3 "Hierarchical G=5, U=3"

# Configuration 7: Hierarchical G=6, U=2
run_eval "hierarchical" 6 2 "Hierarchical G=6, U=2"

# Configuration 8: Hierarchical G=7, U=1
run_eval "hierarchical" 7 1 "Hierarchical G=7, U=1"

# Configuration 9: Group-only G=8, U=0
run_eval "hierarchical" 8 0 "Group-only G=8, U=0"

echo ""
echo "=========================================="
echo "All synonym attack evaluations complete!"
echo "=========================================="

