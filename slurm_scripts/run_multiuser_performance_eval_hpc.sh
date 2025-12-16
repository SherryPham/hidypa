#!/bin/bash

#SBATCH --job-name=multiuser_perf_eval
#SBATCH --account=oz411
#SBATCH -p volta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
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
run_py evaluation_scripts/evaluate_multiuser_performance.py \
    --users-file assets/users.csv \
    --model gpt2 \
    --l-bits 10 \
    --delta 3.5 \
    --entropy-threshold 2.5 \
    --hashing-context 5 \
    --z-threshold 4.0 \
    --max-new-tokens 512 \
    --prompts-file assets/prompts.txt \
    --max-prompts 300 \
    --user-id 0 \
    --output-dir evaluation/multiuser_performance
