#!/bin/bash

# =============================================================================
# SLURM Job — Tracing-time evaluation: Hi-DyPa (2/3/4 layers) vs MAU vs Segment
#
# Step 1 (GPU) measures the REAL channel: it generates with GPT-2, detects, and
# traces, so the tracing times come from codewords the detector actually
# produced. Step 2 (CPU) is the controlled comparison: same three schemes,
# synthetic channel, swept over user count and erasure rate, which is where the
# O(N) vs O(D) scaling shows up.
#
# Step 2 needs no GPU. It runs here only so both halves land in one result set;
# if the GPU queue is long you can run step 2 alone on a CPU partition in
# minutes (see the header of the CPU-only job script).
#
# Hi-DyPa configurations — all L = 16, all capacity 1024, <= 8 bits per layer,
# so DEPTH is the only variable:
#   l16_d2   8+8        d=(4,2)      fanout (16,64)     tables 131,072 entries
#   l16_d3   8+4+4      d=(4,2,2)    fanout (16,8,8)    tables  66,048 entries
#   l16_d4   4+4+4+4    d=(2,2,2,2)  fanout (8,8,4,4)   tables   1,024 entries
#
# Hi-DyPa rows use the factorised decoder: each layer is resolved from its own
# bits in a single table lookup, independently of the others. --include-scan
# adds the sequential coarse-to-fine decoder so the gain is visible.
# =============================================================================
#SBATCH --job-name=hidypa_trace_depth
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/trpham/hidypa/slurm_out/slurm-%j.out

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
mkdir -p slurm_out evaluation/tracing_depth

APPTAINER_RUN="apptainer exec --nv \
    --bind ${CODE_DIR}:/workspace \
    --bind ${HF_CACHE}:${HF_CACHE} \
    --env HF_HOME=${HF_CACHE} \
    --env HF_HUB_CACHE=${HF_HUB} \
    --env TRANSFORMERS_CACHE=${HF_HUB} \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONIOENCODING=utf-8 \
    --env LC_ALL=C.UTF-8 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    ${SIF}"

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUT_DIR="/workspace/evaluation/tracing_depth"
MODEL=${MODEL:-gpt2}
NUM_USERS=${NUM_USERS:-1000}
NUM_PROMPTS=${NUM_PROMPTS:-60}

echo "=========================================="
echo "Run tag : ${RUN_TAG}"
echo "Model   : ${MODEL}   users: ${NUM_USERS}   prompts: ${NUM_PROMPTS}"
echo "Node    : $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    scontrol show node "$(hostname -s)" 2>/dev/null | grep -oiE "Gres=[^ ]*" || true
    echo "(nvidia-smi not on host PATH; the 'gpu   :' line in step 1 is authoritative)"
fi
echo "=========================================="

# -----------------------------------------------------------------------------
# Step 0: correctness gate. Timings are meaningless if the decoder is wrong.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 0: hierarchy unit tests"
${APPTAINER_RUN} python3 /workspace/tests/test_hierarchy.py
if [ $? -ne 0 ]; then
    echo "ERROR: hierarchy tests failed — aborting before any measurement."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1 (GPU): real channel. Hi-DyPa at each depth, plus MAU and Segment.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 1: real-channel run (GPU), L=16"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hierarchical_gpt2.py \
    --mode full \
    --model ${MODEL} \
    --prompts-file /workspace/assets/prompts.txt \
    --users-file /workspace/assets/users.csv \
    --num-users ${NUM_USERS} \
    --num-prompts ${NUM_PROMPTS} \
    --l-bits 16 \
    --max-new-tokens 960 \
    --schemes hier:l16_d2,hier:l16_d3,hier:l16_d4,naive,segment \
    --delta 3.5 --entropy-threshold 2.5 --hashing-context 5 --z-threshold 4.0 \
    --run-tag ${RUN_TAG} \
    --output-dir ${OUT_DIR}

# -----------------------------------------------------------------------------
# Step 2 (CPU): controlled tracing-time comparison, swept over erasure rate.
# --include-scan contrasts the factorised (layer-independent) decoder against
# the sequential one at each depth.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 2: controlled benchmark, N=1000, erasure sweep"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/benchmark_tracing_time.py \
    --users-file /workspace/assets/users.csv \
    --num-users ${NUM_USERS} \
    --l-bits 16 \
    --depths 2,3,4 \
    --include-scan \
    --trials 5000 \
    --repeat 9 \
    --erasure-rates 0.0,0.02,0.05,0.10,0.15,0.20,0.30 \
    --run-tag ${RUN_TAG} \
    --output ${OUT_DIR}/tracing_time_main_${RUN_TAG}.json

# -----------------------------------------------------------------------------
# Step 3 (CPU): scaling in N. This is where O(N) and O(D) separate.
# Capacity of every config is 1024, so the sweep stops there.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 3: scaling in N"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/benchmark_tracing_time.py \
    --users-file /workspace/assets/users.csv \
    --n-sweep 64,128,256,512,1000 \
    --l-bits 16 \
    --depths 2,3,4 \
    --trials 5000 \
    --repeat 9 \
    --erasure-rates 0.0,0.10 \
    --run-tag ${RUN_TAG} \
    --output ${OUT_DIR}/tracing_time_scaling_${RUN_TAG}.json

echo ""
echo "=========================================="
echo "Done. Results:"
ls -la ${CODE_DIR}/evaluation/tracing_depth/
echo ""
echo "Render the tables with:"
echo "  python helper_scripts/show_hierarchical_results.py evaluation/tracing_depth"
echo "  python helper_scripts/show_hierarchical_results.py evaluation/tracing_depth --markdown"
echo "=========================================="
