#!/bin/bash

# =============================================================================
# SLURM Job — Tracing time at a FIXED 8 bits per layer, varying depth
#
#   2 layers x 8 bits = L=16   (current Hi-DyPa payload width)
#   3 layers x 8 bits = L=24
#   4 layers x 8 bits = L=32
#
# Unlike the earlier depth sweep, L grows with depth here, so each width needs
# its own run: MAU and Segment-WM must use the same payload width as the Hi-DyPa
# arm they are compared against. Three runs, one per L, then compare across them.
#
# N = 4096 users, not 1000. 4096 is the only value that is a perfect square,
# cube AND fourth power, so every tree comes out COMPLETE:
#   L=16  2 layers, fanout 64 each  -> 64^2  = 4096
#   L=24  3 layers, fanout 16 each  -> 16^3  = 4096
#   L=32  4 layers, fanout  8 each  ->  8^4  = 4096
# With the full 8-bit codebook (128 words) instead, capacity would be 128^4 =
# 268M, leaving the tree almost entirely empty; every node would be ragged, the
# factorised fast path would never fire, and the measurement would be of the
# fallback path rather than the decoder under test.
#
# Requested on milan-gpu so the run sits on the same A100 hardware as the rest of
# the results. Note the benchmark itself is pure CPU -- it maps codewords to
# identities and never loads a language model -- so the GPU stays idle. The
# timings are unaffected by that; only the queue wait is.
#
# OzSTAR does not allow --exclusive, so the node is shared. Shared nodes add
# timing jitter, which is why --repeat defaults to 9 and the reported figure is
# the MEDIAN of those runs rather than the mean.
# =============================================================================
#SBATCH --job-name=hidypa_8bit_layers
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/trpham/hidypa/slurm_out/slurm-%j.out

module --force purge
module load apptainer

CODE_DIR="/home/trpham/hidypa"
SIF="/fred/oz411/trpham/hidypa.sif"

cd ${CODE_DIR}
mkdir -p slurm_out evaluation/tracing_8bit

# single-threaded Python; stop BLAS from adding timing jitter
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

APPTAINER_RUN="apptainer exec \
    --bind ${CODE_DIR}:/workspace \
    --env PYTHONIOENCODING=utf-8 \
    --env LC_ALL=C.UTF-8 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONHASHSEED=0 \
    ${SIF}"

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUT_DIR="/workspace/evaluation/tracing_8bit"
USERS=${USERS:-/workspace/assets/users_4096.csv}
NUM_USERS=${NUM_USERS:-4096}
TRIALS=${TRIALS:-5000}
REPEAT=${REPEAT:-9}
RATES=${RATES:-0.0,0.02,0.05,0.10,0.15,0.20}

echo "=========================================="
echo "Run tag : ${RUN_TAG}"
echo "Node    : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (allocated, unused)"
else
    echo "GPU     : $(scontrol show node "$(hostname -s)" 2>/dev/null | grep -oiE 'Gres=[^ ]*' | head -1) (allocated, unused)"
fi
echo "Users   : ${NUM_USERS} from ${USERS}"
echo "=========================================="

echo ""
echo "### Step 0: hierarchy unit tests"
${APPTAINER_RUN} python3 /workspace/tests/test_hierarchy.py
if [ $? -ne 0 ]; then
    echo "ERROR: hierarchy tests failed — aborting before any measurement."
    exit 1
fi

# -----------------------------------------------------------------------------
# One run per payload width. Each compares Hi-DyPa (at that depth) against MAU
# and Segment-WM at the SAME width, so every row inside a run is comparable.
# -----------------------------------------------------------------------------
run_width () {
    local L=$1 CFG=$2 D=$3
    echo ""
    echo "############################################################"
    echo "# ${D} layers x 8 bits  ->  L = ${L}   (config ${CFG})"
    echo "############################################################"
    ${APPTAINER_RUN} python3 /workspace/evaluation_scripts/benchmark_tracing_time.py \
        --users-file ${USERS} \
        --num-users ${NUM_USERS} \
        --l-bits ${L} \
        --hier-configs ${CFG} \
        --include-scan \
        --trials ${TRIALS} \
        --repeat ${REPEAT} \
        --erasure-rates ${RATES} \
        --run-tag ${RUN_TAG} \
        --output ${OUT_DIR}/tracing_L${L}_${D}layers_${RUN_TAG}.json
}

run_width 16 l16_8x2 2
run_width 24 l24_8x3 3
run_width 32 l32_8x4 4

echo ""
echo "=========================================="
echo "Done. Results:"
ls -la ${CODE_DIR}/evaluation/tracing_8bit/
echo ""
echo "Render with:"
echo "  python helper_scripts/show_hierarchical_results.py evaluation/tracing_8bit"
echo "  python helper_scripts/show_hierarchical_results.py evaluation/tracing_8bit --markdown"
echo "=========================================="
