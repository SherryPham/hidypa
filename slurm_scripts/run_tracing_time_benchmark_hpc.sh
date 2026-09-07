#!/bin/bash

# =============================================================================
# SLURM Job — Tracing-stage time benchmark (Hi-DyPa 3-layer vs Segment-WM, L=16)
#
# This benchmark is CPU-ONLY: it measures codeword -> identity decoding and
# never loads a language model, so it does NOT request a GPU. Do not run it on
# milan-gpu; that would hold an A100 idle for the whole job.
#
# Timing stability matters here, so the job asks for a whole node (--exclusive).
# If your allocation cannot do that, drop --exclusive and raise --repeat instead.
# =============================================================================
#SBATCH --job-name=hidypa_tracing_time
#SBATCH --account=oz411
#SBATCH -p milan                       # CPU partition — check `sinfo -s` if this name differs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/home/trpham/hidypa/slurm_out/slurm-%j.out

# =============================================================================
# Environment
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
mkdir -p evaluation/tracing_time

# Keep BLAS/OpenMP from stealing cores; the benchmark is single-threaded Python
# and background threads add timing jitter.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# No --nv: this job needs no GPU.
APPTAINER_RUN="apptainer exec \
    --bind ${CODE_DIR}:/workspace \
    --bind ${HF_CACHE}:${HF_CACHE} \
    --env HF_HOME=${HF_CACHE} \
    --env HF_HUB_CACHE=${HF_HUB} \
    --env TRANSFORMERS_CACHE=${HF_HUB} \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_OFFLINE=1 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env PYTHONHASHSEED=0 \
    --env PYTHONIOENCODING=utf-8 \
    --env LC_ALL=C.UTF-8 \
    ${SIF}"

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUT_DIR="/workspace/evaluation/tracing_time"

echo "=========================================="
echo "Run tag        : ${RUN_TAG}"
echo "Node           : $(hostname)"
echo "CPUs allocated : ${SLURM_CPUS_PER_TASK:-?}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Step 0: correctness gate. If the hierarchy tests fail, the timings are
# meaningless, so stop here.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 0: hierarchy unit tests"
${APPTAINER_RUN} python3 /workspace/tests/test_hierarchy.py
if [ $? -ne 0 ]; then
    echo "ERROR: hierarchy tests failed — aborting before timing."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: main comparison at N = 1000, sweeping the erasure rate.
# --include-asis additionally times the classes in src/watermark.py exactly as
# they ship today (needs torch, which the container has).
# -----------------------------------------------------------------------------
echo ""
echo "### Step 1: all schemes, N=1000, erasure sweep"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/benchmark_tracing_time.py \
    --users-file /workspace/assets/users.csv \
    --num-users 1000 \
    --trials 5000 \
    --repeat 9 \
    --erasure-rates 0.0,0.02,0.05,0.10,0.15,0.20,0.30 \
    --include-asis \
    --run-tag ${RUN_TAG} \
    --output ${OUT_DIR}/tracing_time_main_${RUN_TAG}.json

# -----------------------------------------------------------------------------
# Step 2: scaling in N — this is where O(N) and O(1) separate.
# Capacity of the 3-layer config is 1024, so the sweep stops there.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 2: scaling in N"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/benchmark_tracing_time.py \
    --users-file /workspace/assets/users.csv \
    --n-sweep 64,128,256,512,1000 \
    --trials 5000 \
    --repeat 9 \
    --erasure-rates 0.0,0.10 \
    --include-asis \
    --run-tag ${RUN_TAG} \
    --output ${OUT_DIR}/tracing_time_scaling_${RUN_TAG}.json

# -----------------------------------------------------------------------------
# Step 3: mixed erasure + flip channel, to check the d=4 top layer earns its bit.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 3: erasure + flip channel"
for FLIP in 0.02 0.05; do
    ${APPTAINER_RUN} python3 /workspace/evaluation_scripts/benchmark_tracing_time.py \
        --users-file /workspace/assets/users.csv \
        --num-users 1000 \
        --trials 5000 \
        --repeat 5 \
        --erasure-rates 0.0,0.05,0.10,0.20 \
        --flip-rate ${FLIP} \
        --run-tag ${RUN_TAG} \
        --output ${OUT_DIR}/tracing_time_flip${FLIP}_${RUN_TAG}.json
done

echo ""
echo "=========================================="
echo "Done. Results in ${OUT_DIR}"
ls -la ${CODE_DIR}/evaluation/tracing_time/
echo "=========================================="
