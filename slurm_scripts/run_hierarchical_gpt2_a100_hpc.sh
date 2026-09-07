#!/bin/bash

# =============================================================================
# SLURM Job — End-to-end 3-layer Hi-DyPa on GPT-2 (A100)
#
# Stage 1 (calibrate) answers the gating question: is a 16-bit payload carryable
# on GPT-2 at all? Per-bit z scales as sqrt(blocks)/L, and GPT-2's context is
# capped at 1024 tokens, so blocks cannot exceed ~960 no matter what we ask for.
# If stage 1 says L=16 is not usable, stage 2's numbers are meaningless -- read
# the stage 1 table before trusting stage 2.
#
# Stage 2 (full) compares all schemes at the chosen operating point and reports
# per-level accuracy, containment size, and the tracing time on the REAL channel.
# =============================================================================
#SBATCH --job-name=hidypa_hier_gpt2
#SBATCH --account=oz411
#SBATCH -p milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
mkdir -p evaluation/hierarchical_gpt2

# The recovered-codeword symbols are non-ASCII; without a UTF-8 stdout the run
# dies with UnicodeEncodeError partway through.
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
    ${SIF}"

RUN_TAG=${RUN_TAG:-job_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUT_DIR="/workspace/evaluation/hierarchical_gpt2"

MODEL=${MODEL:-gpt2}
NUM_USERS=${NUM_USERS:-1000}
DELTA=${DELTA:-3.5}
ENTROPY=${ENTROPY:-2.5}
HC=${HC:-5}
ZT=${ZT:-4.0}

echo "=========================================="
echo "Run tag : ${RUN_TAG}"
echo "Model   : ${MODEL}"
echo "Node    : $(hostname)"
# nvidia-smi is not on the host PATH on every GPU node; fall back to slurm.
# torch reports the real device from inside the container in step 1.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \n  || scontrol show node "$(hostname -s)" 2>/dev/null | grep -oiE "Gres=[^ ]*" \n  || echo "GPU info unavailable on host (see 'gpu :' line in step 1)"
echo "=========================================="

# -----------------------------------------------------------------------------
# Step 0: correctness gate (CPU, seconds). Timings and accuracy mean nothing if
# the codebooks or the decoder are wrong.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 0: hierarchy unit tests"
${APPTAINER_RUN} python3 /workspace/tests/test_hierarchy.py
if [ $? -ne 0 ]; then
    echo "ERROR: hierarchy tests failed — aborting."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: CALIBRATION. Sweep payload width against token budget.
# GPT-2's 1024-token context means 960 is the practical ceiling.
# Read the printed decision rule at the end of this stage.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 1: calibration sweep (L x max_new_tokens)"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hierarchical_gpt2.py \
    --mode calibrate \
    --model ${MODEL} \
    --prompts-file /workspace/assets/prompts.txt \
    --users-file /workspace/assets/users.csv \
    --num-users ${NUM_USERS} \
    --num-prompts 20 \
    --l-bits-sweep 8,12,16 \
    --token-sweep 256,512,960 \
    --delta ${DELTA} \
    --entropy-threshold ${ENTROPY} \
    --hashing-context ${HC} \
    --z-threshold ${ZT} \
    --run-tag ${RUN_TAG} \
    --output-dir ${OUT_DIR}

# -----------------------------------------------------------------------------
# Step 2: FULL comparison at L=16, all schemes, same prompts and same user draw.
# If step 1 shows L=16 is not carryable, rerun this with --l-bits 12 and
# --schemes hier:l12_6_3_3,... instead.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 2: full comparison, L=16, 100 prompts"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hierarchical_gpt2.py \
    --mode full \
    --model ${MODEL} \
    --prompts-file /workspace/assets/prompts.txt \
    --users-file /workspace/assets/users.csv \
    --num-users ${NUM_USERS} \
    --num-prompts 100 \
    --l-bits 16 \
    --max-new-tokens 960 \
    --schemes hier:l16_8_4_4,hier:l16_8_8_optionC,hi_dypa_2layer,naive \
    --delta ${DELTA} \
    --entropy-threshold ${ENTROPY} \
    --hashing-context ${HC} \
    --z-threshold ${ZT} \
    --run-tag ${RUN_TAG} \
    --output-dir ${OUT_DIR}

# -----------------------------------------------------------------------------
# Step 3: L=12 fallback, run unconditionally so the comparison exists either way.
# If L=16 turns out marginal on GPT-2, this is the configuration to report.
# -----------------------------------------------------------------------------
echo ""
echo "### Step 3: L=12 fallback comparison"
${APPTAINER_RUN} python3 /workspace/evaluation_scripts/evaluate_hierarchical_gpt2.py \
    --mode full \
    --model ${MODEL} \
    --prompts-file /workspace/assets/prompts.txt \
    --users-file /workspace/assets/users.csv \
    --num-users ${NUM_USERS} \
    --num-prompts 100 \
    --l-bits 12 \
    --max-new-tokens 960 \
    --schemes hier:l12_6_3_3,naive \
    --delta ${DELTA} \
    --entropy-threshold ${ENTROPY} \
    --hashing-context ${HC} \
    --z-threshold ${ZT} \
    --run-tag ${RUN_TAG}_L12 \
    --output-dir ${OUT_DIR}

echo ""
echo "=========================================="
echo "Done. Results:"
ls -la ${CODE_DIR}/evaluation/hierarchical_gpt2/
echo "=========================================="
