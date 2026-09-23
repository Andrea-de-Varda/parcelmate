#!/bin/bash
#
# Minimal-pair task screen for a model (LOG.md Iteration 23): every task, both-correct
# accuracy, and neuron attribution for the tasks that pass the inclusion rule.
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_patching.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_patching.sh submit'
#
# Two GPU jobs: GPT-2 (the validation against the original repository's GPT-2 baselines,
# BOS off as in the original) and LFM2.5-350M (BOS both ways on the raw-text tasks). About
# 50,000 items x 4 short forward passes each; well under an hour per model on an a6000.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
shift || true

# Model screen (LOG.md Iteration 23, second round): accuracy on all 46 tasks, no
# attribution, one GPU job per model. Larger models get more time; all run in float32.
# Attribution runs (LOG.md Iteration 29): the models whose networks are compared with circuits.
ATTR_MODELS=${ATTR_MODELS:-"Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"}
# jagupard32 has a GPU held by a foreign process (LOG.md Iteration 28).
EXCLUDE=${EXCLUDE:-jagupard32}
SCREEN_MODELS=${SCREEN_MODELS:-"Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B LiquidAI/LFM2.5-1.2B-Instruct ibm-granite/granite-4.2-3b google/gemma-4-e4b-it"}

case "$MODE" in
    generate|submit|generate_screen|submit_screen|generate_attribution|submit_attribution) ;;
    *) echo "usage: $0 generate | submit | generate_screen | submit_screen | generate_attribution | submit_attribution  (SCREEN_MODELS=, ATTR_MODELS= to override)" >&2; exit 2 ;;
esac

cd "$WORK"

job() {
    # $1 name, $2 hours, $3 GB, $4.. command
    local name=$1 hours=$2 mem=$3; shift 3
    cat > jobs/$name.pbs <<EOF
#!/bin/bash
#
#SBATCH --job-name=$name
#SBATCH --output=$WORK/logs/$name-%N-%j.out
#SBATCH --error=$WORK/logs/$name-%N-%j.err
#SBATCH --time=$hours:00:00
#SBATCH --mem=${mem}gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=$EXCLUDE

set -e
umask 002

mkdir -p $WORK/logs
cd $WORK
source $CONDA_SH
conda activate $CONDA_ENV
export HF_HOME=/juice6/u/nlp/climblab/devarda/.hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

$@
EOF
    echo jobs/$name.pbs
}

generate() {
    mkdir -p jobs logs
    job patching.gpt2 3 32 python -m parcelmate.bin.patch_eval --model openai-community/gpt2 --bos none --attribution --out results/patching
    job patching.lfm2_5-350m 4 32 python -m parcelmate.bin.patch_eval --model LiquidAI/LFM2.5-350M --bos both --attribution --out results/patching
}

submit() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local id
    for n in patching.gpt2 patching.lfm2_5-350m; do
        id=$(sbatch --parsable jobs/$n.pbs)
        echo "$n -> $id"
    done
    squeue -u "$USER" -o "%.9i %.40j %.9T %.10M %R" | grep patching
}

screen_name() { echo "screen.$(echo "$1" | tr '/' '_' | tr '.' '-')"; }

generate_screen() {
    mkdir -p jobs logs
    local m
    local dtype
    for m in $SCREEN_MODELS; do
        # Gemma 4 E4B has about 8B raw parameters (per-layer embeddings), which do not fit
        # an a6000 in float32; everything else runs in the protocol's float32.
        case "$m" in google/gemma-4*) dtype=bfloat16 ;; *) dtype=float32 ;; esac
        job "$(screen_name $m)" 6 48 python -m parcelmate.bin.patch_eval --model $m --bos both --dtype $dtype --out results/patching
    done
}

submit_screen() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local m id
    for m in $SCREEN_MODELS; do
        id=$(sbatch --parsable jobs/$(screen_name $m).pbs)
        echo "$m -> $id"
    done
    squeue -u "$USER" -o "%.9i %.40j %.9T %.10M %R" | grep screen
}

attr_name() { echo "attr.$(echo "$1" | tr '/' '_' | tr '.' '-')"; }

generate_attribution() {
    # --bos none: Qwen has no BOS token, so the BOS variants of the raw-text tasks are the
    # same prompts. The accuracies are cached from the screen, so only the attribution runs:
    # three passes (two without gradient) per batch of both-correct items, in float32.
    # Batch 8 for evaluation, 4 for attribution, for the backward pass at 4B.
    mkdir -p jobs logs
    local m hours
    for m in $ATTR_MODELS; do
        case "$m" in *4B*) hours=8 ;; *) hours=5 ;; esac
        job "$(attr_name $m)" $hours 64 python -m parcelmate.bin.patch_eval --model $m --bos none --attribution --batch-size 8 --out results/patching
    done
}

submit_attribution() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local m id
    for m in $ATTR_MODELS; do
        id=$(sbatch --parsable jobs/$(attr_name $m).pbs)
        echo "$m attribution -> $id"
    done
}

case "$MODE" in
    generate) generate ;;
    generate_attribution) generate_attribution ;;
    submit_attribution)   submit_attribution ;;
    submit)   submit ;;
    generate_screen) generate_screen ;;
    submit_screen)   submit_screen ;;
esac
