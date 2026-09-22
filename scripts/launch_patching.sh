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

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}

case "$MODE" in
    generate|submit) ;;
    *) echo "usage: $0 generate | submit" >&2; exit 2 ;;
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

set -e

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

case "$MODE" in
    generate) generate ;;
    submit)   submit ;;
esac
