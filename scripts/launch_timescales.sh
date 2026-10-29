#!/bin/bash
#
# Intrinsic timescales (LOG.md Iteration 47, info/TIMESCALES_PLAN.md).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_timescales.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_timescales.sh submit qwen3.5-2b'
#
# acf (GPU, a6000):  windows of wikitext and bookcorpus, one token-and-position fit per condition
#                    (intact, shuffled), the ACF of all 147,456 units for each dataset and half.
#                    About 1.2M tokens of forward passes; the fit's types x units table and its
#                    copy on the GPU are smaller than in Iteration 43, which ran on the same card.
# analyze (CPU):     reliability, depth, networks, circuits; starts when acf ends (afterok).
#
# Disk: acf.h5 about 1.3 GB (8 ACFs of 147,456 x 257 float32) plus the windows; the non-streaming
# wikitext and bookcorpus downloads go to the shared HF cache.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
NAME=${2:-}

case "$MODE" in
    generate|submit) ;;
    *) echo "usage: $0 generate | submit qwen3.5-2b" >&2; exit 2 ;;
esac

cd "$WORK"

write_pbs() {
    # write_pbs <job name> <partition> <gres line or empty> <hours> <GB> <cores> <command...>
    local name=$1 part=$2 gres=$3 hours=$4 mem=$5 cores=$6
    shift 6
    cat > jobs/$name.pbs <<EOF
#!/bin/bash
#
#SBATCH --job-name=$name
#SBATCH --output=$WORK/logs/$name-%N-%j.out
#SBATCH --error=$WORK/logs/$name-%N-%j.err
#SBATCH --time=$hours:00:00
#SBATCH --mem=${mem}gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cores
#SBATCH --account=nlp
#SBATCH --partition=$part
$gres

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
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
git rev-parse HEAD

$*
EOF
}

generate() {
    mkdir -p jobs logs
    local m=qwen3.5-2b out=results/timescales/qwen3.5-2b
    write_pbs timescales-$m-acf jag-standard "#SBATCH --gres=gpu:a6000:1" 8 64 4 \
        python -m parcelmate.bin.timescales acf --model Qwen/Qwen3.5-2B --out $out
    write_pbs timescales-$m-analyze john "" 6 48 16 \
        python -m parcelmate.bin.timescales analyze --out $out --cores 16 \
            --networks results/qwen35/qwen3.5-2b-resid-pool4 \
            --patching results/patching_bank/Qwen_Qwen3-5-2B results/patching/Qwen_Qwen3-5-2B \
            --richclub results/richclub/qwen3.5-2b-resid-pool4-tiles/metrics/richclub
    echo "wrote jobs/timescales-$m-{acf,analyze}.pbs"
}

submit() {
    [ "$NAME" = qwen3.5-2b ] || { echo "only qwen3.5-2b is configured" >&2; exit 2; }
    local a
    a=$(sbatch --parsable jobs/timescales-$NAME-acf.pbs)
    sbatch --parsable --dependency=afterok:$a jobs/timescales-$NAME-analyze.pbs
    echo "acf $a"
}

$MODE
