#!/bin/bash
#
# Training-dynamics measures for the Pythia checkpoints (LOG.md Iteration 28).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_dynamics.sh generate 70m'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_dynamics.sh submit 70m'
#
# Per size: one GPU job per checkpoint (connectome recomputed without the null, activation
# measures and loss in the same forward passes, nothing N x N kept but a 4,096-unit
# subsample), one CPU job for the partition-only measures (it reads the stored
# parcellations and can start at once), and one CPU job that combines everything after the
# checkpoints. The GPU jobs are throttled: each waits (afterany) on the one MAX_GPU places
# before it, so at most MAX_GPU run at a time (Andrea: keep the concurrent request polite).
#
# Sizing from the Pythia connectivity anchors (LOG.md Iteration 24): 70m took 13-35 min per
# checkpoint with the null, at under 16 GB; 160m 1 h 14 to 2 h 49. Without the null
# correlation and with one extra (loss) forward pass, about the same. 70m: 2 h, 32 GB.
# 160m: 4 h, 96 GB (a sample's timecourses are 14.5 GB, the halves 5.4 GB each). Disk: the
# subsamples, 33 MB per half, about 3 GB per size.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
SIZE=${2:-}
MAX_GPU=${MAX_GPU:-4}
# Nodes to avoid (comma-separated). jagupard32 had a GPU held by a stale 43 GB process on
# 2026-09-23 that killed three jobs (LOG.md Iteration 28).
EXCLUDE=${EXCLUDE:-jagupard32}
STEPS="0 1 4 16 64 256 1000 4000 16000 64000 143000"

case "$MODE" in
    generate|submit) ;;
    *) echo "usage: $0 generate|submit {70m|160m}   (MAX_GPU=$MAX_GPU concurrent GPU jobs)" >&2; exit 2 ;;
esac
case "$SIZE" in
    70m)  GPU_T=2; GPU_M=32 ;;
    160m) GPU_T=4; GPU_M=64 ;;   # 70m peaked at 19 GB; 160m about 30 GB after the Iteration 28 memory fixes
    *) echo "size must be 70m or 160m" >&2; exit 2 ;;
esac
cd "$WORK"
OUT=results/pythia/pythia-$SIZE/dynamics

header() {
    # $1 name, $2 hours, $3 GB, $4 cpus, $5 partition, $6 gres line or empty
    cat <<EOF
#!/bin/bash
#
#SBATCH --job-name=$1
#SBATCH --output=$WORK/logs/$1-%N-%j.out
#SBATCH --error=$WORK/logs/$1-%N-%j.err
#SBATCH --time=$2:00:00
#SBATCH --mem=$3gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$4
#SBATCH --account=nlp
#SBATCH --partition=$5
$6

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

EOF
}

generate() {
    mkdir -p jobs logs
    local step name
    for step in $STEPS; do
        name=dynamics.pythia-$SIZE.step$step
        { header $name $GPU_T $GPU_M 4 jag-standard "#SBATCH --gres=gpu:a6000:1${EXCLUDE:+
#SBATCH --exclude=$EXCLUDE}"
          echo "python -m parcelmate.bin.dynamics checkpoint configs/pythia/pythia-${SIZE}_step${step}.yml --out $OUT"
        } > jobs/$name.pbs
    done
    name=dynamics.pythia-$SIZE.partitions
    { header $name 2 16 4 john ""
      echo "python -m parcelmate.bin.dynamics partitions --root results/pythia/pythia-$SIZE --out $OUT"
    } > jobs/$name.pbs
    name=dynamics.pythia-$SIZE.combine
    { header $name 1 16 2 john ""
      echo "python -m parcelmate.bin.dynamics combine --out $OUT"
    } > jobs/$name.pbs
    ls -1 jobs/dynamics.pythia-$SIZE.*.pbs
}

submit() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local step id ids="" queue=()
    for step in $STEPS; do
        dep=""
        if [ ${#queue[@]} -ge "$MAX_GPU" ]; then
            dep="--dependency=afterany:${queue[$(( ${#queue[@]} - MAX_GPU ))]}"
        fi
        id=$(sbatch --parsable $dep jobs/dynamics.pythia-$SIZE.step$step.pbs)
        echo "pythia-$SIZE step$step -> $id ${dep:+($dep)}"
        queue+=("$id")
        ids="$ids:$id"
    done
    id=$(sbatch --parsable jobs/dynamics.pythia-$SIZE.partitions.pbs)
    echo "pythia-$SIZE partitions -> $id"
    id=$(sbatch --parsable --dependency=afterok$ids jobs/dynamics.pythia-$SIZE.combine.pbs)
    echo "pythia-$SIZE combine -> $id"
}

case "$MODE" in
    generate) generate ;;
    submit)   submit ;;
esac
