#!/bin/bash
#
# Word-shuffled Pythia connectomes (LOG.md Iteration 32).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_shuffle.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_shuffle.sh submit'
#
# One GPU job per (size, checkpoint): the connectivity on word-shuffled text (no null), then
# the split-half and across-domain connectome correlations, then the connectivity is
# deleted; only metrics/connectome_consistency.csv stays. Throttled: each job waits
# (afterany) on the one MAX_GPU places before it. Sizing from the Pythia connectivity
# anchors (with the null): 70m 13-35 min, 160m 1 h 14 to 2 h 49; halves 0.6 GB (70m) and
# 5.4 GB (160m) each, 8 per checkpoint, deleted at the end.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
MAX_GPU=${MAX_GPU:-3}
EXCLUDE=${EXCLUDE:-jagupard32}
SIZES=${SIZES:-"70m 160m"}
STEPS="0 1 4 16 64 256 1000 4000 16000 64000 143000"

cd "$WORK"

generate() {
    mkdir -p jobs logs
    local size step name T M
    for size in $SIZES; do
        case "$size" in 70m) T=2; M=32 ;; 160m) T=5; M=64 ;; esac
        for step in $STEPS; do
            name=shuffle.pythia-$size.step$step
            cat > jobs/$name.pbs <<PBS
#!/bin/bash
#
#SBATCH --job-name=$name
#SBATCH --output=$WORK/logs/$name-%N-%j.out
#SBATCH --error=$WORK/logs/$name-%N-%j.err
#SBATCH --time=$T:00:00
#SBATCH --mem=${M}gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=$EXCLUDE

set -e
umask 002
cd $WORK
export TMPDIR=$WORK/tmp/\$SLURM_JOB_ID
mkdir -p \$TMPDIR
trap 'rm -rf \$TMPDIR' EXIT
source $CONDA_SH
conda activate $CONDA_ENV
export HF_HOME=/juice6/u/nlp/climblab/devarda/.hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

cfg=configs/pythia_shuffled/pythia-${size}_step${step}.yml
python -m parcelmate.bin.main \$cfg -s connectivity
python -m parcelmate.bin.connectome_consistency \$cfg --purge
PBS
            echo jobs/$name.pbs
        done
    done
}

submit() {
    echo "code at $(git log --oneline | head -1)"
    local size step id dep queue=()
    for size in $SIZES; do
        for step in $STEPS; do
            dep=""
            if [ ${#queue[@]} -ge "$MAX_GPU" ]; then
                dep="--dependency=afterany:${queue[$(( ${#queue[@]} - MAX_GPU ))]}"
            fi
            id=$(sbatch --parsable $dep jobs/shuffle.pythia-$size.step$step.pbs)
            echo "pythia-$size step$step -> $id ${dep:+($dep)}"
            queue+=("$id")
        done
    done
}

case "$MODE" in
    generate) generate ;;
    submit)   submit ;;
    *) echo "usage: $0 generate|submit   (SIZES='$SIZES', MAX_GPU=$MAX_GPU)" >&2; exit 2 ;;
esac
