#!/bin/bash
#
# Network-to-network coupling matrices on the stored rich-club tiles (LOG.md Iteration 46).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_coupling.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_coupling.sh submit qwen3.5-2b'
#
# One CPU job per model: `parcelmate.bin.network_coupling` streams each half of the real and
# the null tree once (four reads of 43 GB at 2B, 174 GB at 4B), with both partitions at once.
# Memory: a 2,048-row float64 block (2.4 GB at 2B, 4.8 GB at 4B) and two one-hot matrices.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
NAME=${2:-}
cd "$WORK"

generate() {
    mkdir -p jobs logs
    local m hours mem
    for m in qwen3.5-2b qwen3.5-4b; do
        case "$m" in *4b) hours=24; mem=32 ;; *) hours=8; mem=16 ;; esac
        cat > jobs/coupling.$m.pbs <<PBS
#!/bin/bash
#
#SBATCH --job-name=coupling.$m
#SBATCH --output=$WORK/logs/coupling.$m-%N-%j.out
#SBATCH --error=$WORK/logs/coupling.$m-%N-%j.err
#SBATCH --time=$hours:00:00
#SBATCH --mem=${mem}gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=nlp
#SBATCH --partition=john

set -e
umask 002
cd $WORK
source $CONDA_SH
conda activate $CONDA_ENV
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
python -m parcelmate.bin.network_coupling --networks results/richclub/$m-resid-pool4-tiles
PBS
        echo jobs/coupling.$m.pbs
    done
}

submit() {
    test -n "$NAME" || { echo "submit needs a model (qwen3.5-2b or qwen3.5-4b)" >&2; exit 2; }
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    echo "coupling.$NAME -> $(sbatch --parsable jobs/coupling.$NAME.pbs)"
}

case "$MODE" in
    generate) generate ;;
    submit) submit ;;
    *) echo "usage: $0 generate | submit {qwen3.5-2b|qwen3.5-4b}" >&2; exit 2 ;;
esac
