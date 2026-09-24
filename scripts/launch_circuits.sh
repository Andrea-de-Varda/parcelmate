#!/bin/bash
#
# Circuits against networks (LOG.md Iteration 30): one CPU job per Qwen3.5 model.
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_circuits.sh generate'
#     ssh sc   'bash .../scripts/launch_circuits.sh submit qwen3.5-2b <afterok job ids...>'
#
# Inputs: results/patching/<model>/<task domain>/<task>/neuron_attribution.npy and the
# stored partitions results/qwen35/<tree>{,_null}/final/parcellation/. The job reads the
# partitions only (not the connectivity), so it can run after the connectivity is purged.
# Sizing: at 4B, 16 partitions of 294,912 x 100 memberships (about 4 GB) and 1,000
# layer-matched draws per task, partition and circuit size (about half an hour).
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
TREE=${2:-}

cd "$WORK"

model_of() {
    case "$1" in
        qwen3.5-2b|qwen3.5-2b-pool5) echo Qwen_Qwen3-5-2B ;;
        qwen3.5-4b|qwen3.5-4b-pool5) echo Qwen_Qwen3-5-4B ;;
        *) echo "unknown tree $1" >&2; exit 2 ;;
    esac
}

generate() {
    mkdir -p jobs logs
    local t name
    for t in qwen3.5-2b qwen3.5-4b qwen3.5-2b-pool5 qwen3.5-4b-pool5; do
        name=circuits.$t
        cat > jobs/$name.pbs <<EOF
#!/bin/bash
#
#SBATCH --job-name=$name
#SBATCH --output=$WORK/logs/$name-%N-%j.out
#SBATCH --error=$WORK/logs/$name-%N-%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=nlp
#SBATCH --partition=john

set -e
umask 002
mkdir -p $WORK/logs
cd $WORK
source $CONDA_SH
conda activate $CONDA_ENV
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

python -m parcelmate.bin.compare_circuits --patching results/patching/$(model_of $t) --networks results/qwen35/$t
EOF
        echo jobs/$name.pbs
    done
}

submit() {
    local t=$1; shift
    local dep=""
    if [ $# -gt 0 ]; then dep="--dependency=afterok:$(echo "$@" | tr ' ' ':')"; fi
    echo "code at $(git log --oneline | head -1)"
    echo "circuits.$t -> $(sbatch --parsable $dep jobs/circuits.$t.pbs) ${dep}"
}

case "$MODE" in
    generate) generate ;;
    submit)   submit "$TREE" "${@:3}" ;;
    *) echo "usage: $0 generate | submit <tree> [afterok job ids...]" >&2; exit 2 ;;
esac
