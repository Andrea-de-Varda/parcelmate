#!/bin/bash
#
# Generate and submit the Pythia training-dynamics runs (LOG.md Iteration 22).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_pythia.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_pythia.sh submit 70m'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_pythia.sh submit 160m'
#
# One chain per (size, checkpoint): a GPU connectivity job (writes the split halves of both
# trees directly), a CPU parcellation job, then a CPU job that scores and purges the
# connectivity. After every chain of a size, one job computes the cross-checkpoint
# agreement and the long score table for that size. `generate` is idempotent; `submit`
# only calls sbatch and can be re-run per size.
#
# Sizing (info/CLUSTER.md rule: pad measured MaxRSS 3-5x, keep wall time near the
# estimate). Anchors: GPT-2 MLP connectivity at 9,984 units took 19 min on an a6000 with
# 15.8 GB peak (17394676). At 36,864 units the timecourse array is 14.5 GB, its shifted copy
# another 14.5, a 5.4 GB matrix per tree and one half's running sums per tree: about 45 GB
# peak, so 64 GB. Parcellation at 36,864 units holds a few copies of the 5.4 GB matrix
# during sparsification and PCA: 48 GB. The scorer streams the fidelity metrics but holds
# two |r| halves and the co-association counts: 64 GB. 70m (12,288 units) is a tenth of
# all of that.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
SIZE=${2:-}

STEPS="0 1 4 16 64 256 1000 4000 16000 64000 143000"
SIZES="70m 160m"

case "$MODE" in
    generate|submit|resubmit_cpu) ;;
    *)
        echo "usage: $0 generate | submit {70m|160m} | resubmit_cpu {70m|160m}" >&2
        echo "  resubmit_cpu  after a failed parcellation stage: cancel that size's orphaned" >&2
        echo "                score and checkpoint jobs, then submit parcellation -> score chains" >&2
        echo "                (connectivity already on disk) and the checkpoints job" >&2
        exit 2
        ;;
esac

cd "$WORK"

# Per-size resources: GPU job (hours, GB), parcellation (hours, GB), score (hours, GB).
resources() {
    case "$1" in
        70m)  GPU_T=2; GPU_M=32; PAR_T=2; PAR_M=16; SCO_T=2; SCO_M=16 ;;
        160m) GPU_T=3; GPU_M=64; PAR_T=5; PAR_M=48; SCO_T=6; SCO_M=64 ;;
        *) echo "unknown size $1" >&2; exit 1 ;;
    esac
}

generate() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    python scripts/make_pythia_configs.py > /dev/null
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    local GPU=configs/cluster/sc.yml
    local size step cfg
    for size in $SIZES; do
        resources $size
        for step in $STEPS; do
            cfg=configs/pythia/pythia-${size}_step${step}.yml
            $M $cfg -c $GPU -s connectivity -t $GPU_T -m $GPU_M -n 4 -o jobs/
            $M $cfg -c $CPU -s parcellation -t $PAR_T -m $PAR_M -n 8 -o jobs/
            $M $cfg -c $CPU -s score purge_connectivity -t $SCO_T -m $SCO_M -n 4 -o jobs/
        done
        # The cross-checkpoint job is a plain script, so make_jobs cannot render it.
        cat > jobs/pythia-${size}.checkpoints.pbs <<EOF
#!/bin/bash
#
#SBATCH --job-name=pythia-${size}.checkpoints
#SBATCH --output=$WORK/logs/pythia-${size}.checkpoints-%N-%j.out
#SBATCH --error=$WORK/logs/pythia-${size}.checkpoints-%N-%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=nlp
#SBATCH --partition=john

set -e
umask 002

mkdir -p $WORK/logs
cd $WORK
source $CONDA_SH
conda activate $CONDA_ENV
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK

python -m parcelmate.bin.score_checkpoints --root results/pythia/pythia-${size} --steps $STEPS
EOF
    done
    echo
    ls -1 jobs/pythia-*.pbs | wc -l
    ls -1 jobs/pythia-*.pbs
}

submit() {
    local size=$1
    test -n "$size" || { echo "submit needs a size: 70m or 160m" >&2; exit 2; }
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local step c p s ids=""
    for step in $STEPS; do
        c=$(sbatch --parsable jobs/pythia-${size}_step${step}.connectivity.pbs)
        p=$(sbatch --parsable --dependency=afterok:$c jobs/pythia-${size}_step${step}.parcellation.pbs)
        s=$(sbatch --parsable --dependency=afterok:$p jobs/pythia-${size}_step${step}.score_purge_connectivity.pbs)
        echo "pythia-${size} step${step}: connectivity $c -> parcellation $p -> score $s"
        ids="$ids:$s"
    done
    c=$(sbatch --parsable --dependency=afterok${ids} jobs/pythia-${size}.checkpoints.pbs)
    echo "pythia-${size} checkpoints -> $c"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R"
}

resubmit_cpu() {
    local size=$1
    test -n "$size" || { echo "resubmit_cpu needs a size: 70m or 160m" >&2; exit 2; }
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local orphans
    orphans=$(squeue -u "$USER" -h -o "%i %j" | grep -E "pythia-${size}(_step[0-9]+\.score_purge_connectivity|\.checkpoints)$" | awk '{print $1}')
    if [ -n "$orphans" ]; then
        echo "cancelling orphaned jobs: $(echo $orphans | tr '\n' ' ')"
        scancel $orphans
    fi
    local step p s ids=""
    for step in $STEPS; do
        for t in "" _null; do
            test "$(ls results/pythia/pythia-${size}/step${step}${t}/connectivity/*.h5 2>/dev/null | wc -l)" -ge 8 || {
                echo "step${step}${t}: connectivity incomplete; not resubmitting" >&2; exit 1; }
        done
        p=$(sbatch --parsable jobs/pythia-${size}_step${step}.parcellation.pbs)
        s=$(sbatch --parsable --dependency=afterok:$p jobs/pythia-${size}_step${step}.score_purge_connectivity.pbs)
        echo "pythia-${size} step${step}: parcellation $p -> score $s"
        ids="$ids:$s"
    done
    p=$(sbatch --parsable --dependency=afterok${ids} jobs/pythia-${size}.checkpoints.pbs)
    echo "pythia-${size} checkpoints -> $p"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R"
}

case "$MODE" in
    generate) generate ;;
    submit)   submit "$SIZE" ;;
    resubmit_cpu) resubmit_cpu "$SIZE" ;;
esac
