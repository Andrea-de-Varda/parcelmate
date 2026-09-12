#!/bin/bash
#
# Generate and submit the Iteration 14 YOLO runs (LOG.md Iteration 14).
#
# `jobs/` is gitignored -- batch scripts are generated artifacts, not source -- so the
# generation step has to run on the cluster. Following the division in info/CLUSTER.md
# (scdt generates, sc submits):
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_yolo.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_yolo.sh submit'
#
# `generate` is idempotent and rewrites every script; `submit` only calls sbatch, so it can
# be re-run after a scancel without regenerating. Assumes results/yolo{,_null}/connectivity
# are hard-linked from results/ladder{,_null} (done 2026-09-11), so split_halves skips and
# YOLO 1+2 use no GPU time.
#
# Sizing follows the rule from the ladder run: pad the measured MaxRSS ~3-5x and keep the
# wall-time request close to the estimate, because padding is paid in queue time. The ladder
# parcellation peaked at 5.16 GB; these ask 16 GB. Wall times are per arm: k=50 MiniBatch
# took 5.5 h for six arms serially, so one Lloyd arm at 40 restarts is ~6 h, Ward far less,
# k=100 up to 10 h.
set -e

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}

case "$MODE" in
    generate|submit) ;;
    *)
        echo "usage: $0 {generate|submit}" >&2
        echo "  generate  write jobs/*.pbs (run on scdt)" >&2
        echo "  submit    sbatch them with dependencies (run on sc)" >&2
        exit 2
        ;;
esac

cd "$WORK"

generate() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    local GPU=configs/cluster/sc.yml

    # YOLO 1 + 2: one job per arm, so the arms run in parallel on `john`.
    for a in vmf_lloyd nopca_lloyd blockmodel; do
        $M configs/yolo.yml -c $CPU -s parcellation -V $a -t 6 -m 16 -n 8 -o jobs/
    done
    for a in fisher_pca_lloyd vmf_ward vmf_ward100; do
        $M configs/yolo.yml -c $CPU -s parcellation -V $a -t 3 -m 16 -n 8 -o jobs/
    done
    for a in vmf_lloyd100 blockmodel100; do
        $M configs/yolo.yml -c $CPU -s parcellation -V $a -t 10 -m 16 -n 8 -o jobs/
    done
    $M configs/yolo.yml -c $CPU -s score -t 4 -m 16 -n 4 -o jobs/

    # YOLO 3: GPU connectivity on MLP units, then two arms, then score.
    $M configs/yolo_mlp.yml -c $GPU -s connectivity split_halves -t 8 -m 32 -n 4 -o jobs/
    $M configs/yolo_mlp.yml -c $CPU -s parcellation -V vmf_profile -t 4 -m 16 -n 8 -o jobs/
    $M configs/yolo_mlp.yml -c $CPU -s parcellation -V vmf_lloyd -t 8 -m 16 -n 8 -o jobs/
    $M configs/yolo_mlp.yml -c $CPU -s score -t 3 -m 16 -n 4 -o jobs/

    # The chain diagnostic is a plain script, not a pipeline step, so make_jobs cannot
    # render it. Same profile values as sc-cpu.yml, one matrix in memory at a time.
    cat > jobs/ladder.residual_chain.pbs <<EOF
#!/bin/bash
#
#SBATCH --job-name=ladder.residual_chain
#SBATCH --output=$WORK/logs/ladder.residual_chain-%N-%j.out
#SBATCH --error=$WORK/logs/ladder.residual_chain-%N-%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=nlp
#SBATCH --partition=john

set -e

mkdir -p $WORK/logs
cd $WORK
source $CONDA_SH
conda activate $CONDA_ENV
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK

PYTHONPATH=. python analysis/residual_chain.py --root results/ladder
EOF
    echo
    ls -1 jobs/*.pbs
}

submit() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"

    local ids=""
    for a in vmf_lloyd nopca_lloyd blockmodel fisher_pca_lloyd vmf_ward vmf_ward100 vmf_lloyd100 blockmodel100; do
        id=$(sbatch --parsable jobs/yolo.parcellation.$a.pbs)
        echo "yolo $a -> $id"
        ids="$ids:$id"
    done
    score=$(sbatch --parsable --dependency=afterok${ids} jobs/yolo.score.pbs)
    echo "yolo score -> $score"

    c=$(sbatch --parsable jobs/yolo_mlp.connectivity_split_halves.pbs)
    echo "mlp connectivity -> $c"
    p1=$(sbatch --parsable --dependency=afterok:$c jobs/yolo_mlp.parcellation.vmf_profile.pbs)
    echo "mlp vmf_profile -> $p1"
    p2=$(sbatch --parsable --dependency=afterok:$c jobs/yolo_mlp.parcellation.vmf_lloyd.pbs)
    echo "mlp vmf_lloyd -> $p2"
    ms=$(sbatch --parsable --dependency=afterok:$p1:$p2 jobs/yolo_mlp.score.pbs)
    echo "mlp score -> $ms"

    rc=$(sbatch --parsable jobs/ladder.residual_chain.pbs)
    echo "residual_chain -> $rc"

    echo
    squeue -u "$USER" -o "%.9i %.40j %.9T %.10M %R"
}

case "$MODE" in
    generate) generate ;;
    submit)   submit ;;
esac
