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
    generate|submit|resume|generate4|submit4) ;;
    *)
        echo "usage: $0 {generate|submit|resume|generate4|submit4}" >&2
        echo "  generate4 YOLO 4 (configs/yolo4.yml): link connectivity from results/yolo" >&2
        echo "            and write jobs/yolo4.*.pbs (run on scdt)" >&2
        echo "  submit4   sbatch the eight YOLO 4 arms and their score job (run on sc)" >&2
        echo "  generate  write jobs/*.pbs (run on scdt)" >&2
        echo "  submit    sbatch them with dependencies (run on sc)" >&2
        echo "  resume    2026-09-12: cancel the stuck yolo.score, score the six finished" >&2
        echo "            arms now, rerun the two block-model arms with the fast refinement" >&2
        echo "            and the full score behind them (run on sc, after generate)" >&2
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
    # The six arms that finished on the first night, scored on their own (-V writes
    # scores_<arms>.csv, never scores.csv) so they can be read before the block-model arms.
    $M configs/yolo.yml -c $CPU -s score -V vmf_lloyd nopca_lloyd fisher_pca_lloyd vmf_ward vmf_ward100 vmf_lloyd100 -t 3 -m 16 -n 4 -o jobs/

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

resume() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    # The first-night score job is waiting on two arms that timed out (jobs 17394669 and
    # 17394674, ~2.9 h per matrix before the refinement was rewritten); its dependency can
    # never be satisfied. Cancel it, then score what is done and rerun what is not.
    for j in $(squeue -u "$USER" -h -o "%i %j %r" | awk '$2=="yolo.score" && $3 ~ /DependencyNeverSatisfied/ {print $1}'); do
        scancel "$j" && echo "cancelled stuck yolo.score $j"
    done
    partial=$(sbatch --parsable jobs/yolo.score.vmf_lloyd_nopca_lloyd_fisher_pca_lloyd_vmf_ward_vmf_ward100_vmf_lloyd100.pbs)
    echo "yolo partial score (six finished arms) -> $partial"
    # The rewritten refinement resumes where the timed-out jobs stopped: run_parcellation
    # skips parcellation files that already exist.
    b1=$(sbatch --parsable jobs/yolo.parcellation.blockmodel.pbs);    echo "yolo blockmodel -> $b1"
    b2=$(sbatch --parsable jobs/yolo.parcellation.blockmodel100.pbs); echo "yolo blockmodel100 -> $b2"
    score=$(sbatch --parsable --dependency=afterok:$b1:$b2 jobs/yolo.score.pbs)
    echo "yolo full score -> $score"
    echo
    squeue -u "$USER" -o "%.9i %.60j %.9T %.10M %R"
}

generate4() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    # Same connectivity as YOLO 1+2 (itself hard-linked from the ladder): prose domains,
    # samples + avg + halves, so split_halves skips and no GPU is used.
    for t in "" _null; do
        mkdir -p results/yolo4$t/connectivity
        for f in results/yolo$t/connectivity/connectivity_*.h5; do
            ln -f "$f" results/yolo4$t/connectivity/
        done
        echo "results/yolo4$t/connectivity: $(ls results/yolo4$t/connectivity | wc -l) files"
    done
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    # ICA: one eigendecomposition (~1-2 min) plus 40 FastICA restarts per matrix, 24 matrices.
    $M configs/yolo4.yml -c $CPU -s parcellation -V ica -t 6 -m 16 -n 8 -o jobs/
    $M configs/yolo4.yml -c $CPU -s parcellation -V ica100 -t 8 -m 16 -n 8 -o jobs/
    # PCA-200 arms: fisher_pca_lloyd took 24 min for all 24 matrices.
    for a in sparse_fisher_lloyd binarize_row_lloyd binarize_global_lloyd; do
        $M configs/yolo4.yml -c $CPU -s parcellation -V $a -t 2 -m 16 -n 8 -o jobs/
    done
    for a in sparse_fisher_lloyd100 binarize_row_lloyd100 binarize_global_lloyd100; do
        $M configs/yolo4.yml -c $CPU -s parcellation -V $a -t 3 -m 16 -n 8 -o jobs/
    done
    $M configs/yolo4.yml -c $CPU -s score -t 4 -m 16 -n 4 -o jobs/
    ls -1 jobs/yolo4.*.pbs
}

submit4() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local ids=""
    for a in ica ica100 sparse_fisher_lloyd binarize_row_lloyd binarize_global_lloyd \
             sparse_fisher_lloyd100 binarize_row_lloyd100 binarize_global_lloyd100; do
        id=$(sbatch --parsable jobs/yolo4.parcellation.$a.pbs)
        echo "yolo4 $a -> $id"
        ids="$ids:$id"
    done
    score=$(sbatch --parsable --dependency=afterok${ids} jobs/yolo4.score.pbs)
    echo "yolo4 score -> $score"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R"
}

case "$MODE" in
    generate)  generate ;;
    submit)    submit ;;
    resume)    resume ;;
    generate4) generate4 ;;
    submit4)   submit4 ;;
esac
