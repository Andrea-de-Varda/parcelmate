#!/bin/bash
#
# Generate and submit the YOLO runs (LOG.md Iterations 14-15) and the final tests T1, T2
# and T5 (Iteration 17).
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
    generate|submit|resume|generate4|submit4|generate_final|submit_final|generate_last|submit_last|generate_confirm|submit_confirm) ;;
    *)
        echo "usage: $0 {generate|submit|resume|generate4|submit4|generate_final|submit_final|generate_last|submit_last|generate_confirm|submit_confirm}" >&2
        echo "  generate_confirm  the confirmation arm of configs/last_mlp.yml and its restricted" >&2
        echo "                    score job (run on scdt, after generate_last linked the connectivity)" >&2
        echo "  submit_confirm    sbatch the arm and its score (run on sc)" >&2
        echo "  generate_last   the last round (configs/last_mlp.yml, pooled_mlp.yml, the yardstick):" >&2
        echo "                  link connectivity from results/yolo_mlp, write jobs (run on scdt)" >&2
        echo "  submit_last     sbatch the pool job, 17 arms, two score jobs and the yardstick (run on sc)" >&2
        echo "  generate_final  final tests T1, T2, T5 (configs/final_mlp.yml, final_resid.yml):" >&2
        echo "                  link connectivity from results/yolo_mlp and results/yolo, write" >&2
        echo "                  jobs/final_*.pbs (run on scdt)" >&2
        echo "  submit_final    sbatch the 17 arms and their three score jobs (run on sc)" >&2
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

# Final tests T1, T2, T5 (LOG.md Iteration 17). The arm lists are grouped by cost, and a test
# checks that together they are exactly the arms of the two configs.
FINAL_MLP_WARD="vmf_ward vmf_ward100 vmf_ward150 vmf_ward200 vmf_sparse_ward100"
FINAL_MLP_WARD_PLUS="vmf_wardlloyd100 vmf_ward100_bm vmf_ward100_bm_double vmf_ward100_bm_degree"
FINAL_MLP_PCA="vmf_pca20_ward100 vmf_pca20_lloyd100 vmf_pca100_ward100 vmf_pca100_lloyd100"
FINAL_MLP_LLOYD="vmf_lloyd100 vmf_sparse_lloyd100"
FINAL_RESID="vmf_ward150 vmf_ward200"

generate_final() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    # Connectivity is already on disk: MLP units from YOLO 3, the residual stream from YOLO 1+2
    # (itself linked from the ladder). Prose domains, samples + avg + halves, both trees.
    local pair dst src t f
    for pair in final_mlp:yolo_mlp final_resid:yolo; do
        dst=${pair%%:*}
        src=${pair##*:}
        for t in "" _null; do
            mkdir -p results/$dst$t/connectivity
            for f in results/$src$t/connectivity/connectivity_*.h5; do
                ln -f "$f" results/$dst$t/connectivity/
            done
            echo "results/$dst$t/connectivity: $(ls results/$dst$t/connectivity | wc -l) files"
        done
    done
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    local a
    # Ward on 9,984 full profiles: scipy's linkage is single-threaded, and YOLO 1 took
    # 2 h 19 (k = 50) and 2 h 54 (k = 100) for 24 matrices. The tree does not depend on k.
    for a in $FINAL_MLP_WARD; do
        $M configs/final_mlp.yml -c $CPU -s parcellation -V $a -t 5 -m 16 -n 4 -o jobs/
    done
    # Ward plus one Lloyd run, or plus the block-model refinement (BLAS sweeps, minutes).
    for a in $FINAL_MLP_WARD_PLUS; do
        $M configs/final_mlp.yml -c $CPU -s parcellation -V $a -t 8 -m 16 -n 8 -o jobs/
    done
    # PCA-20/100 and then clustering in 20-100 dimensions: about a minute per matrix.
    for a in $FINAL_MLP_PCA; do
        $M configs/final_mlp.yml -c $CPU -s parcellation -V $a -t 2 -m 16 -n 8 -o jobs/
    done
    # 40 Lloyd restarts at k = 100 on full profiles: 6 h 07 on the residual stream, and MLP
    # Lloyd ran 1.45x slower than residual at k = 50 (4 h 08 against 2 h 51), so ~9 h.
    for a in $FINAL_MLP_LLOYD; do
        $M configs/final_mlp.yml -c $CPU -s parcellation -V $a -t 14 -m 16 -n 8 -o jobs/
    done
    # An early score of the 13 arms that do not wait on the two Lloyd arms (its name is
    # 13arms_<digest>, from `variants_tag`), then the full table.
    $M configs/final_mlp.yml -c $CPU -s score -V $FINAL_MLP_WARD $FINAL_MLP_WARD_PLUS $FINAL_MLP_PCA \
        -t 6 -m 16 -n 4 -o jobs/
    $M configs/final_mlp.yml -c $CPU -s score -t 6 -m 16 -n 4 -o jobs/
    for a in $FINAL_RESID; do
        $M configs/final_resid.yml -c $CPU -s parcellation -V $a -t 5 -m 16 -n 4 -o jobs/
    done
    $M configs/final_resid.yml -c $CPU -s score -t 2 -m 16 -n 4 -o jobs/
    ls -1 jobs/final_*.pbs
}

submit_final() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local a id early="" all="" resid="" early_pbs
    for a in $FINAL_MLP_WARD $FINAL_MLP_WARD_PLUS $FINAL_MLP_PCA; do
        id=$(sbatch --parsable jobs/final_mlp.parcellation.$a.pbs)
        echo "final_mlp $a -> $id"
        early="$early:$id"
    done
    all=$early
    for a in $FINAL_MLP_LLOYD; do
        id=$(sbatch --parsable jobs/final_mlp.parcellation.$a.pbs)
        echo "final_mlp $a -> $id"
        all="$all:$id"
    done
    early_pbs=$(ls jobs/final_mlp.score.13arms_*.pbs)
    if [ "$(echo "$early_pbs" | wc -l)" -ne 1 ]; then
        echo "expected one early score script, found: $early_pbs" >&2
        exit 1
    fi
    id=$(sbatch --parsable --dependency=afterok${early} "$early_pbs")
    echo "final_mlp early score (13 arms) -> $id"
    id=$(sbatch --parsable --dependency=afterok${all} jobs/final_mlp.score.pbs)
    echo "final_mlp full score -> $id"
    for a in $FINAL_RESID; do
        id=$(sbatch --parsable jobs/final_resid.parcellation.$a.pbs)
        echo "final_resid $a -> $id"
        resid="$resid:$id"
    done
    id=$(sbatch --parsable --dependency=afterok${resid} jobs/final_resid.score.pbs)
    echo "final_resid score -> $id"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R"
}

# The last round (LOG.md Iteration 19). Lists grouped by cost; a test checks they are exactly
# the arms of the two configs.
LAST_MLP_FAST="vmf_pca100_lloyd100 vmf_pca100_lloyd100_bm_raw vmf_pca100_lloyd100_bm_double vmf_pca100_lloyd100_bm_degree vmf_sparse_pca100_lloyd100 vmf_sparse_pca100_lloyd100_bm_degree vmf_pca100_lloyd50 vmf_pca100_lloyd50_bm_degree vmf_pca100_lloyd100_coassoc vmf_pca100_lloyd100_coassoc_bm_degree"
LAST_MLP_K200="vmf_pca100_lloyd200 vmf_pca100_lloyd200_bm_degree"
LAST_MLP_N200="vmf_pca100_lloyd100_n200 vmf_pca100_lloyd100_n200_coassoc"
LAST_MLP_CONFIRM="vmf_sparse_pca100_lloyd100_n200"
POOLED_MLP="vmf_pca100_lloyd100 vmf_pca100_lloyd100_bm_degree vmf_pca100_lloyd100_coassoc"
YARDSTICK_VARIANTS="vmf_lloyd100 vmf_pca100_lloyd100 vmf_ward100"

generate_last() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    local dst t f a
    for dst in last_mlp pooled_mlp; do
        for t in "" _null; do
            mkdir -p results/$dst$t/connectivity
            for f in results/yolo_mlp$t/connectivity/connectivity_*.h5; do
                ln -f "$f" results/$dst$t/connectivity/
            done
            echo "results/$dst$t/connectivity: $(ls results/$dst$t/connectivity | wc -l) files"
        done
    done
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    # Anchors: PCA-100 Lloyd, 40 restarts, 24 matrices took 19 min (job 17417207) at 3.7 GB;
    # the Ward polishes added ~15 min per arm for one refinement per matrix, and a consensus
    # polish refines three labelings per matrix. Co-association and average linkage take
    # seconds per matrix. 200 restarts are 5x the Lloyd time; k = 200 about 2x.
    for a in $LAST_MLP_FAST; do
        $M configs/last_mlp.yml -c $CPU -s parcellation -V $a -t 3 -m 16 -n 8 -o jobs/
    done
    for a in $LAST_MLP_K200; do
        $M configs/last_mlp.yml -c $CPU -s parcellation -V $a -t 4 -m 16 -n 8 -o jobs/
    done
    for a in $LAST_MLP_N200; do
        $M configs/last_mlp.yml -c $CPU -s parcellation -V $a -t 6 -m 16 -n 8 -o jobs/
    done
    # 14 arms, 12 pairs: the 15-arm final_mlp score took 2 h 28.
    $M configs/last_mlp.yml -c $CPU -s score -t 8 -m 16 -n 4 -o jobs/
    # Pools: 11 pools x 3 keys x 2 trees, each a Fisher mean of 2-4 matrices of 400 MB.
    $M configs/pooled_mlp.yml -c $CPU -s pool_domains -t 2 -m 16 -n 4 -o jobs/
    # 15 domains x 3 keys x 2 trees = 90 matrices per arm, 3.75x the single-domain count.
    for a in $POOLED_MLP; do
        $M configs/pooled_mlp.yml -c $CPU -s parcellation -V $a -t 8 -m 16 -n 8 -o jobs/
    done
    $M configs/pooled_mlp.yml -c $CPU -s score -t 8 -m 16 -n 4 -o jobs/

    # The yardstick is a plain script, so make_jobs cannot render it; same profile values.
    cat > jobs/final_mlp.yardstick.pbs <<EOF
#!/bin/bash
#
#SBATCH --job-name=final_mlp.yardstick
#SBATCH --output=$WORK/logs/final_mlp.yardstick-%N-%j.out
#SBATCH --error=$WORK/logs/final_mlp.yardstick-%N-%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

PYTHONPATH=. python analysis/yardstick.py --root results/final_mlp --variants $YARDSTICK_VARIANTS
EOF
    ls -1 jobs/last_mlp.*.pbs jobs/pooled_mlp.*.pbs jobs/final_mlp.yardstick.pbs
}

submit_last() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local a id ids="" pids="" pool
    for a in $LAST_MLP_FAST $LAST_MLP_K200 $LAST_MLP_N200; do
        id=$(sbatch --parsable jobs/last_mlp.parcellation.$a.pbs)
        echo "last_mlp $a -> $id"
        ids="$ids:$id"
    done
    id=$(sbatch --parsable --dependency=afterok${ids} jobs/last_mlp.score.pbs)
    echo "last_mlp score -> $id"
    pool=$(sbatch --parsable jobs/pooled_mlp.pool_domains.pbs)
    echo "pooled_mlp pool_domains -> $pool"
    for a in $POOLED_MLP; do
        id=$(sbatch --parsable --dependency=afterok:$pool jobs/pooled_mlp.parcellation.$a.pbs)
        echo "pooled_mlp $a -> $id"
        pids="$pids:$id"
    done
    id=$(sbatch --parsable --dependency=afterok${pids} jobs/pooled_mlp.score.pbs)
    echo "pooled_mlp score -> $id"
    id=$(sbatch --parsable jobs/final_mlp.yardstick.pbs)
    echo "final_mlp yardstick -> $id"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R"
}

# Confirmation of the chosen pipeline (LOG.md Iteration 21). The connectivity links already
# exist from generate_last; the score is restricted to the one arm, so it writes
# scores_<arm>.csv and leaves the 14-arm scores.csv untouched.
generate_confirm() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    local t
    for t in "" _null; do
        test "$(ls results/last_mlp$t/connectivity | wc -l)" -ge 28 || {
            echo "results/last_mlp$t/connectivity is incomplete; run generate_last first" >&2; exit 1; }
    done
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    # Anchors: 200 restarts took 56 min (17433372), sparse profiles added nothing (16 min
    # against 21 min at 40 restarts); the one-arm score is a fraction of the 14-arm 2 h 26.
    $M configs/last_mlp.yml -c $CPU -s parcellation -V $LAST_MLP_CONFIRM -t 3 -m 16 -n 8 -o jobs/
    $M configs/last_mlp.yml -c $CPU -s score -V $LAST_MLP_CONFIRM -t 3 -m 16 -n 4 -o jobs/
    ls -1 jobs/last_mlp.*.$LAST_MLP_CONFIRM.pbs
}

submit_confirm() {
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local p s
    p=$(sbatch --parsable jobs/last_mlp.parcellation.$LAST_MLP_CONFIRM.pbs)
    echo "last_mlp $LAST_MLP_CONFIRM -> $p"
    s=$(sbatch --parsable --dependency=afterok:$p jobs/last_mlp.score.$LAST_MLP_CONFIRM.pbs)
    echo "last_mlp score $LAST_MLP_CONFIRM -> $s"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R"
}

case "$MODE" in
    generate)  generate ;;
    generate_confirm) generate_confirm ;;
    submit_confirm)   submit_confirm ;;
    submit)    submit ;;
    resume)    resume ;;
    generate4) generate4 ;;
    submit4)   submit4 ;;
    generate_final) generate_final ;;
    submit_final)   submit_final ;;
    generate_last)  generate_last ;;
    submit_last)    submit_last ;;
esac
