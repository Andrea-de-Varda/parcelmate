#!/bin/bash
#
# Rich-club stage (LOG.md Iteration 45, info/RICH_CLUB_PLAN.md): the pooled residual connectome
# recomputed and kept, then the rich-club analysis on the real and circular-shift trees.
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_richclub.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_richclub.sh submit pythia-70m'
#
# pythia-70m (the pilot):  connectivity (GPU) -> 4 parcellations -> score -> richclub
# qwen3.5-2b, qwen3.5-4b:  connectivity (GPU) and a copy of the stored Iteration 43 partitions
#                          -> score (re-scores them on the new tiles) -> richclub, which first
#                          runs scripts/check_tiles_reproduce.py and stops if the scores differ.
#
# <model>-docs (Iteration 48): the same connectomes on document-coherent text, in the trees
# results/richclub/<model>-resid-pool4-docs. Every model runs the full pipeline (connectivity ->
# 4 parcellations -> score -> richclub -> richclub_topk); for Qwen the per-task circuit
# concentration (bank_concentration) runs on the new partitions as soon as they exist.
#
#     ssh sc 'bash .../scripts/launch_richclub.sh submit qwen3.5-2b-docs'
#
# Nothing is purged. Disk on juice6: 1.2 GB (pythia-70m), 172 GB (2B), 696 GB (4B).
# Sizing: connectivity as Iteration 43 (2B 130 GB peak measured, 4B 320 GB on sphinx, on an 80 GB
# A100: the residual fit ran out of memory on a 40 GB card, 18182486); score as
# Iteration 43; richclub streams each half three times per tree (12 passes), row blocks of
# 2,048 rows (1.2 GB at 2B, 2.4 GB at 4B) plus per-unit network strengths (under 1 GB).
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
NAME=${2:-}

case "$MODE" in
    generate|submit|submit-topk) ;;
    *) echo "usage: $0 generate | submit {pythia-70m|qwen3.5-2b|qwen3.5-4b}[-docs] | submit-topk {...}" >&2; exit 2 ;;
esac

cd "$WORK"

# <model> -> the -tiles tree (Iteration 45); <model>-docs -> the -docs tree (Iteration 48).
job_of() { case "$1" in *-docs) echo "${1%-docs}-resid-pool4-docs" ;; *) echo "$1-resid-pool4-tiles" ;; esac; }
config_of() { echo "configs/richclub/$(job_of $1).yml"; }
# Full pipeline in the tree (pythia-70m and every -docs tree) or partitions copied (Qwen -tiles).
full_pipeline() { case "$1" in pythia-70m|*-docs) return 0 ;; *) return 1 ;; esac; }
# Attribution roots of a Qwen model (bank_concentration); empty for Pythia.
patching_of() {
    case "${1%-docs}" in
        qwen3.5-2b) echo "results/patching_bank/Qwen_Qwen3-5-2B results/patching/Qwen_Qwen3-5-2B" ;;
        qwen3.5-4b) echo "results/patching_bank/Qwen_Qwen3-5-4B results/patching/Qwen_Qwen3-5-4B" ;;
    esac
}

write_pbs() {
    # write_pbs <job name> <hours> <GB> <cores> <command...>
    local name=$1 hours=$2 mem=$3 cores=$4
    shift 4
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

$*
EOF
}

generate() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    python scripts/make_richclub_configs.py > /dev/null
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    local GPU=configs/cluster/sc.yml
    local m cfg job src
    for m in pythia-70m qwen3.5-2b qwen3.5-4b pythia-70m-docs qwen3.5-2b-docs qwen3.5-4b-docs; do
        cfg=$(config_of $m); job=$(job_of $m)
        case "${m%-docs}" in
            pythia-70m) GPU_OPTS="-t 3 -m 48"; SCO="-t 4 -m 16"; RC=(4 16 4); TK=(2 16 4 1024); PAR="-t 4 -m 16" ;;
            qwen3.5-2b) GPU_OPTS="-t 10 -m 180"; SCO="-t 24 -m 32"; RC=(24 32 8); TK=(16 32 8 1024); PAR="-t 8 -m 32" ;;
            qwen3.5-4b) GPU_OPTS="-t 16 -m 320 -P sphinx -G a100 -C 80G"; SCO="-t 36 -m 48"; RC=(48 48 8); TK=(40 48 8 512); PAR="-t 24 -m 48" ;;
        esac
        # Fixed-k participation (2026-10-22): reads the kept tiles and the richclub outputs, so it
        # runs on its own after `submit`. Four passes (2 trees x 2 halves), each with a per-row
        # partial sort; sized from the richclub job (12 passes, 14 h at 4B).
        write_pbs $job.richclub_topk ${TK[0]} ${TK[1]} ${TK[2]} \
            "python -m parcelmate.bin.richclub_topk $cfg --block ${TK[3]}"
        $M $cfg -c $GPU -s connectivity $GPU_OPTS -n 8 -o jobs/
        $M $cfg -c $CPU -s score $SCO -n 4 -o jobs/
        if full_pipeline $m; then
            for t in real null; do
                for k in halfA halfB; do
                    $M $cfg -c $CPU -s parcellation $PAR -n 8 -D pooled -T $t -K $k -o jobs/
                done
            done
            write_pbs $job.richclub ${RC[0]} ${RC[1]} ${RC[2]} \
                "python -m parcelmate.bin.richclub $cfg"
            # Circuit concentration (Iteration 44) on the new partitions; reads partitions only.
            # 2B ran locally in 8 min, 4B took 59 min on 4 CPUs (18030767).
            if [ -n "$(patching_of $m)" ]; then
                local size=${m#qwen3.5-}   # figures/bank_concentration_4b_resid_pool4_docs.csv
                write_pbs $job.bankconc 3 16 4 \
                    "python -m parcelmate.bin.bank_concentration --patching $(patching_of $m) --networks results/richclub/$job --out figures/bank_concentration_${size%-docs}_resid_pool4_docs.csv"
            fi
        else
            src=results/qwen35/$m-resid-pool4
            write_pbs $job.copy_partitions 1 4 1 \
                "mkdir -p results/richclub/$job results/richclub/${job}_null
cp -a $src/final results/richclub/$job/
cp -a ${src}_null/final results/richclub/${job}_null/
echo copied partitions from $src"
            write_pbs $job.richclub ${RC[0]} ${RC[1]} ${RC[2]} \
                "python scripts/check_tiles_reproduce.py $src/metrics/scores.csv results/richclub/$job/metrics/scores.csv
python -m parcelmate.bin.richclub $cfg"
        fi
    done
    ls -1 jobs/*-resid-pool4-{tiles,docs}*.pbs
}

submit() {
    local m=$1
    test -n "$m" || { echo "submit needs a model name" >&2; exit 2; }
    local job; job=$(job_of $m)
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    local c p ids="" s r
    c=$(sbatch --parsable jobs/$job.connectivity.pbs)
    echo "$job connectivity -> $c"
    if full_pipeline $m; then
        for t in real null; do
            for k in halfA halfB; do
                p=$(sbatch --parsable --dependency=afterok:$c jobs/$job.parcellation.pooled.$t.$k.pbs)
                echo "$job parcellation $t $k -> $p"
                ids="$ids:$p"
            done
        done
    else
        p=$(sbatch --parsable jobs/$job.copy_partitions.pbs)
        echo "$job copy partitions -> $p"
        ids=":$c:$p"
    fi
    s=$(sbatch --parsable --dependency=afterok${ids} jobs/$job.score.pbs)
    echo "$job score -> $s"
    r=$(sbatch --parsable --dependency=afterok:$s jobs/$job.richclub.pbs)
    echo "$job richclub -> $r"
    case "$m" in
        *-docs)
            # The -docs trees chain the fixed-k participation (it reads the richclub outputs)
            # and, for Qwen, the circuit concentration (it needs only the partitions).
            echo "$job richclub_topk -> $(sbatch --parsable --dependency=afterok:$r jobs/$job.richclub_topk.pbs)"
            if [ -f jobs/$job.bankconc.pbs ]; then
                echo "$job bankconc -> $(sbatch --parsable --dependency=afterok${ids} jobs/$job.bankconc.pbs)"
            fi
            ;;
    esac
}

submit_topk() {
    local m=$1
    test -n "$m" || { echo "submit-topk needs a model name" >&2; exit 2; }
    echo "code at $(git log --oneline | head -1)"
    echo "$(job_of $m) richclub_topk -> $(sbatch --parsable jobs/$(job_of $m).richclub_topk.pbs)"
}

case "$MODE" in
    generate) generate ;;
    submit) submit "$NAME" ;;
    submit-topk) submit_topk "$NAME" ;;
esac
