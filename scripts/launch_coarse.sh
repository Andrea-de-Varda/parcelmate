#!/bin/bash
#
# Coarse networks (k = 10 and 20) on Qwen3.5-2B (LOG.md Iteration 35).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_coarse.sh generate'
#     ssh sc   'cd ... && bash scripts/launch_coarse.sh submit_rest <rest connectivity ids...>'
#     ssh sc   'cd ... && bash scripts/launch_coarse.sh submit qwen3.5-2b-coarse'   (and qwen3.5-2b-pool5-coarse)
#
# One parcellation job per (dataset, tree, half, variant), so k = 10 and k = 20 run in
# parallel and each stays inside the 8 h limit (a 2B half is about 3 h, mostly the PCA).
#
#   submit_rest   the -rest tree: its connectivity is being written by the Iteration 34 chain,
#                 so the coarse parcellations wait on those connectivity jobs, a coarse score
#                 (`-V k10 k20` -> metrics/scores_k10_k20.csv, never the k = 100 table) runs
#                 on the same tiles, and the already queued purges are made to wait for both
#                 (`scontrol update ... Dependency=`), so no tile is deleted early.
#   submit        a -coarse tree (wikitext + bookcorpus, or the pool): connectivity is
#                 recomputed into that tree, then the same chain as launch_qwen.sh.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
cd "$WORK"
VARIANTS="k10 k20"

domains_of() {
    awk '/^  domains:/{f=1; next} f && /^  - /{sub(/^  - /, ""); printf "%s ", $0; next} f{exit}' "configs/qwen35/$1.yml"
}
pool_of() {
    awk '/^  pool_as:/{print $2}' "configs/qwen35/$1.yml"
}
job_domains_of() {
    local p; p=$(pool_of "$1")
    if [ -n "$p" ]; then echo "$p"; else domains_of "$1"; fi
}

generate() {
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    mkdir -p jobs logs
    python scripts/make_qwen_configs.py > /dev/null
    local M="python -m parcelmate.bin.make_jobs" CPU=configs/cluster/sc-cpu.yml GPU=configs/cluster/sc.yml
    local name cfg d t k v
    for name in qwen3.5-2b-rest-coarse qwen3.5-2b-coarse qwen3.5-2b-pool5-coarse; do
        cfg=configs/qwen35/$name.yml
        if [ "$name" != qwen3.5-2b-rest-coarse ]; then
            if [ -n "$(pool_of $name)" ]; then
                $M $cfg -c $GPU -s connectivity -t 6 -m 160 -n 8 -o jobs/
            else
                for d in $(domains_of $name); do $M $cfg -c $GPU -s connectivity -t 6 -m 160 -n 8 -D $d -o jobs/; done
            fi
            for d in $(job_domains_of $name); do
                $M $cfg -c $CPU -s purge_null_connectivity -t 1 -m 4 -n 1 -D $d -o jobs/
            done
            $M $cfg -c $CPU -s score purge_connectivity -t 24 -m 32 -n 4 -o jobs/
        else
            $M $cfg -c $CPU -s score -V $VARIANTS -t 24 -m 32 -n 4 -o jobs/
        fi
        for d in $(job_domains_of $name); do
            for t in real null; do for k in halfA halfB; do for v in $VARIANTS; do
                $M $cfg -c $CPU -s parcellation -t 8 -m 32 -n 8 -V $v -D $d -T $t -K $k -o jobs/
            done; done; done
        done
    done
    ls -1 jobs/qwen3.5-2b*-coarse.*.pbs | wc -l
}

submit_rest() {
    # args: the -rest connectivity job ids, in the order of the -rest datasets
    local name=qwen3.5-2b-rest-coarse
    local conns=("$@") ds=($(domains_of $name)) i d t k v p ids="" nulls
    [ ${#conns[@]} = ${#ds[@]} ] || { echo "need one connectivity id per dataset (${ds[*]})" >&2; exit 2; }
    echo "code at $(git log --oneline | head -1)"
    for i in "${!ds[@]}"; do
        d=${ds[$i]}; nulls=""
        for t in real null; do for k in halfA halfB; do for v in $VARIANTS; do
            p=$(sbatch --parsable --dependency=afterok:${conns[$i]} jobs/$name.parcellation.$v.$d.$t.$k.pbs)
            echo "$name $d $t $k $v -> $p"
            ids="$ids:$p"
            [ "$t" = null ] && nulls="$nulls:$p"
        done; done; done
        echo "$d null coarse:$nulls" >> logs/coarse_rest_null_ids.txt
    done
    p=$(sbatch --parsable --dependency=afterok${ids} jobs/$name.score.k10_k20.pbs)
    echo "$name score -V k10 k20 -> $p"
    echo "parcellations$ids score $p"
}

submit() {
    local name=$1 d t k v c p ids="" nulls q
    echo "code at $(git log --oneline | head -1)"
    for d in $(job_domains_of $name); do
        if [ -n "$(pool_of $name)" ]; then c=$(sbatch --parsable jobs/$name.connectivity.pbs)
        else c=$(sbatch --parsable jobs/$name.connectivity.$d.pbs); fi
        nulls=""
        for t in real null; do for k in halfA halfB; do for v in $VARIANTS; do
            p=$(sbatch --parsable --dependency=afterok:$c jobs/$name.parcellation.$v.$d.$t.$k.pbs)
            echo "$name $d $t $k $v: connectivity $c -> $p"
            if [ "$t" = null ]; then nulls="$nulls:$p"; else ids="$ids:$p"; fi
        done; done; done
        q=$(sbatch --parsable --dependency=afterok$nulls jobs/$name.purge_null_connectivity.$d.pbs)
        echo "$name $d: purge null -> $q"
        ids="$ids:$q"
    done
    p=$(sbatch --parsable --dependency=afterok$ids jobs/$name.score_purge_connectivity.pbs)
    echo "$name score + purge -> $p"
}

case "$MODE" in
    generate) generate ;;
    submit_rest) shift; submit_rest "$@" ;;
    submit) submit "$2" ;;
    *) echo "usage: $0 generate | submit_rest <conn ids> | submit <coarse config>" >&2; exit 2 ;;
esac
