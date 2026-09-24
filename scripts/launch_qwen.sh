#!/bin/bash
#
# Out-of-core runs on Qwen3.5 (LOG.md Iteration 25).
#
#     ssh scdt 'cd /juice6/u/nlp/climblab/devarda/parcelmate && git pull && bash scripts/launch_qwen.sh generate'
#     ssh sc   'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_qwen.sh submit qwen3.5-2b'
#
# Per domain, a chain of three jobs: GPU connectivity (tiled fp16 halves, both trees), CPU
# parcellation of that domain's four halves, then purge of that domain's null tiles. The
# domain chains run in parallel; one score-and-purge job waits on all of them. Disk on
# juice6 (lab share): the real halves of every domain stay until scoring, the null halves
# of a domain go as soon as its null parcellation exists. Peak for 4B about 1.75 TB, 2.8 TB
# written in total; 2B on two domains 350 GB. All files group-writable (umask 002).
#
# Sizing. GPU job: the model in float32 (16 GB at 4B) plus two samples' z-scored
# timecourses in host RAM (2 x N x T x 2 bytes: 116 GB at 4B, 58 GB at 2B), GPU tiles
# under 20 GB. Partition sphinx (a100 80 GB, 1 TB host) for 4B; jag a6000 nodes (510 GB
# host) suffice for 2B. Parcellation streams the half four times for the PCA (175 GB per
# pass at 4B) and holds N x 200 float64 (0.5 GB): 32 GB, disk-bound, several hours.
# Scoring streams every half a few times: 32 GB, several hours.
set -e
umask 002

WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
MODE=${1:-}
NAME=${2:-}

case "$MODE" in
    generate|submit) ;;
    *) echo "usage: $0 generate | submit {qwen3.5-2b|qwen3.5-4b|qwen3.5-2b-pool5|qwen3.5-4b-pool5}" >&2; exit 2 ;;
esac

cd "$WORK"

domains_of() {
    # The head node has no python: read the generated YAML's `domains:` list with awk.
    awk '/^  domains:/{f=1; next} f && /^  - /{sub(/^  - /, ""); printf "%s ", $0; next} f{exit}' \
        "configs/qwen35/$1.yml"
}

pool_of() {
    # A pooled config (`pool_as:`, LOG.md Iteration 31) has ONE connectivity job over all
    # its domains, and its parcellation and purge jobs run on the pseudo-domain it names.
    awk '/^  pool_as:/{print $2}' "configs/qwen35/$1.yml"
}

job_domains_of() {
    local p; p=$(pool_of "$1")
    if [ -n "$p" ]; then echo "$p"; else domains_of "$1"; fi
}

conn_job_of() {
    # jobs/<name>.connectivity[.<domain>].pbs
    if [ -n "$(pool_of "$1")" ]; then echo "jobs/$1.connectivity.pbs"; else echo "jobs/$1.connectivity.$2.pbs"; fi
}

generate() {
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    fi
    mkdir -p jobs logs
    python scripts/make_qwen_configs.py > /dev/null
    local M="python -m parcelmate.bin.make_jobs"
    local CPU=configs/cluster/sc-cpu.yml
    local GPU=configs/cluster/sc.yml
    local name cfg d
    for name in ${NAMES:-qwen3.5-2b qwen3.5-4b qwen3.5-2b-pool5 qwen3.5-4b-pool5}; do
        cfg=configs/qwen35/$name.yml
        # One parcellation job per (domain, tree, half): the streamed PCA takes four passes
        # over a half, measured at 33 min per pass at 147k units (17559496) and four times
        # that at 295k, so one job per domain would be 9 h at 2B and 36 h at 4B, past the
        # partition's limit. Split, the four run in parallel (LOG.md Iteration 26).
        case "$name" in
            # Pooled runs: the same tokens per half as one single-domain run (5 x 40,960
            # against 2 x ~100k), so the same sizes; scoring has no across-domain pairs.
            qwen3.5-2b|qwen3.5-2b-pool5) GPU_OPTS="-t 6 -m 160"; PAR="-t 5 -m 32"; SCO="-t 12 -m 32" ;;
            qwen3.5-4b|qwen3.5-4b-pool5) GPU_OPTS="-t 10 -m 200 -P sphinx -G a100"; PAR="-t 14 -m 48"; SCO="-t 36 -m 48" ;;
        esac
        if [ -n "$(pool_of $name)" ]; then
            $M $cfg -c $GPU -s connectivity $GPU_OPTS -n 8 -o jobs/
        fi
        for d in $(job_domains_of $name); do
            [ -n "$(pool_of $name)" ] || $M $cfg -c $GPU -s connectivity $GPU_OPTS -n 8 -D $d -o jobs/
            for t in real null; do
                for k in halfA halfB; do
                    $M $cfg -c $CPU -s parcellation $PAR -n 8 -D $d -T $t -K $k -o jobs/
                done
            done
            $M $cfg -c $CPU -s purge_null_connectivity -t 1 -m 4 -n 1 -D $d -o jobs/
        done
        $M $cfg -c $CPU -s score purge_connectivity $SCO -n 4 -o jobs/
    done
    ls -1 jobs/qwen3.5-*.pbs
}

submit() {
    local name=$1
    test -n "$name" || { echo "submit needs a config name" >&2; exit 2; }
    mkdir -p logs
    echo "code at $(git log --oneline | head -1)"
    # Serialise the domains where the tiles are large (4B): a domain's connectivity starts
    # only once the previous domain's null tiles are gone, which bounds the disk at the
    # accumulated real halves plus one domain's full set (1.75 TB at 4B) instead of every
    # domain's at once (2.8 TB). At 2B the tiles are small enough to run in parallel.
    local serial=0
    case "$name" in qwen3.5-4b) serial=1 ;; esac
    local d t k c p q dep prev="" ids="" real_ids="" null_ids=""
    for d in $(job_domains_of $name); do
        dep=""
        if [ "$serial" = 1 ] && [ -n "$prev" ]; then dep="--dependency=afterok:$prev"; fi
        c=$(sbatch --parsable $dep $(conn_job_of $name $d))
        real_ids=""; null_ids=""
        for t in real null; do
            for k in halfA halfB; do
                p=$(sbatch --parsable --dependency=afterok:$c jobs/$name.parcellation.$d.$t.$k.pbs)
                echo "$name $d $t $k: connectivity $c -> parcellation $p"
                if [ "$t" = real ]; then real_ids="$real_ids:$p"; else null_ids="$null_ids:$p"; fi
            done
        done
        # The null tiles go as soon as BOTH null halves are parcellated.
        q=$(sbatch --parsable --dependency=afterok${null_ids} jobs/$name.purge_null_connectivity.$d.pbs)
        echo "$name $d: purge null -> $q"
        ids="$ids$real_ids:$q"
        prev=$q
    done
    c=$(sbatch --parsable --dependency=afterok${ids} jobs/$name.score_purge_connectivity.pbs)
    echo "$name score + purge -> $c"
    echo
    squeue -u "$USER" -o "%.9i %.50j %.9T %.10M %R" | grep "$name"
}

case "$MODE" in
    generate) generate ;;
    submit)   submit "$NAME" ;;
esac
