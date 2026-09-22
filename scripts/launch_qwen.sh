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
    *) echo "usage: $0 generate | submit {qwen3.5-2b|qwen3.5-4b}" >&2; exit 2 ;;
esac

cd "$WORK"

domains_of() {
    python - "$1" <<'EOF'
import sys, yaml
print(' '.join(yaml.safe_load(open('configs/qwen35/%s.yml' % sys.argv[1]))['connectivity']['domains']))
EOF
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
    for name in qwen3.5-2b qwen3.5-4b; do
        cfg=configs/qwen35/$name.yml
        case "$name" in
            qwen3.5-2b) GPU_OPTS="-t 6 -m 160"; PAR="-t 12 -m 32"; SCO="-t 12 -m 32" ;;
            qwen3.5-4b) GPU_OPTS="-t 10 -m 200 -P sphinx -G a100"; PAR="-t 24 -m 32"; SCO="-t 24 -m 32" ;;
        esac
        for d in $(domains_of $name); do
            $M $cfg -c $GPU -s connectivity $GPU_OPTS -n 8 -D $d -o jobs/
            $M $cfg -c $CPU -s parcellation $PAR -n 8 -D $d -o jobs/
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
    local d c p q ids=""
    for d in $(domains_of $name); do
        c=$(sbatch --parsable jobs/$name.connectivity.$d.pbs)
        p=$(sbatch --parsable --dependency=afterok:$c jobs/$name.parcellation.$d.pbs)
        q=$(sbatch --parsable --dependency=afterok:$p jobs/$name.purge_null_connectivity.$d.pbs)
        echo "$name $d: connectivity $c -> parcellation $p -> purge null $q"
        ids="$ids:$q"
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
