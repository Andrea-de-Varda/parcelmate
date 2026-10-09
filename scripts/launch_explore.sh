#!/bin/bash
#
# Exploratory GPU jobs (LOG.md Iteration 38): one short a6000 job per call.
#
#     ssh sc 'bash /juice6/u/nlp/climblab/devarda/parcelmate/scripts/launch_explore.sh <name> <hours> <python args...>'
#
# Runs `python <python args>` from the repository root, with the lab-share conventions:
# umask 002, a per-job TMPDIR on the share (removed at exit), the shared HF cache.
set -e
umask 002
WORK=${WORK:-/juice6/u/nlp/climblab/devarda/parcelmate}
CONDA_SH=${CONDA_SH:-/juice6/u/nlp/climblab/devarda/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-parcelmate}
EXCLUDE=${EXCLUDE:-jagupard19,jagupard20,jagupard32}   # any GPU of at least 24 GB (Titan V is 12 GB; jagupard32 bad)
NAME=$1; HOURS=$2; shift 2
cd "$WORK"
mkdir -p jobs logs
cat > jobs/explore.$NAME.pbs <<EOF
#!/bin/bash
#SBATCH --job-name=explore.$NAME
#SBATCH --output=$WORK/logs/explore.$NAME-%N-%j.out
#SBATCH --error=$WORK/logs/explore.$NAME-%N-%j.err
#SBATCH --time=$HOURS:00:00
#SBATCH --mem=96gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --exclude=$EXCLUDE
set -e
umask 002
cd $WORK
export TMPDIR=$WORK/tmp/\$SLURM_JOB_ID
mkdir -p \$TMPDIR
trap 'rm -rf \$TMPDIR' EXIT
source $CONDA_SH
conda activate $CONDA_ENV
export HF_HOME=/juice6/u/nlp/climblab/devarda/.hf_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
python $@
EOF
echo "code at $(git log --oneline | head -1)"
echo "explore.$NAME -> $(sbatch --parsable jobs/explore.$NAME.pbs)"
