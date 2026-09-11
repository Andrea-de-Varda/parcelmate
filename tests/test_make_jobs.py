"""Tests for the SLURM job generator (parcelmate/bin/make_jobs.py)."""
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parcelmate.bin.make_jobs import DEFAULTS, get_profile, resolve, get_job

PROFILE = {
    'workdir': '/work/parcelmate',
    'log_dir': 'logs',
    'conda_sh': '/work/miniforge3/etc/profile.d/conda.sh',
    'conda_env': 'parcelmate',
    'env': {'HF_HOME': '/work/.hf_cache'},
    'slurm': {'account': 'nlp', 'partition': 'jag-standard', 'gpu': 1, 'gpu_type': 'a6000', 'memory': 32},
}


def write_profile(tmpdir, profile):
    path = os.path.join(tmpdir, 'profile.yml')
    with open(path, 'w') as f:
        yaml.safe_dump(profile, f)
    return path


def test_profile_flattens_slurm_section():
    with tempfile.TemporaryDirectory() as tmpdir:
        profile = get_profile(write_profile(tmpdir, PROFILE))
    assert profile['account'] == 'nlp'          # from slurm section
    assert profile['workdir'] == '/work/parcelmate'  # from top level


def test_profile_rejects_unknown_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_profile(tmpdir, {'wrkdir': '/typo'})
        with pytest.raises(ValueError, match='wrkdir'):
            get_profile(path)


def test_no_profile_is_empty():
    assert get_profile(None) == {}


def test_cli_overrides_profile_which_overrides_defaults():
    settings = resolve({'memory': 64, 'partition': None}, {'partition': 'john', 'memory': 32})
    assert settings['memory'] == 64        # CLI wins
    assert settings['partition'] == 'john'  # profile wins over default None
    assert settings['time'] == DEFAULTS['time']  # falls through to default


def test_gpu_job_has_account_partition_gres_and_env():
    settings = resolve({}, get_profile_dict(PROFILE))
    job = get_job('configs/gpt2.yml', settings, steps=['connectivity'])
    assert '#SBATCH --account=nlp' in job
    assert '#SBATCH --partition=jag-standard' in job
    assert '#SBATCH --gres=gpu:a6000:1' in job
    assert 'conda activate parcelmate' in job
    assert 'export HF_HOME=/work/.hf_cache' in job
    assert 'cd /work/parcelmate' in job
    assert job.endswith('python -m parcelmate.bin.main configs/gpt2.yml -s connectivity\n')


def test_cpu_job_requests_no_gpu():
    profile = get_profile_dict({**PROFILE, 'slurm': {'account': 'nlp', 'partition': 'john', 'gpu': 0}})
    job = get_job('configs/gpt2.yml', resolve({}, profile), steps=['parcellation'])
    assert '--gres' not in job
    assert '#SBATCH --partition=john' in job


def test_log_paths_are_absolute_and_job_name_carries_steps():
    job = get_job('configs/gpt2.yml', resolve({}, get_profile_dict(PROFILE)), steps=['parcellation', 'plot_stability'])
    assert '#SBATCH --job-name=gpt2.parcellation_plot_stability' in job
    assert '#SBATCH --output=/work/parcelmate/logs/gpt2.parcellation_plot_stability-%N-%j.out' in job
    assert '#SBATCH --error=/work/parcelmate/logs/' in job


def test_steps_omitted_runs_whole_pipeline():
    job = get_job('configs/gpt2.yml', resolve({}, get_profile_dict(PROFILE)))
    assert job.rstrip().endswith('python -m parcelmate.bin.main configs/gpt2.yml')
    assert '#SBATCH --job-name=gpt2\n' in job


def test_exclude_accepts_list_and_string():
    base = get_profile_dict(PROFILE)
    assert '#SBATCH --exclude=n1,n2' in get_job('c.yml', resolve({'exclude': ['n1', 'n2']}, base))
    assert '#SBATCH --exclude=n1,n2' in get_job('c.yml', resolve({'exclude': 'n1,n2'}, base))


def test_yaml_config_extension_not_truncated():
    """Guards the old `basename[:-4]` behaviour, which mangled `.yaml` names."""
    job = get_job('configs/gpt2.yaml', resolve({}, get_profile_dict(PROFILE)))
    assert '#SBATCH --job-name=gpt2\n' in job


def get_profile_dict(profile):
    """Round-trip a profile dict through get_profile via a temp file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        return get_profile(write_profile(tmpdir, profile))


def test_overwrite_flag_passthrough():
    """-O must reach main.py, or a re-run silently skips every already-cached step."""
    settings = resolve({}, get_profile_dict(PROFILE))
    job = get_job('configs/ladder.yml', settings, steps=['connectivity'], overwrite=True)
    assert job.rstrip().endswith('-s connectivity -O')


def test_overwrite_absent_by_default():
    settings = resolve({}, get_profile_dict(PROFILE))
    job = get_job('configs/ladder.yml', settings, steps=['connectivity'])
    assert ' -O' not in job


def test_seed_flag_passthrough():
    settings = resolve({}, get_profile_dict(PROFILE))
    job = get_job('configs/ladder.yml', settings, steps=['connectivity'], seed=7)
    assert job.rstrip().endswith('--seed 7')
