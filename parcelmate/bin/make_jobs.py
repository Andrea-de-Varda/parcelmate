import os
import argparse

import yaml

from parcelmate.util import variants_tag

# Built-in fallbacks, used when neither the cluster profile nor the CLI supplies a value.
DEFAULTS = {
    'time': 24,
    'n_cores': 1,
    'memory': 8,
    'workdir': None,
    'log_dir': 'logs',
    'python': 'python',
    'conda_sh': None,
    'conda_env': None,
    'account': None,
    'partition': None,
    'qos': None,
    'gpu': 0,
    'gpu_type': None,
    'constraint': None,
    'exclude': None,
    'env': {},
}


def get_profile(path):
    """Load a cluster profile (YAML) and flatten its `slurm` section into the top level."""
    if path is None:
        return {}
    with open(path, 'r') as f:
        profile = yaml.safe_load(f) or {}
    out = {k: v for k, v in profile.items() if k != 'slurm'}
    out.update(profile.get('slurm', {}) or {})
    unknown = set(out) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unrecognized key(s) in cluster profile %s: %s' % (path, ', '.join(sorted(unknown))))

    return out


def resolve(cli, profile):
    """Merge settings. Precedence: CLI argument > cluster profile > DEFAULTS."""
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in profile.items() if v is not None})
    settings.update({k: v for k, v in cli.items() if v is not None})

    return settings


def get_job(config_path, settings, steps=None, overwrite=False, seed=None, variants=None):
    """Render a single SLURM batch script as a string."""
    job_name = os.path.splitext(os.path.basename(config_path))[0]
    if steps:
        job_name = '%s.%s' % (job_name, '_'.join(steps))
    if variants:
        job_name = '%s.%s' % (job_name, variants_tag(variants))
    log_dir = settings['log_dir']
    if settings['workdir'] and not os.path.isabs(log_dir):
        log_dir = os.path.join(settings['workdir'], log_dir)

    out = ['#!/bin/bash', '#']
    out.append('#SBATCH --job-name=%s' % job_name)
    # Absolute log paths, so output does not depend on the directory sbatch was called from.
    out.append('#SBATCH --output=%s' % os.path.join(log_dir, '%s-%%N-%%j.out' % job_name))
    out.append('#SBATCH --error=%s' % os.path.join(log_dir, '%s-%%N-%%j.err' % job_name))
    out.append('#SBATCH --time=%d:00:00' % int(settings['time']))
    out.append('#SBATCH --mem=%dgb' % int(settings['memory']))
    out.append('#SBATCH --ntasks=1')
    out.append('#SBATCH --cpus-per-task=%d' % int(settings['n_cores']))
    if settings['account']:
        out.append('#SBATCH --account=%s' % settings['account'])
    if settings['partition']:
        out.append('#SBATCH --partition=%s' % settings['partition'])
    if settings['qos']:
        out.append('#SBATCH --qos=%s' % settings['qos'])
    if int(settings['gpu']):
        if settings['gpu_type']:
            out.append('#SBATCH --gres=gpu:%s:%d' % (settings['gpu_type'], int(settings['gpu'])))
        else:
            out.append('#SBATCH --gres=gpu:%d' % int(settings['gpu']))
    if settings['constraint']:
        out.append('#SBATCH --constraint=%s' % settings['constraint'])
    if settings['exclude']:
        exclude = settings['exclude']
        if not isinstance(exclude, str):
            exclude = ','.join(exclude)
        out.append('#SBATCH --exclude=%s' % exclude)

    out.append('')
    out.append('set -e')
    # Lab-share etiquette (Cory Shain, 2026-09-22): everything written under the shared
    # store must be group-readable and group-writable.
    out.append('umask 002')
    out.append('')
    out.append('mkdir -p %s' % log_dir)
    if settings['workdir']:
        out.append('cd %s' % settings['workdir'])
    if settings['conda_sh']:
        out.append('source %s' % settings['conda_sh'])
    if settings['conda_env']:
        out.append('conda activate %s' % settings['conda_env'])
    for key in sorted(settings['env'] or {}):
        out.append('export %s=%s' % (key, settings['env'][key]))

    out.append('')
    cmd = '%s -m parcelmate.bin.main %s' % (settings['python'], config_path)
    if steps:
        cmd += ' -s %s' % ' '.join(steps)
    if overwrite:
        cmd += ' -O'
    if seed is not None:
        cmd += ' --seed %d' % int(seed)
    if variants:
        cmd += ' -V %s' % ' '.join(variants)
    out.append(cmd)
    out.append('')

    return '\n'.join(out)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run parcellations specified in one or more config (YAML) files.
    Cluster-specific settings (working directory, conda environment, environment variables, SLURM
    account/partition) come from a cluster profile; any of them can be overridden on the CLI.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s).')
    argparser.add_argument('-c', '--cluster', default=None,
                           help='Path to cluster profile (YAML), e.g. configs/cluster/sc.yml')
    argparser.add_argument('-s', '--steps', nargs='+', default=None,
                           help='Pipeline step(s) to run in this job, passed through to main.py. '
                                'Omit to run all steps. Also appended to the job name.')
    argparser.add_argument('-t', '--time', type=int, default=None, help='Maximum number of hours to run')
    argparser.add_argument('-n', '--n_cores', type=int, default=None, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=None, help='Number of GB of memory to request')
    argparser.add_argument('-g', '--gpu', type=int, default=None, help='Number of GPUs to request (0 for none)')
    argparser.add_argument('-G', '--gpu_type', default=None,
                           help='GPU type to request, e.g. a6000, a100, h100. Omit for any type.')
    argparser.add_argument('-C', '--constraint', default=None,
                           help='Value for SLURM --constraint setting, e.g. 80G')
    argparser.add_argument('-a', '--slurm_account', dest='account', default=None,
                           help='Value for SLURM --account setting')
    argparser.add_argument('-P', '--slurm_partition', dest='partition', default=None,
                           help='Value for SLURM --partition setting')
    argparser.add_argument('-q', '--qos', default=None, help='Value for SLURM --qos setting')
    argparser.add_argument('-e', '--exclude', nargs='+', default=None, help='Nodes to exclude')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts')
    argparser.add_argument('-O', '--overwrite', action='store_true',
                           help='Pass -O to main.py, recomputing outputs that are already cached. '
                                'Without this a re-run silently skips every step whose HDF5 output exists.')
    argparser.add_argument('--seed', type=int, default=None,
                           help='Pass --seed to main.py, overriding the config seed.')
    argparser.add_argument('-V', '--variants', nargs='+', default=None,
                           help='Pass -V to main.py: run only these parcellation variants. '
                                'One job per arm lets the arms of a config run in parallel.')
    args = argparser.parse_args()

    cli = {key: getattr(args, key) for key in (
        'time', 'n_cores', 'memory', 'gpu', 'gpu_type', 'constraint', 'account', 'partition', 'qos', 'exclude'
    )}
    settings = resolve(cli, get_profile(args.cluster))

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for path in args.paths:
        job_name = os.path.splitext(os.path.basename(path))[0]
        if args.steps:
            job_name = '%s.%s' % (job_name, '_'.join(args.steps))
        if args.variants:
            job_name = '%s.%s' % (job_name, variants_tag(args.variants))
        filename = os.path.join(outdir, job_name + '.pbs')
        with open(filename, 'w') as f:
            f.write(get_job(path, settings, steps=args.steps, overwrite=args.overwrite, seed=args.seed,
                            variants=args.variants))
        print(filename)
