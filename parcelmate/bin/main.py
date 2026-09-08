import argparse

from parcelmate.cfg import get_cfg
from parcelmate.model import *
from parcelmate.plot import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Main executable for parcelmate package.''')
    argparser.add_argument('config_path', nargs='?', default=None, help='Path to config file.')
    argparser.add_argument('-s', '--steps', nargs='+', default=['all'], help=
                           'Space-delimited list of steps to run, or `all`.'
                           )
    argparser.add_argument('-O', '--overwrite', action='store_true',
                           help='Recompute all outputs, even if they already exist.')
    argparser.add_argument('--seed', type=int, default=None,
                           help='Master random seed, overriding any `seed` in the config.')
    args = argparser.parse_args()
    config_path = args.config_path
    steps = set(args.steps)
    overwrite = args.overwrite

    if config_path is not None:
        cfg = get_cfg(config_path)
    else:
        cfg = {}

    # Top-level `seed` supplies the default for every step; a step's own section may
    # override it, and --seed on the command line overrides both.
    seed = cfg.get('seed', None) if args.seed is None else args.seed

    def stepcfg(name):
        out = dict(cfg.get(name, {}))
        out.setdefault('seed', seed)
        return out

    if 'all' in steps or 'connectivity' in steps:
        run_connectivity(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            overwrite=overwrite,
            **stepcfg('connectivity')
        )

    if 'all' in steps or 'parcellation' in steps:
        run_parcellation(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            overwrite=overwrite,
            **stepcfg('parcellation')
        )

    if 'all' in steps or 'subnetwork_extraction' in steps:
        run_subnetwork_extraction(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            **cfg.get('subnetwork_extraction', {})
        )

    if 'all' in steps or 'plot_connectivity' in steps:
        plot_connectivity(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'plot_parcellation' in steps:
        plot_parcellation(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'plot_stability' in steps:
        plot_stability(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'subnetwork_knockout' in steps:
        # Reads its own `subnetwork_knockout` section (S3). It previously received
        # `subnetwork_extraction`, whose signature is disjoint from run_knockout's, so any
        # key there raised TypeError in one of the two calls -- including `domains`, which
        # the M6 fix made a key you would actually want to set.
        knockout_kwargs = dict(cfg.get('subnetwork_knockout', {}))
        knockout_kwargs.setdefault('seed', seed)
        # model_name is inherited from the connectivity section as the single source of
        # truth. Previously it was neither forwarded nor overridable: run_knockout defaulted
        # to 'gpt2' regardless of the configured model, and passing the connectivity section
        # wholesale collided with its own model_name argument.
        connectivity_cfg = cfg.get('connectivity', {})
        assert 'model_name' not in knockout_kwargs, \
            'set model_name under `connectivity`, not `subnetwork_knockout` - the lesioned ' \
            'model must be the same model the baseline was measured on'
        if 'model_name' in connectivity_cfg:
            knockout_kwargs['model_name'] = connectivity_cfg['model_name']
        run_knockout(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            connectivity_kwargs=connectivity_cfg,
            overwrite=overwrite,
            **knockout_kwargs
        )

