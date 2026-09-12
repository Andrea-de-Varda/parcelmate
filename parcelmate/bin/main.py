import argparse
import os

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
    argparser.add_argument('-V', '--variants', nargs='+', default=None,
                           help='Restrict the parcellation-level steps (and scoring) to these '
                                'variants, so arms of one config can run as parallel jobs. A '
                                'restricted score writes scores_<arms>.csv, never scores.csv.')
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
        connectivity_kwargs = stepcfg('connectivity')
        # `null_output_dir` defaults to a sibling of output_dir, so a config need only say
        # `null_model: circshift`. Filenames match the real tree exactly, so every later
        # step (parcellation, metrics) runs on either tree unchanged by pointing at it.
        # The key is `null_model`, not `null`: YAML parses a bare `null` key as None, which
        # reaches run_connectivity as a non-string keyword and fails obscurely.
        if connectivity_kwargs.get('null_model') and not connectivity_kwargs.get('null_output_dir'):
            connectivity_kwargs['null_output_dir'] = \
                cfg.get('output_dir', OUTPUT_DIR).rstrip('/') + '_null'
        run_connectivity(
            output_dir=cfg.get('output_dir', OUTPUT_DIR),
            overwrite=overwrite,
            **connectivity_kwargs
        )

    if 'all' in steps or 'split_halves' in steps:
        # Must run after connectivity and before parcellation: it turns the per-sample
        # matrices into the two independent halves that reliability and fidelity need.
        for out_dir in [cfg.get('output_dir', OUTPUT_DIR)] + (
                [cfg.get('output_dir', OUTPUT_DIR).rstrip('/') + '_null']
                if cfg.get('connectivity', {}).get('null_model') else []):
            if os.path.isdir(os.path.join(out_dir, CONNECTIVITY_NAME)):
                run_split_halves(output_dir=out_dir, overwrite=overwrite)

    # Parcellation variants. `parcellation` holds settings common to every arm;
    # `parcellation_variants` maps a variant name to the settings that differ. With no
    # variants block there is a single arm named 'default', so a plain config behaves as
    # before. Each variant writes to <output_dir>/<variant>/, so arms never overwrite one
    # another and one config documents the whole experiment.
    variants = cfg.get('parcellation_variants') or {'default': {}}
    assert isinstance(variants, dict), '`parcellation_variants` must be a mapping of name -> settings'
    if args.variants:
        unknown = [v for v in args.variants if v not in variants]
        assert not unknown, 'unknown variant(s) %s; config has %s' % (unknown, sorted(variants))
        variants = {k: variants[k] for k in args.variants}

    def variant_cfg(name):
        out = stepcfg('parcellation')
        out.update(variants[name] or {})
        out['variant'] = name
        return out

    # Both trees get parcellated: the null is only useful if the *same* pipeline runs on
    # it, and the metrics are reported as real-minus-null.
    def trees():
        out = [cfg.get('output_dir', OUTPUT_DIR)]
        if cfg.get('connectivity', {}).get('null_model'):
            out.append(cfg.get('output_dir', OUTPUT_DIR).rstrip('/') + '_null')
        return [t for t in out if os.path.isdir(t)]

    if 'all' in steps or 'parcellation' in steps:
        for tree in trees():
            for name in variants:
                run_parcellation(
                    output_dir=tree,
                    overwrite=overwrite,
                    **variant_cfg(name)
                )

    if 'all' in steps or 'subnetwork_extraction' in steps:
        for name in variants:
            run_subnetwork_extraction(
                output_dir=cfg.get('output_dir', OUTPUT_DIR),
                variant=name,
                **cfg.get('subnetwork_extraction', {})
            )

    if 'all' in steps or 'plot_connectivity' in steps:
        plot_connectivity(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'plot_parcellation' in steps:
        for name in variants:
            plot_parcellation(
                output_dir=cfg.get('output_dir', OUTPUT_DIR),
                variant=name
            )

    if 'all' in steps or 'plot_stability' in steps:
        plot_stability(
            output_dir=cfg.get('output_dir', OUTPUT_DIR)
        )

    if 'all' in steps or 'score' in steps:
        # Last: needs the parcellations of both trees to exist. Refuses to write a partial
        # table, so a truncated parcellation job fails here loudly instead of producing a
        # scores.csv that looks complete.
        from parcelmate.bin.score import score_config
        # -V restricts scoring too, writing scores_<arms>.csv so the arms that finished can
        # be read before the slow ones do, without ever overwriting the full table.
        score_config(cfg, variants=args.variants)

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
        for name in variants:
            run_knockout(
                output_dir=cfg.get('output_dir', OUTPUT_DIR),
                variant=name,
                connectivity_kwargs=connectivity_cfg,
                overwrite=overwrite,
                **knockout_kwargs
            )

