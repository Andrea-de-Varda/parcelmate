import argparse
import os
import shutil
import time

from parcelmate.cfg import get_cfg
from parcelmate.model import *
from parcelmate.plot import *

# Lab-share rule (info/CLUSTER.md): everything written is group readable and writable.
os.umask(0o002)

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
    argparser.add_argument('-D', '--domains', nargs='+', default=None,
                           help='Restrict the connectivity, parcellation and purge_null_connectivity '
                                'steps to these domains, so domains of one config can run as '
                                'separate jobs (LOG.md Iteration 25). Scoring always covers all.')
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
        if args.domains:
            unknown = [d for d in args.domains if d not in connectivity_kwargs.get('domains', [])]
            assert not unknown, 'unknown domain(s) %s; config has %s' % (unknown, connectivity_kwargs.get('domains'))
            connectivity_kwargs['domains'] = list(args.domains)
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

    if ('all' in steps and cfg.get('pool_domains')) or 'pool_domains' in steps:
        # After split_halves, before parcellation: pooled connectomes become pseudo-domains
        # that the parcellation step picks up like any other (LOG.md Iteration 19, T4). Both
        # trees, so every pooled arm has its matched null partition.
        pool_kwargs = dict(cfg.get('pool_domains') or {})
        assert pool_kwargs.get('pools'), '-s pool_domains needs a `pool_domains: {pools: ...}` section'
        for out_dir in [cfg.get('output_dir', OUTPUT_DIR)] + (
                [cfg.get('output_dir', OUTPUT_DIR).rstrip('/') + '_null']
                if cfg.get('connectivity', {}).get('null_model') else []):
            if os.path.isdir(os.path.join(out_dir, CONNECTIVITY_NAME)):
                run_pool_domains(output_dir=out_dir, overwrite=overwrite, **pool_kwargs)

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
                kw = variant_cfg(name)
                if args.domains:
                    kw['domains'] = list(args.domains)
                run_parcellation(
                    output_dir=tree,
                    overwrite=overwrite,
                    **kw
                )

    if 'purge_null_connectivity' in steps:
        # Never part of `all`. The null tree's connectivity exists only to fit the null
        # partition (the pnull reference is evaluated on REAL data), so once both halves
        # of a domain are parcellated in the null tree it can go (LOG.md Iteration 25).
        # Restricted to -D domains if given. Leaves a manifest like purge_connectivity.
        assert cfg.get('purge_connectivity') is True, \
            '-s purge_null_connectivity needs `purge_connectivity: true` in the config'
        null_tree = cfg.get('output_dir', OUTPUT_DIR).rstrip('/') + '_null'
        conn_dir = os.path.join(null_tree, CONNECTIVITY_NAME)
        domains = args.domains or cfg.get('connectivity', {}).get('domains', [])
        manifest = os.path.join(null_tree, CONNECTIVITY_NAME + '_purged.txt')
        for domain in domains:
            parcs = [os.path.join(null_tree, name, PARCELLATION_NAME,
                                  '%s_%s_%s%s' % (PARCELLATION_NAME, domain, h, EXTENSION))
                     for name in variants for h in HALF_NAMES]
            missing_p = [p for p in parcs if not os.path.exists(p)]
            assert not missing_p, 'refusing to purge %s: null parcellations missing: %s' % (domain, missing_p)
            files = [f for f in sorted(os.listdir(conn_dir)) if f.startswith('%s_%s_' % (CONNECTIVITY_NAME, domain))] \
                if os.path.isdir(conn_dir) else []
            with open(manifest, 'a') as f:
                for name in files:
                    size = os.path.getsize(os.path.join(conn_dir, name))
                    f.write('%s\t%d bytes\tpurged %s after null parcellation\n' % (
                        name, size, time.strftime('%Y-%m-%dT%H:%M:%S')))
                    os.remove(os.path.join(conn_dir, name))
            print('purged %d null connectivity file(s) of %s' % (len(files), domain))

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
        from parcelmate.bin.score_big import score_config_big, tree_is_tiled
        # -V restricts scoring too, writing scores_<arms>.csv so the arms that finished can
        # be read before the slow ones do, without ever overwriting the full table.
        if tree_is_tiled(cfg):
            score_config_big(cfg, variants=args.variants)
        else:
            score_config(cfg, variants=args.variants)

    if 'purge_connectivity' in steps:
        # Never part of `all`. Deletes the connectivity of both trees once the score file
        # exists, for runs whose connectomes are too large to keep (LOG.md Iteration 22:
        # every Pythia checkpoint at all MLP neurons is 86 GB). The parcellations keep a
        # fingerprint of the matrix they came from, and a manifest of what was deleted is
        # left in place of the directory, so the record survives the data.
        assert cfg.get('purge_connectivity') is True, \
            '-s purge_connectivity needs `purge_connectivity: true` in the config'
        score_path = os.path.join(cfg.get('output_dir', OUTPUT_DIR), 'metrics', 'scores.csv')
        assert os.path.exists(score_path), \
            'refusing to purge: %s does not exist, so this run is not scored yet' % score_path
        for tree in trees():
            conn_dir = os.path.join(tree, CONNECTIVITY_NAME)
            if not os.path.isdir(conn_dir):
                continue
            files = sorted(os.listdir(conn_dir))
            sizes = {f: os.path.getsize(os.path.join(conn_dir, f)) for f in files}
            manifest = os.path.join(tree, CONNECTIVITY_NAME + '_purged.txt')
            with open(manifest, 'w') as f:
                f.write('purged %s after %s existed\n' % (time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                        score_path))
                for name in files:
                    f.write('%s\t%d bytes\n' % (name, sizes[name]))
            shutil.rmtree(conn_dir)
            print('purged %s (%d files, %.1f GB); manifest at %s' % (
                conn_dir, len(files), sum(sizes.values()) / 1e9, manifest))

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

