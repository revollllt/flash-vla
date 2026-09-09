"""Run a declared experiment through existing tools; source copies are explicit."""
import argparse
import json
from pathlib import Path

from . import campaign, context, runner, store, transition
from .registry import CampaignRegistry
from .schema import validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'start', 'run', 'reconcile', 'context', 'related',
                                            'campaign-create', 'campaign-status', 'campaign-resume',
                                            'campaign-validate', 'campaign-finalize',
                                            'campaign-render', 'campaign-reanchor', 'campaign-materialize',
                                            'campaign-migrate-legacy', 'campaign-transition', 'campaign-transition-abort',
                                            'campaign-find', 'campaign-open', 'campaign-open-or-create', 'campaign-open-or-seed', 'campaign-fork'))
    parser.add_argument('path', type=Path, nargs='?',
                        help='key JSON for registry commands; spec or run directory otherwise')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path)
    parser.add_argument('--until', choices=('preflight', 'probe', 'check', 'measure', 'qualify'), default='measure')
    parser.add_argument('--recovered-seconds', type=float)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--baseline-evidence', type=Path)
    parser.add_argument('--objective')
    parser.add_argument('--protocol')
    parser.add_argument('--fixture')
    parser.add_argument('--source-input', action='append', default=[])
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--reason')
    parser.add_argument('--fork-id')
    args = parser.parse_args(argv)
    if args.path is None and args.command != 'campaign-create':
        parser.error('this command requires a path')
    if args.command in ('campaign-find', 'campaign-open', 'campaign-open-or-create', 'campaign-open-or-seed', 'campaign-fork'):
        registry = CampaignRegistry(args.root)
        key = store.read(args.path)
        if args.command == 'campaign-find':
            location = registry.find(key)
        elif args.command == 'campaign-open':
            location = registry.open(key, fork_id=args.fork_id)
        elif args.command == 'campaign-fork':
            if not args.reason:
                parser.error('campaign-fork requires --reason')
            location = registry.fork(key, reason=args.reason)
        else:
            baseline = store.read(args.baseline_evidence) if args.baseline_evidence else None
            opener = registry.open_or_seed if args.command == 'campaign-open-or-seed' else registry.open_or_create
            location = opener(key, baseline=baseline, inputs=args.source_input)
        result = dict(directory=str(location) if location else None,
                      state=store.read(location / 'state.json') if location else None)
    elif args.command == 'campaign-create':
        required = (args.baseline_evidence, args.objective, args.protocol, args.fixture)
        if any(value is None for value in required):
            parser.error('campaign-create requires --baseline-evidence, --objective, --protocol and --fixture')
        baseline = store.read(args.baseline_evidence)
        if baseline['identity'].get('schema_version') == 3:
            if args.path is not None:
                parser.error('v3 campaign-create uses the Registry; omit the manual directory')
            if args.fixture != baseline['measurement_context']['fixture']['id']:
                parser.error('--fixture differs from baseline evidence')
            key = campaign.campaign_key(baseline['identity'], objective=args.objective, protocol=args.protocol)
            location = CampaignRegistry(args.root).create(key, baseline=baseline, inputs=args.source_input)
            result = dict(directory=str(location), campaign_key=key,
                          state=store.read(location / 'state.json'))
        else:
            if args.path is None:
                parser.error('legacy campaign-create requires a directory')
            result = campaign.create(args.path, baseline, args.objective,
                                     args.protocol, args.fixture,
                                     root=args.root if args.source_input else None, inputs=args.source_input)
    elif args.command in ('campaign-status', 'campaign-validate'):
        result = campaign.rebuild(args.path)
    elif args.command == 'campaign-transition':
        if args.result is None:
            parser.error('campaign-transition requires --result with a transition request')
        result = campaign.transition_context(args.root, args.path, store.read(args.result))
    elif args.command == 'campaign-transition-abort':
        result = transition.abort(args.path, recovered_seconds=args.recovered_seconds)
    elif args.command == 'campaign-materialize':
        result = campaign.materialize_incumbent(args.root, args.path)
    elif args.command == 'campaign-resume':
        result = campaign.resume(args.path, args.until, args.recovered_seconds)
    elif args.command == 'campaign-finalize':
        if args.iteration is None or args.result is None:
            parser.error('campaign-finalize requires --iteration and --result')
        outcome = store.read(args.result)
        result = campaign.finalize(args.path, args.iteration, outcome['verdict'],
                                   outcome['correctness'], outcome['measurement'],
                                   outcome['qualification'], outcome.get('diagnostics'))
    elif args.command == 'campaign-render':
        from . import render, trace

        result = trace.materialize(args.path)
        metadata, points = render.from_trace(result)
        render.render_optimization_progress(
            metadata=metadata, points=points, output_svg=args.path / 'progress.svg',
            output_png=(args.path / 'progress.png' if args.png else None))
        if args.html:
            render.render_html(metadata, points, args.path / 'progress.html')
    elif args.command == 'campaign-reanchor':
        if args.result is None:
            parser.error('campaign-reanchor requires --result')
        result = campaign.reanchor(args.path, store.read(args.result))
    elif args.command == 'campaign-migrate-legacy':
        from . import migrate

        result = migrate.pi_campaign(args.root, args.path)
    elif args.command in ('preflight', 'start'):
        from . import preflight

        spec = store.read(args.path)
        policy, budget = validate(spec)
        if args.command == 'start':
            if args.out is None and args.campaign is None:
                parser.error('start requires --out')
            resolved = preflight.inspect(spec)
            result = (campaign.start(args.root, spec, args.campaign) if args.campaign
                      else runner.start(args.root, spec, args.out))
            result['preflight'] = resolved
            output = Path(result['directory'])
            store.write(output / 'evidence.json', result)
        else:
            result = preflight.inspect(spec)
    elif args.command == 'run':
        result = runner.run(args.path, args.until)
    elif args.command == 'reconcile':
        result = runner.reconcile(args.path, args.recovered_seconds)
    else:
        record = store.read(args.path / 'evidence.json')
        if args.command == 'context':
            result = context.collect(record['repository'], record['spec'].get('conditions', {}),
                                     record['spec'].get('references', []))
            store.write(args.path / 'context.json', result)
            (args.path / 'context.md').write_text('\n\n'.join(
                f"{item['applicability']}: {item['source']}\n{item['excerpt']}" for item in result))
        else:
            result = [dict(item, duplicate_status=store.duplicate_status(item, record['spec'], record['source']))
                      for item in store.related(args.path.parent / 'index', record['spec'])
                      if item['id'] != record['id']]
    print(json.dumps(result, indent=2))
    return 1 if isinstance(result, dict) and result.get('status') == 'stopped' else 0


if __name__ == '__main__':
    raise SystemExit(main())
