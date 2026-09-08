"""Run a declared experiment through existing tools; source copies are explicit."""
import argparse
import json
from pathlib import Path

from . import context, preflight, runner, store
from .schema import validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'start', 'run', 'reconcile', 'context', 'related'))
    parser.add_argument('path', type=Path, help='spec JSON for preflight/start; run directory otherwise')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path)
    parser.add_argument('--until', choices=('preflight', 'probe', 'check', 'measure', 'qualify'), default='measure')
    parser.add_argument('--recovered-seconds', type=float)
    args = parser.parse_args(argv)
    if args.command in ('preflight', 'start'):
        spec = store.read(args.path)
        policy, budget = validate(spec)
        if args.command == 'start':
            if args.out is None:
                parser.error('start requires --out')
            resolved = preflight.inspect(spec)
            result = runner.start(args.root, spec, args.out)
            result['preflight'] = resolved
            store.write(args.out / 'evidence.json', result)
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
