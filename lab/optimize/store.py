"""Small persistent records and source copies; no content hashes or implicit cache."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .schema import relative_path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    with temp.open('w') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def capture(root, files, destination):
    """Copy declared build inputs once; caller selects the dependency closure."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    inputs = []
    for value in files:
        path = relative_path(value)
        source = (root / path).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f'input escapes repository: {value}')
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs.append(str(path))
    manifest = dict(id=uuid.uuid4().hex, root=str(destination), inputs=inputs,
                    revision=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                                            text=True, capture_output=True, check=True).stdout.strip(),
                    working_tree=subprocess.run(['git', 'status', '--short'], cwd=root,
                                                text=True, capture_output=True, check=True).stdout,
                    coverage='declared inputs only; not a complete runtime snapshot')
    write(destination / 'source.json', manifest)
    return manifest


def changed_inputs(a, b):
    """Compare only explicitly declared build inputs when deciding reuse."""
    names = set(a['inputs']) | set(b['inputs'])
    return sorted(name for name in names if name not in a['inputs'] or name not in b['inputs']
                  or (Path(a['root']) / name).read_bytes() != (Path(b['root']) / name).read_bytes())


def index(root, record):
    """Idempotent small evidence copy; large logs remain in their run directory."""
    item = {key: record[key] for key in ('id', 'target', 'kind', 'status', 'validity',
                                       'correctness', 'conclusion', 'promotion', 'cost')}
    item.update(evidence=str(Path(record['directory']) / 'evidence.json'),
                hypothesis=record['spec']['hypothesis'], conditions=record['spec'].get('conditions', {}),
                options=record['spec'].get('options', {}),
                reopen_when=record['spec'].get('reopen_when', []),
                source=record.get('source'), task_id=record['spec'].get('task_id', record['id']),
                protocol=record['spec'].get('protocol'))
    write(Path(root) / (record['id'] + '.json'), item)
    return item


def related(root, spec):
    items = [read(path) for path in sorted(Path(root).glob('*.json'))]
    return [item for item in items if item['target'] == spec['target']
            and item['hypothesis']['mechanism'] == spec['hypothesis']['mechanism']]


def duplicate_status(item, spec, source):
    """A missing large artifact leaves identity unknown, not a permanent rejection."""
    if (item['conditions'] != spec.get('conditions', {})
            or item.get('options', {}) != spec.get('options', {})):
        return 'reopen_changed_conditions'
    if spec.get('replication_of') == item['id']:
        return 'deliberate_replication'
    previous = item.get('source')
    if previous is None or any(not (Path(previous['root']) / p).is_file() for p in previous['inputs']):
        return 'identity_unknown_missing_artifacts'
    if changed_inputs(previous, source):
        return 'reopen_changed_inputs'
    if item.get('protocol') != spec.get('protocol'):
        return 'reopen_changed_protocol'
    return 'duplicate_review'


def task_budget(root, spec, budget):
    task = spec.get('task_id', spec['id'])
    rows = [read(p) for p in Path(root).glob('*.json')]
    rows = [r for r in rows if r.get('task_id', r['id']) == task]
    candidates = sum(r['kind'] == 'performance' for r in rows)
    non_improving = sum(r['validity'] == 'valid' and r['conclusion'] == 'no_benefit' for r in rows)
    jobs = set(j for r in rows for j in r['cost']['jobs'])
    if spec['kind'] == 'performance' and candidates >= budget['candidates']:
        raise RuntimeError('task candidate budget exhausted')
    if non_improving >= budget['non_improving']:
        raise RuntimeError('task non-improving budget exhausted')
    return dict(candidates=candidates, non_improving=non_improving, jobs=sorted(jobs))
