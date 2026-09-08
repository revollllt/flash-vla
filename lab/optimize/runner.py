"""Sequential, resumable calls to existing experiment tools with durable costs."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import scheduler, store, promotion
from .schema import STAGES, validate


def start(root, spec, directory):
    policy, budget = validate(spec)
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / 'coordinator.lock').open('a') as quota:
        fcntl.flock(quota, fcntl.LOCK_EX)
        store.task_budget(directory.parent / 'index', spec, budget)
        directory.mkdir(parents=True, exist_ok=False)
        record = dict(id=spec['id'], target=spec['target'], kind=spec['kind'], spec=spec,
                      directory=str(directory), repository=str(Path(root).resolve()),
                      status='draft', validity='incomplete', correctness='not_run',
                      conclusion='inconclusive', promotion='not_requested', stages={},
                      cost={'cpu_seconds': 0.0, 'gpu_seconds': 0.0, 'jobs': []},
                      budget=budget, acceptance=policy)
        store.write(directory / 'evidence.json', record)
        record['source'] = store.capture(root, spec['inputs'], directory / 'inputs')
        record['status'] = 'ready'
        store.write(directory / 'evidence.json', record)
        store.index(directory.parent / 'index', record)
        return record


def _command(record, name):
    if name == 'qualify':
        if not record['spec'].get('qualification_sources'):
            raise ValueError('qualification requires explicit incumbent and candidate source records')
        sources = record['spec']['qualification_sources']
        if store.changed_inputs(sources['incumbent'], sources['candidate']):
            raise ValueError('unsupported_protocol: eval.gate supports route variants in one source tree; source-version qualification is not implemented')
        for role in ('incumbent', 'candidate'):
            if Path(sources[role + '_checkout']).resolve() != Path(record['repository']).resolve():
                raise ValueError('eval.gate must load both route variants from the recorded live checkout')
            live = dict(sources[role], root=record['repository'])
            if store.changed_inputs(sources[role], live):
                raise ValueError('qualification source changed before launch; remeasure affected evidence')
        return [sys.executable, '-m', 'eval.gate', '--target', record['target'],
                '--candidate', record['spec'].get('candidate', 'shipped'),
                '--incumbent', record['spec'].get('incumbent', 'shipped'),
                '--baseline', '--out-dir', str(Path(record['directory']) / 'qualify')]
    return [part.replace('{python}', sys.executable).replace('{out}', record['directory'])
            .replace('{inputs}', record['source']['root'])
            for part in record['spec']['stages'][name]['argv']]


def run(directory, until='measure'):
    directory = Path(directory).resolve()
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = store.read(directory / 'evidence.json')
        validate(record['spec'])
        if any(stage['status'] in ('running', 'interrupted', 'failed') for stage in record['stages'].values()):
            raise RuntimeError('run has an unfinished/failed attempt; reconcile it before resuming')
        for name in STAGES[1:STAGES.index(until) + 1]:
            if name not in record['spec']['stages'] or record['stages'].get(name, {}).get('status') == 'completed':
                continue
            if name in ('measure', 'qualify') and record['correctness'] != 'pass':
                raise RuntimeError('required correctness evidence is missing')
            _stage(record, name)
            if record['status'] == 'stopped':
                break
        if record['status'] != 'stopped':
            record['status'] = 'completed' if all(name in record['stages'] for name in record['spec']['stages']) else 'ready'
        store.write(directory / 'evidence.json', record)
        store.index(directory.parent / 'index', record)
        return record


def _stage(record, name):
    directory = Path(record['directory'])
    config = record['spec']['stages'][name]
    resource = 'gpu' if name == 'qualify' else config['resource']
    job = os.environ.get('SLURM_JOB_ID') if resource == 'gpu' else None
    owner = {'job': job, 'node': os.uname().nodename} if job else None
    if owner and owner['job'] not in record['cost']['jobs']:
        with (directory.parent / 'coordinator.lock').open('a') as quota:
            fcntl.flock(quota, fcntl.LOCK_EX)
            reserve = int('qualify' in record['spec']['stages'] and name != 'qualify')
            rows = [store.read(p) for p in (directory.parent / 'index').glob('*.json')]
            task = record['spec'].get('task_id', record['id'])
            jobs = {j for r in rows if r.get('task_id', r['id']) == task for j in r['cost']['jobs']}
            jobs.update(record['cost']['jobs'])
            if owner['job'] not in jobs and len(jobs) >= record['budget']['jobs'] - reserve:
                raise RuntimeError('GPU job budget exhausted; confirmation reserve retained')
            record['cost']['jobs'].append(owner['job'])
            store.write(directory / 'evidence.json', record)
            store.index(directory.parent / 'index', record)
    stage = dict(status='running', argv=None, resource=resource,
                 owner=owner, started=time.time(), worker_pid=os.getpid(), node=os.uname().nodename)
    record['stages'][name] = stage
    record['status'] = 'running'
    store.write(directory / 'evidence.json', record)
    start_time = time.monotonic()
    proc = None
    try:
        stage['argv'] = _command(record, name)
        store.write(directory / 'evidence.json', record)
        if resource == 'gpu':
            stage['owner'] = scheduler.allocation()
            store.write(directory / 'evidence.json', record)
        with (directory / f'{name}.stdout').open('w') as stdout, (directory / f'{name}.stderr').open('w') as stderr:
            proc = subprocess.Popen(stage['argv'], cwd=record['repository'], stdout=stdout,
                                    stderr=stderr, start_new_session=True)
            stage['pid'] = proc.pid
            store.write(directory / 'evidence.json', record)
            stage['returncode'] = proc.wait(timeout=config.get('timeout_s', 1800))
        stage['status'] = 'completed' if stage['returncode'] == 0 else 'failed'
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        stage.update(status='interrupted', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stage['elapsed_s'] = time.monotonic() - start_time
        record['cost'][resource + '_seconds'] += stage['elapsed_s']
        record['status'] = 'ready' if stage['status'] == 'completed' else 'stopped'
        if stage['status'] != 'completed':
            record['validity'] = 'incomplete'
        store.write(directory / 'evidence.json', record)
        store.index(directory.parent / 'index', record)
    if stage['status'] == 'completed':
        if name == 'check':
            record['correctness'] = 'pass'
        record['validity'] = 'valid'
        if name == 'qualify':
            reports = list((directory / 'qualify').rglob('*.json'))
            evidence = store.read(reports[0])
            sources = record['spec']['qualification_sources']
            source_evidence = dict(incumbent_source=sources['incumbent'], candidate_source=sources['candidate'],
                                   acceptance=record['acceptance'], gate_verdict=evidence['verdict'],
                                   qualified_targets=[record['target']])
            latest = {role: dict(source, root=sources[role + '_checkout'])
                      for role, source in ((r, sources[r]) for r in ('incumbent', 'candidate'))}
            policy, _ = validate(record['spec'])
            review = promotion.applicability(source_evidence, latest['incumbent'], latest['candidate'],
                                             policy, sources['affected_targets'])
            record['promotion_review'] = review
            record['promotion'] = 'review_required' if review['applicable'] else 'blocked'
            record['qualification_evidence'] = str(reports[0])
    else:
        record['validity'] = 'incomplete'
        record['conclusion'] = 'implementation_error'
        if name == 'check':
            record['correctness'] = 'incomplete'
        if name == 'qualify':
            record['promotion'] = 'blocked'
    store.write(directory / 'evidence.json', record)


def reconcile(directory, recovered_seconds=None):
    """Archive an uncertain attempt only after its owner can be proven stopped."""
    directory = Path(directory).resolve()
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = store.read(directory / 'evidence.json')
        for name, stage in list(record['stages'].items()):
            if stage['status'] == 'completed':
                continue
            owner = stage.get('owner')
            if owner and not scheduler.released(scheduler.job_state(owner['job'])):
                raise RuntimeError('owner job is still active or unknown; do not reuse its GPU')
            if not owner and stage.get('pid'):
                if stage['node'] != os.uname().nodename:
                    raise RuntimeError('cannot verify remote CPU worker')
                try:
                    os.kill(stage['pid'], 0)
                except ProcessLookupError:
                    pass  # Positive evidence that this recorded process no longer exists.
                else:
                    raise RuntimeError('recorded CPU child is still alive')
            if stage['status'] == 'running':
                if recovered_seconds is None or recovered_seconds < 0:
                    raise RuntimeError('worker died before costing; supply recovered_seconds from accounting')
                stage.update(status='interrupted', elapsed_s=recovered_seconds,
                             cost_source='operator supplied accounting after confirmed owner termination')
                record['cost'][stage['resource'] + '_seconds'] += recovered_seconds
            attempt = directory / f'{name}.attempt{len(record.get("attempts", []))}.json'
            store.write(attempt, stage)
            record.setdefault('attempts', []).append(str(attempt))
            for suffix in ('stdout', 'stderr'):
                path = directory / f'{name}.{suffix}'
                if path.exists():
                    path.rename(attempt.with_suffix('.' + suffix))
            del record['stages'][name]
        record['status'] = 'ready'
        store.write(directory / 'evidence.json', record)
        store.index(directory.parent / 'index', record)
        return record
