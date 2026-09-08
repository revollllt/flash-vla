"""Slurm is the GPU owner; expiry alone never proves a resource is free."""
import os
import subprocess


def command(argv):
    return subprocess.run(argv, text=True, capture_output=True, check=True).stdout.strip()


def job_state(job):
    queued = command(['squeue', '-h', '-j', str(job), '-o', '%T'])
    if queued:
        return queued.splitlines()[0]
    accounting = command(['sacct', '-X', '-n', '-j', str(job), '--format=State', '--parsable2'])
    return accounting.splitlines()[0].split('|')[0].split()[0] if accounting else 'UNKNOWN'


def released(state):
    return state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')


def allocation():
    job = os.environ.get('SLURM_JOB_ID')
    if not job or job_state(job) != 'RUNNING':
        raise RuntimeError('GPU stage needs an active Slurm allocation')
    visible = command(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']).splitlines()
    if len(visible) != 1:
        raise RuntimeError('this worker needs exactly one GPU visible in its allocation')
    identity = visible[0]
    gpu = identity.split(',')[0].strip()
    processes = command(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader'])
    if processes:
        raise RuntimeError(f'allocated GPU has existing compute processes: {processes}')
    return dict(job=job, gpu=identity, node=os.uname().nodename,
                slurm_gpu_ids=os.environ.get('SLURM_JOB_GPUS'))


def conflicts(a, b):
    """Declared paths and atomic groups, not a guessed global dependency graph."""
    from pathlib import Path
    paths = []
    for left in a.get('changes', []):
        for right in b.get('changes', []) + b.get('inputs', []):
            x, y = Path(left), Path(right)
            if x == y or x in y.parents or y in x.parents:
                paths.append((left, right))
    for right in b.get('changes', []):
        for left in a.get('inputs', []):
            if left == right:
                paths.append((left, right))
    groups = sorted(set(a.get('atomic_groups', [])) & set(b.get('atomic_groups', [])))
    return dict(paths=paths, atomic_groups=groups, conflicting=bool(paths or groups))
