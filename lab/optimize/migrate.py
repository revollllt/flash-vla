"""Import the two pre-ledger Pi campaigns without inventing missing evidence."""
import copy
import json
from pathlib import Path
import subprocess

from . import campaign, store


TARGETS = {
    'pi0': dict(model_revision='flash-vla/pi0-random-checkpoint/v1/seed-0',
                history=('expert-pi0', 'siglip', 'gemma_backbone')),
    'pi05': dict(model_revision='flash-vla/pi05-random-checkpoint/v1/seed-0',
                 history=('decoder-fusions', 'decoder-pdl-chain', 'dr-dependency-chain',
                          'dr-wide-tile', 'encoder-attn', 'encoder-qkv-rope',
                          'expert-megakernel', 'ffn-stream-continuity',
                          'ffn-taskloop-inflight', 'gu-copy-column', 'gu-decomposition',
                          'gu-ring-depth', 'mqa-cuda', 'prefix-gemm-epilogue',
                          'prefix-mqa-attn', 'sm90-short-k-gemm', 'vision-gemm-retune',
                          'vision-ln-attn', 'pi05-chunk-tail', 'siglip', 'gemma_backbone')),
}

ACCEPTED = {
    'pi0': {('siglip', 'B2'), ('siglip', 'B3'), ('gemma_backbone', 'c0')},
    'pi05': {('decoder-pdl-chain', 'c3-producer-entry-trigger'),
             ('encoder-qkv-rope', 'c1-fused-rope-epilogue'),
             ('mqa-cuda', 'c5-promote'), ('prefix-gemm-epilogue', 'c3-promote'),
             ('pi05-chunk-tail', 'tail-tabulate'), ('gemma_backbone', 'c0')},
}


def _history(root, target):
    imported = {'accepted': [], 'rejected': [], 'unclassified': [],
                'experiment_evidence': []}
    if target == 'pi05':
        path = root / 'artifacts/optimization/component-isolation-001/evidence.json'
        if path.is_file():
            imported['experiment_evidence'].append(
                {'source': str(path.relative_to(root)), 'record': store.read(path)})
    for lane in TARGETS[target]['history']:
        path = root / f'artifacts/ktasks/{lane}/candidates.jsonl'
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = dict(source=str(path.relative_to(root)), line=line_number,
                        evidence_level='legacy_import', record=row)
            status = str(row.get('status', '')).upper()
            if (lane, row.get('id')) in ACCEPTED[target]:
                imported['accepted'].append(item)
            elif 'REJECTED' in status or 'NOT PROMOTED' in status:
                imported['rejected'].append(item)
            else:
                imported['unclassified'].append(item)
    return imported


def pi_campaign(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    target = directory.name
    if target not in TARGETS:
        raise ValueError(f'legacy Pi campaign directory must be named one of {sorted(TARGETS)}')
    report_path = root / f'artifacts/campaign/600566/latency_h100_{target}.json'
    report = store.read(report_path)
    legacy = report['identity']
    if report['config'].get('seed') != 0:
        raise ValueError('the registered migration is only for the recorded seed-0 fixture')
    revision = subprocess.run(['git', 'rev-parse', legacy['revision']], cwd=root,
                              check=True, text=True, capture_output=True).stdout.strip()
    identity = dict(schema_version=2, target=legacy['target'], hardware=legacy['hardware'],
                    model=legacy['model'], model_revision=TARGETS[target]['model_revision'],
                    shape_profile=legacy['shape_profile'], shape=legacy['shape'],
                    precision=legacy['precision'], plan=legacy['plan'], engine_revision=revision)
    reference = report['deltas']['reference_leg']
    value = report['legs'][reference]['metrics']['chunk_latency']['min']
    env = report['env']
    baseline = dict(
        identity=identity,
        objective={'name': 'e2e_chunk_latency_ms', 'unit': 'ms', 'value': value},
        measurement_segment=dict(id=0, gpu_sku=env['gpu'], driver=env['driver'],
                                 cuda_runtime=env['torch_cuda'], pytorch=env['torch'],
                                 tilelang=env['tilelang'], clock_policy=env['clocks'],
                                 power_policy='not_recorded', benchmark_protocol='latency-v2',
                                 capture_regime='cuda_graph'),
        measurement_context=dict(hostname=env['node'], slurm_job_id=env['job'],
                                 timestamp=report['legs'][reference]['attribution']['started_unix']),
        evidence_level='legacy_import',
        source=dict(priority='machine-readable latency report',
                    path=str(report_path.relative_to(root)), reference_leg=reference),
        continuation=dict(reanchor_required=True,
                          reason='Identity v1 and historical environment require a fresh anchor'),
        raw_measurement=report, legacy_history=_history(root, target))
    result = campaign.create(directory, baseline, 'e2e_chunk_latency_ms', 'latency-v2',
                             'flash-vla-random-inputs-v1/seed-0')
    return dict(campaign=result, state=campaign.rebuild(directory))


def identity_v3(report, *, architecture=None):
    """Explicitly migrate a v2 report; historical latency requires revalidation."""
    from flash_vla.runtime.identity import Identity
    from flash_vla.models.pi0 import spec as pi0
    from flash_vla.models.pi05 import spec as pi05
    from flash_vla.models.lingbot import spec as lingbot

    previous = report['identity']
    if previous.get('schema_version') != 2:
        raise ValueError('identity_v3 migration requires a v2 report')
    known = {
        'hardware/nvidia/h100/pi0': ('pi0', pi0),
        'hardware/nvidia/h100/pi05': ('pi05', pi05),
        'hardware/nvidia/h100/lingbot_vla': ('lingbot-vla', lingbot),
    }
    if architecture is None:
        if previous['target'] not in known:
            raise ValueError('unknown Target: supply an explicit architecture mapping')
        model, spec = known[previous['target']]
        architecture = dict(model=model, model_revision=spec.MODEL_REVISION,
                            inference_signature=spec.INFERENCE_SIGNATURE)
    identity = Identity(
        target=previous['target'], hardware=previous['hardware'],
        model=architecture['model'], model_revision=architecture['model_revision'],
        inference_signature=architecture['inference_signature'],
        shape=previous['shape'], plan=previous['plan'],
        precision=previous['precision'], engine_revision=previous.get('engine_revision'),
    )
    result = copy.deepcopy(report)
    result['identity'] = identity.as_dict()
    context = result.setdefault('measurement_context', {})
    weights = context.setdefault('weights', {})
    old_checkpoint = previous['model_revision']
    if 'checkpoint_id' in weights and weights['checkpoint_id'] != old_checkpoint:
        raise ValueError('legacy checkpoint revision conflicts with existing provenance')
    weights['checkpoint_id'] = old_checkpoint
    # A v2 ID alone does not prove an immutable weights manifest.
    weights.setdefault('checkpoint_digest', None)
    result['legacy_identity'] = copy.deepcopy(previous)
    result['continuation'] = dict(result.get('continuation', {}),
                                  correctness_required=True, reanchor_required=True,
                                  reason='v2 checkpoint identity migrated to architecture semantics')
    return result
