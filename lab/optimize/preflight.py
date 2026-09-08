"""CPU route resolution; equal plans alone never prove an implementation no-op."""
from benchmarks.targets import declare
from .schema import validate


def inspect(spec):
    policy, budget = validate(spec)
    candidate = declare(spec['target'], spec.get('candidate', 'shipped'), **spec.get('options', {}))
    incumbent = declare(spec['target'], spec.get('incumbent', 'shipped'), **spec.get('options', {}))
    changed = sorted(k for k, v in candidate.identity.plan.items() if incumbent.identity.plan.get(k) != v)
    return dict(acceptance=policy, budget=budget, workload=candidate.identity.as_dict(),
                effective_routes=dict(candidate.identity.plan), changed_routes=changed,
                atomic_groups=[sorted(g) for g in candidate.atomic_groups], inputs=spec['inputs'],
                implementation_identity='unknown until artifacts are actually loaded',
                no_op='unknown', protocol=spec.get('protocol', 'unspecified; diagnostic only'))
