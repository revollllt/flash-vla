"""Experiment inputs and read-only acceptance adaptation."""
from pathlib import Path
from eval import acceptance

STAGES = ('preflight', 'probe', 'check', 'measure', 'qualify')
KINDS = ('performance', 'refactor', 'diagnostic_probe', 'control')
WEIGHT_DEPENDENCIES = ('invariant', 'rebuild', 'retune', 'checkpoint_specific')


def validate_applicability(spec):
    """Require executable recovery recipes for weight-dependent v3 candidates."""
    dependency = spec.get('applicability', {}).get('weight_dependency')
    if dependency not in WEIGHT_DEPENDENCIES:
        raise ValueError('applicability.weight_dependency must be invariant, rebuild, '
                         'retune or checkpoint_specific')
    recipe_name = {'rebuild': 'artifact_recipe', 'retune': 'retune_recipe'}.get(dependency)
    if recipe_name:
        recipe = spec.get(recipe_name)
        if not isinstance(recipe, dict):
            raise ValueError(f'{dependency} requires {recipe_name}')
        validate_command(recipe_name, recipe)
    return dependency


def validate_command(name, command):
    """Validate a recorded CPU/GPU command before allocating any work."""
    argv = command.get('argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError(f'{name}: argv must be a nonempty string list')
    if type(command.get('timeout_s')) is not int or command['timeout_s'] < 1:
        raise ValueError(f'{name}: positive timeout_s required')
    if command.get('resource') not in ('cpu', 'gpu'):
        raise ValueError(f'{name}: resource must be cpu or gpu')


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'expected repository-relative path: {value}')
    return path


def validate(spec):
    required = ('version', 'id', 'target', 'kind', 'hypothesis', 'scope', 'inputs', 'stages')
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f'missing experiment fields: {missing}')
    if 'options' in spec.get('conditions', {}):
        raise ValueError('move conditions.options to spec.options before starting or resuming')
    if spec['version'] != 1 or spec['kind'] not in KINDS:
        raise ValueError('unsupported experiment version or kind')
    if not spec['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in spec['id']):
        raise ValueError('experiment id must be a path-safe label')
    for key in ('mechanism', 'alternative', 'falsifier', 'cheapest_probe'):
        if not spec['hypothesis'].get(key):
            raise ValueError(f'missing hypothesis.{key}')
    if spec.get('identity', {}).get('schema_version') == 3 and spec['kind'] == 'performance':
        validate_applicability(spec)
    policy = acceptance.for_target(spec['target'])
    if policy['budget'] is None:
        raise ValueError(f'no acceptance budget for {spec["target"]}')
    budget = dict(policy['budget'])
    for key, value in spec.get('budget', {}).items():
        if key not in budget or type(value) is not int or not 0 < value <= budget[key]:
            raise ValueError(f'unsupported or excessive budget: {key}={value}')
        budget[key] = value
    allowed = [relative_path(p) for p in spec['scope']]
    if not allowed:
        raise ValueError('empty allowed scope')
    for value in spec.get('changes', []):
        path = relative_path(value)
        if path in (Path('eval/acceptance.py'), Path('eval/gate.py')):
            raise ValueError('candidate cannot modify its evaluator')
        if not any(path == prefix or prefix in path.parents for prefix in allowed):
            raise ValueError(f'change outside scope: {value}')
    for value in spec['inputs']:
        relative_path(value)
    stages = spec['stages']
    if not stages or any(name not in STAGES[1:] for name in stages):
        raise ValueError('unknown or empty stage selection')
    if ('measure' in stages or 'qualify' in stages) and 'check' not in stages:
        raise ValueError('measurement requires a correctness stage')
    for name, stage in stages.items():
        if name == 'qualify':
            continue
        validate_command(name, stage)
    return policy, budget
