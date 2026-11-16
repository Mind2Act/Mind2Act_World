"""Release checks shared by CPU admission and the native worker entrypoint."""
from mind2act_eval.package_paths import code_root as installed_code_root
import hashlib
from collections.abc import Mapping
from pathlib import Path

from mind2act_eval.release_contract import (
    ReleaseContractError, load_release, reject_deprecated_paths, path_check_session,
)

NEW15_SOURCE_CLEAN_TASKS = frozenset((
    'E2', 'S2', 'P2', 'F2', 'PR4', 'PR5', 'PR6', 'DR3', 'DR4',
    'CP06', 'CP07', 'CP08', 'CP09', 'CP10', 'M2',
))


def guard_input_paths(args):
    # Preserve the original spelling until the shared guard has checked aliases.
    code_root = installed_code_root()
    paths = [code_root]
    for key in ('spec', 'asset_root', 'h5', 'replay_metadata', 'adapter_config',
                'probe_config', 'native_exit_debug_config'):
        value = getattr(args, key, None)
        if value:
            paths.append(value)
    reject_deprecated_paths(paths)
    return code_root


@path_check_session()
def validate_release_inputs(spec, args, *, require_code_identity=False,
                            verified_code_commit=None):
    """Bind a selected release; unmarked legacy L20 inputs keep their defaults.

    The producer owns asset hashing. Only the selected small spec/metadata and
    RELEASE descriptor are read here, before importing a model or native SDK.
    """
    from mind2act_eval.public_layouts import validate
    validate(spec)
    code_root = guard_input_paths(args)
    root, = reject_deprecated_paths((args.asset_root,))
    paths = [root / value for value in spec.assets.values()]
    variation = getattr(args, 'variation_manifest', None)
    if variation:
        paths.append(root / variation)
    reject_deprecated_paths(paths)
    if not getattr(args, 'case_id', None) and getattr(args, 'policy_kind', None) is None:
        return None
    if (root.name not in {'old15-release-v1', 'new15-release-v1'}
            and not (root / 'RELEASE.json').exists()):
        if getattr(args, 'policy_kind', None) is not None:
            raise ReleaseContractError('versioned model budgets require a frozen formal release')
        return None
    release = load_release(root, code_root=code_root,
                           verified_code_commit=verified_code_commit)
    from mind2act_eval.spec_identity import spec_digest
    SOURCE_CLEAN_TASKS = frozenset(("CP01", "CP02", "CP03", "CP04", "CP05", "DR1", "DR2", "E1", "F1", "M1", "P1", "PR1", "PR2", "PR3", "S1"))
    task_grids = {'old15-release-v1': SOURCE_CLEAN_TASKS,
                  'new15-release-v1': NEW15_SOURCE_CLEAN_TASKS}
    if (release.asset_name not in task_grids
            or {row['task_id'] for row in release.instances} != task_grids[release.asset_name]
            or release.code_commit is None):
        raise ReleaseContractError('formal release requires its exact task grid and code_commit')
    if require_code_identity and not release.code_identity_checked:
        raise ReleaseContractError('formal release requires an actual SOURCE_COMMIT or verified code identity')
    key = (spec.task_id, spec.instance_id, spec.difficulty)
    requested = getattr(args, 'case_id', None)
    if requested:
        row = release.resolve_case(requested)
        if (row['task_id'], row['instance_id'], row['difficulty']) != key:
            raise ReleaseContractError('requested case_id task/instance/difficulty differs from its compiled spec')
    else:
        row = next((r for r in (*release.catalog, *release.instances) if
                    (r['task_id'], r['instance_id'], r['difficulty']) == key), None)
    if row is None:
        raise ReleaseContractError('instance is absent from the release catalog and replay instances')
    if 'seed' in row and spec.seed != row['seed']:
        raise ReleaseContractError('compiled spec seed differs from selected release case')
    if 'source_configuration_sha256' in row:
        source = spec.config.get('test_source')
        if (spec.source_id == 'cp07-training-layout-v1'
                and spec.config.get('cp07_mode') == 'train_layout_test'):
            from .cp07_layout_identity import validate_identity
            validate_identity(spec)
            source = spec.config['fixed50_origin']
        expected = dict(instance_set='test50-r5',split='test',**{k:row[k] for k in
            ('logical_slot','native_case_id')})
        if not isinstance(source,Mapping) or any(source.get(k) != v for k,v in expected.items()):
            raise ReleaseContractError('compiled test source identity differs from selected source index')
    limits = row.get('limits') if 'case_id' in row else None
    if getattr(args, 'policy_kind', None) is not None and limits is None:
        raise ReleaseContractError('versioned model case requires original frozen limits')
    if limits is not None:
        from .budgets import selected_limits, validate_execution_limits
        kind = getattr(args, 'policy_kind', None)
        try:
            limits = selected_limits(row, kind, getattr(args, 'budget_version', None))
            # Preserve wall extensions only for legacy jobs. Explicit protocols
            # use the uniform infrastructure allowance.
            validate_execution_limits(args, limits, allow_longer_wall=kind is None)
        except ValueError as exc:
            message = str(exc) if kind else 'execution limits differ from selected case original release budgets'
            raise ReleaseContractError(message) from exc

    return release


def resolve_case_arguments(args):
    """Shared CLI/consumer seam for exact --case-id selection."""
    case_id = getattr(args, 'case_id', None)
    if not case_id:
        if not getattr(args, 'spec', None):
            raise ReleaseContractError('provide --case-id or --spec')
        return args
    release = load_release(args.asset_root, code_root=guard_input_paths(args))
    row = release.resolve_case(case_id)
    if getattr(args, 'policy_kind', None) is not None:
        from .budgets import selected_limits
        try:
            limits = selected_limits(row, args.policy_kind, getattr(args, 'budget_version', None))
        except ValueError as exc:
            raise ReleaseContractError(str(exc)) from exc
        for key, value in limits.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    path = release.asset_root / row['spec']['path']
    from .cli import read_spec
    if getattr(args, 'spec', None) and Path(args.spec).resolve() != path.resolve():
        # An overlay has its own file; the shared validator below must still
        # bind its complete clean source to this exact catalog case.
        supplied = Path(args.spec)
        if read_spec(supplied).variation.mode != 'random':
            raise ReleaseContractError('--case-id and --spec select different instances')
        path = supplied
    args.spec = str(path)
    if getattr(args, 'h5', None):
        metadata = row.get('replay_metadata')
        if not metadata:
            raise ReleaseContractError('expert replay requires replay_metadata for the selected case_id')
        if not getattr(args, 'replay_metadata', None):
            args.replay_metadata = str(release.asset_root / metadata['path'])
    validate_release_inputs(read_spec(path), args)
    return args
