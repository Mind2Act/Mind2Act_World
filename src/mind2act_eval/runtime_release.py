"""Release checks shared by CPU admission and the native worker entrypoint."""
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
    code_root = Path(__file__).parents[2]
    paths = [code_root]
    for key in ('spec', 'asset_root', 'h5', 'replay_metadata', 'adapter_config',
                'probe_config', 'gpu_allocation', 'native_exit_debug_config'):
        value = getattr(args, key, None)
        if value:
            paths.append(value)
    reject_deprecated_paths(paths)
    return code_root


@path_check_session()
def validate_release_inputs(spec, args, *, require_code_identity=False,
                            verified_code_commit=None):
    """Bind a selected release; unmarked unmarked inputs keep their defaults.

    The producer owns asset hashing. Only the selected small spec/metadata and
    RELEASE descriptor are read here, before importing a model or native SDK.
    """
    code_root = guard_input_paths(args)
    root, = reject_deprecated_paths((args.asset_root,))
    paths = [root / value for value in spec.assets.values()]
    variation = getattr(args, 'variation_manifest', None)
    if variation:
        paths.append(root / variation)
    reject_deprecated_paths(paths)
    if (root.name not in {'old15-release-v1', 'new15-release-v1'}
            and not (root / 'RELEASE.json').exists()):
        if getattr(args, 'policy_kind', None) is not None:
            raise ReleaseContractError('versioned model budgets require a frozen formal release')
        return None
    release = load_release(root, code_root=code_root,
                           verified_code_commit=verified_code_commit)
    from mind2act_eval.scene_probe import SOURCE_CLEAN_TASKS, spec_digest
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
            ('logical_slot','native_case_id','source_configuration_sha256')})
        if not isinstance(source,Mapping) or any(source.get(k) != v for k,v in expected.items()):
            raise ReleaseContractError('compiled test source identity differs from selected source index')
    limits = row.get('limits') if 'case_id' in row else None
    if getattr(args, 'policy_kind', None) is not None and limits is None:
        raise ReleaseContractError('versioned model case requires original frozen limits')
    if getattr(args, 'consumer', None) == 'scene_probe':
        # Diagnostics have their own fixed 600-tick admission, never a model score.
        # Code, case and asset identity checks below still apply unchanged.
        from .cli import _probe_inputs
        _probe_inputs(args, spec)
    elif limits is not None:
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
    clean = spec
    if spec.variation.mode == 'random':
        from mind2act_eval.variations import base_spec_digest, clean_spec
        clean = clean_spec(spec)
        if base_spec_digest(clean) != spec.variation.base_spec_sha256:
            raise ReleaseContractError('random base digest differs from selected RELEASE instance')
    descriptor = row.get('spec')
    if not isinstance(descriptor, dict) or descriptor.get('semantic_sha256') != spec_digest(clean):
        raise ReleaseContractError('clean source spec differs from selected RELEASE')
    relative = descriptor.get('path')
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ReleaseContractError('release spec path must be relative')
    declared, = reject_deprecated_paths((release.asset_root / relative,))
    if not declared.is_relative_to(release.asset_root):
        raise ReleaseContractError('release spec escapes selected asset root')
    if (not declared.is_file()
            or hashlib.sha256(declared.read_bytes()).hexdigest() != descriptor.get('sha256')):
        raise ReleaseContractError('release clean spec bytes differ from descriptor')
    # Check the declared clean spec itself, rather than trusting a supplied digest.
    from mind2act_eval.cli import read_spec
    if spec_digest(read_spec(declared)) != spec_digest(clean):
        raise ReleaseContractError('release clean spec semantic identity differs')
    if 'case_id' in row:
        from .instance_catalog import verify_case_assets
        verify_case_assets(release.asset_root, row, clean)
    metadata = getattr(args, 'replay_metadata', None)
    if getattr(args, 'h5', None) and not row.get('replay_metadata'):
        raise ReleaseContractError('expert replay requires catalog replay_metadata; this test case has no expert trajectory')
    if metadata:
        recorded = row.get('replay_metadata')
        if not isinstance(recorded, dict) or not isinstance(recorded.get('path'), str):
            raise ReleaseContractError('RELEASE has no accepted replay metadata for this instance')
        relative = Path(recorded['path'])
        target, = reject_deprecated_paths((release.asset_root / relative,))
        if relative.is_absolute() or not target.is_relative_to(release.asset_root):
            raise ReleaseContractError('release replay metadata escapes asset root')
        if (Path(metadata).resolve() != target
                or hashlib.sha256(target.read_bytes()).hexdigest() != recorded.get('sha256')):
            raise ReleaseContractError('replay metadata differs from selected RELEASE')
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
            args.replay_metadata_sha256 = metadata['sha256']
    validate_release_inputs(read_spec(path), args)
    return args
