"""Explicit, hash-pinned PRO admission; absence preserves L20 defaults.

This module uses only CPU/sysfs/proc and nvidia-smi. It never imports a GPU SDK.
The profile is configuration, not permission: a separate pinned execution receipt
and the not-before time are required before even querying the GPU.
"""
from mind2act_eval.package_paths import code_root as installed_code_root
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from .verified_files import verified_bytes, verify_sha256
from .release_contract import path_check_session

PROFILE_ENV = 'MIND2ACT_EVAL_CLUSTER_PROFILE'
PROFILE_SHA_ENV = PROFILE_ENV + '_SHA256'
NOT_BEFORE = 1793556000  # 2026-11-01 18:00 UTC / Nov 2 02:00 Asia/Shanghai
GIB = 1024 ** 3


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _pinned_bytes(path, expected):
    return verified_bytes(path, expected)


def pinned_json(path, expected):
    return json.loads(_pinned_bytes(path, expected))


def _process_policy(config):
    if config is None:
        return 'error', None, 36 * GIB, 36 * GIB * 24
    if (not isinstance(config, dict) or set(config) != {'path', 'sha256'}
            or not isinstance(config['path'], str) or not Path(config['path']).is_absolute()):
        raise ValueError('invalid pinned PRO cluster configuration')
    raw = _pinned_bytes(config['path'], config['sha256']).decode('utf-8')
    try:
        import tomllib
    except ModuleNotFoundError:  # Existing Isaac dependency on Python 3.10.
        import toml as tomllib
    plan = tomllib.loads(raw)
    supervision = plan.get('process_supervision', {})
    if (plan.get('schema') != 'mind2act.cluster.plan.v1' or plan.get('cluster_id') != 'pro5000'
            or not isinstance(supervision, dict) or set(supervision) != {'unregistered_process_policy'}
            or supervision['unregistered_process_policy'] not in ('error', 'warn')):
        raise ValueError('invalid PRO process supervision policy')
    resources = plan.get('resources', dict(rss_bytes_per_worker=36 * GIB,
                                          max_campaign_rss_bytes=36 * GIB * 24))
    if (not isinstance(resources, dict) or set(resources) != {
            'rss_bytes_per_worker', 'max_campaign_rss_bytes'}
            or any(type(v) is not int for v in resources.values())
            or not 8 * GIB <= resources['rss_bytes_per_worker'] <= 64 * GIB
            or not 8 * GIB <= resources['max_campaign_rss_bytes'] <= 960 * GIB):
        raise ValueError('invalid pinned PRO memory capacity')
    return (supervision['unregistered_process_policy'], config['sha256'],
            resources['rss_bytes_per_worker'], resources['max_campaign_rss_bytes'])


def pro_campaign_rss_budget(config, max_parallel, *, device_pool_size=1):
    """Size an explicit budget from the pinned plan and validated device pool.

    Pool capacity reserves memory without increasing the actual worker limit.
    The profile loader remains the authority for the complete device identities.
    """
    if type(max_parallel) is not int or not 1 <= max_parallel <= 24:
        raise ValueError('PRO worker count must be an integer from 1 to 24')
    if type(device_pool_size) is not int or not 1 <= device_pool_size <= 8:
        raise ValueError('PRO device pool size must be an integer from 1 to 8')
    _, _, per_worker, campaign_max = _process_policy(config)
    return min(per_worker * max(max_parallel, device_pool_size), campaign_max)


def process_identity(pid):
    """Kernel birth ticks, parent and session, parsed past a possibly spaced comm."""
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rindex(')') + 2:].split()
    return dict(pid=int(pid), birth=int(fields[19]), ppid=int(fields[1]), state=fields[0],
                session=int(fields[3]), rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))


@dataclass(frozen=True)
class ClusterProfile:
    sha256: str
    devices: tuple
    min_free_vram_mib: int
    host_reserve_bytes: int
    rss_limit_bytes: int
    storage_root: Path
    cgroup_parent: Path
    hostname: str
    machine_id_sha256: str
    first_episode: dict
    max_parallel: int = 1
    job_manifest: dict | None = None
    unregistered_process_policy: str = 'error'
    cluster_config_sha256: str | None = None
    max_workers_per_gpu: int = 1
    rss_bytes_per_worker: int = 36 * GIB
    max_campaign_rss_bytes: int = 36 * GIB * 24
    startup_capacity: int = 1
    queue_control_root: Path | None = None
    cluster_config_path: Path | None = None
    external_model_owners: dict | None = None
    shared_gpu_admission_policy: str = 'error'
    startup_host_capacity: int = 2
    startup_constructor_capacity: int = 1
    concurrency_experiment: str | None = None

    def verify_host(self):
        if socket.gethostname() != self.hostname or hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest() != self.machine_id_sha256:
            raise ValueError('PRO profile does not match the actual host/machine')
        if not self.storage_root.is_absolute() or self.storage_root.resolve() != self.storage_root:
            raise ValueError('PRO storage root must be the pinned physical NVMe prefix')
        if not self.cgroup_parent.is_relative_to(Path("/sys/fs/cgroup")) or self.cgroup_parent == Path("/sys/fs/cgroup") or self.cgroup_parent.resolve() != self.cgroup_parent:
            raise ValueError('PRO cgroup parent differs from the inspected hierarchy')

    def lock_root(self):
        return self.storage_root / f'.mind2act-eval-{os.getuid()}'

    def verify_storage(self, output, lock_root):
        self.verify_host()
        output, lock_root = Path(output).resolve(), Path(lock_root).resolve()
        if not output.is_relative_to(self.storage_root) or output == self.storage_root or output.is_relative_to(self.lock_root()):
            raise ValueError('PRO output must be inside its own runs prefix, outside lock storage')
        if lock_root != self.lock_root():
            raise ValueError('PRO requires its one stable per-host/UID lock root')

    def verify_queue_control(self, path):
        """Only an explicitly frozen control prefix may hold the shared ledger.

        This does not authorize worker outputs or substitute the canonical
        per-host launcher locks. Receipt SHA binds this opt-in root as usual.
        """
        self.verify_host()
        root = self.queue_control_root
        if (root is None or not root.is_absolute() or root.resolve() != root
                or root == self.storage_root.parent or not root.is_relative_to(self.storage_root.parent)
                or root.is_relative_to(self.storage_root) or self.storage_root.is_relative_to(root)):
            raise ValueError('PRO rolling control root must be an explicit separate sibling prefix')
        if not Path(path).resolve().is_relative_to(root):
            raise ValueError('PRO rolling control path is outside its pinned control root')
        for input_path in self.queue_input_paths():
            self.verify_control_input(input_path)

    def verify_control_input(self, path):
        root, path = self.queue_control_root.resolve(), Path(path).resolve()
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError('PRO rolling control and input domains must be disjoint')

    def queue_input_paths(self):
        """Actual immutable inputs; no Job/Spec or authorization result is cached."""
        paths = {installed_code_root()}
        for name in (PROFILE_ENV, 'MIND2ACT_PRO_INPUTS', 'MIND2ACT_PRO_CODE_MANIFEST',
                     'MIND2ACT_PRO_EXECUTION_RECEIPT'):
            if os.environ.get(name):
                paths.add(Path(os.environ[name]))
        if self.cluster_config_path is not None:
            paths.add(self.cluster_config_path)
        if self.external_model_owners is not None:
            from .pro_external_models import ExternalModelOwners
            paths.update(ExternalModelOwners(self).input_paths())
        if self.job_manifest is not None:
            paths.add(Path(self.job_manifest['path']))
            entries = pinned_json(self.job_manifest['path'], self.job_manifest['sha256'])
            if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
                raise ValueError('rolling input catalog must be bounded to 64 real cases')
            for entry in entries:
                for key in ('asset_root', 'spec', 'h5', 'replay_metadata', 'adapter_config'):
                    if entry.get(key):
                        paths.add(Path(entry[key]))
                if entry.get('variation_manifest'):
                    paths.add(Path(entry['asset_root']) / entry['variation_manifest'])
        return tuple(paths)

    def verify_job(self, job):
        from mind2act_eval.contracts import ConsumerKind, to_json
        if self.job_manifest is not None:
            if job not in self.frozen_jobs():
                raise ValueError('PRO job differs from its frozen manifest/spec/source/budget/adapter')
            return
        semantic = hashlib.sha256(json.dumps(to_json(job.spec),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (job.consumer_kind != ConsumerKind.EXPERT_REPLAY or job.wall_s != self.first_episode['wall_s'] or job.retries != 0
                or semantic != self.first_episode['semantic_spec_sha256']):
            raise ValueError('PRO first stage requires the exact original PR1/easy spec/budget')
        for flag, expected in [('--h5-sha256',self.first_episode['h5_sha256']),
                               ('--replay-metadata-sha256',self.first_episode['metadata_sha256']),
                               ('--physics-ticks',str(self.first_episode['physics_ticks'])), ('--backend','isaac')]:
            if job.argv.count(flag) != 1 or job.argv[job.argv.index(flag)+1] != expected:
                raise ValueError('PRO first-stage worker argv/input pins mismatch')

    def verify_queue_job(self, job):
        """A shared PRO queue may bind only the physical GPU, never inputs/clocks."""
        from dataclasses import replace
        if self.job_manifest is None or type(job.gpu) is not int or job.gpu not in {d['index'] for d in self.devices}:
            raise ValueError('PRO queue requires frozen jobs and an authorized physical GPU')
        for frozen in self.frozen_jobs(dynamic_gpu=True):
            argv = list(frozen.argv)
            argv[argv.index('--gpu') + 1] = str(job.gpu)
            if job == replace(frozen, gpu=job.gpu, argv=tuple(argv)):
                return
        raise ValueError('PRO queue job differs from frozen spec/source/adapter/budget')

    @path_check_session()
    def frozen_jobs(self, *, dynamic_gpu=False, _entries=None, _code_commit=None):
        """Bind standard CLI commands without opening H5 or connecting an adapter.

        The CLI/worker still hash and validate replay contents. Rechecking the
        small pinned spec/metadata/config here prevents direct API substitutions
        and keeps each original seed, clock, source and budget unchanged.
        """
        from argparse import Namespace
        from math import isfinite
        from mind2act_eval.batch import EpisodeJob
        from mind2act_eval.cli import _job_defaults, _worker_argv
        from mind2act_eval.runtime_release import guard_input_paths, validate_release_inputs
        from mind2act_eval.contracts import ConsumerKind, InstanceSpec
        from mind2act_eval.replay import ReplayMetadata

        entries = (pinned_json(self.job_manifest['path'], self.job_manifest['sha256'])
                   if _entries is None else _entries)
        if not isinstance(entries, list) or not entries:
            raise ValueError('PRO frozen job manifest must be a nonempty standard batch list')
        required = {'spec', 'spec_sha256', 'asset_root', 'backend', 'adapter',
                    'gpu', 'physics_ticks', 'wall_s', 'retries'}
        jobs, keys = [], set()
        for entry in entries:
            values = _job_defaults()
            if (not isinstance(entry, dict) or not required <= set(entry)
                    or set(entry) - (set(values) | {'episode_key'})):
                raise ValueError('PRO frozen job requires explicit standard batch options')
            values.update(entry)
            if dynamic_gpu:
                if type(values['gpu']) is not int or not 0 <= values['gpu'] <= 7:
                    raise ValueError('invalid frozen physical GPU template')
                values['gpu'] = self.devices[0]['index']
            if (values['backend'] != 'isaac' or type(values['gpu']) is not int
                    or values['gpu'] not in {d['index'] for d in self.devices}
                    or type(values['retries']) is not int or values['retries'] != 0
                    or type(values['physics_ticks']) is not int or values['physics_ticks'] <= 0
                    or type(values['wall_s']) not in (int, float)
                    or not isfinite(values['wall_s']) or values['wall_s'] <= 0
                    or any(values[k] is not None for k in ('gpu_allocation', 'gpu_allocation_sha256'))):
                raise ValueError('invalid PRO frozen GPU/budget/execution options')
            for key in ('spec', 'asset_root', 'h5', 'replay_metadata', 'adapter_config'):
                if values[key] is not None and (not isinstance(values[key], str)
                        or not Path(values[key]).is_absolute()):
                    raise ValueError('PRO frozen input paths must be explicit absolute paths')
            guard_input_paths(Namespace(**values))
            spec = InstanceSpec.from_dict(pinned_json(values['spec'], values['spec_sha256']))
            validate_release_inputs(spec, Namespace(**values),
                require_code_identity=_code_commit is not None, verified_code_commit=_code_commit)
            key = (spec.task_id, spec.instance_id, spec.difficulty)
            if key in keys:
                raise ValueError('duplicate PRO frozen episode')
            keys.add(key)
            episode_key = dict(zip(('task_id', 'instance_id', 'difficulty'), key))
            if 'episode_key' in entry and entry['episode_key'] != episode_key:
                raise ValueError('PRO frozen episode_key differs from its spec')
            if values['h5']:
                if (not _is_sha(values['h5_sha256']) or not values['replay_metadata']
                        or values['adapter_config'] is not None or values['adapter_config_sha256'] is not None):
                    raise ValueError('PRO frozen replay requires original H5/metadata pins')
                metadata = ReplayMetadata(**pinned_json(values['replay_metadata'], values['replay_metadata_sha256']))
                if spec.clock != metadata.clock or spec.source_id != metadata.source_id:
                    raise ValueError('PRO replay clock/source differs from frozen instance')
                if spec.variation.mode == 'random':
                    from mind2act_eval.variations import base_spec_digest, clean_spec
                    if metadata.instance_spec.variation.mode != 'clean':
                        raise ValueError('PRO random replay requires the exact clean expert instance')
                    clean_spec(spec).assert_binding(metadata.instance_spec)
                    if base_spec_digest(metadata.instance_spec) != spec.variation.base_spec_sha256:
                        raise ValueError('PRO random replay base digest differs from expert instance')
                else:
                    spec.assert_binding(metadata.instance_spec)
                consumer = ConsumerKind.EXPERT_REPLAY
            else:
                if (values['adapter'] != 'mind2act_eval.adapters.xpolicylab:create'
                        or not values['adapter_config'] or any(values[k] is not None
                            for k in ('h5_sha256', 'replay_metadata', 'replay_metadata_sha256'))):
                    raise ValueError('PRO frozen model requires the formal G05 adapter and config pin')
                config = pinned_json(values['adapter_config'], values['adapter_config_sha256'])
                if (not isinstance(config, dict) or config.get('transport') != 'g05-native'
                        or not _is_sha(config.get('checkpoint_sha256'))
                        or not isinstance(config.get('checkpoint'), str) or not config['checkpoint'].strip()):
                    raise ValueError('PRO G05 config must pin its native transport/checkpoint identity')
                consumer = ConsumerKind.MODEL
            jobs.append(EpisodeJob(spec, _worker_argv(Namespace(**values)), consumer,
                                   values['wall_s'], values['retries'], values['gpu'],
                                   policy_kind=values['policy_kind'], budget_version=values['budget_version'],
                                   physics_ticks=values['physics_ticks'] if values['policy_kind'] else None))
        return jobs

    def verify_jobs(self, jobs):
        if self.job_manifest is None:
            if len(jobs) != 1 or jobs[0].spec.task_id != 'PR1' or jobs[0].spec.difficulty != 'easy':
                raise ValueError('PRO first stage requires exactly one PR1/easy episode')
            self.verify_job(jobs[0])
        elif jobs != self.frozen_jobs():
            raise ValueError('PRO batch must match its complete ordered frozen job manifest')

    @path_check_session()
    def _authorize_deployment(self):
        self.verify_host()
        if time.time() < NOT_BEFORE:
            raise ValueError('PRO GPU execution is forbidden before Nov 2 02:00 Asia/Shanghai')
        path = os.environ.get('MIND2ACT_PRO_EXECUTION_RECEIPT')
        digest = os.environ.get('MIND2ACT_PRO_EXECUTION_RECEIPT_SHA256')
        if not path or not digest:
            raise ValueError('a new-stage pinned PRO execution receipt is required')
        value = pinned_json(path, digest)
        keys = {'schema', 'phase', 'task_id', 'dispatch_id', 'profile_sha256', 'code_manifest_sha256', 'inputs_sha256'}
        if (set(value) != keys or value['schema'] != 'mind2act_eval.pro_execution.v1'
                or value['phase'] != ('post_02_single_pr1' if self.job_manifest is None else 'post_02_frozen_jobs')
                or value['profile_sha256'] != self.sha256
                or not all(isinstance(value[k], str) and value[k].strip() for k in ('task_id', 'dispatch_id'))
                or value['code_manifest_sha256'] != os.environ.get('MIND2ACT_PRO_CODE_MANIFEST_SHA256')
                or value['inputs_sha256'] != os.environ.get('MIND2ACT_PRO_INPUTS_SHA256')):
            raise ValueError('PRO execution receipt identity/profile/code/stage mismatch')
        if self.job_manifest is not None and value['inputs_sha256'] != self.job_manifest['sha256']:
            raise ValueError('PRO execution receipt must bind the frozen job manifest')
        pinned_json(os.environ.get('MIND2ACT_PRO_INPUTS',''), value['inputs_sha256'])
        # The launcher verifies every source byte, and the library requires the
        # exact same pinned manifest even when called through the standard CLI.
        manifest = pinned_json(os.environ.get('MIND2ACT_PRO_CODE_MANIFEST', ''), value['code_manifest_sha256'])
        root = installed_code_root()
        if (manifest.get('schema') != 'mind2act_eval.pro_code.v1' or not manifest.get('files')
                or not isinstance(manifest.get('code_sha'), str) or len(manifest['code_sha']) != 40):
            raise ValueError('invalid PRO code manifest')
        pinned_paths = {r['path'] for r in manifest['files']}
        actual_python = {str(p.relative_to(root)) for folder in ('eval', 'task', 'env', 'env_cfg', 'policies') for p in (root / folder).rglob('*.py')}
        if not actual_python <= pinned_paths or 'eval/cluster_profile.py' not in pinned_paths:
            raise ValueError('PRO code manifest omits/introduces executable source')
        for row in manifest['files']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe PRO code manifest path')
            file_path = root / relative
            try:
                verify_sha256(file_path, row['sha256'])
            except (OSError, ValueError) as exc:
                raise ValueError('PRO deployment differs from its pinned code manifest') from exc

        return dict(code_sha=manifest['code_sha'], code_manifest_sha256=value['code_manifest_sha256'],
                    inputs_sha256=value['inputs_sha256'], profile_sha256=self.sha256,
                    task_id=value['task_id'], dispatch_id=value['dispatch_id'])

    @path_check_session()
    def authorize(self):
        identity = self._authorize_deployment()
        if self.job_manifest is not None:
            from argparse import Namespace
            from mind2act_eval.cli import _job_defaults, read_spec
            from mind2act_eval.runtime_release import validate_release_inputs
            for entry in pinned_json(self.job_manifest['path'], self.job_manifest['sha256']):
                values = _job_defaults()
                values.update(entry)
                validate_release_inputs(read_spec(values['spec']), Namespace(**values),
                    require_code_identity=True, verified_code_commit=identity['code_sha'])

        return identity


def load_cluster_profile():
    path, digest = os.environ.get(PROFILE_ENV), os.environ.get(PROFILE_SHA_ENV)
    if path is None and digest is None:
        return None
    if not path or not digest:
        raise ValueError('PRO profile path and SHA256 must be supplied together')
    data = pinned_json(path, digest)
    keys = {'schema', 'cluster', 'hostname', 'machine_id_sha256', 'storage_root',
            'cgroup_parent', 'devices', 'min_free_vram_mib', 'host_reserve_bytes',
            'rss_limit_bytes', 'max_parallel', 'not_before_unix_s', 'first_episode'}
    workers_per_gpu = data.get('max_workers_per_gpu', 1)
    startup_capacity = data.get('startup_capacity', 1)
    startup_host_capacity = data.get('startup_host_capacity', 2)
    constructor_capacity = data.get('startup_constructor_capacity', 1)
    if type(constructor_capacity) is not int or constructor_capacity not in (1, 2, 3):
        raise ValueError('PRO constructor capacity must be an explicit integer 1..3')
    if constructor_capacity > 1 and (startup_capacity != 2
                                     or type(startup_host_capacity) is not int
                                     or startup_host_capacity < constructor_capacity):
        raise ValueError('PRO constructor concurrency requires parallel initialization and sufficient host slots')
    if type(startup_capacity) is not int or startup_capacity not in (1, 2):
        raise ValueError('PRO startup capacity must be an explicit integer 1 or 2')
    if (type(startup_host_capacity) is not int or not 1 <= startup_host_capacity <= 8
            or 'startup_host_capacity' in data and startup_capacity != 2):
        raise ValueError('PRO host startup capacity requires integer 1..8 and parallel initialization')
    experiment = data.get('concurrency_experiment')
    if experiment is not None and experiment != 'g05-clean-concurrency-4-6-8-v1':
        raise ValueError('unsupported PRO concurrency experiment')
    if experiment is not None and (startup_capacity != 2
                                    or startup_host_capacity not in ((2, 3) if constructor_capacity > 1 else (2,))
                                    or len(data['devices']) != 1):
        raise ValueError('concurrency experiment requires one GPU and its bounded SDK startup slots')
    allowed_workers = (1, 2, 3, 4, 6, 8) if experiment else (1, 2, 3)
    if type(workers_per_gpu) is not int or workers_per_gpu not in allowed_workers:
        raise ValueError('PRO max_workers_per_gpu exceeds its explicit policy')
    if (not keys <= set(data) or set(data) - (keys | {'job_manifest', 'cluster_config', 'max_workers_per_gpu', 'startup_capacity', 'startup_host_capacity', 'startup_constructor_capacity', 'concurrency_experiment', 'queue_control_root', 'external_model_owners', 'shared_gpu_admission_policy'})
            or data['schema'] != 'mind2act_eval.cluster_profile.v1'
            or data['cluster'] != 'pro5000-single-host' or type(data['max_parallel']) is not int
            or not 1 <= data['max_parallel'] <= 8 * workers_per_gpu
            or data['not_before_unix_s'] != NOT_BEFORE):
        raise ValueError('invalid or unsupported PRO cluster profile')
    shared_policy = data.get('shared_gpu_admission_policy', 'error')
    if shared_policy not in ('error', 'warn'):
        raise ValueError('invalid PRO shared GPU admission policy')
    manifest = data.get('job_manifest')
    if manifest is not None:
        if (not isinstance(manifest, dict) or set(manifest) != {'path', 'sha256'}
                or not isinstance(manifest['path'], str) or not Path(manifest['path']).is_absolute()
                or not _is_sha(manifest['sha256'])):
            raise ValueError('invalid pinned PRO job manifest')
    elif data['max_parallel'] != 1:
        raise ValueError('PRO parallel admission requires a frozen job manifest')
    external_models = data.get('external_model_owners')
    if external_models is not None:
        from .pro_external_models import _path_pin
        if not _path_pin(external_models):
            raise ValueError('external model owner receipt must be explicitly pinned')
    if (not isinstance(data['hostname'], str) or not data['hostname'].strip()
            or not _is_sha(data['machine_id_sha256'])):
        raise ValueError('explicit inspected host identity is required')
    for key in ('storage_root', 'cgroup_parent'):
        path_value = Path(data[key])
        if not path_value.is_absolute() or '..' in path_value.parts or str(path_value) != data[key]:
            raise ValueError('profile roots must be canonical absolute paths')
    control_root = data.get('queue_control_root')
    if control_root is not None:
        if (not isinstance(control_root, str) or not Path(control_root).is_absolute()
                or '..' in Path(control_root).parts or str(Path(control_root)) != control_root
                or manifest is None):
            raise ValueError('rolling control root requires a canonical path and frozen catalog')
    first = data['first_episode']
    first_keys = {'spec_sha256','semantic_spec_sha256','metadata_sha256','h5_sha256',
                  'instance_id','seed','physics_ticks','wall_s','release_sha256','release_tag'}
    if (not isinstance(first, dict) or set(first) != first_keys
            or any(not _is_sha(first[k]) for k in first_keys if k.endswith('sha256'))
            or not isinstance(first['instance_id'], str) or not first['instance_id']
            or not isinstance(first['release_tag'], str) or not first['release_tag']
            or type(first['seed']) is not int or first['seed'] != 0
            or type(first['physics_ticks']) is not int or first['physics_ticks'] != 60000
            or type(first['wall_s']) is not int or first['wall_s'] != 7200):
        raise ValueError('invalid exact first-episode source/budget bindings')
    policy, config_sha, rss_per_worker, campaign_max = _process_policy(data.get('cluster_config'))
    for key, minimum, maximum in [('min_free_vram_mib', 24576, 73728),
                                  ('host_reserve_bytes', 64 * GIB, 256 * GIB)]:
        if type(data[key]) is not int or not minimum <= data[key] <= maximum:
            raise ValueError('invalid PRO resource limit: ' + key)
    devices = data['devices']
    if not isinstance(devices, list) or not devices:
        raise ValueError('explicit PRO devices are required')
    if any(not isinstance(d, dict) or set(d) != {'index', 'uuid', 'pci_bus_id'}
           or type(d['index']) is not int or not 0 <= d['index'] <= 7
           or not isinstance(d['uuid'], str) or not d['uuid'].startswith('GPU-')
           or not isinstance(d['pci_bus_id'], str) for d in devices):
        raise ValueError('invalid PRO device identity')
    for key in ('index', 'uuid', 'pci_bus_id'):
        if len({d[key] for d in devices}) != len(devices):
            raise ValueError('duplicate PRO device identity')
    if data['max_parallel'] > len(devices) * workers_per_gpu:
        raise ValueError('PRO parallel count exceeds its pinned device pool')
    if startup_capacity > data['max_parallel']:
        raise ValueError('PRO startup capacity exceeds authorized parallel slots')
    # Count only the complete, unique, in-range pool from the pinned profile.
    # Reserving pool memory never changes max_parallel or creates extra jobs.
    rss_max = min(campaign_max, rss_per_worker * max(data['max_parallel'], len(devices)))
    if type(data['rss_limit_bytes']) is not int or not 8 * GIB <= data['rss_limit_bytes'] <= rss_max:
        raise ValueError('invalid PRO resource limit: rss_limit_bytes')
    profile = ClusterProfile(digest, tuple(devices), data['min_free_vram_mib'],
                             data['host_reserve_bytes'], data['rss_limit_bytes'],
                             Path(data['storage_root']), Path(data['cgroup_parent']),
                             data['hostname'], data['machine_id_sha256'], dict(first),
                             data['max_parallel'], manifest, policy, config_sha, workers_per_gpu,
                             rss_bytes_per_worker=rss_per_worker, max_campaign_rss_bytes=campaign_max,
                             startup_capacity=startup_capacity,
                             startup_host_capacity=startup_host_capacity,
                             startup_constructor_capacity=constructor_capacity,
                             concurrency_experiment=experiment,
                             queue_control_root=None if control_root is None else Path(control_root),
                             cluster_config_path=None if not data.get('cluster_config') else Path(data['cluster_config']['path']),
                             external_model_owners=external_models,
                             shared_gpu_admission_policy=shared_policy)
    profile.verify_host()
    if profile.queue_control_root is not None:
        profile.verify_queue_control(profile.queue_control_root)
    return profile


def _query(arguments):
    return subprocess.run(['nvidia-smi', *arguments], check=True, capture_output=True,
                          text=True, timeout=15).stdout


def _device_pids(index):
    return _host_device_pids().get(str(index), set())


def _host_device_pids():
    """Include graphics/compute device users, not just compute-apps queries."""
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith('/dev/nvidia') and target[11:].isdecimal():
                        found.setdefault(target[11:], set()).add(int(proc.name))
                except FileNotFoundError:
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Permission failures are deliberately not converted to an empty set.
    return found


def _context_pids(profile, index, expected, *, xml=None):
    """Read all compute/graphics contexts, including idle contexts, from NVML.

    An open device descriptor alone also describes persistence and monitoring
    clients. Only a complete, identity-checked context table can distinguish it
    from actual GPU ownership; malformed/unsupported tables never mean idle.
    """
    root = ET.fromstring(_query(['-q', '-x', '-i', str(index)]) if xml is None else xml)
    devices = root.findall('gpu')
    if xml is not None:
        devices = [d for d in devices if d.findtext('uuid') == expected['uuid']]
    if len(devices) != 1 or devices[0].findtext('uuid') != expected['uuid']:
        raise ValueError('PRO context snapshot GPU identity mismatch')
    table = devices[0].find('processes')
    if table is None or (table.text or '').strip():
        raise ValueError('PRO requires a complete supported GPU context table')
    pids = set()
    for process in table:
        raw = process.findtext('pid')
        if process.tag != 'process_info' or raw is None or not raw.isdecimal() or int(raw) <= 0:
            raise ValueError('invalid PRO GPU context process identity')
        pids.add(int(raw))
    return pids


def check_pro_shared_gpu(profile, index, *, admitted, _epoch=None, _snapshot=None):
    if profile.max_workers_per_gpu <= 1:
        raise ValueError('PRO sharing requires explicit max_workers_per_gpu')
    capacity_warn = profile.shared_gpu_admission_policy == 'warn'
    # Operator warn preserves the original process policy without requiring a
    # model-owner chain; strict admission still requires verified ownership.
    receipt = check_pro_gpu(profile, index, admitted=admitted, require_idle=not capacity_warn,
                           _epoch=_epoch, _snapshot=_snapshot)
    device = receipt['device']
    shortage = (device['free_mib'] < profile.min_free_vram_mib
                or not receipt['processes'] and device['utilization'] >= 10)
    if capacity_warn and shortage:
        receipt['capacity_warnings'] = [dict(
            reason='PRO shared GPU lacks registered sibling or VRAM headroom',
            policy='warn', gpu_index=index, free_mib=device['free_mib'],
            required_free_mib=profile.min_free_vram_mib, utilization=device['utilization'],
            registered_processes=len(receipt['processes']))]
    return receipt


def check_pro_gpu(profile, index, *, admitted=None, require_idle=True, _epoch=None, _snapshot=None):
    # Only CampaignResources supplies its live epoch's verified identity. Direct
    # callers retain the full admission path before any GPU query.
    if _epoch is None:
        authorization = profile.authorize()
    else:
        if profile != _epoch.profile:
            raise ValueError('PRO GPU profile differs from owned epoch')
        authorization = _epoch.admission.authorization(_epoch)
    expected = next((d for d in profile.devices if d['index'] == index), None)
    if expected is None:
        raise ValueError('GPU is outside the pinned PRO device allowlist')
    from .pro_external_models import ExternalModelOwners
    model_owners = ExternalModelOwners(profile) if _epoch is None else _epoch.admission.models
    models = model_owners.snapshot(index)  # fresh birth/UID/argv; never a cached verdict
    admitted = dict(admitted or {})
    for pid, current in models.items():
        if pid in admitted and admitted[pid] != current['birth']:
            raise ValueError('external model conflicts with admitted process birth')
        admitted[pid] = current['birth']
    if _snapshot is not None:
        from .pro_gpu_snapshot import validate_age
        validate_age(_snapshot)
    rows = (_query(['--query-gpu=index,name,uuid,pci.bus_id,memory.free,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits']) if _snapshot is None else _snapshot['devices'])
    matched = []
    for line in rows.splitlines():
        gpu, name, uuid, pci, free, used, util = [v.strip() for v in line.split(',')]
        if int(gpu) == index:
            matched.append(dict(index=int(gpu), name=name, uuid=uuid, pci_bus_id=pci.lower(),
                                free_mib=int(free), used_mib=int(used), utilization=int(util)))
    if len(matched) != 1:
        raise ValueError('selected PRO GPU is missing/ambiguous')
    device = matched[0]
    def pci_key(value):
        domain, bus, slot = value.lower().rsplit(':', 2)
        return int(domain, 16), int(bus, 16), slot
    if ('RTX PRO 5000' not in device['name'] or device['uuid'] != expected['uuid']
            or pci_key(device['pci_bus_id']) != pci_key(expected['pci_bus_id'])):
        raise ValueError('PRO GPU index/UUID/PCI/name identity mismatch')
    if require_idle and device['free_mib'] < profile.min_free_vram_mib:
        raise ValueError('PRO GPU lacks fresh idle/VRAM headroom')
    fd_pids = (_device_pids(index) if _snapshot is None
               else set(_snapshot['fd_pids'].get(str(index), [])))
    contexts = (_context_pids(profile, index, expected) if _snapshot is None
                else _context_pids(profile, index, expected, xml=_snapshot['contexts']))
    if not set(models) <= contexts:
        raise ValueError('pinned external model is absent from the complete GPU context table')
    pids = set(contexts)
    compute = (_query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
               if _snapshot is None else _snapshot['compute'])
    for line in compute.splitlines():
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid == expected['uuid']:
            pids.add(int(pid))
    # NVML's complete compute+graphics table and the independent compute/FD
    # checks above are mandatory. Only this redundant diagnostic may fail.
    diagnostic_warnings = []
    try:
        # The host snapshot already includes complete graphics/compute plus
        # the independent compute and FD tables. pmon adds no ownership proof.
        for line in (_query(['pmon', '-c', '1', '-s', 'um']) if _snapshot is None else '').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if int(fields[0]) == index and fields[1] != '-':
                pid = int(fields[1])
                if pid <= 0:
                    raise ValueError('invalid pmon PID')
                pids.add(pid)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        diagnostic_warnings.append(dict(reason='redundant pmon diagnostic unavailable',
            detail=type(exc).__name__, gpu_index=index))
    descriptor_only = []
    for pid in sorted(fd_pids - pids):
        try:
            descriptor_only.append(process_identity(pid))
        except (FileNotFoundError, ProcessLookupError):
            continue
    identities, process_warnings = [], []
    for pid in sorted(pids):
        try:
            identity = process_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            if pid in models:
                raise ValueError('pinned external model exited during GPU sampling')
            continue
        if _snapshot is not None and _snapshot['births'].get(str(pid)) != identity['birth']:
            raise ValueError('GPU snapshot process PID/birth changed; resample required')
        if pid in models and identity['birth'] != models[pid]['birth']:
            raise ValueError('pinned external model birth changed during GPU sampling')
        if admitted is None or admitted.get(pid) != identity['birth']:
            if profile.unregistered_process_policy != 'warn':
                raise ValueError(f'foreign/unregistered GPU process {pid} birth={identity["birth"]}')
            warning = dict(schema='mind2act_eval.pro_gpu_process_warning.v1',
                reason='foreign/unregistered GPU process', policy='warn',
                profile_sha256=profile.sha256, cluster_config_sha256=profile.cluster_config_sha256,
                gpu_index=index, process=identity, observed_unix_s=time.time(),
                ownership='unverified; observation only, never admitted or signalled')
            process_warnings.append(warning)
            logging.getLogger(__name__).warning(
                'PRO foreign/unregistered GPU process %s birth=%s on GPU %s; policy=warn; '
                'observation only, never admitted or signalled', pid, identity['birth'], index)
            continue
        identities.append(identity)
    if require_idle and process_warnings:
        raise ValueError('PRO launch requires complete registered sibling ownership; foreign/unregistered GPU process')
    if require_idle and device['utilization'] >= 10:
        diagnostic_warnings.append(dict(reason='utilization with verified GPU ownership',
            gpu_index=index, utilization=device['utilization'], registered_contexts=len(identities),
            qualification='complete compute/graphics ownership, device identity and free VRAM verified'))
    if _snapshot is not None:
        validate_age(_snapshot)  # Ownership verification cannot extend the telemetry TTL.
    return dict(device=device, processes=[p for p in identities if p['pid'] not in models],
                external_model_processes=list(models.values()),
                external_model_owners_sha256=None if model_owners.pin is None else model_owners.pin['sha256'],
                descriptor_only_observations=descriptor_only,
                process_warnings=process_warnings,
                diagnostic_warnings=diagnostic_warnings,
                context_source="complete NVML compute/graphics table plus compute/FD",
                sampled_unix_s=time.time() if _snapshot is None else _snapshot["sampled_unix_s"],
                snapshot_age_s=0 if _snapshot is None else time.monotonic() - _snapshot["sampled_monotonic_s"],
                authorization=authorization,
                qualification='bounded observation; ownership reverified, not reservation against external launchers')
