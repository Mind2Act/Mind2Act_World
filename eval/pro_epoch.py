"""Immutable admission for one live, owned PRO CampaignResources epoch.

Root-owned releases/code/profile/receipts remain immutable for that epoch by
deployment contract. This is deliberately not a process-global PASS cache:
every new scope reauthorizes, and selected input bytes are checked at launch.
Only serialized bytes are retained; callers cannot mutate a cached Job/Spec.
"""
from dataclasses import replace
import json
import os
from types import MappingProxyType

from .contracts import to_json

_MAX_INDEX_BYTES = 32 * 1024**2


def _bytes(value):
    return json.dumps(to_json(value), sort_keys=True, separators=(',', ':')).encode()


def _key(job):
    return job.spec.task_id, job.spec.instance_id, job.spec.difficulty


class ProEpochAdmission:
    def __init__(self, owner):
        from .cluster_profile import pinned_json
        self._owner = owner
        self._directory = owner.directory
        self._pid = os.getpid()
        self._profile = _bytes(owner.profile)
        self._environment = self._env()
        identity = owner.profile._authorize_deployment()
        self._authorization = _bytes(identity)
        from .pro_external_models import ExternalModelOwners
        self.models = ExternalModelOwners(owner.profile)
        entries = []
        if owner.profile.job_manifest is not None:
            pin = owner.profile.job_manifest
            entries = pinned_json(pin['path'], pin['sha256'])
            if not isinstance(entries, list) or not 1 <= len(entries) <= 4096:
                raise ValueError('PRO epoch requires a bounded frozen job catalog')
        # Dynamic queues may use templates from another authorized host's pool.
        # Normalize only GPU for the index; all other fields remain exact.
        jobs = owner.profile.frozen_jobs(dynamic_gpu=True, _entries=entries,
            _code_commit=identity['code_sha']) if entries else []
        records, retained = {}, 0
        for entry, job in zip(entries, jobs):
            record = _bytes(entry), _bytes(job)
            retained += sum(map(len, record))
            if retained > _MAX_INDEX_BYTES:
                raise ValueError('PRO epoch frozen job index exceeds its byte bound')
            records[_key(job)] = record
        self._records = MappingProxyType(records)

    @staticmethod
    def _env():
        return tuple(sorted((k, v) for k, v in os.environ.items()
            if k.startswith(('MIND2ACT_PRO_', 'MIND2ACT_EVAL_CLUSTER_PROFILE'))))

    def authorization(self, owner):
        if (owner is not self._owner or owner.directory != self._directory
                or not owner.epoch_active or os.getpid() != self._pid
                or self._env() != self._environment or _bytes(owner.profile) != self._profile):
            raise ValueError('PRO admission is not this live owned epoch/profile/execution identity')
        return dict(json.loads(self._authorization), epoch_id=owner.directory.name)

    def verify_job(self, job, *, dynamic_gpu=False):
        from .cluster_profile import verify_sha256
        from .cli import _job_defaults
        self.authorization(self._owner)
        profile = self._owner.profile
        if profile.job_manifest is None:
            profile.verify_job(job)
            return
        record = self._records.get(_key(job))
        if record is None or type(job.gpu) is not int or job.gpu not in {d['index'] for d in profile.devices}:
            raise ValueError('PRO job differs from frozen epoch identity/device')
        entry = json.loads(record[0])
        # Rebind exactly one GPU field and the corresponding standard argv.
        normalized = job
        if dynamic_gpu:
            argv = list(job.argv)
            if argv.count('--gpu') != 1:
                raise ValueError('PRO queue job differs from frozen GPU argv')
            argv[argv.index('--gpu') + 1] = str(profile.devices[0]['index'])
            normalized = replace(job, gpu=profile.devices[0]['index'], argv=tuple(argv))
        else:
            template = json.loads(record[1])
            template['gpu'] = entry['gpu']
            template['argv'][template['argv'].index('--gpu') + 1] = str(entry['gpu'])
            if _bytes(job) != _bytes(template):
                raise ValueError('PRO job differs from frozen spec/source/adapter/budget')
        if dynamic_gpu and _bytes(normalized) != record[1]:
            raise ValueError('PRO queue job differs from frozen spec/source/adapter/budget')
        identity = json.loads(self._authorization)
        selected = profile.frozen_jobs(dynamic_gpu=dynamic_gpu, _entries=[entry],
            _code_commit=identity['code_sha'])[0]
        if dynamic_gpu:
            argv = list(selected.argv)
            argv[argv.index('--gpu') + 1] = str(job.gpu)
            selected = replace(selected, gpu=job.gpu, argv=tuple(argv))
        if selected != job:
            raise ValueError('PRO selected inputs differ from frozen epoch job')
        # Replay data is too large for the index. Hash actual bytes through the
        # verified-file reader (which always checks current inode identity).
        values = _job_defaults()
        values.update(entry)
        self.models.verify_job(values, job.gpu)
        if values['h5']:
            verify_sha256(values['h5'], values['h5_sha256'])

