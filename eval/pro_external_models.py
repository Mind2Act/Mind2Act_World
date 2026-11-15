"""Root-reviewed external G05 ownership, without adopting model processes.

Receipt/control evidence is immutable for one epoch. Process identity and GPU
membership are never cached. Only argv hashes enter receipts, never argv bytes.
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType


def _identity(pid):
    from .cluster_profile import process_identity
    before = process_identity(pid)
    uid_rows = [line.split()[1:] for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                if line.startswith('Uid:')]
    with Path(f'/proc/{pid}/cmdline').open('rb') as stream:
        argv = stream.read(1024**2 + 1)
    after = process_identity(pid)
    if (len(uid_rows) != 1 or len(uid_rows[0]) != 4
            or any(not value.isdecimal() for value in uid_rows[0])
            or not argv or len(argv) > 1024**2 or before['birth'] != after['birth']
            or after.get('state') in {'Z', 'X'}):
        raise ValueError('external model process identity is incomplete/changed')
    return dict(after, uids=tuple(map(int, uid_rows[0])),
                argv_sha256=hashlib.sha256(argv).hexdigest())


def _path_pin(value):
    from .cluster_profile import _is_sha
    return (isinstance(value, dict) and set(value) == {'path', 'sha256'}
            and isinstance(value['path'], str) and Path(value['path']).is_absolute()
            and '..' not in Path(value['path']).parts and _is_sha(value['sha256']))


def _pci(value):
    domain, bus, slot = value.lower().rsplit(':', 2)
    return int(domain, 16), int(bus, 16), slot


class ExternalModelOwners:
    def __init__(self, profile):
        from .cluster_profile import _is_sha, pinned_json, _pinned_bytes
        pin = profile.external_model_owners
        self.pin = None if pin is None else MappingProxyType(dict(pin))
        records = {}
        if pin is not None:
            if not _path_pin(pin):
                raise ValueError('external model owner receipt must be explicitly pinned')
            value = pinned_json(pin['path'], pin['sha256'])
            if (not isinstance(value, dict) or set(value) != {'schema', 'hostname', 'machine_id_sha256', 'models'}
                    or value['schema'] != 'mind2act_eval.pro_external_model_owners.v1'
                    or value['hostname'] != profile.hostname
                    or value['machine_id_sha256'] != profile.machine_id_sha256
                    or not isinstance(value['models'], list)
                    or not 1 <= len(value['models']) <= len(profile.devices)):
                raise ValueError('external model owner receipt host/schema/pool mismatch')
            devices = {device['uuid']: device for device in profile.devices}
            pids = set()
            keys = {'kind', 'gpu_uuid', 'pci_bus_id', 'pid', 'birth', 'uid', 'argv_sha256',
                    'checkpoint', 'export_sha256', 'service_control'}
            for row in value['models']:
                if (not isinstance(row, dict) or set(row) != keys or row['kind'] != 'g05-native'
                        or type(row['pid']) is not int or not 0 < row['pid'] < 2**31
                        or type(row['birth']) is not int or row['birth'] <= 0
                        or type(row['uid']) is not int or not 0 <= row['uid'] < 2**32
                        or not _is_sha(row['argv_sha256']) or not _is_sha(row['export_sha256'])
                        or not _path_pin(row['checkpoint']) or not _path_pin(row['service_control'])
                        or not isinstance(row['gpu_uuid'], str) or row['gpu_uuid'] not in devices
                        or not isinstance(row['pci_bus_id'], str)):
                    raise ValueError('invalid pinned external G05 model owner')
                device = devices[row['gpu_uuid']]
                if (_pci(row['pci_bus_id']) != _pci(device['pci_bus_id'])
                        or device['index'] in records or row['pid'] in pids):
                    raise ValueError('external model owner GPU/PCI/PID is ambiguous')
                # Root reviews this immutable control evidence and the original
                # checkpoint/export pins. Large model weights are not reread.
                proof = row['service_control']
                _pinned_bytes(proof['path'], proof['sha256'])
                records[device['index']] = json.dumps(row, sort_keys=True).encode()
                pids.add(row['pid'])
        self._records = MappingProxyType(records)

    def input_paths(self):
        paths = set() if self.pin is None else {Path(self.pin['path'])}
        for raw in self._records.values():
            row = json.loads(raw)
            paths.update(Path(row[key]['path']) for key in ('checkpoint', 'service_control'))
        return tuple(paths)

    def snapshot(self, index):
        raw = self._records.get(index)
        if raw is None:
            return {}
        row = json.loads(raw)
        try:
            current = _identity(row['pid'])
        except (OSError, ValueError) as exc:
            raise ValueError('pinned external model identity is unavailable') from exc
        if (current['birth'] != row['birth'] or any(uid != row['uid'] for uid in current['uids'])
                or current['argv_sha256'] != row['argv_sha256']):
            raise ValueError('pinned external model birth/UID/argv identity changed')
        return {row['pid']: dict(current, uid=row['uid'], checkpoint=row['checkpoint'],
            export_sha256=row['export_sha256'], gpu_uuid=row['gpu_uuid'],
            ownership='profile-pinned external G05 model; never adopted or signalled')}

    def verify_job(self, values, index):
        if index not in self._records or values['adapter_config'] is None or values['h5']:
            return
        from .cluster_profile import pinned_json
        config = pinned_json(values['adapter_config'], values['adapter_config_sha256'])
        checkpoint = json.loads(self._records[index])['checkpoint']
        if config.get('checkpoint') != checkpoint['path'] or config.get('checkpoint_sha256') != checkpoint['sha256']:
            raise ValueError('selected G05 checkpoint differs from its external model owner receipt')
