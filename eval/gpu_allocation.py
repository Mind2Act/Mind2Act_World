"""Live single-GPU Slurm identity binding; importing this module never touches a GPU.

CUDA UUID/PCI: NVIDIA CUDA Driver API. Vulkan IDs: VkPhysicalDeviceIDProperties
and VkPhysicalDevicePCIBusInfoPropertiesEXT (Khronos). Neither metadata probe
creates a CUDA context or Vulkan logical device. Slurm's GRES IDs are opaque;
they must not be reinterpreted as NVML or renderer ordinals.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import ctypes as C
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import uuid

SCHEMA = 'mind2act.slurm-gpu-allocation.v1'
ENV_KEYS = ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
            'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')


def _ok(code):
    if code != 0:
        raise RuntimeError(f'GPU metadata API failed with status {code}')


def _uuid(raw):
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


def _pci(value):
    fields = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if not fields:
        raise ValueError('invalid PCI identity')
    return tuple(int(s, 16) for s in fields.groups())


def allocation_identity(environ=None):
    env = os.environ if environ is None else environ
    identity = {key: env.get(key) for key in ENV_KEYS}
    if not all(identity[k] for k in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'CUDA_VISIBLE_DEVICES')):
        raise ValueError('explicit Slurm step and original CUDA visibility required')
    if not (identity['SLURM_STEP_GPUS'] or identity['SLURM_JOB_GPUS']):
        raise ValueError('Slurm allocated GPU identity missing')
    storage = env.get('M2A_BSCC_STORAGE_ROOT')
    if not storage or not Path(storage).is_absolute():
        raise ValueError('explicit approved BSCC storage root required')
    try:
        root = Path(storage).resolve(strict=True)
    except OSError:
        raise ValueError('approved storage root unavailable') from None
    if str(root) != storage or not root.is_dir() or root.stat().st_uid != os.getuid():
        raise ValueError('approved storage root must be canonical and owned')
    identity['storage_root_sha256'] = hashlib.sha256(str(root).encode()).hexdigest()
    identity['storage_device'] = root.stat().st_dev
    identity['uid'] = os.getuid()
    identity['hostname'] = socket.gethostname()
    identity['machine_id_sha256'] = hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest()
    identity['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return identity


def cuda_devices():
    class Uuid(C.Structure):
        _fields_ = [('bytes', C.c_ubyte * 16)]
    driver = C.CDLL('libcuda.so.1')
    _ok(driver.cuInit(0))
    count = C.c_int()
    _ok(driver.cuDeviceGetCount(C.byref(count)))
    if count.value != 1:
        raise ValueError('first Slurm integration supports one allocated CUDA device')
    device = C.c_int()
    _ok(driver.cuDeviceGet(C.byref(device), 0))
    identity = Uuid()
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', driver.cuDeviceGetUuid)
    _ok(get_uuid(C.byref(identity), device))
    pci = C.create_string_buffer(32)
    _ok(driver.cuDeviceGetPCIBusId(pci, 32, device))
    return [dict(logical_index=0, uuid=_uuid(identity.bytes), pci_bus_id=pci.value.decode())]


def vulkan_devices():
    class App(C.Structure):
        _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('name', C.c_char_p),
                    ('version', C.c_uint32), ('engine', C.c_char_p),
                    ('engineVersion', C.c_uint32), ('apiVersion', C.c_uint32)]
    class Create(C.Structure):
        _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('flags', C.c_uint32),
                    ('app', C.POINTER(App)), ('layerCount', C.c_uint32), ('layers', C.c_void_p),
                    ('extensionCount', C.c_uint32), ('extensions', C.c_void_p)]
    class Pci(C.Structure):
        _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('domain', C.c_uint32),
                    ('bus', C.c_uint32), ('device', C.c_uint32), ('function', C.c_uint32)]
    class Id(C.Structure):
        _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('deviceUUID', C.c_ubyte * 16),
                    ('driverUUID', C.c_ubyte * 16), ('deviceLUID', C.c_ubyte * 8),
                    ('nodeMask', C.c_uint32), ('luidValid', C.c_uint32)]
    class Properties(C.Structure):
        # VkPhysicalDeviceProperties has a stable header and a bounded native
        # limits/sparse-properties tail. Extra storage avoids mirroring its ABI.
        _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('properties', C.c_ubyte * 4096)]
    driver = C.CDLL('libvulkan.so.1')
    driver.vkCreateInstance.argtypes = [C.POINTER(Create), C.c_void_p, C.POINTER(C.c_void_p)]
    driver.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)]
    driver.vkGetPhysicalDeviceProperties2.argtypes = [C.c_void_p, C.POINTER(Properties)]
    driver.vkGetPhysicalDeviceProperties2.restype = None
    driver.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
    driver.vkDestroyInstance.restype = None
    instance = C.c_void_p()
    app = App(0, None, b'mind2act-allocation-metadata', 1, b'none', 1, (1 << 22) | (1 << 12))
    create = Create(1, None, 0, C.pointer(app), 0, None, 0, None)
    _ok(driver.vkCreateInstance(C.byref(create), None, C.byref(instance)))
    try:
        count = C.c_uint32()
        _ok(driver.vkEnumeratePhysicalDevices(instance, C.byref(count), None))
        if not 1 <= count.value <= 256:
            raise ValueError('invalid Vulkan device count')
        devices = (C.c_void_p * count.value)()
        _ok(driver.vkEnumeratePhysicalDevices(instance, C.byref(count), devices))
        result = []
        for index, device in enumerate(devices):
            pci = Pci(1000212000, None, *([0xffffffff] * 4))
            identity = Id()
            identity.sType = 1000071004
            identity.pNext = C.cast(C.pointer(pci), C.c_void_p)
            props = Properties()
            props.sType = 1000059001
            props.pNext = C.cast(C.pointer(identity), C.c_void_p)
            driver.vkGetPhysicalDeviceProperties2(device, C.byref(props))
            data = bytes(props.properties)
            result.append(dict(enumeration_index=index, vendor_id=int.from_bytes(data[8:12], 'little'),
                               device_type=int.from_bytes(data[16:20], 'little'),
                               uuid=_uuid(identity.deviceUUID),
                               pci=[pci.domain, pci.bus, pci.device, pci.function]))
        return result
    finally:
        driver.vkDestroyInstance(instance, None)


def _nvidia_query(gpu_uuid, fields, group='gpu'):
    result = subprocess.run(['nvidia-smi', '-i', gpu_uuid, f'--query-{group}={fields}',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('allocated GPU NVML metadata query failed')
    return list(csv.reader(io.StringIO(result.stdout)))


def probe_current():
    cuda = cuda_devices()
    selected = _nvidia_query(cuda[0]['uuid'], 'index,uuid,pci.bus_id,driver_version')
    if len(selected) != 1 or len(selected[0]) != 4:
        raise ValueError('ambiguous allocated GPU NVML identity')
    row = [s.strip() for s in selected[0]]
    return dict(cuda=cuda, vulkan=vulkan_devices(),
                nvidia=dict(index=int(row[0]), uuid=row[1], pci_bus_id=row[2], driver_version=row[3]))


def _selection(probe):
    cuda = probe['cuda']
    nvidia = probe['nvidia']
    candidates = [d for d in probe['vulkan'] if d['vendor_id'] == 0x10de]
    if len(cuda) != 1 or len(candidates) != 1:
        raise ValueError('single allocated CUDA/NVIDIA Vulkan device required')
    c, v = cuda[0], candidates[0]
    for entry in (c, v, nvidia):
        value = entry['uuid']
        if not isinstance(value, str) or not value.lower().startswith('gpu-'):
            raise ValueError('invalid GPU UUID')
        uuid.UUID(value[4:])
    for entry, key in ((c, 'logical_index'), (v, 'enumeration_index')):
        if type(entry[key]) is not int:
            raise ValueError('GPU index must be integer')
    # In the measured first-single-GPU domain the sole NVIDIA adapter is also
    # Vulkan enumeration zero; remaining adapters may only be CPU software.
    # Other orderings need a separately verified Kit-index provider, not a guess.
    if v['enumeration_index'] != 0 or any(d['device_type'] != 4 for d in probe['vulkan'] if d is not v):
        raise ValueError('Vulkan ordering needs an explicit supported Kit mapping provider')
    if c['logical_index'] != 0 or len({c['uuid'].lower(), v['uuid'].lower(), nvidia['uuid'].lower()}) != 1:
        raise ValueError('CUDA/Vulkan/NVML UUID mismatch')
    if _pci(c['pci_bus_id']) != _pci(nvidia['pci_bus_id']) or _pci(c['pci_bus_id']) != tuple(v['pci']):
        raise ValueError('CUDA/Vulkan/NVML PCI mismatch')
    if type(nvidia['index']) is not int or nvidia['index'] < 0:
        raise ValueError('invalid NVML physical index')
    return dict(physical_gpu=nvidia['index'], cuda_index=c['logical_index'], renderer_index=v['enumeration_index'],
                uuid=c['uuid'].lower(), pci=list(_pci(c['pci_bus_id'])), driver_version=nvidia['driver_version'])


def capture_receipt():
    return dict(schema=SCHEMA, observed_at=datetime.now(timezone.utc).isoformat(),
                allocation=allocation_identity(), probe=probe_current())


@dataclass(frozen=True)
class SlurmGpuSelection:
    allocation_json: str
    selection_json: str
    receipt_sha256: str

    @property
    def storage_root(self):
        root = Path(os.environ['M2A_BSCC_STORAGE_ROOT']).resolve(strict=True)
        expected = json.loads(self.allocation_json)['storage_root_sha256']
        if hashlib.sha256(str(root).encode()).hexdigest() != expected:
            raise ValueError('approved storage root does not match pinned identity')
        return root

    def verify_storage(self, output, canonical_uid_lock):
        if allocation_identity() != json.loads(self.allocation_json):
            raise ValueError('approved storage/allocation identity changed')
        root = self.storage_root
        lock = Path(canonical_uid_lock).resolve()
        if lock != root / f'.mind2act-eval-{os.getuid()}':
            raise ValueError('Slurm storage requires canonical shared per-UID lock root')
        if not Path(output).resolve().is_relative_to(root):
            raise ValueError('output outside approved Slurm storage root')
        return root

    @property
    def cuda_index(self):
        return json.loads(self.selection_json)['cuda_index']

    @property
    def renderer_index(self):
        return json.loads(self.selection_json)['renderer_index']

    def assert_current(self, physical_gpu):
        if allocation_identity() != json.loads(self.allocation_json):
            raise ValueError('Slurm allocation/environment/node identity changed')
        selection = _selection(probe_current())
        if type(physical_gpu) is not int or selection['physical_gpu'] != physical_gpu:
            raise ValueError('requested GPU differs from allocated physical identity')
        if selection != json.loads(self.selection_json):
            raise ValueError('live GPU mapping changed from pinned receipt')
        return selection


def load_gpu_selection(path, sha256, *, physical_gpu):
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raise ValueError('GPU allocation receipt unavailable') from None
    if not re.fullmatch('[0-9a-f]{64}', sha256 or '') or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('GPU allocation receipt hash mismatch')
    receipt = json.loads(raw)
    if receipt['schema'] != SCHEMA:
        raise ValueError('GPU allocation receipt schema mismatch')
    selection = _selection(receipt['probe'])
    result = SlurmGpuSelection(json.dumps(receipt['allocation'], sort_keys=True),
                               json.dumps(selection, sort_keys=True), sha256)
    result.assert_current(physical_gpu)
    return result




def worker_cuda_mask(physical_gpu, argv, environ):
    """Parent GPU-env seam only; preserve original Slurm visibility after live validation."""
    flags = ('--gpu-allocation', '--gpu-allocation-sha256')
    present = [flag in argv for flag in flags]
    if not environ.get('SLURM_JOB_ID'):
        if any(present):
            raise ValueError('Slurm GPU mapping supplied outside a Slurm allocation')
        return str(physical_gpu)
    if not all(present) or any(argv.count(flag) != 1 for flag in flags):
        raise ValueError('Slurm worker requires an explicit pinned GPU allocation')
    try:
        path, digest = (argv[argv.index(flag) + 1] for flag in flags)
    except IndexError as error:
        raise ValueError('GPU allocation flag value missing') from error
    if environ.get('CUDA_VISIBLE_DEVICES') != os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('parent CUDA visibility already changed')
    load_gpu_selection(path, digest, physical_gpu=physical_gpu)
    return environ['CUDA_VISIBLE_DEVICES']


def optional_gpu_selection(physical_gpu, argv, environ):
    """Explicit opt-in for parent storage admission; a malformed opt-in never falls back."""
    flags = ('--gpu-allocation', '--gpu-allocation-sha256')
    if not any(flag in argv for flag in flags):
        return None
    if not environ.get('SLURM_JOB_ID') or any(argv.count(flag) != 1 for flag in flags):
        raise ValueError('invalid explicit Slurm allocation flags')
    try:
        path, digest = (argv[argv.index(flag) + 1] for flag in flags)
    except IndexError as error:
        raise ValueError('GPU allocation flag value missing') from error
    return load_gpu_selection(path, digest, physical_gpu=physical_gpu)


def verify_kit_gpu_table(path, selection):
    """Read the current owned Kit log's actual Active GPU row, never settings echoes."""
    expected = json.loads(selection.selection_json)
    with Path(path).open('r', errors='replace') as stream:
        text = stream.read(4 * 1024 * 1024)
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if '| GPU |' in line and '| Active |' in line]
    if not headers:
        raise ValueError('Kit actual GPU table header unavailable')
    lines = lines[headers[-1]:]
    records = []
    for index, line in enumerate(lines):
        fields = [field.strip() for field in line.split('|')[1:-1]]
        if len(fields) != 7 or not fields[0].isdigit():
            continue
        if fields[5].lower() != '10de' or not fields[1].startswith('NVIDIA'):
            continue
        if index + 1 >= len(lines):
            continue
        detail = [field.strip() for field in lines[index + 1].split('|')[1:-1]]
        if len(detail) != 7:
            continue
        prefix = detail[6].lower().removesuffix('..').replace('-', '')
        if not re.fullmatch('[0-9a-f]{8,32}', prefix):
            raise ValueError('Kit GPU UUID readback unavailable')
        records.append(dict(index=int(fields[0]), active=fields[2].startswith('Yes'), uuid_prefix=prefix))
    if len(records) != 1:
        raise ValueError('Kit GPU table differs from unique live NVIDIA device')
    row = records[0]
    uuid_hex = expected['uuid'][4:].replace('-', '')
    if not row['active'] or row['index'] != expected['renderer_index'] or not uuid_hex.startswith(row['uuid_prefix']):
        raise ValueError('Kit actual active GPU differs from allocated CUDA/Vulkan identity')
    return dict(schema='mind2act.kit-gpu-readback.v1', receipt_sha256=selection.receipt_sha256,
                kit_index=row['index'], kit_uuid_prefix=row['uuid_prefix'],
                current_cuda_vulkan_uuid=expected['uuid'], pci=expected['pci'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Capture only allocated CUDA/Vulkan metadata; no Kit or GPU workload')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = capture_receipt()
    _selection(receipt['probe'])
    raw = (json.dumps(receipt, indent=2) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())

