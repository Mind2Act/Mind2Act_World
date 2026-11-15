"""Bounded numeric fleet qualification, submitted by the trusted operator."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from math import isfinite
from pathlib import Path
import time

from mind2act_eval.cluster_profile import pinned_json


def _fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError('invalid fleet stability fields')


def _number(value, *, integer=False, minimum=0):
    if type(value) not in ((int,) if integer else (int, float)) or not isfinite(value) or value < minimum:
        raise ValueError('invalid fleet numeric observation')
    return value


def _pin(value):
    _fields(value, 'path sha256')
    return pinned_json(value['path'], value['sha256'])


def _attempt(state, job_id, token, bindings):
    row = state['jobs'].get(job_id)
    matches = [a for a in row.get('attempts', []) if a['token'] == token] if row else []
    if len(matches) != 1:
        raise ValueError('fleet worker must bind a real queue attempt')
    attempt = matches[0]
    host = attempt['owner_hostname']
    if (host not in bindings or attempt['profile_sha256'] != bindings[host]['profile']['sha256']
            or attempt['owner_pid'] != bindings[host]['pid']
            or attempt['owner_starttime'] != bindings[host]['birth']
            or attempt['inputs']['code_sha'] != bindings[host]['code_sha']
            or attempt.get('child_started_at_unix_s') is None):
        raise ValueError('fleet worker generation/owner/start identity mismatch')
    return row, attempt


def _historical_exit(attempt, admission_sha256):
    if attempt.get('finished_at') is None:
        return
    receipt = attempt.get('execution_receipt') or {}
    exit_path = Path(attempt['directory']) / 'batch-exit.json'
    value = pinned_json(exit_path, receipt.get('batch_exit_sha256'))
    seal = Path(attempt['directory']) / 'worker-seal.json'
    if (value.get('worker_exit_verified') is not True or value.get('worker_seal_valid') is not True
            or receipt.get('admission_sha256') != admission_sha256
            or hashlib.sha256(seal.read_bytes()).hexdigest() != value.get('worker_seal_sha256')
            or (attempt.get('result') or {}).get('infra_valid') is not True):
        raise ValueError('historical fleet worker lacks its original verified exit/seal evidence')


def _group(row):
    options = row['definition']['options']
    spec = pinned_json(options['spec'], options['spec_sha256'])
    checkpoint = None
    if options.get('adapter_config'):
        checkpoint = pinned_json(options['adapter_config'], options['adapter_config_sha256']).get('checkpoint_sha256')
    return (spec['task_id'], spec['difficulty'], json.dumps(spec.get('clock'), sort_keys=True),
            options['adapter'], checkpoint, options['physics_ticks'], options['wall_s'])


def _pci(value):
    domain, bus, slot = value.lower().rsplit(':', 2)
    return int(domain, 16), int(bus, 16), slot


def _memory(value, profile, uid):
    _fields(value, 'uid uid_rss_bytes rss_bytes cgroup_path cgroup_current_bytes cgroup_max_bytes cgroup_high_bytes host_available_bytes ancestors')
    budget, reserve = profile['rss_limit_bytes'], profile['host_reserve_bytes']
    scope = Path(value['cgroup_path'])
    if scope.parent != Path(profile['cgroup_parent']) or not scope.name.startswith(f'mind2act-pro5000-{uid}-'):
        raise ValueError('fleet campaign counters must belong to the actual owned child cgroup')
    if (type(value['uid']) is not int or value['uid'] != uid or _number(value['uid_rss_bytes'], integer=True) > budget
            or _number(value['rss_bytes'], integer=True) > budget
            or _number(value['cgroup_current_bytes'], integer=True) > budget
            or _number(value['cgroup_max_bytes'], integer=True) != budget
            or _number(value['host_available_bytes'], integer=True) < reserve):
        raise ValueError('fleet host/campaign memory qualification failed')
    if value['cgroup_high_bytes'] is not None and _number(value['cgroup_high_bytes'], integer=True) - value['cgroup_current_bytes'] < reserve:
        raise ValueError('fleet campaign memory.high headroom exhausted')
    parent = Path(profile['cgroup_parent'])
    paths = []
    while parent.is_relative_to('/sys/fs/cgroup'):
        paths.append(str(parent))
        if parent == Path('/sys/fs/cgroup'): break
        parent = parent.parent
    rows = value['ancestors']
    if not isinstance(rows, list) or len(rows) != len(paths) or {r.get('path') for r in rows} != set(paths):
        raise ValueError('fleet evidence must enumerate every cgroup ancestor')
    for row in rows:
        _fields(row, 'path limit_bytes high_bytes current_bytes interface')
        if row['interface'] == 'root-no-memory-interface':
            if row['path'] != '/sys/fs/cgroup' or any(row[k] is not None for k in ('limit_bytes', 'high_bytes', 'current_bytes')):
                raise ValueError('only the actual kernel root may explicitly lack the memory interface')
            continue
        if row['interface'] != 'present':
            raise ValueError('unreadable ancestor is not an unlimited ancestor')
        current = _number(row['current_bytes'], integer=True)
        if row['limit_bytes'] is not None and _number(row['limit_bytes'], integer=True) - current < reserve:
            raise ValueError('fleet ancestor memory headroom exhausted')
        if row['high_bytes'] is not None and _number(row['high_bytes'], integer=True) - current < reserve:
            raise ValueError('fleet ancestor memory.high headroom exhausted')


def validate_fleet_stability(queue, state, receipt):
    """Validate observations, never infer fleet load from a driver uptime timer.

    Profiles, READY files, baseline and the outer receipt are SHA-pinned. This
    validates trusted operator samples, not a new authentication/collector system.
    It runs only on explicit expansion; active monitors and first-two intake do
    not read the sample window or an unbounded master catalog.
    """
    _fields(receipt, 'schema queue_id queue_manifest_sha256 drivers samples baseline')
    if (receipt['schema'] != 'mind2act_eval.pro_rolling_stability.v2' or receipt['queue_id'] != queue.queue_id
            or receipt['queue_manifest_sha256'] != queue.manifest_sha256):
        raise ValueError('fleet stability requires the current pinned queue generation')
    bindings = receipt['drivers']
    if not isinstance(bindings, dict) or set(bindings) != set(queue.nodes) or len(bindings) != 3:
        raise ValueError('fleet stability requires all three current hosts')
    profiles = {}
    code = state['drivers'].get(queue.node.hostname, {}).get('code_sha')
    for host, binding in bindings.items():
        _fields(binding, 'pid birth uid profile code_sha')
        _fields(binding['profile'], 'path sha256')
        _number(binding['pid'], integer=True, minimum=1)
        queue.profile.verify_control_input(binding['profile']['path'])
        if not any(Path(binding['profile']['path']).resolve().is_relative_to(p.resolve()) for p in queue.nodes[host].input_roots):
            raise ValueError('fleet profile path is outside the frozen node input domains')
        driver = state['drivers'].get(host, {})
        if (driver.get('status') != 'running' or time.time() - driver.get('heartbeat_unix_s', 0) > 30
                or binding['pid'] != driver.get('pid') or binding['birth'] is None
                or type(binding['uid']) is not int or binding['uid'] != driver.get('uid')
                or binding['birth'] != driver.get('process_birth') or binding['code_sha'] != driver.get('code_sha')
                or not isinstance(code, str) or len(code) != 40 or binding['code_sha'] != code
                or binding['profile']['sha256'] != driver.get('profile_sha256')
                or driver.get('queue_manifest_sha256') != queue.manifest_sha256):
            raise ValueError('fleet driver generation/PID/birth/profile/code/heartbeat mismatch')
        profile = _pin(binding['profile'])
        if (profile['hostname'].lower() != host or profile['machine_id_sha256'] != queue.nodes[host].machine_id_sha256
                or tuple(d['index'] for d in profile['devices']) != queue.nodes[host].gpu_indices
                or profile['rss_limit_bytes'] != queue.nodes[host].rss_limit_bytes
                or profile.get('max_workers_per_gpu') != 3
                or profile.get('max_parallel') != queue.nodes[host].stable_parallel):
            raise ValueError('fleet profile must bind the actual frozen GPU pool and budget')
        profiles[host] = profile
    samples = receipt['samples']
    if not isinstance(samples, list) or not 21 <= len(samples) <= 61:
        raise ValueError('fleet window requires 21..61 complete samples')
    times = [_number(s.get('at_unix_s')) for s in samples]
    if (not 1200 <= times[-1] - times[0] <= 1800 or not time.time() - 30 <= times[-1] <= time.time()
            or any(not 0 < b - a <= 60 for a, b in zip(times, times[1:]))):
        raise ValueError('fleet evidence must cover a fresh uninterrupted 20-minute full-load window')
    if any(state['drivers'][h].get('healthy_since_unix_s', time.time()) > times[0] for h in bindings):
        raise ValueError('fleet window predates current generation or its last infrastructure failure')
    progress = defaultdict(list)
    infra = {}
    for sample, at in zip(samples, times):
        _fields(sample, 'at_unix_s nodes')
        if not isinstance(sample['nodes'], dict) or set(sample['nodes']) != set(bindings):
            raise ValueError('fleet sample omits a host')
        for host, value in sample['nodes'].items():
            _fields(value, 'workers gpus memory infra_failures')
            profile = profiles[host]
            pool = set(queue.nodes[host].gpu_indices)
            _memory(value['memory'], profile, bindings[host]['uid'])
            failures = _number(value['infra_failures'], integer=True)
            expected = 0
            for job in queue.jobs:
                for attempt in state['jobs'][job['job_id']]['attempts']:
                    if (attempt['owner_hostname'] == host and attempt['profile_sha256'] == bindings[host]['profile']['sha256']
                            and (attempt.get('result') or {}).get('infra_valid') is False
                            and datetime.fromisoformat(attempt['finished_at']).timestamp() <= at):
                        expected += 1
            if failures != expected or failures != infra.setdefault(host, failures):
                raise ValueError('fleet infrastructure failures worsened or disagree with the queue')
            gpus = value['gpus']
            if not isinstance(gpus, list) or len(gpus) != len(pool) or {g.get('index') for g in gpus} != pool:
                raise ValueError('fleet sample omits a physical GPU')
            for gpu in gpus:
                _fields(gpu, 'index uuid pci_bus_id used_mib total_mib')
                _number(gpu['index'], integer=True)
                device = next(d for d in profile['devices'] if d['index'] == gpu['index'])
                used, total = _number(gpu['used_mib'], integer=True), _number(gpu['total_mib'], integer=True)
                if (gpu['uuid'] != device['uuid'] or _pci(gpu['pci_bus_id']) != _pci(device['pci_bus_id'])
                        or used > 72 * 1024 or total - used < profile['min_free_vram_mib']):
                    raise ValueError('fleet physical GPU identity/VRAM qualification failed')
            workers = value['workers']
            if not isinstance(workers, list) or len(workers) != 2 * len(pool):
                raise ValueError('fleet stability requires two actual READY/RPC workers per frozen GPU')
            counts, identities, tokens = defaultdict(int), set(), set()
            for worker in workers:
                _fields(worker, 'job_id token gpu pid birth process_state ready_sha256 admission_sha256 rpc_count rpc_errors physics_tick')
                row, attempt = _attempt(state, worker['job_id'], worker['token'], bindings)
                gpu = _number(worker['gpu'], integer=True)
                if (attempt['owner_hostname'] != host or gpu != attempt['gpu']
                        or gpu != row['definition']['eligible_gpu'] or host not in row['definition']['eligible_nodes']
                        or attempt['child_started_at_unix_s'] > at or not isinstance(worker['birth'], str) or not worker['birth']
                        or (attempt.get('finished_at') is not None and datetime.fromisoformat(attempt['finished_at']).timestamp() <= at)
                        or worker['token'] in tokens or (worker['pid'], worker['birth']) in identities):
                    raise ValueError('fleet worker occupancy/identity/fixed route mismatch')
                _number(worker['pid'], integer=True, minimum=1)
                if worker['process_state'] not in {'R', 'S', 'D', 'I'}:
                    raise ValueError('fleet sample process is not actually alive/runnable')
                admission = pinned_json(Path(attempt['directory']) / 'pro-admission.json', worker['admission_sha256'])
                if (admission.get('profile_sha256') != attempt['profile_sha256']
                        or admission.get('resources', {}).get('cgroup') != value['memory']['cgroup_path']
                        or admission.get('device', {}).get('index') != gpu
                        or admission.get('device', {}).get('uuid') != next(d['uuid'] for d in profile['devices'] if d['index'] == gpu)):
                    raise ValueError('fleet memory scope lacks its actual pinned worker admission binding')
                _historical_exit(attempt, worker['admission_sha256'])
                ready = pinned_json(Path(attempt['directory']) / 'ready.json', worker['ready_sha256'])
                if ready.get('ready') is not True or ready.get('pid') != worker['pid']:
                    raise ValueError('fleet worker READY receipt/PID mismatch')
                _number(worker['rpc_count'], integer=True, minimum=1)
                if _number(worker['rpc_errors'], integer=True) != 0:
                    raise ValueError('fleet RPC infrastructure worsened')
                tick = worker['physics_tick']
                if tick is not None and _number(tick, integer=True) > row['definition']['options']['physics_ticks']:
                    raise ValueError('fleet progress exceeds original physics budget')
                counts[gpu] += 1; tokens.add(worker['token']); identities.add((worker['pid'], worker['birth']))
                progress[(worker['job_id'], worker['token'])].append((at, worker))
            if dict(counts) != {i: 2 for i in pool}:
                raise ValueError('fleet window requires two actual workers on every physical GPU')
    baseline = _pin(receipt['baseline'])
    _fields(baseline, 'schema code_sha observations')
    if baseline['schema'] != 'mind2act_eval.pro_rolling_baseline.v1' or baseline['code_sha'] != code:
        raise ValueError('fleet baseline code generation mismatch')
    observations = baseline['observations']
    if not isinstance(observations, list) or not 1 <= len(observations) <= 192:
        raise ValueError('fleet qualification requires bounded comparable duration baseline observations')
    reference, measured, seen_groups = defaultdict(list), defaultdict(list), set()
    baseline_tokens = set()
    for observation in observations:
        _fields(observation, 'job_id token from_unix_s until_unix_s from_tick until_tick from_rpc_count until_rpc_count')
        row, attempt = _attempt(state, observation['job_id'], observation['token'], bindings)
        if observation['token'] in baseline_tokens:
            raise ValueError('duplicate baseline observation cannot add comparable evidence')
        baseline_tokens.add(observation['token'])
        a, b = _number(observation['from_unix_s']), _number(observation['until_unix_s'])
        x, y = observation['from_tick'], observation['until_tick']
        r, s = _number(observation['from_rpc_count'], integer=True), _number(observation['until_rpc_count'], integer=True)
        if not attempt['child_started_at_unix_s'] <= a < b < times[0] or s < r:
            raise ValueError('fleet baseline must contain actual earlier comparable progress')
        group = _group(row)
        physical = x is not None and y is not None
        if physical:
            _number(x, integer=True); _number(y, integer=True)
            if not x <= y <= row['definition']['options']['physics_ticks']:
                raise ValueError('fleet baseline physics progress is invalid')
        elif x is not None or y is not None:
            raise ValueError('fleet baseline cannot invent a missing physics endpoint')
        if physical and y > x: reference[group + ('physics',)].append((b - a) / (y - x))
        if s > r: reference[group + ('rpc',)].append((b - a) / (s - r))
        if not (physical and y > x) and s == r:
            raise ValueError('fleet baseline has no actual physical or usable RPC progress')
    for (job_id, token), points in progress.items():
        row, _ = _attempt(state, job_id, token, bindings)
        group = _group(row)
        seen_groups.add(group)
        first_worker = points[0][1]
        identity = (first_worker['pid'], first_worker['birth'])
        last_tick, last_rpc = None, None
        for _, current in points:
            tick, rpc = current['physics_tick'], current['rpc_count']
            if ((current['pid'], current['birth']) != identity
                    or (last_rpc is not None and rpc < last_rpc)
                    or (tick is not None and last_tick is not None and tick < last_tick)):
                raise ValueError('fleet worker PID/birth or progress regressed')
            if tick is not None:
                last_tick = tick
            last_rpc = rpc
        a, first = points[0]; b, last = points[-1]
        ticks = (last['physics_tick'] - first['physics_tick']
                 if first['physics_tick'] is not None and last['physics_tick'] is not None else 0)
        replies = last['rpc_count'] - first['rpc_count']
        # Natural completion/replacement may leave one fresh observation at
        # the window edge. It still needs actual progress, and its task family
        # must have measured duration pairs elsewhere in this same window.
        if len(points) == 1 and (first['rpc_count'] > 0 or (first['physics_tick'] or 0) > 0):
            continue
        if ticks <= 0 and replies <= 0:
            raise ValueError('fleet worker lacks physical OR usable RPC progress across observations')
        if ticks > 0: measured[group + ('physics',)].append((b - a) / ticks)
        if replies > 0: measured[group + ('rpc',)].append((b - a) / replies)
    for group in seen_groups:
        comparable = False
        for unit in ('physics', 'rpc'):
            values, prior = measured[group + (unit,)], reference[group + (unit,)]
            if values and prior:
                comparable = True
                if sum(values) / len(values) > sum(prior) / len(prior):
                    raise ValueError('fleet comparable task duration worsened')
        if not comparable:
            raise ValueError('fleet comparable task duration baseline is missing')
