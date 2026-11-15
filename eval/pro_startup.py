"""Opt-in PRO initialization leases inside the canonical startup domain.

Old host Kit writers remain excluded. New workers retain one device lease and
one bounded host slot until their own verified first frame; SDK construction
and its crash-safe 60-second spacing still have a short shared launch gate.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import os
import json
import stat
import time
from types import SimpleNamespace
from math import isfinite

from mind2act_eval.startup_paths import read_gate_state, verify_startup_domain

LEASE_ENV = 'MIND2ACT_EVAL_PRO_STARTUP_LEASE'


def startup_interval(profile):
    """Read an explicit spacing from the profile's verified cluster plan.

    Defaults and legacy interop recovery retain 60 seconds. Only a newly frozen
    plan can lower normal PRO spacing; no environment flag changes a live pin.
    """
    path = getattr(profile, 'cluster_config_path', None)
    if path is None:
        return 60.
    from mind2act_eval.verified_files import verified_bytes
    try:
        import tomllib
    except ModuleNotFoundError:
        import toml as tomllib
    plan = tomllib.loads(verified_bytes(path, profile.cluster_config_sha256).decode())
    if plan.get('schema') != 'mind2act.cluster.plan.v1' or plan.get('cluster_id') != 'pro5000':
        raise ValueError('PRO startup spacing requires its pinned cluster plan')
    value = plan.get('startup_interval_s', 60.)
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 60:
        raise ValueError('PRO startup interval must be finite seconds from 0 to 60')
    return float(value)


@contextmanager
def _legacy_guard(paths, *, cancel_event, clock):
    from mind2act_eval.sim import recover_startup_state, startup_gate
    while True:
        with startup_gate(paths.kit, shared=True, cancel_event=cancel_event, clock=clock) as shared_mark:
            verify_startup_domain(paths)
            state = read_gate_state(paths.kit.read_text())
            if not (state in {'active', 'cooling'} or state is None and paths.inherited_kit_cooldown):
                yield shared_mark
                return
        # Never discard a prior owner's durable crash/death cooldown. Recovery
        # takes the original EX domain before any new per-device lease starts.
        with startup_gate(paths.kit, cancel_event=cancel_event, clock=clock):
            verify_startup_domain(paths)
            with paths.kit.open('r+') as stream:
                recover_startup_state(stream, paths.kit, 60., clock=clock,
                    inherited_cooldown=paths.inherited_kit_cooldown, cancel_event=cancel_event)


@contextmanager
def _capacity_slot(directory, capacity, *, cancel_event, clock):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as files:
        descriptors = []
        for index in range(capacity):
            fd = os.open(directory / f'slot-{index}.lock',
                         os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            files.callback(os.close, fd)
            descriptors.append(fd)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError('PRO initialization cancelled before capacity acquisition')
            for fd in descriptors:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                lease = SimpleNamespace(fd=fd, transferred=False)
                try:
                    yield lease
                finally:
                    if not lease.transferred:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                return
            clock.sleep(.1)


@contextmanager
def initialization_gate(paths, physical_gpu, *, capacity, cancel_event=None, clock=time, profile=None):
    from mind2act_eval.sim import startup_gate
    if capacity != 2 or type(capacity) is not int or type(physical_gpu) is not int or not 0 <= physical_gpu <= 7:
        raise ValueError('PRO initialization requires explicit capacity 2 and an authorized GPU index')
    directory = paths.kit.parent / 'pro-startup'
    with _legacy_guard(paths, cancel_event=cancel_event, clock=clock) as legacy_mark, \
            startup_gate(directory / f'gpu-{physical_gpu}.lock', minimum_interval_s=startup_interval(profile), track_state=True,
                         cancel_event=cancel_event, clock=clock,
                         domain_check=lambda: verify_startup_domain(paths)) as mark:
        host_capacity = getattr(profile, 'startup_host_capacity', 2)
        with _capacity_slot(directory, host_capacity, cancel_event=cancel_event, clock=clock) as slot:
            mark.lease_fds = (mark.lock_fd, slot.fd, legacy_mark.lock_fd)
            def handoff():
                mark.transfer_to_child()
                slot.transferred = True
                legacy_mark.transfer_to_child()
            mark.handoff = handoff
            yield mark


def parent_lease(profile, paths, physical_gpu, output_dir, *, cancel_event=None):
    from mind2act_eval.cluster_profile import ClusterProfile
    if (type(profile) is not ClusterProfile or profile.startup_capacity != 2
            or profile.job_manifest is None):
        raise ValueError('startup handoff requires a recognized frozen PRO capacity-two profile')
    profile.verify_storage(output_dir, paths.root)
    if physical_gpu not in {device['index'] for device in profile.devices}:
        raise ValueError('startup handoff GPU differs from the frozen physical route')
    return initialization_gate(paths, physical_gpu, capacity=2, cancel_event=cancel_event, profile=profile)


def handoff_environment(profile, paths, physical_gpu, output_dir, mark):
    from mind2act_eval.cluster_profile import process_identity
    parent = process_identity(os.getpid())
    return json.dumps(dict(schema='mind2act_eval.pro_startup_lease.v1',
        profile_sha256=profile.sha256, uid=os.getuid(), parent_pid=parent['pid'],
        parent_birth=parent['birth'], gpu=physical_gpu,
        uuid=next(d['uuid'] for d in profile.devices if d['index'] == physical_gpu),
        output=str(output_dir), marker_sha256=paths.marker_sha256,
        gpu_fd=mark.lease_fds[0], slot_fd=mark.lease_fds[1], legacy_fd=mark.lease_fds[2]), sort_keys=True)


@contextmanager
def dispatch_gate(paths, *, interval_s=60., cancel_event=None, clock=time, profile=None):
    """Keep the existing global spacing; the next launcher pays its cooldown.

    Closing after Popen never blocks the current child's dynamic monitor.
    Durable active/cooling state retains a full fresh interval after death.
    """
    from mind2act_eval.sim import startup_gate, recover_startup_state, _write_gate_state
    from mind2act_eval.startup_paths import startup_compatibility
    minimum = 60. if profile is None else startup_interval(profile)
    if profile is not None:
        interval_s = minimum
    if type(interval_s) not in (int, float) or not isfinite(interval_s) or interval_s < minimum:
        raise ValueError('PRO dispatch requires at least 60 seconds of startup spacing' if profile is None
                         else 'PRO dispatch requires its pinned startup spacing')
    with startup_compatibility(paths, cancel_event=cancel_event), \
            startup_gate(paths.worker_launch, cancel_event=cancel_event, clock=clock,
                         domain_check=lambda: verify_startup_domain(paths)):
        with paths.worker_launch.open('r+') as stream:
            stream.seek(0)
            legacy_unpaid = paths.inherited_worker_cooldown and read_gate_state(stream.read()) is None
            recover_startup_state(stream, paths.worker_launch, 60. if legacy_unpaid else interval_s,
                inherited_cooldown=paths.inherited_worker_cooldown, clock=clock, cancel_event=cancel_event)
            _write_gate_state(stream, paths.worker_launch, 'active')
            try:
                yield lambda: None
            finally:
                _write_gate_state(stream, paths.worker_launch, 'cooling')


@contextmanager
def _borrowed_initialization(profile, paths, physical_gpu, output_dir, raw, *, clock):
    import fcntl
    from mind2act_eval.cluster_profile import ClusterProfile, process_identity
    from mind2act_eval.sim import _gate_wait, _write_gate_state
    row = json.loads(raw)
    keys = {'schema', 'profile_sha256', 'uid', 'parent_pid', 'parent_birth', 'gpu',
            'uuid', 'output', 'marker_sha256', 'gpu_fd', 'slot_fd', 'legacy_fd'}
    if (type(profile) is not ClusterProfile or profile.job_manifest is None
            or profile.startup_capacity != 2 or not isinstance(row, dict) or set(row) != keys
            or row['schema'] != 'mind2act_eval.pro_startup_lease.v1'
            or row['profile_sha256'] != profile.sha256 or row['uid'] != os.getuid()
            or row['parent_pid'] != os.getppid() or row['gpu'] != physical_gpu
            or row['uuid'] != next(d['uuid'] for d in profile.devices if d['index'] == physical_gpu)
            or row['output'] != str(output_dir) or row['marker_sha256'] != paths.marker_sha256
            or any(type(row[k]) is not int for k in ('uid', 'parent_pid', 'parent_birth', 'gpu'))
            or any(type(row[k]) is not int or row[k] < 3 for k in ('gpu_fd', 'slot_fd', 'legacy_fd'))
            or len({row[k] for k in ('gpu_fd', 'slot_fd', 'legacy_fd')}) != 3
            or process_identity(os.getppid())['birth'] != row['parent_birth']):
        raise ValueError('inherited PRO initialization lease identity mismatch')
    directory = paths.kit.parent / 'pro-startup'
    def matches(fd, path):
        held, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
        return (stat.S_ISREG(held.st_mode) and held.st_uid == os.getuid()
                and (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino))
    gpu_path = directory / f'gpu-{physical_gpu}.lock'
    if (not matches(row['gpu_fd'], gpu_path) or not matches(row['legacy_fd'], paths.kit)
            or not any(matches(row['slot_fd'], directory / f'slot-{i}.lock')
                       for i in range(profile.startup_host_capacity))):
        raise ValueError('inherited PRO initialization descriptor is outside its canonical lease')
    for fd in (row['gpu_fd'], row['slot_fd']):
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(row['legacy_fd'], fcntl.LOCK_SH | fcntl.LOCK_NB)
    verify_startup_domain(paths)
    with os.fdopen(os.dup(row['gpu_fd']), 'r+') as stream:
        stream.seek(0)
        if read_gate_state(stream.read()) != 'active':
            raise ValueError('inherited PRO initialization gate is not active')
    started = None
    def mark():
        nonlocal started
        started = clock.monotonic()
    try:
        yield mark
    finally:
        # Close-only is essential while parent/child still share the FD owner.
        try:
            with os.fdopen(os.dup(row['gpu_fd']), 'r+') as stream:
                _write_gate_state(stream, gpu_path, 'cooling')
                interval = startup_interval(profile)
                remaining = interval if started is None else max(0., interval - (clock.monotonic() - started))
                if remaining:
                    _gate_wait(remaining, clock=clock)
                _write_gate_state(stream, gpu_path, 'idle')
        finally:
            os.close(row['gpu_fd']); os.close(row['slot_fd']); os.close(row['legacy_fd'])


@contextmanager
def _serial_pro_gate(paths, *, clock):
    """A new serial PRO epoch also inherits dead parallel initialization state."""
    from mind2act_eval.sim import startup_gate, _gate_wait, _write_gate_state
    with startup_gate(paths.kit, minimum_interval_s=60., clock=clock,
                      inherited_cooldown=paths.inherited_kit_cooldown,
                      domain_check=lambda: verify_startup_domain(paths)) as mark:
        # EX on the original domain excludes every live parallel SH owner.
        # Inspect only existing canonical gates; no per-GPU/slot roots migrate.
        directory = paths.kit.parent / 'pro-startup'
        names = ['kit-launch.lock'] + [f'gpu-{gpu}.lock' for gpu in range(8)]
        with ExitStack() as locks:
            unfinished = []
            for name in names:
                path = directory / name
                if not path.exists():
                    continue
                locks.enter_context(startup_gate(path, clock=clock))
                stream = locks.enter_context(path.open('r+'))
                if read_gate_state(stream.read()) in {'active', 'cooling'}:
                    unfinished.append((stream, path))
            if unfinished:
                _gate_wait(60., clock=clock)
                for stream, path in unfinished:
                    _write_gate_state(stream, path, 'idle')
        yield mark


@contextmanager
def _constructor_gate(paths, profile, *, clock=time):
    """Bound opt-in constructors and recover prior EX owners before dispatch."""
    import fcntl
    from mind2act_eval.sim import startup_gate, _gate_wait, _write_gate_state
    directory = paths.kit.parent / 'pro-startup'
    legacy_path = directory / 'kit-launch.lock'
    dispatch = directory / 'constructor-dispatch.lock'
    interval = startup_interval(profile)
    # Waiting for a slot owns no startup gate. Ingress takes dispatch before
    # Kit; a live SDK constructor retains only Kit SH and the live-cohort SH.
    with _capacity_slot(directory / 'constructors', profile.startup_constructor_capacity,
                        cancel_event=None, clock=clock), ExitStack() as lifetime:
        with startup_gate(dispatch, domain_check=lambda: verify_startup_domain(paths), clock=clock):
            legacy = lifetime.enter_context(legacy_path.open('a+'))
            fcntl.flock(legacy, fcntl.LOCK_SH)
            active_fd = os.open(directory / 'constructors-active.lock',
                                os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            # Close the live-cohort descriptor before Kit: observing a live
            # cohort must imply at least one surviving Kit SH owner.
            lifetime.callback(os.close, active_fd)
            try:
                fcntl.flock(active_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                alone = False
            else:
                alone = True
            with dispatch.open('r+') as stream:
                if alone:
                    # No prior opt-in owner survives. Drop SH and recover the
                    # old/default EX domain, including a crash with no dispatch
                    # file. No other ingress can hold Kit while waiting here.
                    fcntl.flock(legacy, fcntl.LOCK_UN)
                    fcntl.flock(legacy, fcntl.LOCK_EX)
                    while True:
                        verify_startup_domain(paths)
                        legacy.seek(0); stream.seek(0)
                        states = [(legacy, legacy_path, read_gate_state(legacy.read())),
                                  (stream, dispatch, read_gate_state(stream.read()))]
                        # Validate both files before rewriting either one. One
                        # fresh interval covers both durable crash markers.
                        unfinished = [(handle, path) for handle, path, state in states
                                      if state in {'active', 'cooling'}]
                        if unfinished:
                            _gate_wait(interval, clock=clock)
                            for handle, path in unfinished:
                                _write_gate_state(handle, path, 'idle')
                        fcntl.flock(legacy, fcntl.LOCK_SH)
                        verify_startup_domain(paths)
                        legacy.seek(0)
                        if read_gate_state(legacy.read()) in {None, 'idle'}:
                            break
                        # flock conversion need not be atomic: a legacy EX
                        # owner may have intervened. Recheck/recover under EX.
                        fcntl.flock(legacy, fcntl.LOCK_UN)
                        fcntl.flock(legacy, fcntl.LOCK_EX)
                else:
                    # Kit SH was acquired before probing the live cohort.
                    # Its other SH owner proves no legacy EX writer intervened;
                    # still reject corrupt state before any persistent write.
                    verify_startup_domain(paths)
                    legacy.seek(0); stream.seek(0)
                    read_gate_state(legacy.read())
                    state = read_gate_state(stream.read())
                    if state in {'active', 'cooling'}:
                        _gate_wait(interval, clock=clock)
                fcntl.flock(active_fd, fcntl.LOCK_SH)
                _write_gate_state(stream, dispatch, 'cooling')
                _write_gate_state(legacy, legacy_path, 'cooling')
        yield lambda: None


def simulator_gates(paths, physical_gpu, output_dir, *, gpu_selection=None):
    """Resolve only the hash-pinned PRO profile; environment flags cannot opt in."""
    from mind2act_eval.cluster_profile import load_cluster_profile
    from mind2act_eval.sim import startup_gate
    profile = load_cluster_profile()
    inherited = os.environ.get(LEASE_ENV)
    if inherited is not None and (profile is None or profile.startup_capacity != 2):
        raise ValueError('inherited startup lease requires its frozen PRO capacity-two profile')
    if profile is None:
        return (startup_gate(paths.kit, minimum_interval_s=60.,
                             inherited_cooldown=paths.inherited_kit_cooldown,
                             domain_check=lambda: verify_startup_domain(paths)),
                nullcontext(lambda: None))
    if profile.startup_capacity == 1:
        return _serial_pro_gate(paths, clock=time), nullcontext(lambda: None)
    if gpu_selection is not None:
        raise ValueError('PRO startup capacity cannot be combined with Slurm allocation')
    profile.verify_storage(output_dir, paths.root)
    if physical_gpu not in {device['index'] for device in profile.devices}:
        raise ValueError('PRO initialization GPU is outside the pinned device pool')
    gate = (initialization_gate(paths, physical_gpu, capacity=profile.startup_capacity, profile=profile) if inherited is None else
            _borrowed_initialization(profile, paths, physical_gpu, output_dir, inherited, clock=time))
    constructor = (startup_gate(paths.kit.parent / 'pro-startup' / 'kit-launch.lock',
                                minimum_interval_s=startup_interval(profile),
                                domain_check=lambda: verify_startup_domain(paths))
                   if getattr(profile, 'startup_constructor_capacity', 1) == 1
                   else _constructor_gate(paths, profile))
    return gate, constructor
