"""Opt-in cooperative PRO capacity in the existing canonical launcher domain.

Every participating parent holds batch.lock SH; old EX launchers still block.
Reservations cover complete campaign caps, with a fixed shared 512 GiB ceiling.
This module never adopts, signals, or changes another campaign's processes.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid


GIB = 1024 ** 3
MAX_HOST_RSS = 512 * GIB


class CapacityLaunchError(RuntimeError):
    """A child exists even though its durable lease commit failed."""
    def __init__(self, child, identity, cause):
        super().__init__('started worker capacity commit failed: ' + str(cause))
        self.child, self.identity = child, dict(identity)


class CapacityUnavailable(ValueError):
    """Known occupied capacity; waiting here has not executed an episode."""


class LinuxKernel:
    """Fresh system observations; the CPU tests replace only this OS adapter."""
    def identity(self, pid):
        from mind2act_eval.cluster_profile import process_identity
        row = process_identity(pid)
        status = dict(line.split(':', 1) for line in Path('/proc', str(pid), 'status').read_text().splitlines() if ':' in line)
        row['uid'] = int(status['Uid'].split()[0])
        return row

    def members(self, path):
        result = []
        for pid in Path(path, 'cgroup.procs').read_text().split():
            try:
                result.append(self.identity(int(pid)))
            except (FileNotFoundError, ProcessLookupError):
                continue
        return result

    def memory(self, parents, scopes):
        from mind2act_eval.batch_resources import _memory_scope
        from mind2act_eval.campaign_resources import _host_available, _membership
        _membership()  # Prove the real unremapped cgroup2 mount.
        mount = Path('/sys/fs/cgroup')
        ancestors = {}
        for parent in parents:
            parent = Path(parent)
            if not parent.is_relative_to(mount) or parent.resolve() != parent:
                raise ValueError('capacity parent is outside the inspected hierarchy')
            path = parent
            while path.is_relative_to(mount):
                ancestors[str(path)] = _memory_scope(path, mount=mount, leaf=parent)
                if path == mount:
                    break
                path = path.parent
        return dict(host_available_bytes=_host_available(), ancestors=list(ancestors.values()),
                    scopes={p:_memory_scope(Path(p), mount=mount, leaf=Path(p)) for p in scopes})

    def gpu(self, profile, index, *, admitted, scope, require_headroom=True):
        from mind2act_eval.cluster_profile import check_pro_gpu, check_pro_shared_gpu
        capacity_warn = (getattr(profile, 'shared_gpu_admission_policy', 'error') == 'warn'
                         and profile.max_workers_per_gpu > 1)
        observation_warn = (getattr(profile, 'shared_gpu_admission_policy', 'error') == 'warn'
                            and getattr(profile, 'unregistered_process_policy', 'error') == 'warn')
        # Authorize before even collecting telemetry, as for direct checks.
        if hasattr(scope, 'check_gpu'):
            if profile != scope.profile:
                raise ValueError('PRO GPU profile differs from owned epoch')
            scope.admission.authorization(scope)
        else:
            profile.authorize()
        from mind2act_eval.pro_gpu_snapshot import host_gpu_snapshot, SnapshotExpired
        options = dict(admitted=admitted)
        if hasattr(scope, 'check_gpu'):
            options['_epoch'] = scope  # O1-O3's already verified admission.
        for attempt in range(3):
            try:
                options['_snapshot'] = host_gpu_snapshot(profile.lock_root(), force_refresh=attempt > 0)
                if capacity_warn and require_headroom:
                    receipt = check_pro_shared_gpu(profile, index, **options)
                else:
                    receipt = check_pro_gpu(profile, index, require_idle=False, **options)
                break
            except SnapshotExpired:
                # A valid observation may age out during publication/identity
                # checks. Refresh and repeat every ownership check; never turn
                # this clock race directly into termination of a healthy child.
                if attempt == 2:
                    raise
        # Explicit shared WARN retains the profile's original observation
        # policy. Observed foreign contexts never enter registry ownership.
        if receipt.get('process_warnings') and not observation_warn:
            raise ValueError('capacity launch cannot adopt foreign GPU contexts')
        if require_headroom and not capacity_warn and receipt['device']['free_mib'] < profile.min_free_vram_mib:
            raise ValueError('capacity launch lacks free VRAM')
        return receipt


def _pinned(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('capacity epoch requires a path/SHA pin')
    raw = Path(pin['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('capacity epoch SHA mismatch')
    return json.loads(raw)


def _lock(path, operation, *, blocking=False):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError('capacity lock must be an owned regular inode')
        fcntl.flock(fd, operation if blocking else operation | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


class ProCampaignAdmission:
    @classmethod
    def from_env(cls, profile, *, kernel=None):
        path = os.environ.get('MIND2ACT_EVAL_PRO_CAPACITY_EPOCH')
        digest = os.environ.get('MIND2ACT_EVAL_PRO_CAPACITY_EPOCH_SHA256')
        if path is None and digest is None:
            return None
        if not path or not digest:
            raise ValueError('capacity epoch requires both path and SHA')
        if profile is None:
            raise ValueError('capacity epochs are PRO-only')
        return cls(profile, dict(path=path, sha256=digest), kernel=kernel)

    def __init__(self, profile, epoch_pin, *, kernel=None):
        self.profile, self.pin = profile, epoch_pin
        self.kernel = LinuxKernel() if kernel is None else kernel
        self.policy = _pinned(epoch_pin)
        profile.verify_host()
        self.root = Path(profile.lock_root())
        p = self.policy
        experimental = getattr(profile, 'concurrency_experiment', None) == 'g05-clean-concurrency-4-6-8-v1'
        host_limit = 1024 * GIB if experimental else MAX_HOST_RSS
        if (p.get('schema') != 'mind2act_eval.pro_capacity_epoch.v1'
                or p.get('hostname') != profile.hostname or p.get('machine_id_sha256') != profile.machine_id_sha256
                or p.get('uid') != os.getuid() or p.get('lock_root') != str(self.root)
                or type(p.get('host_rss_limit_bytes')) is not int or not 0 < p['host_rss_limit_bytes'] <= host_limit):
            raise ValueError('capacity epoch differs from actual host/canonical budget')
        expected = dict(sha256=profile.sha256, rss_limit_bytes=profile.rss_limit_bytes,
            cgroup_parent=str(profile.cgroup_parent), max_workers_per_gpu=profile.max_workers_per_gpu,
            devices=[d['uuid'] for d in profile.devices])
        if expected not in p['profiles']:
            raise ValueError('profile is outside the frozen capacity epoch')
        self.devices = {d['index']:d for d in p['devices']}
        for device in profile.devices:
            declared = self.devices.get(device['index'], {})
            if (any(declared.get(k) != device[k] for k in ('uuid', 'pci_bus_id'))
                    or type(declared.get('slots')) is not int or not 1 <= declared['slots'] <= (8 if experimental else 3)
                    or profile.max_workers_per_gpu > declared['slots']):
                raise ValueError('capacity device/slot declaration mismatch')
        self.owner = self.kernel.identity(os.getpid())
        self.token = uuid.uuid4().hex
        self.fd = None
        self.workers = {}
        self.scope = None
        self.receipts = {}
        self.observation_warnings = []
        self.path = self.root / 'pro-capacity.json'
        self.lifetime = None

    @staticmethod
    def _regular(fd, path):
        info = os.fstat(fd)
        linked = os.stat(path, follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (linked.st_dev, linked.st_ino)):
            raise ValueError('capacity state/lock must retain its owned regular identity')
        return [info.st_dev, info.st_ino]

    def _initialize(self):
        """Only exclusive creation starts a lifetime; the mutex is its tombstone."""
        path = self.root / 'pro-capacity.lock'
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            return  # Joining an established lifetime must only read/verify it.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            identity = self._regular(fd, path)
            lifetime = uuid.uuid4().hex
            marker = dict(schema='mind2act_eval.pro_capacity_lifetime.v1', lifetime_id=lifetime)
            with os.fdopen(os.dup(fd), 'w') as stream:
                json.dump(marker, stream, sort_keys=True)
                stream.flush(); os.fsync(stream.fileno())
            state = dict(schema='mind2act_eval.pro_capacity_state.v1', lifetime_id=lifetime,
                mutex_identity=identity, epoch_sha256=self.pin['sha256'], campaigns={})
            # Existing state (even without its mutex) can never be reset/adopted.
            output = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(output, 'w') as stream:
                json.dump(state, stream, sort_keys=True)
                stream.flush(); os.fsync(stream.fileno())
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            os.close(fd)  # Partial initialization keeps the tombstone, fails closed.

    @contextmanager
    def _transaction(self, *, write=True):
        path = self.root / 'pro-capacity.lock'
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)  # Never create during a transaction.
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            identity = self._regular(fd, path)
            marker = json.loads(os.read(fd, 4096))
            if (marker.get('schema') != 'mind2act_eval.pro_capacity_lifetime.v1'
                    or not isinstance(marker.get('lifetime_id'), str) or not marker['lifetime_id']):
                raise ValueError('invalid capacity lifetime marker')
            state_fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(state_fd, 'rb') as stream:
                self._regular(stream.fileno(), self.path)
                state = json.load(stream)
            if (state.get('schema') != 'mind2act_eval.pro_capacity_state.v1'
                    or state.get('lifetime_id') != marker['lifetime_id']
                    or state.get('mutex_identity') != identity or not isinstance(state.get('campaigns'), dict)
                    or self.lifetime is not None and self.lifetime != marker['lifetime_id']):
                raise ValueError('invalid capacity state/lifetime identity')
            self.lifetime = marker['lifetime_id']
            if state['epoch_sha256'] != self.pin['sha256']:
                if state['campaigns']:
                    raise ValueError('another capacity epoch still owns reservations')
                state['epoch_sha256'] = self.pin['sha256']
            yield state
            if not write:
                return
            self._regular(fd, path)
            temporary = self.path.with_name('.pro-capacity-' + uuid.uuid4().hex)
            fd_out = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd_out, 'w') as stream:
                json.dump(state, stream, sort_keys=True)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            os.close(fd)

    def _retire_exited_owners(self, state):
        # Called under the capacity mutex. A disappeared owner is an exited
        # owner, not a failed /proc read; never discard its surviving workers.
        for token, row in list(state['campaigns'].items()):
            try:
                owner = self.kernel.identity(row['owner']['pid'])
            except (FileNotFoundError, ProcessLookupError):
                owner = None
            if owner is not None and owner['birth'] == row['owner']['birth']:
                if owner['uid'] != self.policy['uid']:
                    raise ValueError('capacity owner PID/birth changed')
                continue
            if row['owner']['pid'] == self.owner['pid']:
                raise ValueError('capacity owner PID/birth changed')
            for worker in row['workers'].values():
                try:
                    observed = self.kernel.identity(worker['identity']['pid'])
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if observed['birth'] == worker['identity']['birth']:
                    raise CapacityUnavailable('exited capacity owner retains a live worker')
            if row['cgroup'] is not None:
                try:
                    members = self.kernel.members(row['cgroup'])
                except (FileNotFoundError, ProcessLookupError):
                    members = []
                if members:
                    raise CapacityUnavailable('exited capacity owner retains campaign members')
            del state['campaigns'][token]
    def _memory(self, state):
        # Observation is read-only: _launch uses a ledger copy outside its
        # mutex and must compare it unchanged at commit. Retirement belongs
        # to _retire_exited_owners in the existing admission/bind transaction.
        campaigns = state['campaigns'].values()
        total = sum(row['cap'] for row in campaigns)
        if total > self.policy['host_rss_limit_bytes']:
            raise ValueError('joint campaign caps exceed capacity budget')
        parents = {row['parent'] for row in campaigns}
        scopes = [row['cgroup'] for row in campaigns if row['cgroup'] is not None]
        observed = self.kernel.memory(parents, scopes)
        remaining = 0
        remaining_rows = []
        for row in campaigns:
            try:
                owner = self.kernel.identity(row['owner']['pid'])
            except (FileNotFoundError, ProcessLookupError) as exc:
                raise CapacityUnavailable('capacity owner exited during observation; retry') from exc
            if owner['birth'] != row['owner']['birth']:
                raise CapacityUnavailable('capacity owner changed during observation; retry')
            if owner['uid'] != self.policy['uid']:
                raise ValueError('capacity owner PID/birth changed')
            if row['cgroup'] is None:
                remaining += row['cap']  # Explicit reserved/not-launched state, never missing telemetry.
                remaining_rows.append((row, row['cap']))
                continue
            memory = observed['scopes'][row['cgroup']]
            if memory['limit_bytes'] != row['cap'] or memory['current_bytes'] is None:
                raise ValueError('unknown or changed campaign cgroup cap')
            if memory['current_bytes'] >= row['cap']:
                raise ValueError('campaign cgroup budget exhausted')
            if memory['high_bytes'] is not None and memory['high_bytes'] - memory['current_bytes'] < row['reserve']:
                raise ValueError('campaign memory.high lacks admission headroom')
            remaining += row['cap'] - memory['current_bytes']
            remaining_rows.append((row, row['cap'] - memory['current_bytes']))
        reserve = max((row['reserve'] for row in campaigns), default=self.profile.host_reserve_bytes)
        if observed['host_available_bytes'] < remaining + reserve:
            raise ValueError('host lacks jointly reserved memory headroom')
        for ancestor in observed['ancestors']:
            affected = [(row, left) for row, left in remaining_rows
                        if Path(row['cgroup'] or row['parent']).is_relative_to(Path(ancestor['path']))]
            if not affected:
                continue
            required = sum(left for _, left in affected) + max(row['reserve'] for row, _ in affected)
            current = ancestor['current_bytes']
            if current is None and (ancestor.get('path') != '/sys/fs/cgroup'
                    or not ancestor.get('root_controller_evidence')):
                raise ValueError('unknown memory ancestor is not unlimited')
            for field in ('limit_bytes', 'high_bytes'):
                if ancestor[field] is not None and (current is None or ancestor[field] - current < required):
                    raise ValueError('finite ancestor lacks jointly reserved memory headroom')
        return observed

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.fd = _lock(self.root / 'batch.lock', fcntl.LOCK_SH)
        try:
            self._initialize()
            with self._transaction() as state:
                self._retire_exited_owners(state)
                state['campaigns'][self.token] = dict(owner=self.owner, profile_sha256=self.profile.sha256,
                    cap=self.profile.rss_limit_bytes, parent=str(self.profile.cgroup_parent),
                    reserve=self.profile.host_reserve_bytes, cgroup=None, workers={})
                self._memory(state)
            return self
        except BaseException:
            os.close(self.fd); self.fd = None
            raise

    def launch(self, index, scope, argv, *, receipt_path=None, case_key=None,
               wait_for_slot=False, cancel_event=None, before_launch=None, **options):
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError('capacity intake cancelled before Popen')
            started = []
            try:
                return self._launch(index, scope, argv, receipt_path=receipt_path, case_key=case_key,
                                    _started=started, before_launch=before_launch, **options)
            except BaseException as exc:
                if started:
                    child, identity = started[0]
                    self.workers[child.pid]['uncommitted'] = True
                    raise CapacityLaunchError(child, identity, exc) from exc
                if not wait_for_slot or not isinstance(exc, CapacityUnavailable):
                    raise
                time.sleep(.1)

    def _launch(self, index, scope, argv, *, receipt_path, case_key, _started, before_launch=None, **options):
        if self.fd is None or scope.profile.sha256 != self.profile.sha256:
            raise ValueError('launch requires this admitted campaign scope')
        self._bind_scope(scope)
        device = next((d for d in self.profile.devices if d['index'] == index), None)
        if device is None:
            raise ValueError('launch GPU is outside profile')
        # Observe outside the host mutex. Commit only against the exact ledger
        # that was inspected, so a concurrent launch/exit cannot spend a slot
        # or change ownership between verification and the atomic handoff.
        with self._transaction(write=False) as observed:
            pass
        self._assert_owner(observed['campaigns'][self.token])
        with self._observing_ledger(observed):
            self._memory(observed)
            admitted = self._admitted(observed, device['uuid'])
            receipt = self.kernel.gpu(self.profile, index, admitted=admitted, scope=scope)
        with self._transaction() as state:
            if state != observed:
                raise CapacityUnavailable('capacity ledger changed during GPU observation; retry')
            row = state['campaigns'][self.token]
            self._assert_owner(row)
            row['cgroup'] = str(scope.directory)
            if sum(w['uuid'] == device['uuid'] for w in row['workers'].values()) >= self.profile.max_workers_per_gpu:
                raise CapacityUnavailable('frozen profile worker capacity is already leased')
            used = {w['slot'] for campaign in state['campaigns'].values() for w in campaign['workers'].values()
                    if w['uuid'] == device['uuid']}
            slot = next((s for s in range(self.devices[index]['slots']) if s not in used), None)
            if slot is None:
                raise CapacityUnavailable('physical GPU capacity is already leased')
            if self.observation_warnings:
                receipt = dict(receipt, capacity_observation_warnings=list(self.observation_warnings))
            lease_id = uuid.uuid4().hex
            safe_uuid = hashlib.sha256(device['uuid'].encode()).hexdigest()
            slot_fd = _lock(self.root / ('pro-device-' + safe_uuid + '-%d.lock' % slot), fcntl.LOCK_EX)
            case_key = case_key if case_key is not None else dict(profile=self.profile.sha256, gpu=index, argv=argv)
            case_sha = hashlib.sha256(json.dumps(case_key, sort_keys=True, allow_nan=False).encode()).hexdigest()
            case_fd = None
            try:
                try:
                    case_fd = _lock(self.root / ('pro-case-' + case_sha + '.lock'), fcntl.LOCK_EX)
                except BlockingIOError as exc:
                    raise CapacityUnavailable('original case is already claimed') from exc
                options['pass_fds'] = tuple(options.get('pass_fds', ())) + (self.fd, slot_fd, case_fd)
                if receipt_path is not None:
                    payload = dict(receipt, profile_sha256=self.profile.sha256, resources=scope.snapshot(),
                        capacity_epoch=self.pin, capacity_lease=dict(lease_id=lease_id, uuid=device['uuid'],
                        slot=slot, case_sha256=case_sha, owner=self.owner))
                    with Path(receipt_path).open('x') as stream:
                        json.dump(payload, stream, sort_keys=True, allow_nan=False)
                        stream.flush(); os.fsync(stream.fileno())
                # All cooperative capacity/case waits precede selected input
                # validation and startup handoff, still under the caller's guard.
                if before_launch is not None:
                    before_launch()
                child, identity = scope.launch(argv, **options)
            except BaseException:
                os.close(slot_fd)
                if case_fd is not None:
                    os.close(case_fd)
                raise
            self.workers[child.pid] = dict(fd=slot_fd, case_fd=case_fd, lease_id=lease_id, identity=dict(identity), child=child)
            _started.append((child, dict(identity)))
            row['workers'][lease_id] = dict(uuid=device['uuid'], slot=slot, case_sha256=case_sha, identity=dict(identity))
            self.scope = scope
            self.receipts[child.pid] = receipt
            return child, identity

    def _bind_scope(self, scope):
        path = Path(scope.directory)
        if (path.parent != self.profile.cgroup_parent
                or not path.name.startswith('mind2act-pro5000-%d-' % self.policy['uid'])
                or (scope.parent_identity['pid'], scope.parent_identity['birth']) != (self.owner['pid'], self.owner['birth'])
                or self.scope is not None and self.scope.directory != path):
            raise ValueError('scope is not this parent\'s owned campaign cgroup')
        with self._transaction() as state:
            self._retire_exited_owners(state)
            if any(k != self.token and row['cgroup'] == str(path) for k, row in state['campaigns'].items()):
                raise ValueError('campaign cgroup already belongs to another reservation')
            state['campaigns'][self.token]['cgroup'] = str(path)
        self.scope = scope
        from mind2act_eval.campaign_resources import CampaignResources
        if type(scope) is CampaignResources:
            scope.capacity_admission = self

    @contextmanager
    def _observing_ledger(self, observed):
        try:
            yield
        except (FileNotFoundError, ProcessLookupError) as exc:
            # A peer may remove its empty scope and retire its reservation
            # after our unlocked ledger copy. Only a committed ledger change
            # proves that this observation can be retried; never treat a
            # missing/unreadable scope as empty or waive ownership checks.
            with self._transaction(write=False) as state:
                self._assert_owner(state['campaigns'][self.token])
                if state != observed:
                    raise CapacityUnavailable('capacity ledger changed during disappearing observation; retry') from exc
            raise

    def _admitted(self, state, gpu_uuid=None):
        admitted = {}
        for campaign in state['campaigns'].values():
            if campaign['cgroup'] is None:
                continue
            members = {p['pid']:p for p in self.kernel.members(campaign['cgroup'])}
            roots = {w['identity']['pid']:w for w in campaign['workers'].values()}
            for member in members.values():
                current, seen, lease = member, set(), None
                while current['pid'] not in seen:
                    seen.add(current['pid'])
                    if current['uid'] != self.policy['uid']:
                        raise ValueError('unregistered other-UID campaign member')
                    candidate = roots.get(current['pid'])
                    if candidate is not None:
                        if current['birth'] != candidate['identity']['birth']:
                            raise ValueError('leased worker PID was reused')
                        lease = candidate
                        break
                    current = members.get(current['ppid'])
                    if current is None:
                        break
                if lease is None:
                    if (getattr(self.profile, 'unregistered_process_policy', 'error') == 'warn'
                            and getattr(self.profile, 'shared_gpu_admission_policy', 'error') == 'warn'):
                        warning = dict(pid=member['pid'], birth=member['birth'],
                            cgroup=campaign['cgroup'], adopted=False,
                            reason='unleased same-UID campaign member; observation only')
                        if not hasattr(self, 'observation_warnings'):
                            self.observation_warnings = []
                        if warning not in self.observation_warnings:
                            self.observation_warnings.append(warning)
                        continue
                    raise ValueError('unleased/unregistered campaign process; never adopted')
                if gpu_uuid is None or lease['uuid'] == gpu_uuid:
                    admitted[member['pid']] = member['birth']
        return admitted

    def monitor_gpu(self, index, scope):
        if self.fd is None or self.scope is not scope:
            raise ValueError('monitor requires this owned campaign scope')
        device = next((d for d in self.profile.devices if d['index'] == index), None)
        if device is None:
            raise ValueError('monitor GPU is outside profile')
        while True:
            with self._transaction(write=False) as observed:
                self._assert_owner(observed['campaigns'][self.token])
            try:
                with self._observing_ledger(observed):
                    admitted = self._admitted(observed, device['uuid'])
                    receipt = self.kernel.gpu(self.profile, index, admitted=admitted, scope=scope, require_headroom=False)
            except CapacityUnavailable:
                continue
            with self._transaction(write=False) as state:
                self._assert_owner(state['campaigns'][self.token])
                if state != observed:
                    continue
            if self.observation_warnings:
                receipt = dict(receipt, capacity_observation_warnings=list(self.observation_warnings))
            return receipt

    def worker_exited(self, child):
        record = self.workers.get(child.pid)
        if record is None or record['child'] is not child or child.poll() is None:
            raise ValueError('lease release requires the original exited worker')
        self.scope.assert_worker_exited(child.pid)
        with self._transaction() as state:
            row = state['campaigns'][self.token]
            self._assert_owner(row)
            self._admitted(dict(campaigns={self.token:row}))
            committed = row['workers'].get(record['lease_id'])
            if committed is None and not record.get('uncommitted'):
                raise ValueError('worker lease disappeared without owned commit-failure evidence')
            if committed is not None and committed['identity'] != record['identity']:
                raise ValueError('worker lease ownership changed')
            row['workers'].pop(record['lease_id'], None)
            # A new launcher must not see the free ledger slot while its old
            # kernel lock remains held. The transaction spans both releases.
            os.close(record.pop('fd'))
            os.close(record.pop('case_fd'))
            if self.workers.get(child.pid) is record:
                del self.workers[child.pid]
            self.receipts.pop(child.pid, None)
        return dict(status='released', epoch_sha256=self.pin['sha256'], lease_id=record['lease_id'],
                    pid=child.pid, birth=record['identity']['birth'])

    def __exit__(self, *error):
        if self.fd is None:
            return
        if self.workers:
            raise ValueError('live/unknown worker leases retained')
        with self._transaction() as state:
            row = state['campaigns'][self.token]
            self._assert_owner(row)
            if row['workers']:
                raise ValueError('durable worker leases are still owned')
            members = []
            if self.scope is not None:
                removed = getattr(self.scope, 'empty_removal', None)
                if removed is not None:
                    from mind2act_eval.campaign_resources import CampaignResources
                    if (type(self.scope) is not CampaignResources or not isinstance(removed,dict)
                            or removed.get('verified_empty') is not True
                            or removed.get('directory') != str(self.scope.directory)
                            or (removed.get('owner_pid'),removed.get('owner_birth')) !=
                               (self.owner['pid'],self.owner['birth'])
                            or self.scope.directory.exists()):
                        raise ValueError('owned empty-scope removal proof differs from reservation')
                else:
                    members = self.kernel.members(str(self.scope.directory))
            if members:
                if (getattr(self.profile, 'unregistered_process_policy', 'error') != 'warn'
                        or getattr(self.profile, 'shared_gpu_admission_policy', 'error') != 'warn'):
                    raise ValueError('unknown campaign members retain their bounded reservation')
                # Validate UID and census identity without admitting or signalling
                # these processes. Keep the cap, scope and SH reservation intact.
                self._admitted(dict(campaigns={self.token:row}))
                row['cleanup'] = dict(status='retained_unknown_observation',
                    observations=list(self.observation_warnings), at_unix_s=time.time())
                return
            if self.scope is not None:
                from mind2act_eval.campaign_resources import CampaignResources
                if type(self.scope) is CampaignResources and getattr(self.scope, 'empty_removal', None) is None:
                    # No peer may sample a ledger row after its cgroup vanished.
                    # Kernel removal proves empty before the reservation retires.
                    self.scope._remove_empty_scope()
                    if self.scope.empty_removal is None:
                        return  # Unknown-only members appeared; preserve their bound.
            del state['campaigns'][self.token]
        os.close(self.fd); self.fd = None

    def _assert_owner(self, row):
        observed = self.kernel.identity(os.getpid())
        if (row['owner']['pid'], row['owner']['birth'], observed['birth'], observed['uid']) != (
                self.owner['pid'], self.owner['birth'], self.owner['birth'], self.policy['uid']):
            raise ValueError('capacity owner PID/birth changed')

    def close(self):
        self.__exit__(None, None, None)
