"""PRO-only owned process scope. Kernel membership bounds descendants, including daemons.

Only a newly created campaign cgroup is changed. Parent controllers, old processes,
and other campaigns are never moved, signalled, or reconfigured. Failure to obtain
this containment is an admission failure, not a fallback to partial RSS accounting.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time
import uuid
import signal

from mind2act_eval.cluster_profile import process_identity


def _counter(path, *, optional=False):
    try:
        value = Path(path).read_text().strip()
    except FileNotFoundError:
        if optional:
            return None
        raise
    if value == 'max':
        return None
    if not value.isdecimal():
        raise ValueError('invalid cgroup memory counter')
    return int(value)


def _host_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    fields = values['MemAvailable'].split()
    if len(fields) != 2 or fields[1] != 'kB':
        raise ValueError('host MemAvailable is unavailable')
    return int(fields[0]) * 1024


def _membership():
    rows = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    if len(rows) != 1 or not rows[0].startswith('/') or '..' in Path(rows[0]).parts:
        raise ValueError('unmappable unified cgroup membership')
    # The host profile pins this root. A remapped mount/namespace is unsupported.
    mounts = [line for line in Path('/proc/self/mountinfo').read_text().splitlines()
              if ' - cgroup2 ' in line]
    if len(mounts) != 1 or mounts[0].split()[3:5] != ['/', '/sys/fs/cgroup']:
        raise ValueError('PRO requires its inspected unremapped cgroup mount')
    return Path('/sys/fs/cgroup') / rows[0].lstrip('/')


def verify_parent(parent):
    current = _membership()
    if not current.is_relative_to(parent):
        raise ValueError('campaign cgroup parent is not a current membership ancestor')
    # Moving the new child to a sibling must not bypass any finite session cap.
    while current != parent:
        if _counter(current / 'memory.max') is not None:
            raise ValueError('PRO scope would bypass a finite inherited memory cap')
        current = current.parent
    if 'memory' not in (parent / 'cgroup.subtree_control').read_text().split():
        raise ValueError('memory controller is not already delegated; parent is not changed')


def memory_headroom(parent):
    scopes = []
    path = parent
    while True:
        limit = _counter(path / 'memory.max', optional=path == Path('/sys/fs/cgroup'))
        if limit is not None:
            current = _counter(path / 'memory.current')
            if _counter(path / 'memory.max') != limit:
                raise ValueError('cgroup limit changed during admission')
            scopes.append(dict(path=str(path), limit_bytes=limit, current_bytes=current))
        if path == Path('/sys/fs/cgroup'):
            break
        if not path.is_relative_to('/sys/fs/cgroup'):
            raise ValueError('cgroup ancestor escape')
        path = path.parent
    return scopes


class CampaignResources:
    def __init__(self, profile, receipt_path):
        from copy import deepcopy
        self.profile = deepcopy(profile)
        self.receipt_path = Path(receipt_path)
        self.directory = profile.cgroup_parent / f'mind2act-pro5000-{os.getuid()}-{uuid.uuid4().hex}'
        self.parent_identity = process_identity(os.getpid())
        self.registered = {}
        self.roots = set()
        self.mutex = threading.RLock()
        self.peak_rss_bytes = 0
        self.last_receipt = None
        self.unregistered_observations = {}
        self.epoch_active = False
        self.admission = None
        self.gpu_diagnostic_warnings = {}
        self.empty_removal = None
        self.capacity_admission = None

    def _observe_unregistered(self, current):
        key = (current['pid'], current['birth'])
        if key in self.unregistered_observations:
            return
        warning = dict(schema='mind2act_eval.pro_process_warning.v1',
            reason='unregistered/orphan process in campaign cgroup',
            policy=getattr(self.profile, 'unregistered_process_policy', 'error'),
            profile_sha256=getattr(self.profile, 'sha256', None),
            cluster_config_sha256=getattr(self.profile, 'cluster_config_sha256', None),
            cgroup=str(self.directory), process=current, observed_unix_s=time.time(),
            ownership='unverified; observation only, never admitted or signalled')
        # Persist the first identity before any later budget/GPU check can fail.
        path = self.receipt_path.with_suffix('.process-warnings.jsonl')
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a') as stream:
            stream.write(json.dumps(warning, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.unregistered_observations[key] = warning

    def admit_memory(self, *, include_budget=False):
        available = _host_available()
        reserve = self.profile.host_reserve_bytes
        required = reserve + (self.profile.rss_limit_bytes if include_budget else 0)
        if available < required:
            raise ValueError('PRO host MemAvailable below reserve/budget')
        scopes = memory_headroom(self.profile.cgroup_parent)
        if any(s['limit_bytes'] - s['current_bytes'] < required for s in scopes):
            raise ValueError('PRO ancestor cgroup headroom below reserve/budget')
        return dict(host_available_bytes=available, reserve_bytes=reserve, ancestor_limits=scopes)

    def _record_rss_exhaustion(self, rss, supervisor, live, unregistered):
        """Best-effort first failure evidence; the original guard still fails."""
        counters = {}
        for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.events'):
            try:
                counters[name] = (self.directory / name).read_text().strip()
            except OSError as exc:
                counters[name] = dict(unavailable=type(exc).__name__)
        value = dict(schema='mind2act_eval.pro_rss_exhaustion.v1',
            sampled_unix_s=time.time(), cgroup=str(self.directory),
            aggregate_rss_bytes=rss, rss_limit_bytes=self.profile.rss_limit_bytes,
            supervisor=supervisor, registered_processes=list(live.values()),
            unregistered_processes=list(unregistered.values()), cgroup_counters=counters,
            qualification='Failure-time observations; individual proc/sysfs reads are not atomic')
        path = self.receipt_path.with_suffix('.rss-exhausted.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())

    def __enter__(self):
        from .cluster_profile import load_cluster_profile
        from .pro_epoch import ProEpochAdmission
        if self.admission is not None or load_cluster_profile() != self.profile:
            raise ValueError('PRO epoch requires its current pinned profile and a new scope')
        self.admission = ProEpochAdmission(self)
        verify_parent(self.profile.cgroup_parent)
        self.admit_memory(include_budget=True)
        self.directory.mkdir(mode=0o700)  # exclusive; never adopt an existing scope
        try:
            if (self.directory / 'cgroup.procs').read_text().strip():
                raise ValueError('new campaign cgroup unexpectedly contains a process')
            (self.directory / 'memory.max').write_text(str(self.profile.rss_limit_bytes))
            if _counter(self.directory / 'memory.max') != self.profile.rss_limit_bytes:
                raise ValueError('own campaign memory.max did not apply')
            self.snapshot()
            self.epoch_active = True
            return self
        except BaseException:
            self.directory.rmdir()  # only our new empty directory, never cgroup.kill
            raise

    def register(self, child):
        with self.mutex:
            identity = process_identity(child.pid)
            if identity.get('state') in {'Z','X'} or identity['ppid'] != os.getpid() or identity['session'] != child.pid:
                raise ValueError('Popen child parent/session identity mismatch')
            if identity['birth'] < self.parent_identity['birth']:
                raise ValueError('Popen child birth predates its supervisor')
            self.registered[child.pid] = dict(identity, root=child.pid, registered_at=time.monotonic())
            self.roots.add(child.pid)
            return identity

    def snapshot(self):
        with self.mutex:
            verify_parent(self.profile.cgroup_parent)
            if any(p.is_dir() for p in self.directory.iterdir()):
                raise ValueError('unregistered nested campaign cgroup')
            members = {int(p) for p in (self.directory / 'cgroup.procs').read_text().split()}
            live = {}
            for pid, old in self.registered.items():
                try:
                    current = process_identity(pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if current['birth'] != old['birth']:
                    if pid in members:
                        raise ValueError('registered campaign PID was reused inside its cgroup')
                    continue  # Retired PID reused elsewhere is foreign, never adopted/signalled.
                if current.get('state') in {'Z','X'}:
                    continue  # Zombies have left cgroup.procs but have not been reaped yet.
                # A registered root can briefly be blocked at its join pipe.
                if pid not in members:
                    joined = old.get('joined', False)
                    if joined or pid not in self.roots:
                        raise ValueError('registered live process escaped campaign cgroup')
                    if time.monotonic() - old['registered_at'] > 5:
                        raise ValueError('registered worker failed to join campaign cgroup')
                live[pid] = dict(current, root=old['root'], joined=pid in members,
                                 registered_at=old['registered_at'])
            pending = set(members) - set(live)
            unregistered = {}
            while pending:
                progress = False
                for pid in list(pending):
                    try:
                        current = process_identity(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        pending.remove(pid)
                        progress = True
                        continue
                    parent = live.get(current['ppid'])
                    if current.get('state') in {'Z','X'}:
                        pending.remove(pid)
                        progress = True
                        continue
                    unregistered[pid] = current
                    if parent is not None and parent['birth'] <= current['birth']:
                        live[pid] = dict(current, root=parent['root'], joined=True,
                                         registered_at=time.monotonic())
                        pending.remove(pid)
                        unregistered.pop(pid, None)
                        progress = True
                if not progress:
                    for pid in pending:
                        self._observe_unregistered(unregistered[pid])
                    if getattr(self.profile, 'unregistered_process_policy', 'error') != 'warn':
                        raise ValueError('unregistered/orphan process in campaign cgroup')
                    break
            unregistered = {pid: unregistered[pid] for pid in pending}
            # Preserve retired birth identities so PID reuse never adopts another job.
            self.registered.update(live)
            supervisor = process_identity(os.getpid())
            if supervisor['birth'] != self.parent_identity['birth']:
                raise ValueError('campaign supervisor birth changed')
            rss = supervisor['rss_bytes'] + sum(x['rss_bytes'] for x in [*live.values(), *unregistered.values()])
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            if rss >= self.profile.rss_limit_bytes:
                try:
                    self._record_rss_exhaustion(rss, supervisor, live, unregistered)
                except OSError:
                    pass  # Existing receipt or unavailable output cannot waive the guard.
                raise ValueError('aggregate registered campaign RSS exhausted (including unregistered observations)')
            memory = self.admit_memory()
            cap = _counter(self.directory / 'memory.max')
            used = _counter(self.directory / 'memory.current')
            if cap != self.profile.rss_limit_bytes or used >= cap:
                raise ValueError('campaign memory cap changed/exhausted')
            self.last_receipt = dict(schema='mind2act_eval.pro_resources.v1',
                cgroup=str(self.directory), processes=list(live.values()), rss_bytes=rss,
                registered_processes=list(self.registered.values()), supervisor=supervisor,
                unregistered_process_policy=getattr(self.profile, 'unregistered_process_policy', 'error'),
                cluster_config_sha256=getattr(self.profile, 'cluster_config_sha256', None),
                unregistered_processes=list(unregistered.values()),
                process_warnings=list(self.unregistered_observations.values()),
                gpu_diagnostic_warnings=list(self.gpu_diagnostic_warnings.values()),
                peak_sampled_rss_bytes=self.peak_rss_bytes, cgroup_memory_current_bytes=used,
                memory=memory, sampled_unix_s=time.time())
            return self.last_receipt

    def admitted_births(self):
        self.snapshot()
        return {pid: record['birth'] for pid, record in self.registered.items()}

    def check_gpu(self, index, *, require_idle=True, snapshot=None):
        from .cluster_profile import check_pro_gpu, check_pro_shared_gpu
        if self.admission is None:
            raise ValueError('PRO GPU check requires this owned epoch admission')
        self.admission.authorization(self)
        if snapshot is None:
            snapshot = self.snapshot()
        # Use only live, freshly observed admitted identities, not retired PIDs.
        admitted = {p['pid']: p['birth'] for p in snapshot['processes']}
        check = check_pro_shared_gpu if require_idle and self.profile.max_workers_per_gpu > 1 else check_pro_gpu
        options = {} if check is check_pro_shared_gpu else dict(require_idle=require_idle)
        receipt = check(self.profile, index, admitted=admitted, _epoch=self, **options)
        if require_idle and self.profile.max_workers_per_gpu == 1 and receipt['processes']:
            raise ValueError('PRO exclusive GPU already has registered contexts')
        with self.mutex:
            for warning in receipt.get('diagnostic_warnings', ()):
                # At most two fixed diagnostic reasons per authorized GPU. The
                # normal resource receipt overwrites; no per-poll log appends.
                self.gpu_diagnostic_warnings.setdefault((index, warning['reason']), warning)
        return receipt

    def verify_job(self, job, *, dynamic_gpu=False):
        if self.admission is None:
            raise ValueError('PRO selected job requires this owned epoch admission')
        self.admission.verify_job(job, dynamic_gpu=dynamic_gpu)

    def assert_worker_exited(self, pid):
        if any(p['root'] == pid for p in self.snapshot()['processes']):
            raise ValueError('owned worker descendants remain after process exit')

    def stop_owned(self, child, *, grace_s=5):
        """Signal only pinned registered identities; never a potentially reused PGID."""
        import subprocess
        try:
            self.snapshot()
        except (OSError, ValueError):
            pass  # Cleanup may signal already registered identities only.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            for pid, identity in list(self.registered.items()):
                if identity['root'] != child.pid:
                    continue
                fd = None
                try:
                    fd = os.pidfd_open(pid)
                    current = process_identity(pid)
                    if current['birth'] != identity['birth']:
                        continue  # A reused identity is never signalled.
                    signal.pidfd_send_signal(fd, signum)
                except (FileNotFoundError, ProcessLookupError):
                    pass
                finally:
                    if fd is not None:
                        os.close(fd)
            try:
                child.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
        child.wait(timeout=10)

    def launch(self, argv, **options):
        import subprocess
        import sys
        read_fd, write_fd = os.pipe()
        try:
            wrapped = [sys.executable, '-m', 'mind2act_eval.campaign_resources',
                       '--join', str(self.directory), '--release-fd', str(read_fd),
                       '--parent-pid', str(os.getpid()), '--parent-birth', str(self.parent_identity['birth']),
                       '--', *argv]
            descriptors = tuple(options.pop('pass_fds', ())) + (read_fd,)
            child = subprocess.Popen(wrapped, pass_fds=descriptors, **options)
            try:
                identity = self.register(child)
                self.registered[child.pid]['registered_at'] = time.monotonic()
                os.write(write_fd, b'1')  # only after PID/birth/session registration
            except BaseException:
                os.close(write_fd)
                write_fd = None
                child.wait(timeout=10)  # EOF causes the CPU join shim to exit
                raise
            return child, identity
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def persist(self):
        from mind2act_eval.runner import write_json
        write_json(self.receipt_path, self.snapshot())
        with self.receipt_path.open('rb') as stream:
            os.fsync(stream.fileno())
        fd = os.open(self.receipt_path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def __exit__(self, *error):
        try:
            self._close()
        finally:
            self.epoch_active = False

    def _close(self):
        if self.empty_removal is not None:
            if self.directory.exists():
                raise ValueError('removed campaign scope was recreated')
            return
        self.persist()
        capacity = getattr(self, 'capacity_admission', None)
        if capacity is not None:
            # Capacity owns the canonical mutex across removal and retirement.
            capacity.close()
            return
        self._remove_empty_scope()

    def _remove_empty_scope(self):
        members = {int(p) for p in (self.directory / 'cgroup.procs').read_text().split()}
        if members:
            if (getattr(self.profile, 'unregistered_process_policy', 'error') == 'warn'
                    and not members.intersection(self.registered)):
                return  # Unknown-only scope remains bounded; no ownership claim or signals.
            raise ValueError('campaign still has registered/unknown live processes; scope retained')
        identity = self.directory.stat()
        self.directory.rmdir()  # Kernel removal succeeds only for an empty scope.
        self.empty_removal = dict(directory=str(self.directory), verified_empty=True,
            device=identity.st_dev, inode=identity.st_ino,
            owner_pid=self.parent_identity['pid'], owner_birth=self.parent_identity['birth'])


def _join_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--join', required=True)
    parser.add_argument('--release-fd', required=True, type=int)
    parser.add_argument('--parent-pid', required=True, type=int)
    parser.add_argument('--parent-birth', required=True, type=int)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.read(args.release_fd, 1) != b'1':
        raise ValueError('supervisor did not register this worker')
    os.close(args.release_fd)
    parent = process_identity(args.parent_pid)
    if parent['birth'] != args.parent_birth or os.getppid() != args.parent_pid:
        raise ValueError('supervisor PID/birth changed before worker join')
    from mind2act_eval.cluster_profile import load_cluster_profile
    profile = load_cluster_profile()
    if profile is None:
        raise ValueError('cgroup join requires explicit PRO profile')
    verify_parent(profile.cgroup_parent)
    path = Path(args.join)
    if (path.parent != profile.cgroup_parent or path.is_symlink()
            or not path.name.startswith(f'mind2act-pro5000-{os.getuid()}-')
            or _counter(path / 'memory.max') != profile.rss_limit_bytes):
        raise ValueError('worker cgroup is not the owned bounded scope')
    (path / 'cgroup.procs').write_text(str(os.getpid()))
    # The live parent authorized this immutable deployment once for this owned
    # epoch and rechecked the selected job before Popen. The registration pipe,
    # parent birth and bounded scope above bind this shim to that parent; the
    # standard worker still validates its selected actual inputs before native.
    command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not command:
        raise ValueError('missing standard worker argv')
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    _join_main()
