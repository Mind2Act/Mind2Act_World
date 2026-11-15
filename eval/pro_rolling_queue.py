"""Bounded PRO intake controls; immutable profiles remain the execution authority."""
from contextlib import contextmanager
import os
import time


def validate_manifest(raw, profile):
    if profile is None or raw != dict(controller_hostname=raw.get('controller_hostname'),
                                     automatic_retries=0, stable_window_s=1200) or (
            type(raw.get('automatic_retries')) is not int or type(raw.get('stable_window_s')) is not int):
        raise ValueError('rolling dispatch requires PRO, one controller, zero retries and 1200s stability')
    if not isinstance(raw['controller_hostname'], str) or not raw['controller_hostname']:
        raise ValueError('rolling controller hostname is required')


def initial_controls(queue):
    return {host: dict(revision=0, drain=False,
                       workers_per_gpu=node.initial_parallel // len(node.gpu_indices))
            for host, node in queue.nodes.items()}


def control(store, state, hostname=None):
    hostname = store.queue.node.hostname if hostname is None else hostname
    value = state['dispatch'][hostname]
    node = store.queue.nodes[hostname]
    ceiling = node.stable_parallel // len(node.gpu_indices)
    if (set(value) != {'revision', 'drain', 'workers_per_gpu'}
            or type(value['revision']) is not int or value['revision'] < 0
            or type(value['drain']) is not bool
            or type(value['workers_per_gpu']) is not int
            or not 1 <= value['workers_per_gpu'] <= ceiling):
        raise ValueError('rolling dispatch control exceeds its frozen node ceiling')
    return value


def set_control(store, *, hostname, expected_revision, workers_per_gpu, drain, evidence,
                stability_receipt=None):
    queue = store.queue
    if queue.rolling is None or queue.node.hostname != queue.rolling['controller_hostname']:
        raise ValueError('only the frozen rolling controller may adjust intake')
    queue.profile.authorize()
    stability = None
    if stability_receipt is not None:
        from mind2act_eval.cluster_profile import pinned_json
        if not isinstance(stability_receipt, dict) or set(stability_receipt) != {'path', 'sha256'}:
            raise ValueError('stability receipt requires an exact path and SHA256')
        queue.profile.verify_queue_control(stability_receipt['path'])
        stability = pinned_json(stability_receipt['path'], stability_receipt['sha256'])
    if (hostname not in queue.nodes or type(expected_revision) is not int
            or type(workers_per_gpu) is not int or type(drain) is not bool
            or not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError('rolling control requires exact node/revision/target/drain and evidence')
    with store._transaction(operation='rolling_control') as state:
        old = dict(control(store, state, hostname))
        if old['revision'] != expected_revision:
            raise ValueError('stale rolling control revision')
        new = dict(revision=expected_revision + 1, workers_per_gpu=workers_per_gpu, drain=drain)
        state['dispatch'][hostname] = new
        control(store, state, hostname)
        if workers_per_gpu > 2 and (workers_per_gpu > old['workers_per_gpu']
                                    or (old['drain'] and not drain)):
            from mind2act_eval.pro_fleet_stability import validate_fleet_stability
            validate_fleet_stability(queue, state, stability)
        state.setdefault('dispatch_history', []).append(dict(hostname=hostname, before=old,
            after=new, evidence=evidence, at_unix_s=time.time(), controller=queue.node.hostname,
            stability_receipt=stability_receipt,
            # Audit only; every later expansion freshly validates the pinned
            # samples. Do not copy the whole fleet window into the hot ledger.
            stability=None if stability is None else dict(schema=stability['schema'],
                queue_manifest_sha256=stability['queue_manifest_sha256'], samples=len(stability['samples']),
                from_unix_s=stability['samples'][0]['at_unix_s'], until_unix_s=stability['samples'][-1]['at_unix_s'])))
        return dict(new)


def append_catalog(store, *, expected_revision, evidence):
    """Register a new immutable catalog, retaining all old readers and claims.

    Each host verifies exactly its own <=64 frozen jobs and proxy routes when
    loading this catalog. Other hosts must load it with their new receipts
    after draining; registering it never changes any running scope/profile.
    """
    import hashlib
    from mind2act_eval.shared_queue import _json_bytes
    queue = store.queue
    if (queue.rolling is None or queue.node.hostname != queue.rolling['controller_hostname']
            or type(expected_revision) is not int or not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError('catalog registration requires frozen controller, revision and evidence')
    authorization = queue.profile.authorize()
    with store._transaction(operation='rolling_append_catalog', allow_unregistered_catalog=True) as state:
        if state['catalog_revision'] != expected_revision:
            raise ValueError('stale catalog revision')
        if queue.manifest_sha256 in state['catalog_manifests']:
            raise ValueError('catalog already registered')
        def key(job):
            return tuple(job['episode_key'].get(k) for k in ('task_id', 'instance_id', 'difficulty', 'variation_sha256'))
        keys = {key(row['definition']) for row in state['jobs'].values()}
        new_rows = store._initial_state()['jobs']
        added = []
        for job in queue.jobs:
            old = state['jobs'].get(job['job_id'])
            if old is not None:
                if old['definition'] != job:
                    raise ValueError('successor catalog cannot change an existing case definition')
                continue
            if key(job) in keys:
                raise ValueError('successor catalog duplicates a logical case')
            keys.add(key(job))
            state['jobs'][job['job_id']] = new_rows[job['job_id']]
            added.append(job['job_id'])
        if not added:
            raise ValueError('successor catalog must add real cases')
        state['catalog_revision'] += 1
        state['catalog_manifests'][queue.manifest_sha256] = dict(
            job_ids=[job['job_id'] for job in queue.jobs], added_job_ids=added,
            manifest_path=str(queue.manifest_path), revision=state['catalog_revision'],
            evidence=evidence, authorization=authorization, at_unix_s=time.time(),
            definitions_sha256=hashlib.sha256(_json_bytes(list(queue.jobs))).hexdigest())
        return dict(catalog_revision=state['catalog_revision'], added_job_ids=added,
                    manifest_sha256=queue.manifest_sha256)


def _owned_attempt(store, state, claim):
    from mind2act_eval.shared_queue import _process_birth
    row = state['jobs'][claim.job_id]
    attempt = row['attempts'][-1] if row['attempts'] else {}
    if (row['status'] != 'running' or attempt.get('token') != claim.token
            or attempt.get('owner_hostname') != store.queue.node.hostname
            or attempt.get('owner_pid') != os.getpid()
            or attempt.get('owner_starttime') is None
            or attempt['owner_starttime'] != _process_birth(os.getpid())
            or attempt.get('gpu') != claim.gpu or attempt.get('directory') != str(claim.directory)
            or attempt.get('attempt_number') != claim.attempt_number
            or row['definition']['options'] != claim.options
            or row['definition']['episode_key'] != claim.episode_key
            or row['definition']['eligible_gpu'] != claim.gpu
            or store.queue.node.hostname not in row['definition']['eligible_nodes']):
        raise ValueError('rolling claim ownership mismatch')
    held = os.fstat(claim.lock_fd)
    named = os.stat(store._job_lock_path(claim.job_id), follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError('rolling claim lease identity changed')
    return row, attempt


@contextmanager
def launch_guard(store, claim):
    """Serialize drain against physical Popen, not merely ThreadPool submission.

    The queue lock spans Popen plus canonical PID/birth registration. A failed
    commit after Popen leaves the claim owned; only physical-exit evidence can
    finish it. No monitor takes this lock, and no authorization is cached.
    """
    from mind2act_eval.batch import AttemptDeferred
    with store._transaction(operation='rolling_start') as state:
        _row, attempt = _owned_attempt(store, state, claim)
        if attempt.get('child_started_at_unix_s') is not None:
            raise ValueError('rolling claim already started a child')
        current = control(store, state)
        if current['drain']:
            raise AttemptDeferred('rolling node is draining before child launch')
        # Reducing a target leaves existing children alone; pending claims wait.
        started = sum(1 for row in state['jobs'].values() if row['status'] == 'running'
                      and row['attempts'][-1].get('owner_hostname') == store.queue.node.hostname
                      and row['attempts'][-1].get('gpu') == claim.gpu
                      and row['attempts'][-1].get('child_started_at_unix_s') is not None)
        if started >= current['workers_per_gpu']:
            raise AttemptDeferred('rolling soft GPU target reduced before child launch')
        yield
        attempt['child_started_at_unix_s'] = time.time()


def release_unstarted(store, claim):
    """Keep never-started preparation evidence without consuming an attempt."""
    import json
    receipt = json.loads((claim.directory / 'batch-exit.json').read_text())
    if (receipt.get('dispatch_deferred') is not True or receipt.get('worker_started') is not False
            or receipt.get('worker_exit_verified') is not True):
        raise ValueError('release needs affirmative never-started deferred exit evidence')
    try:
        with store._transaction(operation='rolling_release_unstarted') as state:
            row, attempt = _owned_attempt(store, state, claim)
            if attempt.get('child_started_at_unix_s') is not None:
                raise ValueError('cannot release a started rolling claim')
            row['history'].append(dict(kind='released_before_start', claim=row['attempts'].pop(),
                                       exit_receipt=receipt))
            row['status'] = 'queued'
    finally:
        claim.close()
