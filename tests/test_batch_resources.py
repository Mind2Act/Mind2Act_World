"""CPU resource admission tests; no GPU/native concurrency qualification."""
import json
import sys
from pathlib import Path

import pytest

import mind2act_eval.batch as batch
import mind2act_eval.batch_resources as resources
from mind2act_eval.batch_resources import BatchResourceConfig, GIB


def cgroup_tree(tmp_path, monkeypatch, *, leaf_limit=256*GIB, leaf_current=20*GIB,
                parent_limit='max', parent_current=20*GIB):
    mount = tmp_path / 'cgroup with space'
    leaf = mount / 'campaign'
    leaf.mkdir(parents=True)
    for path, cap, current in ((mount, parent_limit, parent_current), (leaf, leaf_limit, leaf_current)):
        (path / 'memory.max').write_text(str(cap))
        (path / 'memory.high').write_text('max')
        (path / 'memory.current').write_text(str(current))
    proc = tmp_path / 'proc'; proc.mkdir()
    (proc / 'cgroup').write_text('0::/campaign\n')
    escaped = str(mount).replace(' ', r'\040')
    (proc / 'mountinfo').write_text(f'1 0 0:1 / {escaped} rw - cgroup2 cgroup rw\n')
    monkeypatch.setattr(resources, 'PROC_CGROUP', proc / 'cgroup')
    monkeypatch.setattr(resources, 'PROC_MOUNTINFO', proc / 'mountinfo')
    return mount, leaf


def one_job():
    from mind2act_eval.contracts import InstanceSpec, ConsumerKind
    spec = InstanceSpec("demo", "resource-test", "easy", "synthetic", "v1", "none", "v1")
    return [batch.EpisodeJob(spec, (sys.executable, "-c", "raise AssertionError(\"must not start\")"), ConsumerKind.DIAGNOSTIC, 1, 0)]


def slurm_tree(tmp_path, monkeypatch, *, job_limit=60000*1024**2):
    mount, leaf = cgroup_tree(tmp_path, monkeypatch, leaf_limit='max', leaf_current=1024)
    job = mount/'job_1763371'; job.mkdir()
    leaf.rename(job/'step_0'); leaf = job/'step_0'
    for name, value in [('memory.max',job_limit),('memory.high',job_limit),('memory.current',2048)]:
        (job/name).write_text(str(value))
    resources.PROC_CGROUP.write_text('0::/job_1763371/step_0\n')
    (mount/'cgroup.controllers').write_text('cpuset cpu io memory hugetlb pids rdma misc')
    (mount/'cgroup.subtree_control').write_text('cpuset cpu io memory pids')
    for name in ('memory.max','memory.high','memory.current'):(mount/name).unlink()
    return mount,job,leaf


@pytest.mark.parametrize('cap',[60000*1024**2,120000*1024**2])
def test_real_root_has_no_memory_controls_but_slurm_job_cap_is_preserved(tmp_path,monkeypatch,cap):
    mount,job,leaf=slurm_tree(tmp_path,monkeypatch,job_limit=cap)
    snapshot=resources.read_memory_snapshot()
    assert snapshot['status']=='verified',snapshot
    assert snapshot['effective_limit_bytes']==cap
    assert snapshot['effective_high_bytes']==cap
    assert snapshot['available_bytes']==cap-2048
    assert snapshot['current_bytes']==1024
    root=snapshot['scopes'][-1]
    assert root['path']==str(mount)
    assert root['limit_bytes'] is root['high_bytes'] is root['current_bytes'] is None
    assert 'memory' in root['root_controller_evidence']['cgroup.controllers']
    rss,parallel,receipt=resources.resolve_resources(36*GIB,1,BatchResourceConfig(36*GIB,1))
    assert rss==36*GIB and parallel==1 and receipt['admission']=='admitted'
    (job/'memory.current').write_text(str(cap-8*GIB))
    with pytest.raises(resources.ResourceAdmissionError,match='headroom'):
        resources.resolve_resources(36*GIB,1,BatchResourceConfig(36*GIB,1))


@pytest.mark.parametrize('scope',['job','leaf'])
@pytest.mark.parametrize('name',['memory.max','memory.high','memory.current'])
def test_missing_nonroot_counter_never_inherits_root_exception(tmp_path,monkeypatch,scope,name):
    _,job,leaf=slurm_tree(tmp_path,monkeypatch)
    ((job if scope=='job' else leaf)/name).unlink()
    assert resources.read_memory_snapshot()['status']=='unavailable'


@pytest.mark.parametrize('kind',['no_controller','not_delegated','partial_controls','hidden_mount','namespace_root','root_permission'])
def test_root_exception_needs_positive_full_v2_root_evidence(tmp_path,monkeypatch,kind):
    mount,job,leaf=slurm_tree(tmp_path,monkeypatch)
    if kind=='no_controller':(mount/'cgroup.controllers').write_text('cpu io')
    elif kind=='not_delegated':(mount/'cgroup.subtree_control').write_text('cpu io')
    elif kind=='partial_controls':(mount/'memory.current').write_text('0')
    elif kind=='hidden_mount':
        resources.PROC_CGROUP.write_text('0::/hidden/job_1763371/step_0\n')
        resources.PROC_MOUNTINFO.write_text(resources.PROC_MOUNTINFO.read_text().replace('0:1 / ','0:1 /hidden '))
    elif kind=='namespace_root':resources.PROC_CGROUP.write_text('0::/\n')
    else:
        original=Path.read_text
        def denied(path,*a,**kw):
            if path==mount/'memory.max':raise PermissionError('root control unreadable')
            return original(path,*a,**kw)
        monkeypatch.setattr(Path,'read_text',denied)
    assert resources.read_memory_snapshot()['status']=='unavailable'


def test_finite_high_headroom_does_not_replace_the_required_hard_cap(tmp_path,monkeypatch):
    mount,leaf=cgroup_tree(tmp_path,monkeypatch,parent_limit=128*GIB,parent_current=40*GIB)
    (mount/'memory.high').write_text(str(64*GIB))
    snapshot=resources.read_memory_snapshot()
    assert snapshot['status']=='verified'
    assert snapshot['effective_limit_bytes']==128*GIB
    assert snapshot['effective_high_bytes']==64*GIB
    assert snapshot['available_bytes']==24*GIB
    with pytest.raises(resources.ResourceAdmissionError,match='headroom'):
        resources.resolve_resources(36*GIB,1,BatchResourceConfig(32*GIB,1))
    (mount/'memory.max').write_text('max');(leaf/'memory.max').write_text('max')
    assert resources.read_memory_snapshot()['status']=='unavailable'


def test_high_change_during_read_is_unavailable(tmp_path,monkeypatch):
    _,leaf=cgroup_tree(tmp_path,monkeypatch)
    original=resources._counter;reads=[]
    def changing(path,*,limit=False):
        if path==leaf/'memory.high':
            reads.append(1)
            if len(reads)==2:return 64*GIB
        return original(path,limit=limit)
    monkeypatch.setattr(resources,'_counter',changing)
    assert resources.read_memory_snapshot()['status']=='unavailable'


def test_membership_change_during_final_limit_recheck_is_unavailable(tmp_path,monkeypatch):
    _,leaf=cgroup_tree(tmp_path,monkeypatch)
    original=resources._counter;reads=[]
    def changing(path,*,limit=False):
        if path==leaf/'memory.max':
            reads.append(1)
            if len(reads)==3:resources.PROC_CGROUP.write_text('0::/other\n')
        return original(path,limit=limit)
    monkeypatch.setattr(resources,'_counter',changing)
    assert resources.read_memory_snapshot()['status']=='unavailable'


def test_root_controller_change_during_snapshot_is_unavailable(tmp_path,monkeypatch):
    mount,_,_=slurm_tree(tmp_path,monkeypatch)
    original=Path.read_text;reads=[]
    def changing(path,*a,**kw):
        if path==mount/'cgroup.subtree_control':
            reads.append(1)
            if len(reads)==2:return 'cpu io'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',changing)
    assert resources.read_memory_snapshot()['status']=='unavailable'


def test_root_memory_interface_appearing_during_snapshot_is_unavailable(tmp_path,monkeypatch):
    mount,_,leaf=slurm_tree(tmp_path,monkeypatch)
    original=resources._counter;reads=[]
    def changing(path,*,limit=False):
        if path==leaf/'memory.max':
            reads.append(1)
            if len(reads)==3:(mount/'memory.current').write_text('0')
        return original(path,limit=limit)
    monkeypatch.setattr(resources,'_counter',changing)
    assert resources.read_memory_snapshot()['status']=='unavailable'


def test_headroom_uses_the_larger_current_from_both_reads(tmp_path,monkeypatch):
    _,leaf=cgroup_tree(tmp_path,monkeypatch)
    original=resources._counter;reads=[]
    def changing(path,*,limit=False):
        if path==leaf/'memory.current':
            reads.append(1)
            return (20 if len(reads)==1 else 250)*GIB
        return original(path,limit=limit)
    monkeypatch.setattr(resources,'_counter',changing)
    snapshot=resources.read_memory_snapshot()
    assert snapshot['status']=='verified'
    assert snapshot['current_bytes']==250*GIB
    assert snapshot['available_bytes']==6*GIB


def refuse_worker(*args, **kwargs):
    pytest.fail('resource rejection must happen before any worker launch')


@pytest.mark.parametrize('rss,parallel', [(96*GIB, 3), (96*GIB, 4), (128*GIB, 6)])
def test_explicit_config_admits_headroom_and_writes_campaign_receipt(tmp_path, monkeypatch, rss, parallel):
    cgroup_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    campaign, summary = batch.run_batch([], tmp_path/'out', require_shared=False,
        start_interval_s=0, resource_config=BatchResourceConfig(rss, parallel))
    receipt = json.loads((campaign/'resource-receipt.json').read_text())
    assert receipt['opt_in'] and receipt['rss_limit_bytes'] == rss
    assert receipt['max_parallel'] == parallel and receipt['reserve_bytes'] == 16*GIB
    assert receipt['memory']['effective_limit_bytes'] == 256*GIB
    assert receipt['memory']['current_bytes'] == 20*GIB
    assert receipt['gpu_assignments'] == [] and summary == []
    assert 'EpisodeJob.gpu' in receipt['gpu_selection_source']


@pytest.mark.parametrize('rss', [True, False, 0, -1, float('nan'), float('inf'), 96.0*GIB, 129*GIB])
def test_invalid_explicit_budget_rejected_before_worker(tmp_path, monkeypatch, rss):
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    with pytest.raises(ValueError):
        config = BatchResourceConfig(rss, 4)
        batch.run_batch(one_job(), tmp_path/'out', require_shared=False, resource_config=config)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('parallel', [True, 0, -1, 1.5, float('nan'), float('inf'), 7])
def test_invalid_explicit_parallel_rejected_before_worker(tmp_path, monkeypatch, parallel):
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    with pytest.raises(ValueError):
        config = BatchResourceConfig(96*GIB, parallel)
        batch.run_batch(one_job(), tmp_path/'out', require_shared=False, resource_config=config)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('rss,parallel', [(True, 1), (float('nan'), 1), (37*GIB, 1), (36*GIB, True), (36*GIB, 5)])
def test_legacy_api_cannot_bypass_opt_in(tmp_path, monkeypatch, rss, parallel):
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    with pytest.raises(ValueError):
        batch.run_batch(one_job(), tmp_path/'out', require_shared=False, rss_limit=rss, max_parallel=parallel)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('mode', ['missing', 'unlimited', 'malformed', 'unmappable', 'hidden_ancestors'])
def test_expansion_fails_closed_without_finite_cgroup_evidence(tmp_path, monkeypatch, mode):
    mount, leaf = cgroup_tree(tmp_path, monkeypatch)
    if mode == 'missing':
        (leaf/'memory.current').unlink()
    elif mode == 'unlimited':
        (leaf/'memory.max').write_text('max')
    elif mode == 'malformed':
        (leaf/'memory.max').write_text('NaN')
    elif mode == 'unmappable':
        resources.PROC_CGROUP.write_text('0::/../outside\n')
    else:
        resources.PROC_CGROUP.write_text('0::/hidden/campaign\n')
        resources.PROC_MOUNTINFO.write_text(resources.PROC_MOUNTINFO.read_text().replace('0:1 / ', '0:1 /hidden '))
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    with pytest.raises(ValueError, match='verified cgroup'):
        batch.run_batch(one_job(), tmp_path/'out', require_shared=False, resource_config=BatchResourceConfig(96*GIB, 4))
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('leaf_limit,leaf_current,parent_limit,parent_current', [
    (64*GIB, 0, 'max', 0),  # budget exceeds effective cap
    (256*GIB, 150*GIB, 'max', 0),  # budget plus reserve exceeds available
    (256*GIB, 0, 128*GIB, 40*GIB),  # tighter ancestor cap
    (256*GIB, 0, 512*GIB, 440*GIB),  # roomy leaf, occupied ancestor
])
def test_every_ancestor_headroom_and_reserve_bound_is_enforced(tmp_path, monkeypatch,
        leaf_limit, leaf_current, parent_limit, parent_current):
    cgroup_tree(tmp_path, monkeypatch, leaf_limit=leaf_limit, leaf_current=leaf_current,
                parent_limit=parent_limit, parent_current=parent_current)
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    with pytest.raises(ValueError, match='headroom'):
        batch.run_batch(one_job(), tmp_path/'out', require_shared=False, resource_config=BatchResourceConfig(96*GIB, 4))
    assert not (tmp_path/'out').exists()


def test_default_cpu_no_cgroup_keeps_limits_and_does_not_require_expansion_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, 'PROC_CGROUP', tmp_path/'absent')
    campaign, _ = batch.run_batch([], tmp_path/'out', require_shared=False, start_interval_s=0, max_parallel=4)
    receipt = json.loads((campaign/'resource-receipt.json').read_text())
    assert receipt['rss_limit_bytes'] == 36*GIB and receipt['max_parallel'] == 4
    assert receipt['opt_in'] is False and receipt['memory']['status'] == 'unavailable'


def test_rechecks_after_original_batch_lock_before_submission(tmp_path, monkeypatch):
    mount, leaf = cgroup_tree(tmp_path, monkeypatch)
    from contextlib import contextmanager
    gates = []
    @contextmanager
    def gate(path):
        gates.append(Path(path))
        (leaf/'memory.current').write_text(str(200*GIB))
        yield
    monkeypatch.setattr(batch, 'startup_gate', gate)
    monkeypatch.setattr(batch, 'run_attempt', refuse_worker)
    with pytest.raises(ValueError, match='headroom'):
        batch.run_batch(one_job(), tmp_path/'out', require_shared=False, resource_config=BatchResourceConfig(96*GIB, 4))
    assert len(gates) == 1 and gates[0].name == 'batch.lock'
    receipt_path = next((tmp_path/'out').glob('batch-*/resource-receipt.json'))
    refused = json.loads(receipt_path.read_text())
    assert refused['admission'] == 'rejected' and refused['memory']['current_bytes'] == 200*GIB


def test_cgroup_limit_change_during_read_is_unsupported(tmp_path, monkeypatch):
    _, leaf = cgroup_tree(tmp_path, monkeypatch)
    original = resources._counter
    reads = 0
    def changing(path, *, limit=False):
        nonlocal reads
        if path == leaf/'memory.max':
            reads += 1
            if reads == 2:
                return 128*GIB
        return original(path, limit=limit)
    monkeypatch.setattr(resources, '_counter', changing)
    assert resources.read_memory_snapshot()['status'] == 'unavailable'


def test_config_file_pins_bytes_and_rejects_unknown_fields(tmp_path):
    path = tmp_path/'resources.json'
    data = dict(schema='mind2act_eval.batch_resources.v1', rss_limit_bytes=96*GIB, max_parallel=4)
    path.write_text(json.dumps(data))
    config = BatchResourceConfig.read(path)
    assert config.source == str(path.resolve()) and len(config.source_sha256) == 64
    data['unchecked_override'] = True; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='schema/fields'):
        BatchResourceConfig.read(path)


def test_resource_config_cannot_silently_override_legacy_api_arguments(tmp_path, monkeypatch):
    cgroup_tree(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='combined'):
        batch.run_batch([], tmp_path/'out', require_shared=False, max_parallel=2,
                        resource_config=BatchResourceConfig(96*GIB, 4))


def test_cli_threads_explicit_resource_config_without_changing_default_parallel(tmp_path, monkeypatch):
    from mind2act_eval import cli
    config = tmp_path/'resources.json'
    config.write_text(json.dumps(dict(schema='mind2act_eval.batch_resources.v1', rss_limit_bytes=96*GIB, max_parallel=4)))
    manifest = tmp_path/'manifest.json';manifest.write_text('[{}]')
    monkeypatch.setattr(cli, '_job', lambda args: one_job()[0])
    calls = []
    def run(jobs, root, **kwargs):
        calls.append(kwargs)
        return root/'cpu-placeholder', []
    monkeypatch.setattr(batch, 'run_batch', run)
    assert cli.main(['batch', '--manifest', str(manifest), '--asset-root', str(tmp_path),
                     '--output-root', str(tmp_path/'out'), '--backend', 'demo', '--resource-config', str(config)]) == 0
    assert calls[0]['max_parallel'] == 1
    assert calls[0]['resource_config'].rss_limit_bytes == 96*GIB
    assert calls[0]['resource_config'].max_parallel == 4
    assert calls[0]['start_interval_s'] == 0
    config.write_text(json.dumps(dict(schema='mind2act_eval.batch_resources.v1', rss_limit_bytes=True, max_parallel=4)))
    calls.clear()
    assert cli.main(['batch', '--manifest', str(manifest), '--asset-root', str(tmp_path),
                     '--output-root', str(tmp_path/'out'), '--backend', 'demo', '--resource-config', str(config)]) == 2
    assert calls == []
