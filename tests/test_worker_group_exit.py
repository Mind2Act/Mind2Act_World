"""Real child/seal/owned-descendant settlement regressions; no GPU processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from mind2act_eval import batch
from mind2act_eval.contracts import ConsumerKind, InstanceSpec


def alive(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    return result.returncode==0 and result.stdout.strip()[:1] not in {'Z','X',''}


@pytest.mark.parametrize('tamper', [False, True])
def test_sealed_worker_cleans_its_surviving_group_before_valid_settlement(tmp_path, monkeypatch, tamper):
    root=Path(__file__).parents[1]
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join([str(root/'src'),str(root/'tests'),os.environ.get('PYTHONPATH','')]))
    script=tmp_path/'worker.py'
    script.write_text('''import argparse,json,os,subprocess,sys
from pathlib import Path
from test_sealing import _successful_attempt
from mind2act_eval.sealing import seal_worker_outputs
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();out=Path(a.output)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(out/'descendant.json').write_text(json.dumps({'pid':child.pid}))
_successful_attempt(out,pid=os.getpid());seal_worker_outputs(out);os._exit(0)
''')
    foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    directory=tmp_path/'attempt'
    if tamper:
        stop=batch._stop_owned_group
        def mutate_then_stop(child,**kwargs):
            (directory/'video.mp4').write_bytes(b'changed after initial seal verification')
            stop(child,**kwargs)
        monkeypatch.setattr(batch,'_stop_owned_group',mutate_then_stop)
    try:
        spec=InstanceSpec('demo','seal-001','easy','synthetic-public-v1','v1','none','v1')
        job=batch.EpisodeJob(spec,(sys.executable,str(script)),ConsumerKind.DIAGNOSTIC,10,0)
        result=batch.run_attempt(job,directory,poll_s=.01,stop_grace_s=.05)
        receipt=json.loads((directory/'batch-exit.json').read_text())
        pid=json.loads((directory/'descendant.json').read_text())['pid']
        assert receipt['worker_exit_verified'] is True
        assert not alive(pid)
        if tamper:
            assert not result.infra_valid and result.score is None
            assert 'post-exit worker seal validation failed' in result.error
        else:
            assert result.infra_valid and result.score==25
        assert foreign.poll() is None
    finally:
        foreign.terminate();foreign.wait(timeout=5)
        record=directory/'descendant.json'
        if record.exists():
            pid=json.loads(record.read_text())['pid']
            if alive(pid):
                os.kill(pid,signal.SIGKILL)


def test_unverifiable_group_cannot_return_effective_score_or_reclaim_scratch(tmp_path, monkeypatch):
    from mind2act_eval import episode_storage
    root=Path(__file__).parents[1]
    monkeypatch.setenv('PYTHONPATH',str(root/'src'))
    monkeypatch.setattr(episode_storage,'process_group_exited',lambda pid:False)
    spec=InstanceSpec('demo','demo-001','easy','synthetic-public-v1','v1','none','v1')
    source=tmp_path/'source.json'
    from mind2act_eval.contracts import to_json
    source.write_text(json.dumps(to_json(spec)))
    argv=(sys.executable,'-m','mind2act_eval.cli','_worker','--spec',str(source),
          '--asset-root',str(tmp_path),'--backend','demo','--wall-s','10')
    directory=tmp_path/'attempt'
    result=batch.run_attempt(batch.EpisodeJob(spec,argv,ConsumerKind.DIAGNOSTIC,10,0),directory,
        lock_root=tmp_path/'locks',poll_s=.01,stop_grace_s=0,cleanup_timeout_s=.05)
    receipt=json.loads((directory/'batch-exit.json').read_text())
    assert not result.infra_valid and result.score is None and 'exit unknown' in result.error
    assert receipt['worker_exit_verified'] is False and receipt['worker_seal_valid'] is False
    assert (directory/'worker-result-before-supervisor.json').is_file()
    assert (directory/'cache').is_dir() and not (directory/'scratch-ledger.json').exists()


def test_batch_does_not_release_slot_or_retry_unknown_exit(tmp_path, monkeypatch):
    spec=InstanceSpec('demo','demo-001','easy','synthetic-public-v1','v1','none','v1')
    job=batch.EpisodeJob(spec,('unused',),ConsumerKind.DIAGNOSTIC,10,1)
    launched=[]
    def attempt(job,directory,**kwargs):
        launched.append(directory);directory.mkdir()
        (directory/'batch-exit.json').write_text('{"worker_exit_verified":false}')
        from mind2act_eval.contracts import Termination
        return batch._invalid(job,Termination.RESOURCE_LIMIT,'owned exit unknown')
    monkeypatch.setattr(batch,'run_attempt',attempt)
    with pytest.raises(RuntimeError,match='without releasing a batch slot or retrying'):
        batch.run_batch([job,job],tmp_path/'runs',require_shared=False,start_interval_s=0)
    assert len(launched)==1




@pytest.mark.parametrize('change',['score','instance','spec','consumer'])
def test_settlement_uses_final_coherent_seal_and_rechecks_job_binding(tmp_path,monkeypatch,change):
    from mind2act_eval.sealing import seal_worker_outputs, validate_worker_seal
    root=Path(__file__).parents[1]
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join([str(root/'src'),str(root/'tests'),os.environ.get('PYTHONPATH','')]))
    script=tmp_path/'worker.py'
    script.write_text('''import argparse,json,os,subprocess,sys
from pathlib import Path
from test_sealing import _successful_attempt
from mind2act_eval.sealing import seal_worker_outputs
p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();out=Path(a.output)
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(out/'descendant.json').write_text(json.dumps({'pid':child.pid}))
_successful_attempt(out,pid=os.getpid())
debug=out/'native-exit-debug';debug.mkdir();(debug/'status.json').write_text('{}')
seal_worker_outputs(out);os._exit(0)
''')
    directory=tmp_path/'attempt'
    stop=batch._stop_owned_group
    resealed=[]
    def stop_then_reseal(child,**kwargs):
        stop(child,**kwargs)
        value=json.loads((directory/'result.json').read_text())
        assert value['score']==25  # The supervisor has already read this seal.
        value.update(score=75.,ticks=8,score_components={'final':75},task_metadata={'seal':'final'})
        spec=json.loads((directory/'spec.json').read_text())
        if change=='instance':
            value['instance_id']='other-sealed-instance'
        elif change=='spec':
            spec['seed']+=1
        elif change=='consumer':
            value['consumer_kind']='expert_replay'
        (directory/'result.json').write_text(json.dumps(value))
        (directory/'score.json').write_text(json.dumps({'value':75.}))
        (directory/'evidence.json').write_text(json.dumps({'task_id':'demo','tick':8}))
        (directory/'spec.json').write_text(json.dumps(spec))
        seal_worker_outputs(directory)
        final,_=validate_worker_seal(directory,durable=True)
        assert final.score==75  # A real, internally valid second seal, not a fake validator.
        resealed.append(final)
    monkeypatch.setattr(batch,'_stop_owned_group',stop_then_reseal)
    spec=InstanceSpec('demo','seal-001','easy','synthetic-public-v1','v1','none','v1')
    job=batch.EpisodeJob(spec,(sys.executable,str(script)),ConsumerKind.DIAGNOSTIC,10,0)
    try:
        result=batch.run_attempt(job,directory,poll_s=.01,stop_grace_s=.05)
        assert len(resealed)==1
        receipt=json.loads((directory/'batch-exit.json').read_text())
        assert receipt['worker_exit_verified'] is True
        if change=='score':
            assert result.infra_valid and result.score==75 and result.ticks==8
            assert result.score_components=={'final':75} and result.task_metadata=={'seal':'final'}
            assert result.artifacts['native_exit_debug_status']=='native-exit-debug/status.json'
            assert receipt['worker_seal_valid'] is True
        else:
            assert not result.infra_valid and result.score is None
            assert receipt['worker_seal_valid'] is False
            assert 'post-exit worker seal validation failed' in result.error
            expected={'instance':'worker result identity/consumer mismatch',
                      'spec':'requested/resolved instance binding mismatch',
                      'consumer':'worker consumer differs from the trusted parent job'}
            assert expected[change] in result.error
            assert (directory/'cache').is_dir() and not (directory/'scratch-ledger.json').exists()
    finally:
        record=directory/'descendant.json'
        if record.exists():
            pid=json.loads(record.read_text())['pid']
            if alive(pid):os.kill(pid,signal.SIGKILL)
