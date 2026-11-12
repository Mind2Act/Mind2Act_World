"""Exercise actual batch scheduling with CPU attempts and explicit GPU identities."""
from contextlib import nullcontext
import json
import hashlib
import threading
import time
import pytest

from mind2act_eval import batch
from mind2act_eval.contracts import InstanceSpec, Termination, to_json


def test_non_pro_busy_gpu_queues_while_a_different_card_can_start(tmp_path,monkeypatch):
    jobs=[batch.EpisodeJob(InstanceSpec('demo',f'case-{i}','easy','synthetic','v1','v1','v1'),
                          ('CPU-only-fixture',),gpu=gpu) for i,gpu in enumerate((0,0,1))]
    monkeypatch.setattr(batch,'resolve_resources',lambda *args:(batch.RSS_LIMIT_BYTES,2,{}))
    monkeypatch.setattr(batch,'_campaign_launcher_gates',lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(batch,'_archive_attempt',lambda *args:None)
    active={};peaks={};started=[];mutex=threading.Lock();peer=threading.Event()
    def attempt(job,directory,**kwargs):
        directory.mkdir();(directory/'ready.json').write_text('{}')
        with mutex:
            active[job.gpu]=active.get(job.gpu,0)+1
            peaks[job.gpu]=max(peaks.get(job.gpu,0),active[job.gpu])
            started.append(job.spec.instance_id)
            if job.gpu==1:peer.set()
        peer.wait(3);time.sleep(.02)
        with mutex:active[job.gpu]-=1
        (directory/'batch-exit.json').write_text(json.dumps(dict(worker_exit_verified=True)))
        return batch._invalid(job,Termination.ERROR,'CPU scheduling fixture; no GPU workload')
    monkeypatch.setattr(batch,'run_attempt',attempt)
    _,summary=batch.run_batch(jobs,tmp_path/'runs',require_shared=False,max_parallel=2,start_interval_s=0)
    assert peaks=={0:1,1:1}
    assert started==['case-0','case-2','case-1']
    assert len(summary)==3


