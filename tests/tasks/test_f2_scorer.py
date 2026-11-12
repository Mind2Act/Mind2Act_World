"""Measurement-stream to score regressions using explicitly synthetic facts."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parent))
from test_f2_mechanism import Facts
from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.F2.evidence import Rows, validate_frame
from mind2act_eval.tasks.F2.provenance import VERSION
from mind2act_eval.tasks.F2.scorer import score


class Recorded(Facts):
    def __init__(self,tmp_path,level='easy',offset=0):
        super().__init__(level,offset)
        self.offset=offset
        self.initial=self.packet()
        self.writer=Rows(tmp_path/'measurements.jsonl')
        self.terminal_tick=None
        self.origin=None
        if offset:
            source=dict(public_task='F2',cell='F2_'+level,instance_id='synthetic-input',rows=1000,
                actual_h5_sha256='a'*64,instance_sha256='b'*64,
                source_clock=dict(physics_hz=120,control_hz=30,rgb_hz=30),
                timeline=[dict(state_physics_step=offset,action_physics_step=offset+4,state_control_index=0,action_control_index=4)])
            path=tmp_path/'synthetic-origin.json';raw=json.dumps(dict(cells=[source])).encode();path.write_bytes(raw)
            self.origin=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),instance_id='synthetic-input')

    def packet(self,**kwargs):
        tick=self.tick-self.offset
        raw=self.frame(**kwargs)
        return dict(physics_tick=tick,source_tick=self.tick,simulation_time_s=.025+tick/120,f2_raw=raw)

    def step(self,count=1,**kwargs):
        for _ in range(count):
            self.tick+=1
            row=self.packet(**kwargs)
            validate_frame(self.c,row,native_tick=self.tick-self.offset,source_start=self.offset)
            if self.machine.status=='running':
                self.machine.sample(row['f2_raw'])
                if self.machine.status!='running':self.terminal_tick=self.tick-self.offset
            self.writer.append(row)

    def evidence(self):
        return NativeEvidence('F2','synthetic',self.tick-self.offset,
            dict(source_case=self.source,input_protocol_version=VERSION,source_start_tick=self.offset,
                 source_origin_receipt=self.origin,initial=self.initial,measurements=self.writer.receipt(),terminal_tick=self.terminal_tick))


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_measured_success_replays_independently_to_100(tmp_path,level):
    f=Recorded(tmp_path,level);f.step(600);f.place(f.c['target']);f.confirm()
    result=score(f.evidence())
    assert result.value==100 and result.components['rubric']==VERSION
    assert result.components['attempts']==1
    assert result.components['native_physics_ticks']==f.tick
    f.writer.close()


def test_source_600_and_native_zero_are_distinct_with_verified_projection(tmp_path):
    f=Recorded(tmp_path,offset=600)
    assert f.initial['physics_tick']==0 and f.initial['source_tick']==600
    f.place(f.c['target']);f.confirm()
    result=score(f.evidence())
    assert result.value==100
    assert result.components['source_last_tick']==600+f.tick-f.offset
    assert result.components['native_physics_ticks']==result.components['machine_physics_ticks']==64
    f.writer.close()


def test_unreleased_or_partial_episode_is_zero_without_invented_completion(tmp_path):
    f=Recorded(tmp_path);f.step(600);f.place(f.c['target']);f.step(3,pressed=True)
    result=score(f.evidence())
    assert result.value==0 and result.components['terminal_tick'] is None
    assert result.components['attempts']==0
    f.step();assert score(f.evidence()).value==100
    f.writer.close()


@pytest.mark.parametrize('fault',['offset_without_receipt','clock_gap','joint_claim','contact_claim','tamper'])
def test_missing_contradictory_and_changed_native_evidence_are_rejected(tmp_path,fault):
    f=Recorded(tmp_path,offset=600);f.place(f.c['target']);f.confirm()
    native=f.evidence();v=to_json(native.values)
    if fault=='offset_without_receipt':v['source_origin_receipt']=None
    else:
        path=f.writer.path;rows=[json.loads(line) for line in path.read_text().splitlines()]
        if fault=='clock_gap':rows[0]['f2_raw']['tick']+=1
        elif fault=='joint_claim':rows[0]['f2_raw']['button']['depth']=.004
        elif fault=='contact_claim':rows[0]['f2_raw']['button']['robot_contact']=True
        else:rows[-1]['simulation_time_s']+=.1
        raw=b''.join((json.dumps(row)+'\n').encode() for row in rows);path.write_bytes(raw)
        if fault!='tamper':v['measurements'].update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    with pytest.raises(ValueError):score(NativeEvidence('F2','synthetic',native.tick,v))
    f.writer.close()


def test_native_notice_contact_can_replay_to_later_success(tmp_path):
    f = Recorded(tmp_path)
    f.step(10, held='block1')
    assert f.machine.status == 'running'
    f.step(600 - f.tick)
    f.place(f.c['target'])
    f.confirm()
    result = score(f.evidence())
    assert result.value == 100 and result.components['attempts'] == 1
    f.writer.close()


def test_confirmed_success_is_rechecked_against_actual_terminal_arrangement(tmp_path):
    # A complete confirmation does not license a different terminal arrangement.
    f=Recorded(tmp_path);f.step(600);f.place(f.c['target']);f.confirm()
    f.layout=list(f.c['objects'])
    f.step()
    assert score(f.evidence()).value == 0
    f.writer.close()


def test_mechanical_confirmation_needs_no_finger_proof_or_dwell(tmp_path):
    class MechanismOnly(Recorded):
        def packet(self, **kwargs):
            packet=super().packet(**kwargs)
            packet['f2_raw']['button']['robot_contact']=False
            packet['f2_raw']['contact_events']=[]
            return packet
    f=MechanismOnly(tmp_path);f.step(600)
    f.layout=list(f.c['target']);f.step(1)
    f.step(1,pressed=True);f.step()
    assert score(f.evidence()).value == 100
    f.writer.close()
