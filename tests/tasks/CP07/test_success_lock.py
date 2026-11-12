"""A native successful crossing must survive the replay tail."""
import pytest

from test_physical import Trace


def successful_crossing(difficulty, trace=None):
    trace = Trace(difficulty) if trace is None else trace
    trace.assemble()
    trace.press()
    trace.state['hands_clear'] = True
    trace.step()
    trace.state['car']['p_m'][0] = .4
    trace.step()
    assert trace.result().value == 100
    return trace


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_t0_can_lose_arrival_after_later_physical_displacement(difficulty):
    trace = successful_crossing(difficulty)
    # The reported native failure is driven past the far platform ~1.3s later.
    trace.state['car'].update(p_m=[.7, 0., .4], up=0.)
    trace.step(156)
    assert trace.result().value == 0
    assert not trace.scene.is_complete()


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_success_stops_task_owned_drive_at_the_arrival_tick(difficulty):
    trace = successful_crossing(difficulty)
    assert trace.drives[-1] == (False,)
    drive_count = len(trace.drives)
    trace.step(156)
    assert (True,) not in trace.drives[drive_count:]


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize('consumer', ['model', 'expert_replay'])
def test_runner_keeps_cp07_to_t0_after_native_success_including_h5_replay(tmp_path, difficulty, consumer):
    from copy import deepcopy
    from types import SimpleNamespace
    import json
    import h5py
    import numpy as np
    from mind2act_eval.contracts import BuildContext, ConsumerKind, Termination, to_json
    from mind2act_eval.replay import H5Replay, ReplayMetadata
    from mind2act_eval.runner import EpisodeLimits, run_episode
    from mind2act_eval.sim.demo import DemoSimulator, DemoRobot
    from mind2act_eval.tasks import TaskDefinition
    from mind2act_eval.tasks.CP07.scene import BridgeScene
    from mind2act_eval.tasks.CP07.scorer import score
    from mind2act_eval.adapters.demo import DemoAdapter

    trace = Trace(difficulty)
    initial = deepcopy(trace.state)
    samples = []
    def record():
        state=deepcopy(trace.state);car=state['car']
        car['far_bank_center_inside']=bool(trace.binding['finish_x_m']<=car['p_m'][0]<=.6 and abs(car['p_m'][1])<=.3)
        samples.append(state)
        return samples[-1]
    trace.scene.sensors.read = record
    successful_crossing(difficulty, trace)
    success_tick = trace.tick
    trace.state['car'].update(p_m=[.7, 0., .4], up=0.)
    trace.step(156)
    sim = DemoSimulator()
    robot = DemoRobot()
    drives = []
    sensors = SimpleNamespace(read=lambda: initial if sim.ticks == 0 else samples[sim.ticks-1],
                              drive=lambda *args: drives.append(args))
    ctx = BuildContext(sim, robot, None, tmp_path)
    task = TaskDefinition('CP07', lambda spec, context: BridgeScene(spec, context, trace.binding, sensors), score)
    adapter = DemoAdapter()
    if consumer == 'expert_replay':
        rows = (len(samples) + 3) // 4
        path = tmp_path / 'success-then-overrun.h5'
        with h5py.File(path, 'w') as h5:
            for side in ('left', 'right'):
                h5.create_dataset(f'action/{side}_arm_joint_states', data=np.zeros((rows, 6)))
                h5.create_dataset(f'action/{side}_ee_joint_states', data=np.ones((rows, 1)))
        adapter = H5Replay(path, ReplayMetadata('synthetic', trace.spec.source_id, 120, 30, 30,
            'rad', 'normalized', 1., 0., instance_spec=to_json(trace.spec)))
    result = run_episode(trace.spec, task, ctx, adapter, consumer_kind=ConsumerKind(consumer),
                         limits=EpisodeLimits(len(samples), 10))
    assert result.infra_valid, result.error
    assert result.score == 0
    assert result.termination == Termination.PHYSICS_BUDGET
    assert result.ticks == len(samples)
    assert sim.ticks == len(samples)
    assert result.task_metadata['success_lock_tick'] is None
    assert result.task_metadata['arrival_brake_tick'] == success_tick
    assert drives[-1] == (False,)
    if consumer == 'expert_replay':
        receipt = json.loads((tmp_path / 'replay-coverage.json').read_text())
        assert receipt.get('stopped_on_native_success', False) is False
        assert receipt['task_first_complete_tick'] == success_tick
        assert receipt['executed_physics_ticks'] == len(samples)
        assert receipt['clock_coverage_complete'] is False
        assert receipt['adapter_eof'] is False


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_result_task_success_ends_at_next_complete_model_action(tmp_path,difficulty):
    from copy import deepcopy
    from types import SimpleNamespace
    from mind2act_eval.contracts import BuildContext,ConsumerKind,Termination
    from mind2act_eval.runner import EpisodeLimits,run_episode
    from mind2act_eval.sim.demo import DemoSimulator,DemoRobot
    from mind2act_eval.tasks import TaskDefinition
    from mind2act_eval.tasks.CP07.scene import BridgeScene
    from mind2act_eval.tasks.CP07.scorer import score
    from mind2act_eval.adapters.demo import DemoAdapter
    trace=Trace(difficulty);initial=deepcopy(trace.state);samples=[]
    def record():
        row=deepcopy(trace.state);p=row['car']['p_m']
        row['car']['far_bank_center_inside']=bool(.4<=p[0]<=.6 and abs(p[1])<=.3)
        samples.append(row);return row
    trace.scene.sensors.read=record
    successful_crossing(difficulty,trace);first=trace.tick
    trace.step(100)
    sim=DemoSimulator();robot=DemoRobot()
    sensors=SimpleNamespace(read=lambda:initial if sim.ticks==0 else samples[sim.ticks-1],drive=lambda *args:None)
    task=TaskDefinition('CP07',lambda spec,ctx:BridgeScene(spec,ctx,trace.binding,sensors),score)
    result=run_episode(trace.spec,task,BuildContext(sim,robot,None,tmp_path),DemoAdapter(),
        consumer_kind=ConsumerKind.MODEL,limits=EpisodeLimits(len(samples),10))
    assert result.infra_valid,result.error
    assert result.score==100 and result.termination==Termination.COMPLETED
    assert first<=result.ticks<first+4
