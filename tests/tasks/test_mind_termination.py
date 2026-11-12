"""Task-side predicates for the two action-boundary termination families."""
from types import SimpleNamespace as NS
from mind2act_eval.tasks.E1.scene import EScene
from mind2act_eval.tasks.E2.scene import ArithmeticScene
from mind2act_eval.tasks.E2.mechanism.keys import Keys
from mind2act_eval.tasks.F1.scene import SliderScene
from mind2act_eval.tasks.F2.scene import ExperimentScene
from mind2act_eval.tasks.S2.scene import NavigationScene
from .test_s2 import PhysicsFixture


def test_e1_stops_when_last_batch_submitted_without_waiting_for_transport_departure():
    scene=EScene.__new__(EScene)
    scene.spec=NS(difficulty='easy')
    scene.evidence_recorder=NS(results=[dict(pass_=False),dict(pass_=True)])
    assert scene.events_complete()


def test_e2_completed_wrong_password_ends_the_event_sequence_after_return():
    scene=ArithmeticScene.__new__(ArithmeticScene)
    scene.keys=Keys();scene.case={'password':[1,2]}
    for tick,slot in enumerate([2,None,3,None],1):
        scene.keys.sample(tick,[-.004 if i==slot else 0. for i in range(9)],[False]*9,enabled=True)
    assert not scene.keys.complete([1,2])
    assert scene.events_complete()


def test_f2_only_confirmation_or_used_attempt_budget_ends_sequence():
    scene=ExperimentScene.__new__(ExperimentScene)
    scene.c={'attempt_budget':5}
    scene.machine=NS(status='failure',pending_release=False,attempts=[])
    assert not scene.events_complete()
    scene.machine.attempts=[[]]*5
    assert scene.events_complete()
    scene.machine.pending_release=True
    assert not scene.events_complete()


def test_violations_are_recorded_without_terminating_result_tasks(tmp_path):
    scene=SliderScene.__new__(SliderScene)
    scene._grade={'task_success':False,'protocol_violations':['bad']}
    assert not scene.is_success()
    fixture=PhysicsFixture(tmp_path);fixture.step(240);fixture.press('up')
    assert fixture.session.violations and fixture.session.status=='running'
    nav=NavigationScene.__new__(NavigationScene);nav.session=fixture.session
    assert not nav.is_success()
    before=fixture.tick;fixture.step(3)
    assert fixture.tick==before+3
    fixture.close()
