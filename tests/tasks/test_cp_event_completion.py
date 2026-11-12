"""Task public event-completion hooks do not confuse a wrong result with abort."""
from types import SimpleNamespace
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP04.scene import CookingScene
from mind2act_eval.tasks.CP05.scene import PianoScene


def test_cp04_off_and_returns_finish_even_if_recipe_and_thermal_are_wrong():
    scene=object.__new__(CookingScene)
    scene.native={'counter':{'step':10}}
    scene.clock=SimpleNamespace(substeps=1)
    scene.spec=SimpleNamespace(config={'recipe':['a','b']})
    raw=dict(thermal=dict(events=[dict(mode='OFF'),dict(mode='HEAT'),dict(mode='OFF')]),
        terminal_measurements=dict(jars={n:dict(in_drawer=True,in_tray=True,detached=True) for n in ('a','b')}))
    scene.evidence=lambda tick:NativeEvidence('CP04','cpu',tick,dict(cp04_evidence=raw))
    assert scene.events_complete()
    raw['terminal_measurements']['jars']['b']['in_tray']=False
    assert not scene.events_complete()


def test_cp05_wrong_note_sequence_completes_after_count_and_return():
    scene=object.__new__(PianoScene);scene.tick=10
    scene.spec=SimpleNamespace(difficulty='easy',config={'setup':{'sequence':[0]*8}})
    raw=dict(demonstration_completed=True,replay_events=[dict(key=1)]*8,
        tools=[dict(tool_id='single',returned=True)])
    scene.evidence=lambda tick:NativeEvidence('CP05','cpu',tick,dict(cp05_evidence=raw))
    assert scene.events_complete()
    raw['tools'][0]['returned']=False
    assert not scene.events_complete()


def test_cp04_initial_off_with_caps_in_tray_does_not_finish():
    scene=object.__new__(CookingScene);scene.native={'counter':{'step':10}}
    scene.clock=SimpleNamespace(substeps=1);scene.spec=SimpleNamespace(config={'recipe':['a','b']})
    raw=dict(thermal=dict(events=[dict(mode='OFF')]),terminal_measurements=dict(
        jars={n:dict(in_drawer=True,in_tray=True,detached=True) for n in ('a','b')}))
    scene.evidence=lambda tick:NativeEvidence('CP04','cpu',tick,dict(cp04_evidence=raw))
    assert not scene.events_complete()
