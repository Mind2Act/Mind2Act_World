"""CPU sampling-to-scoring integration; not native GPU acceptance."""
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from mind2act_eval.contracts import InstanceSpec, ClockSpec, freeze, to_json
from mind2act_eval.tasks.CP05.scene import PianoScene
from mind2act_eval.tasks.CP05.scorer import score


class Body:
    def __init__(self, pos):self.pos=np.array(pos,dtype=float)
    def get_world_pose(self):return self.pos.copy(),np.array([1.,0.,0.,0.])


class Materials:
    @staticmethod
    def Apply(prim):return NS(Bind=lambda material:None)


def piano(level='easy'):
    sequence=[1,2,1,3,2,4,3,4]+([1,2,3,4] if level=='hard' else [])
    setup=dict(level=level,sequence=sequence,cloth=False,lid=False,
               volume_required=level=='hard',volume_axis='Y',volume_travel_m=.07,asset_package_scene=True)
    spec=InstanceSpec('CP05','synthetic',level,'synthetic','v1','v1','v1',
        clock=ClockSpec(120,30,30),config={'setup':setup,'instruction':'Synthetic test'})
    contacts=set()
    def touch(root,other=None,robot_only=False,link=None):
        return (root,other,robot_only,link) in contacts
    w=NS(tools={'Left':Body([0,0,1])},robots={'Left':None,'Right':None},
         volume=Body([0,0,1]),button=Body([0,0,1]),
         keys={i:Body([0,0,.822]) for i in range(29)},
         control_positions={'VolumeCap':np.array([0.,0.,1.]),'StartFace':np.array([0.,0.,1.])},
         rack_contains=lambda side,p,q:bool(abs(p[0]-(0 if side=='Left' else .2))<=.04 and abs(p[1])<=.02),
         touch=touch,supported=touch,shared=NS(material=lambda *a,**kw:None),
         stage=NS(GetPrimAtPath=lambda p:p),UsdShade=NS(MaterialBindingAPI=Materials),clothmesh=None)
    if level!='easy':w.tools['Right']=Body([.2,0,1])
    return PianoScene(spec,None,w),contacts


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_calibrated_scene_metadata_survives_runner_freeze_and_json(level):
    scene, _ = piano(level)
    for tick in range(1, 181):
        scene.after_step(tick)
    assert scene.w.key_press_policy.ready

    # Exercise the actual result_metadata boundary used by runner cleanup.
    metadata = json.loads(json.dumps(to_json(freeze(scene.result_metadata())), allow_nan=False))
    calibration = metadata['key_calibration']
    assert calibration['ready'] is True
    assert calibration['calibrated_at_sample'] == 180
    assert len(calibration['baseline_depths_m']) == len(calibration['idle_noise_ranges_m']) == 29
    assert calibration['baseline_depths_m']['0'] == calibration['baseline_depths_m']['28'] == 0.
    assert calibration['idle_noise_ranges_m']['0'] == calibration['idle_noise_ranges_m']['28'] == 0.
    assert metadata['initialization_physics_ticks'] == 0  # CPU fixture performs no native initialization.
    assert metadata['cue_period_ticks'] == 240


@pytest.mark.parametrize('ticks', [0, 179])
def test_uncalibrated_scene_metadata_keeps_unknown_depths(ticks):
    scene, _ = piano()
    for tick in range(1, ticks + 1):
        scene.after_step(tick)
    calibration = freeze(scene.result_metadata())['key_calibration']
    assert calibration['ready'] is False
    assert calibration['baseline_depths_m'] is None
    assert calibration['idle_noise_ranges_m'] is None
    assert calibration['calibrated_at_sample'] is None


def test_export_preserves_calibrated_measurements_and_integer_sensor_ids():
    scene, _ = piano()
    policy = scene.w.key_press_policy
    for tick in range(180):
        policy.observe_idle({key: .0005 + (tick % 2) * .00008 for key in range(29)}, clear=True)
    metadata = scene.result_metadata()
    calibration = to_json(freeze(metadata))['key_calibration']
    assert calibration['baseline_depths_m']['0'] == pytest.approx(.00054)
    assert calibration['idle_noise_ranges_m']['28'] == pytest.approx(.00008)

    # Mutating an outgoing metadata object must not rewrite live calibration.
    metadata['key_calibration']['baseline_depths_m']['0'] = 1.
    metadata['key_calibration']['idle_noise_ranges_m']['28'] = 1.
    depths = policy.relative_depths({key: .00254 for key in range(29)})
    assert depths[0] == pytest.approx(.002)
    assert depths[28] == pytest.approx(.002)
    assert set(depths) == set(range(29))
    assert policy.snapshot()['idle_noise_ranges_m']['28'] == pytest.approx(.00008)
    with pytest.raises(ValueError, match='Missing or extra key measurement'):
        policy.relative_depths({str(key): .00254 for key in range(29)})
    with pytest.raises(ValueError, match='configuration keys must be strings'):
        freeze({0: .002})


@pytest.mark.parametrize('wrong,drop_demo',[(False,False),(True,False),(False,True)])
def test_piano_sampling_uses_lcs_and_does_not_require_demo_holding(wrong,drop_demo):
    scene,contacts=piano();w=scene.w;path='/World/CP12/Tools/Left/Mallet'
    tick=0
    def step(count=1):
        nonlocal tick
        for _ in range(count):tick+=1;scene.after_step(tick)
    step(180)
    assert w.key_press_policy.ready
    held=(path,'/World/RobotLeft',False,'link7');contacts.add(held)
    w.tools['Left'].pos[0]=.02;step()
    button=(path,'/World/CP12/StartFace',False,None);contacts.add(button)
    w.button.pos[2]=.996;step();contacts.remove(button);w.button.pos[2]=1;step()
    if drop_demo:contacts.remove(held)
    step(36+8*240+1)
    if drop_demo:contacts.add(held)
    assert w.clock.state=='replay'
    seq=list(w.clock.sequence)
    if wrong:seq[2]=7
    for key in seq:
        tip=(path+'/SoftTip',f'/World/CP12/Keyboard/Keys/White{key:02d}',False,None)
        contacts.add(tip);w.keys[key].pos[2]=.819;step()
        contacts.remove(tip);w.keys[key].pos[2]=.822;step()
    contacts.remove(held);w.tools['Left'].pos[0]=0
    contacts.add((path,'/World/CP12/Tools/Left',False,None));step(120)
    result=score(scene.evidence(tick))
    assert result.components['completion']['k']==(7 if wrong else 8)
    assert result.value==(15*7/8 if wrong else 100)
    assert not w.clock.failures
    assert len(scene.events.events)==8
    assert all(e['release_tick']>e['press_tick'] for e in scene.events.events)


class Demonstration:
    """Drive native contact/depth samples through the task's actual scene hooks."""
    def __init__(self,level='easy',factory=piano):
        self.scene,self.contacts=factory(level);self.tick=0
        self.step(180)
        w=self.scene.w
        if level=='hard':
            finger=('/World/CP12/VolumeCap','/World/RobotLeft',False,'link7')
            robot=('/World/CP12/VolumeCap',None,True,None)
            self.contacts.update((finger,robot));w.volume.pos[1]=.07;self.step()
            self.contacts.difference_update((finger,robot));self.step()
            assert self.scene.state['volume_max']
        for side in w.tools:
            self.contacts.add(self.held(side));w.tools[side].pos[0]+=.02
        self.step()
        button=(self.scene.paths['Left'],'/World/CP12/StartFace',False,None)
        self.contacts.add(button);w.button.pos[2]=.996;self.step()
        self.contacts.remove(button);w.button.pos[2]=1;self.step()
        assert w.clock.state=='transition'

    def held(self,side):
        return (self.scene.paths[side],'/World/Robot'+side,False,'link7')

    def step(self,count=1):
        for _ in range(count):
            self.tick+=1;self.scene.after_step(self.tick)


    def finish(self):
        w=self.scene.w
        self.step(max(0,w.clock.cue_start+len(w.clock.sequence)*w.clock.period_ticks+1-self.tick))
        for key in w.clock.sequence:
            tip=(self.scene.paths['Left']+'/SoftTip',f'/World/CP12/Keyboard/Keys/White{key:02d}',False,None)
            self.contacts.add(tip);w.keys[key].pos[2]=.819;self.step()
            self.contacts.remove(tip);w.keys[key].pos[2]=.822;self.step()
        for side in w.tools:
            self.contacts.remove(self.held(side));w.tools[side].pos[0]-=.02
            self.contacts.add((self.scene.paths[side],'/World/CP12/Tools/'+side,False,None))
        self.step(120)
        return score(self.scene.evidence(self.tick))


@pytest.mark.parametrize('drop_tick,restore_tick,failed', [
    (218,219,False),  # transition: holding restored before the first cue sample
    (219,220,True),   # first actual demonstration sample
    (2138,2139,True), # final actual demonstration sample
    (2139,2140,False),# replay begins: returning/releasing is permitted
])
def test_demo_holding_is_not_a_scoring_gate_at_any_boundary(drop_tick,restore_tick,failed):
    demo=Demonstration();clock=demo.scene.w.clock
    assert clock.cue_start==218
    demo.step(drop_tick-demo.tick-1)
    demo.contacts.remove(demo.held('Left'))
    demo.step(restore_tick-demo.tick-1)
    demo.contacts.add(demo.held('Left'));demo.step()
    assert not clock.failures
    assert demo.finish().value == 100


@pytest.mark.parametrize('level,missing,prefix,n', [
    ('medium','Left',1,8),('medium','Right',1,8),
    ('hard','Left',2,12),('hard','Right',2,12),
])
def test_missing_mallet_hold_during_demo_does_not_cancel_formal_notes(level,missing,prefix,n):
    demo=Demonstration(level);clock=demo.scene.w.clock
    demo.step(clock.cue_start-demo.tick-1)
    demo.contacts.remove(demo.held(missing));demo.step()
    demo.contacts.add(demo.held(missing))
    demo.step(len(clock.sequence)*clock.period_ticks)
    assert not clock.failures
    result=demo.finish()
    assert len(demo.scene.events.events)==(12 if level=='hard' else 8)
    assert result.components['completion']['k']==n
    assert result.components['completion']['n']==n
    assert result.value==100
    assert result.components['task_success'] is True


@pytest.mark.parametrize('level,n',[('easy',8),('medium',8),('hard',12)])
def test_legal_complete_demo_allows_required_tools_to_be_returned_and_released(level,n):
    demo=Demonstration(level)
    result=demo.finish()
    assert demo.scene.w.clock.failures==[]
    assert result.value==100
    assert result.components['completion']['k']==n
    assert all(tool['returned'] and tool['released'] for tool in demo.scene.evidence(demo.tick).values['cp05_evidence']['tools'])


def usd_piano(level='easy', stage=None):
    """Real USD material relationships; deterministic CPU contact/depth inputs."""
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdShade
    from mind2act_eval.tasks.CP05.scene import _material

    original, contacts = piano(level)
    w = original.w
    w.stage = Usd.Stage.CreateInMemory() if stage is None else stage
    for path in [f'/World/CP12/Keyboard/Keys/White{i:02d}/Cap' for i in w.keys] + ['/World/CP12/StatusLamp']:
        UsdGeom.Cube.Define(w.stage, path)
    calls, applications = [], []

    class BindingSpy:
        @staticmethod
        def Apply(prim):
            applications.append(str(prim.GetPath()))
            api = UsdShade.MaterialBindingAPI.Apply(prim)
            def bind(material):
                calls.append((str(prim.GetPath()), material.GetPath().name.removeprefix('FormalCP05_')))
                return api.Bind(material)
            return NS(Bind=bind)

    w.UsdShade = NS(MaterialBindingAPI=BindingSpy)
    w.shared = NS(material=lambda name, color, **kw: _material(w.stage, name, color, **kw))
    scene = PianoScene(original.spec, None, w)
    scene.material_spy = NS(calls=calls, applications=applications)
    return scene, contacts


def bound_colors(scene):
    from pxr import UsdShade
    paths = [f'/World/CP12/Keyboard/Keys/White{i:02d}/Cap' for i in scene.w.keys] + ['/World/CP12/StatusLamp']
    return {path: UsdShade.MaterialBindingAPI(scene.w.stage.GetPrimAtPath(path)).ComputeBoundMaterial()[0]
            .GetPath().name.removeprefix('FormalCP05_') for path in paths}


def test_static_ticks_do_not_rebind_real_usd_materials():
    scene, _ = usd_piano()
    scene.after_step(1)
    assert len(scene.material_spy.calls) == len(scene.material_spy.applications) == 30
    initial = bound_colors(scene)
    assert list(initial.values()) == ['white'] * 29 + ['yellow']
    for tick in range(2, 181):
        scene.after_step(tick)
    assert bound_colors(scene) == initial
    assert len(scene.material_spy.calls) == len(scene.material_spy.applications) == 30
    assert scene.w.key_press_policy.ready  # Sampling/calibration still runs every tick.


@pytest.mark.parametrize('reuse_stage', [False, True])
def test_new_scene_binds_initial_state_independently(reuse_stage):
    first, _ = usd_piano()
    first.after_step(1)
    first.w.clock.state = 'replay'
    first.w.clock.flash_until[1] = 100
    first.after_step(2)
    assert bound_colors(first)['/World/CP12/StatusLamp'] == 'green'
    second, _ = usd_piano(stage=first.w.stage if reuse_stage else None)
    assert second.w.stage == first.w.stage if reuse_stage else second.w.stage != first.w.stage
    second.after_step(1)
    assert list(bound_colors(second).values()) == ['white'] * 29 + ['yellow']
    assert len(second.material_spy.calls) == 30
    second.after_step(2)
    assert len(second.material_spy.calls) == 30


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_real_usd_material_edges_preserve_complete_tick_and_event_sequence(level):
    from pxr import UsdShade
    rgb = dict(white=(.89, .87, .8), cyan=(.02, .8, .85),
               green=(.05, .75, .15), yellow=(.9, .65, .02))
    previous, transitions, frames = {}, [], []

    def factory(tier):
        scene, contacts = usd_piano(tier)
        sample = scene.after_step
        def after_step(tick):
            start = len(scene.material_spy.calls)
            sample(tick)
            clock = scene.w.clock
            cue = clock.cue_key(tick)
            # Preserve the original per-tick material/RGB selection, including
            # cyan priority and green flash expiry after press/release.
            expected = {f'/World/CP12/Keyboard/Keys/White{i:02d}/Cap':
                        'cyan' if i == cue else 'green' if clock.flash_until.get(i, 0) > tick else 'white'
                        for i in scene.w.keys}
            expected['/World/CP12/StatusLamp'] = 'green' if clock.state == 'replay' else 'yellow'
            changed = [(path, color) for path, color in expected.items() if previous.get(path) != color]
            assert scene.material_spy.calls[start:] == changed
            assert bound_colors(scene) == expected
            for path, color in changed:
                prim = scene.w.stage.GetPrimAtPath(path)
                material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                shader = UsdShade.Shader(scene.w.stage.GetPrimAtPath(material.GetPath().AppendChild('Shader')))
                assert tuple(shader.GetInput('diffuseColor').Get()) == pytest.approx(rgb[color])
                transitions.append((tick, path, previous.get(path), color))
            previous.update(expected)
            frames.append((tick, clock.state, cue))
        scene.after_step = after_step
        return scene, contacts

    demo = Demonstration(level, factory=factory)
    result = demo.finish()
    assert result.value == 100
    assert [tick for tick, _, _ in frames] == list(range(1, demo.tick + 1))
    events = demo.scene.events.events
    assert [event['key'] for event in events] == demo.scene.w.clock.sequence
    assert all(event['nib_contact'] and event['release_tick'] == event['press_tick'] + 1 for event in events)
    assert demo.scene.w.clock.failures == []
    assert any(old == 'cyan' and new == 'white' for _, _, old, new in transitions)
    assert any(old == 'white' and new == 'cyan' for _, _, old, new in transitions)
    assert any(old == 'white' and new == 'green' for _, _, old, new in transitions)
    assert any(old == 'green' and new == 'white' for _, _, old, new in transitions)
    assert len(demo.scene.material_spy.applications) == 30


def test_final_play_does_not_require_picked_up_mallet_and_key_depth_accepts_finger():
    scene,contacts=piano();w=scene.w
    for tick in range(1,181):scene.after_step(tick)
    w.button.pos[2]=.997
    scene.after_step(181)
    assert w.clock.cue_start is not None
    end=w.clock.cue_start+len(w.clock.sequence)*w.clock.period_ticks
    for tick in range(182,end):scene.after_step(tick)
    key=w.clock.sequence[0]
    contacts.add((f'/World/CP12/Keyboard/Keys/White{key:02d}',None,True,None))
    w.keys[key].pos[2]=.819
    scene.after_step(end)
    assert [e['key'] for e in scene.events.events]==[key]
    scene.after_step(end+1)
    assert len(scene.events.events)==1
    w.keys[key].pos[2]=.822;scene.after_step(end+2)
    w.keys[key].pos[2]=.819;scene.after_step(end+3)
    assert len(scene.events.events)==2
    assert not w.clock.failures


def test_return_uses_real_rack_bounds_instead_of_hidden_ten_mm_target():
    scene,contacts=piano()
    scene.w.rack_contains=lambda side,p,q:bool(-.04<=p[0]<=.04 and -.02<=p[1]<=.02)
    scene.w.tools['Left'].pos[0]=.02
    contacts.add(('/World/CP12/Tools/Left/Mallet','/World/CP12/Tools/Left',False,None))
    scene.after_step(1)
    assert scene.evidence(1).values['cp05_evidence']['tools'][0]['returned']
