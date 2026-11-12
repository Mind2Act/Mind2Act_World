"""CPU-only synthetic measurements; these fixtures are not runtime instances."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.contracts import BuildContext, InstanceSpec, NativeEvidence
from mind2act_eval.tasks.CP10.machine import ArenaMachine
from mind2act_eval.tasks.CP10.prefix import PhysicalPrefix
from mind2act_eval.tasks.CP10.protocol import PROTOCOL_ID, RETENTION
from mind2act_eval.tasks.CP10.scene import ArenaScene
from mind2act_eval.tasks.CP10.scorer import score
from mind2act_eval.tasks.CP10.retention import floor_z


def configuration(difficulty):
    total, white = {'easy': (5, 3), 'medium': (8, 4), 'hard': (12, 6)}[difficulty]
    colors = ['white', 'red'] + ['white'] * white + ['red'] * (total - white)
    objects = {'paddle': dict(position=[0., -.22, .84], size=[.052, .014, .048])}
    objects.update({'ball' + str(i): dict(position=[0., .4, .831], size=[.036] * 3,
                    asset='authored_sphere') for i in range(len(colors))})
    return dict(task='CP10', difficulty=difficulty, seed=0, physics_hz=120, max_ticks=10000,
        support_tolerance=.004, objects=objects, arena=dict(shot_colors=colors,
        practice_shots=2, scored_shots=total, net_settle_ticks=60, per_shot_timeout_ticks=2400,
        initial_yaw_deg=0., initial_slide_x_m=0., court_center=[0., .15], court_size=[.65, .88],
        surface_z=.813, net_depth_m=.12, collector_length_m=dict(near=.075, far=.135),
        heading_sampling=dict(distribution='uniform', stream='cpu-only-fixture', range_deg=[0., 0.]),
        launcher=dict(base_xy=[0., .755], yaw_limits_deg=[-58., 58.], muzzle_distance_m=.36,
                      stowed_pitch_deg=-90.),
        countdown=dict(seconds=1, settle_ticks=12, heading_tolerance_deg=.5,
                       max_yaw_speed_deg_s=1.)))


@pytest.fixture(autouse=True)
def authorize_synthetic_prefix_source(monkeypatch):
    # These CPU mechanism fixtures are explicitly not runtime instances. Replace
    # source authority at this unit seam; actual original-input integration and
    # marker-deletion tests use the unmodified production cohort pins.
    from mind2act_eval.tasks.CP10 import scoring_envelope
    from mind2act_eval.tasks.CP10.scene import encode_config
    monkeypatch.setattr(scoring_envelope, 'TRAINING_CONFIG_SHA256', {
        level: hashlib.sha256(encode_config(configuration(level))).hexdigest()
        for level in ('easy', 'medium', 'hard')})


def measured(config, tick, moved=False):
    bodies = {}
    for name, spec in config['objects'].items():
        center = list(spec['position'])
        if moved and name == 'ball0':
            center[0] += .006
        half = [s / 2 for s in spec['size']]
        bodies[name] = dict(lower=[p - h for p, h in zip(center, half)],
                            upper=[p + h for p, h in zip(center, half)], contacts=[],physical_contacts=[],physical_robot_contact=False,support_nets=dict(near=False,far=False))
    return dict(tick=tick, bodies=bodies, emitted_balls=[], launcher=dict(
        yaw_deg=0., yaw_speed_deg_s=0., pitch_deg=0., pitch_speed_deg_s=0.,
        slide_x_m=0., slide_speed_m_s=0.))


def evidence(journal, difficulty, values=None, tick=None):
    values = journal.values() if values is None else deepcopy(values)
    values.update(difficulty=difficulty, collection_protocol=dict(RETENTION),
                  rule_version=PROTOCOL_ID, normal_termination=True)
    return NativeEvidence('CP10', 'cpu-only-fixture', journal.tick if tick is None else tick, values)


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_eof_with_running_native_machine_scores_actual_prefix(tmp_path, difficulty):
    config = configuration(difficulty)
    journal = PhysicalPrefix(ArenaMachine(config, RETENTION, PROTOCOL_ID), tmp_path)
    for tick in range(10):
        journal.observe(measured(config, tick))
    values = journal.values()
    assert values['native_terminal_tick'] is None
    assert values['frozen_native_snapshot'] is None
    assert values['physical_prefix_receipt']['count'] == 10
    assert score(evidence(journal, difficulty)).value == 0
    journal.close()


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
def test_waiting_ball_drift_keeps_complete_real_journal(tmp_path, difficulty):
    config=configuration(difficulty)
    journal=PhysicalPrefix(ArenaMachine(config,RETENTION,PROTOCOL_ID),tmp_path)
    for tick in range(8):journal.observe(measured(config,tick,moved=tick>0))
    v=journal.values()
    assert v['native_terminal_tick'] is None and v['scored_until_tick']==7
    assert v['physical_prefix_receipt']['count']==8 and v['postterminal_replay']['count']==0
    assert score(evidence(journal,difficulty)).value==0
    journal.close()


@pytest.mark.parametrize('key,bad', [('sha256', '0' * 64), ('bytes', 1), ('count', 1),
                                    ('first_tick', False), ('last_tick', True), ('last_tick', 8)])
def test_bad_prefix_receipt_is_invalid(tmp_path, key, bad):
    config = configuration('easy')
    journal = PhysicalPrefix(ArenaMachine(config, RETENTION, PROTOCOL_ID), tmp_path)
    journal.observe(measured(config, 0))
    journal.observe(measured(config, 1, moved=True))
    values = journal.values()
    values['physical_prefix_receipt'][key] = bad
    with pytest.raises(ValueError):
        score(evidence(journal, 'easy', values))
    journal.close()


def test_unfrozen_eof_cannot_claim_an_earlier_scored_cutoff(tmp_path):
    config = configuration('easy')
    journal = PhysicalPrefix(ArenaMachine(config, RETENTION, PROTOCOL_ID), tmp_path)
    journal.observe(measured(config, 0))
    with pytest.raises(ValueError):
        score(evidence(journal, 'easy', tick=1))
    journal.close()


def test_appending_to_closed_prefix_invalidates_receipt(tmp_path):
    config = configuration('easy')
    journal = PhysicalPrefix(ArenaMachine(config, RETENTION, PROTOCOL_ID), tmp_path)
    journal.observe(measured(config, 0))
    journal.observe(measured(config, 1, moved=True))
    with journal.path.open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError):
        score(evidence(journal, 'easy'))
    journal.close()


def test_forged_frozen_snapshot_and_tail_clock_are_invalid(tmp_path):
    config = configuration('easy')
    journal = PhysicalPrefix(ArenaMachine(config, RETENTION, PROTOCOL_ID), tmp_path)
    journal.observe(measured(config, 0))
    journal.observe(measured(config, 1, moved=True))
    journal.observe(measured(config, 2, moved=True))
    values = journal.values()
    values['cp10']['failure'] = dict(tick=1,reason='robot_touched_ball')
    with pytest.raises(ValueError):
        score(evidence(journal, 'easy', values))
    values = journal.values()
    values['postterminal_replay']['count'] += 1
    with pytest.raises(ValueError):
        score(evidence(journal, 'easy', values))
    journal.close()


@pytest.mark.parametrize('difficulty', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize('wrong_third_white', [False, True])
def test_native_receipts_feed_success_and_authoritative_partial_score(difficulty, wrong_third_white):
    config = configuration(difficulty)
    machine = ArenaMachine(config, RETENTION, PROTOCOL_ID)
    colors = config['arena']['shot_colors']
    scene = object.__new__(ArenaScene)
    scene.machine = machine
    white_indices = [i for i, color in enumerate(colors) if i >= 2 and color == 'white']
    touched = set()
    for tick in range(config['max_ticks']):
        frame = measured(config, tick)
        frame['emitted_balls'] = sorted(machine.expected_emitted)
        h = machine.hardware
        frame['launcher'].update(yaw_deg=h['target_yaw_deg'], pitch_deg=h['target_pitch_deg'])
        for name in machine.expected_emitted:
            index = int(name[4:])
            net = 'far' if colors[index] == 'white' else 'near'
            if wrong_third_white and index == white_indices[2]:
                net = 'near'
            a = config['arena']
            sign = -1 if net == 'near' else 1
            center = [0., a['court_center'][1] + sign * (a['court_size'][1] / 2
                       + a['collector_length_m'][net] / 2), floor_z(config, net) + .018]
            contacts = []
            if colors[index] == 'white' and name not in touched:
                contacts.append('/World/Task/Objects/paddle')
                touched.add(name)
            frame['bodies'][name] = dict(lower=[p - .018 for p in center],
                                        upper=[p + .018 for p in center], contacts=contacts,physical_contacts=contacts,physical_robot_contact=False,support_nets={n:n==net for n in ('near','far')})
        machine.sample(frame)
        # The real source cycle needs one further sample after its final
        # receipt, including a wrong final outcome. Earlier EOF gets no grant.
        if len(machine.shots) < len(colors) or machine.status != 'running':
            assert scene.replay_eof_settle_ticks() == 0
        else:
            assert scene.replay_eof_settle_ticks() == 1
        if machine.status != 'running':
            break
    assert machine.status == ('failure' if wrong_third_white else 'success')
    assert len(machine.shots) == len(colors)
    values = dict(difficulty=difficulty, cp10=machine.snapshot(), normal_termination=True,
                  collection_protocol=dict(RETENTION), rule_version=PROTOCOL_ID)
    result = score(NativeEvidence('CP10', 'cpu-only-fixture', machine.tick, values))
    n = {'easy': 4, 'medium': 5, 'hard': 7}[difficulty]
    assert result.value == pytest.approx(dict(easy=15,medium=10,hard=7)[difficulty] * 2 / (n-1) if wrong_third_white else 100)


def test_global_actor_cutoff_cannot_freeze_pending_collection_window():
    config = configuration('easy')
    config['max_ticks'] = 4
    machine = ArenaMachine(config, RETENTION, PROTOCOL_ID)
    for tick in range(5):
        machine.sample(measured(config, tick))
    assert machine.status == 'running'


def test_native_practice_timeout_advances_without_terminating():
    config=configuration('easy');machine=ArenaMachine(config,RETENTION,PROTOCOL_ID)
    first=None
    for tick in range(5000):
        frame=measured(config,tick);frame['emitted_balls']=sorted(machine.expected_emitted)
        h=machine.hardware;frame['launcher'].update(yaw_deg=h['target_yaw_deg'],pitch_deg=h['target_pitch_deg'])
        machine.sample(frame)
        if first is None and machine.fired_tick is not None:first=machine.fired_tick
        if machine.shots:break
    assert machine.status=='running' and machine.failure is None
    assert machine.shots[0]['collection_tick']-first==2401
    assert machine.shots[0]['net'] is None


def test_scene_publishes_final_native_state_once_and_retains_real_outer_tick(tmp_path):
    config = configuration('easy')
    spec = InstanceSpec('CP10', 'cpu-only-fixture', 'easy', 'cpu-only-fixture',
        'cpu-only-fixture', 'cpu-only-fixture', 'cpu-only-fixture', seed=0,
        config=dict(source_config=config, collection_protocol=dict(RETENTION), rule_version=PROTOCOL_ID))

    class Objects:
        def __init__(self):
            self.names = list(config['objects'])
            self.positions = np.array([v['position'] for v in config['objects'].values()])

        def get_world_poses(self):
            return self.positions, np.tile([1., 0., 0., 0.], (len(self.names), 1))

        def get_velocities(self):
            return np.zeros((len(self.names), 6))

    class Launcher:
        def get_joint_positions(self):
            return np.zeros(3)

        def get_joint_velocities(self):
            return np.zeros(3)

    objects = Objects()
    scene = ArenaScene(spec, SimpleNamespace(output_dir=tmp_path,sim=SimpleNamespace(contacts=SimpleNamespace(supported=lambda *a,**kw:False))), None, objects, Launcher(),
        objects.names, dict(Slide=0, Yaw=1, Pitch=2), config, 'CPU fixture instruction')
    published = []
    scene._update_public = lambda: published.append(scene.machine.status)
    scene.prefix.observe(scene._measure(0))
    objects.positions[objects.names.index('ball0'), 0] += .006
    scene.after_step(1)
    assert not scene.events_complete() and scene.health().ok
    assert published == ['running']
    for tick in range(2, 6):
        scene.after_step(tick)
    assert published == ['running']*5
    assert scene.evidence(5).tick == 5
    assert score(scene.evidence(5)).value == 0
    assert scene.result_metadata()['scored_until_tick'] == 5
    assert scene.result_metadata()['postterminal_replay_ticks'] == 0
    scene.close()


def test_real_builder_writes_only_task_owned_physics(monkeypatch, tmp_path):
    """Call the product builder with recording native API bindings, not Kit."""
    import sys
    from types import ModuleType
    from mind2act_eval.tasks.CP10 import scene as product

    config = configuration('easy')
    config['prop_solver'] = dict(position_iterations=128, velocity_iterations=8)
    spec = InstanceSpec('CP10', 'cpu-only-fixture', 'easy', 'cpu-only-fixture',
        'cpu-only-fixture', 'cpu-only-fixture', 'v1', config=dict(source_config=config,
        collection_protocol=dict(RETENTION), rule_version=PROTOCOL_ID))
    rigid_api, collision_api = object(), object()
    mutations = []

    class Prim:
        def __init__(self, path, apis=(), attributes=None):
            self.path, self.apis, self.attributes = path, apis, attributes or {}

        def GetPath(self):
            return self.path

        def HasAPI(self, api):
            return api in self.apis

        def GetAttribute(self, name):
            return SimpleNamespace(Get=lambda: self.attributes.get(name))

    task = Prim('/World/Task')
    ball = Prim('/World/Task/Objects/ball0', (rigid_api, collision_api))
    net = Prim('/World/Task/Design/Nets/Near/Bottom', (collision_api,))
    physics = Prim('/World/Physics', attributes={'physxScene:timeStepsPerSecond': 60,
                   'physxScene:enableCCD': False, 'physxScene:solverType': 'PGS'})
    floor = Prim('/World/Floor', (rigid_api, collision_api))
    robot = Prim('/World/RobotLeft/Asset/link7', (rigid_api, collision_api))
    station = Prim('/World/Worktable', (rigid_api, collision_api))
    prims = [task, ball, net, physics, floor, robot, station]
    stage = SimpleNamespace(GetPrimAtPath=lambda path: next((p for p in prims if p.path == path), None),
                            Traverse=lambda: iter(prims))

    class API:
        def __init__(self, prim):
            self.prim = prim

        def __getattr__(self, name):
            assert name.startswith('Create') and name.endswith('Attr')

            def create(value):
                mutations.append((self.prim.path, name, value))
                return SimpleNamespace(Set=lambda v: mutations.append((self.prim.path, name, v)))
            return create

    def schema():
        def apply(prim):
            mutations.append((prim.path, 'ApplyAPI', None))
            return API(prim)
        return SimpleNamespace(Apply=apply)

    def module(name, **attributes):
        value = ModuleType(name)
        value.__path__ = []
        for key, attribute in attributes.items():
            setattr(value, key, attribute)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    physx = SimpleNamespace(PhysxSceneAPI=schema(), PhysxContactReportAPI=schema(),
                           PhysxRigidBodyAPI=schema(), PhysxCollisionAPI=schema())
    module('pxr', PhysxSchema=physx, PhysicsSchemaTools=SimpleNamespace(intToSdfPath=str),
        Usd=SimpleNamespace(PrimRange=lambda root: iter([task, ball, net]) if root is task else iter([])),
        UsdPhysics=SimpleNamespace(RigidBodyAPI=rigid_api, CollisionAPI=collision_api))
    names = list(config['objects'])
    objects = SimpleNamespace(prim_paths=['/World/Task/Objects/' + n for n in names],
        get_velocities=lambda: np.zeros((len(names), 6)), set_velocities=lambda values: None)
    launcher = SimpleNamespace(num_dof=3,
        get_dof_index=lambda name: {'Slide': 0, 'Yaw': 1, 'Pitch': 2}[name],
        set_joint_positions=lambda q: None, set_joint_velocities=lambda q: None,
        apply_action=lambda action: None)
    module('isaacsim')
    module('isaacsim.core')
    module('isaacsim.core.prims', RigidPrim=lambda *a, **kw: objects,
           SingleArticulation=lambda *a, **kw: launcher)
    module('isaacsim.core.utils')
    module('isaacsim.core.utils.types', ArticulationAction=lambda **kw: SimpleNamespace(**kw))
    module('omni')
    module('omni.physx',
        get_physx_interface=lambda: SimpleNamespace(update_transformations=lambda *a: None),
        get_physx_simulation_interface=lambda: SimpleNamespace(
            subscribe_contact_report_events=lambda callback: SimpleNamespace(unsubscribe=lambda: None)))
    module('omni.physx.bindings')
    module('omni.physx.bindings._physx', ContactEventType=SimpleNamespace(CONTACT_LOST=0))
    world = SimpleNamespace(scene=SimpleNamespace(add=lambda obj: obj),
        physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: None))
    ctx = BuildContext(SimpleNamespace(stage=stage, world=world),
        SimpleNamespace(prepare_stage=lambda stage: None, initialize=lambda world: None), object(), tmp_path)
    monkeypatch.setattr(product, 'validate_binding', lambda spec, assets: config)

    class MeasuredScene:
        def __init__(self, *args):
            self.prefix = SimpleNamespace(observe=lambda measurement: None)

        def _measure(self, tick):
            return dict(tick=tick)

        def _update_public(self):
            pass

    monkeypatch.setattr(product, 'ArenaScene', MeasuredScene)
    built = product.build(spec, ctx)
    assert mutations
    assert all(path.startswith('/World/Task/') for path, name, value in mutations)
    assert (ball.path, 'CreateEnableCCDAttr', True) in mutations
    assert (ball.path, 'CreateSolverPositionIterationCountAttr', 128) in mutations
    assert (net.path, 'CreateContactOffsetAttr', .001) in mutations
    assert (net.path, 'CreateRestOffsetAttr', 0.) in mutations
    diagnostics = built.source_physics_diagnostics
    assert diagnostics['source_wrapper_global_requested']['physxScene:timeStepsPerSecond'] == 120
    assert diagnostics['observed_shared_scene']['physxScene:timeStepsPerSecond'] == 60
    assert diagnostics['observed_shared_scene']['physxScene:enableCCD'] is False
    assert diagnostics['global_and_station_opinions_applied'] is False


def drive_complete_machine(*, robot_contact=False):
    config=configuration('easy');m=ArenaMachine(config,RETENTION,PROTOCOL_ID)
    for tick in range(4000):
        f=measured(config,tick);f['emitted_balls']=sorted(m.expected_emitted)
        h=m.hardware;f['launcher'].update(yaw_deg=h['target_yaw_deg'],pitch_deg=h['target_pitch_deg'])
        for name in m.expected_emitted:
            index=int(name[4:]);net='far' if m.colors[index]=='white' else 'near'
            a=config['arena'];sign=-1 if net=='near' else 1
            center=[0.,a['court_center'][1]+sign*(a['court_size'][1]/2+a['collector_length_m'][net]/2),.8]
            f['bodies'][name]=dict(lower=[p-.018 for p in center],upper=[p+.018 for p in center],
                contacts=[],physical_contacts=['/World/Task/Objects/paddle'] if m.colors[index]=='white' else [],
                physical_robot_contact=robot_contact and index==2,support_nets={n:n==net for n in ('near','far')})
        m.sample(f)
        if m.status!='running':return m,f
    raise AssertionError('sequence did not complete')


def test_formal_robot_contact_scores_zero_but_every_ball_still_runs():
    m,f=drive_complete_machine(robot_contact=True)
    assert len(m.shots)==len(m.colors) and m.failure['reason']=='robot_touched_ball'
    e=NativeEvidence('CP10','cpu',m.tick,dict(difficulty='easy',cp10=m.snapshot(),normal_termination=True,
        collection_protocol=dict(RETENTION),rule_version=PROTOCOL_ID))
    assert score(e).value==0


def test_final_net_state_is_measured_after_sequence_completion():
    m,f=drive_complete_machine()
    assert m.status=='success'
    # A collected formal white later leaves its actual net before the outer action ends.
    f['tick']+=1;f['emitted_balls']=sorted(m.expected_emitted)
    f['bodies']['ball2']['support_nets']=dict(near=False,far=False)
    m.sample(f)
    assert m.final_nets['ball2'] is None
    e=NativeEvidence('CP10','cpu',m.tick,dict(difficulty='easy',cp10=m.snapshot(),normal_termination=True,
        collection_protocol=dict(RETENTION),rule_version=PROTOCOL_ID))
    assert score(e).value==0
