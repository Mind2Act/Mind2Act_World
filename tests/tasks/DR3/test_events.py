"""Environment-event regressions use synthetic analytic motions, never expert events."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from mind2act_eval.tasks.DR3.events import rebuild_events as _rebuild_events
from mind2act_eval.tasks.DR3.geometry import swept_aperture
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.DR3.audit import AppendOnlyTrace, write_manifest
from mind2act_eval.tasks.DR3.audit import read_trace
from mind2act_eval.tasks.DR3.instances import SOURCE_ID
from mind2act_eval.tasks.DR3.scorer import score
from mind2act_eval.tasks.DR3.scene import GateScene


def rebuild_events(rows,manifest):
    return _rebuild_events(rows,manifest,allow_test_certificates=True)


def fixture(hard=False, end=240):
    vertices=[[x,y,z] for x in (-.005,.005) for y in (-.005,.005) for z in (-.005,.005)]
    gate=lambda name,x:dict(id=name,origin=[x,0,0],aperture_low_yz=[-.5,-.5],aperture_high_yz=[.5,.5],
        half_thickness_m=.001,fully_open_position_m=1.,open_tolerance_m=.001,closed_tolerance_m=.001)
    gates=[gate("blue",0)] + ([gate("orange",.06)] if hard else [])
    manifest=dict(physics_dt_s=1/120,native_runtime_validated=True,
        bodies=dict(item=dict(vertices=vertices,role="item"),grip=dict(vertices=vertices,role="robot",arm="Left",gripper=True)),
        coverage=dict(expected_colliders=["item","grip"],represented_colliders=["item","grip"],missing_colliders=[]),
        item_root="/World/Task/Item",gates=gates,gate_colliders=["/World/Task/Gate/Collider"],
        frame_colliders=["/World/Task/Rail/Collider"],tray_colliders=["/World/Task/Tray/Collider"],
        structural_contact_pairs=[],thresholds=dict(contact_impulse_ns=1e-5,support_normal_z=.5,motion_epsilon_m=1e-6),
        tray=dict(origin=[.15,0,0],low=[-.06,-.1,-.01],high=[.06,.1,.1],
                  floor_triangles_xy=[[[-.06,-.1],[.06,-.1],[.06,.1]],
                                      [[-.06,-.1],[.06,.1],[-.06,.1]]]))
    def pose(x):
        return dict(position=[x,0,0],quaternion_wxyz=[1,0,0,0],linear_speed_m_s=0.,angular_speed_rad_s=0.)
    def contact(path,normal=None):
        return dict(pair=["/World/Task/Item/Collider",path],active=True,impulse_ns=.001,
                    item_upward_normals=[] if normal is None else [normal])
    rows=[]
    for i in range(end+1):
        x=-.1 if i==0 else min(.15,-.1+.025*i)
        gx=x if i<=12 else max(-.04,.15-.025*(i-12))
        contacts=[]
        if i<=10:
            contacts=[contact("/World/RobotLeft/Asset/link7/Collider"),contact("/World/RobotLeft/Asset/link8/Collider")]
        if i>=10:contacts.append(contact("/World/Task/Tray/Collider",1.))
        frame=dict(poses=dict(item=pose(x),grip=pose(gx)),gates={g["id"]:1. if i<60 else 0. for g in gates},
            contacts=contacts,contacts_complete=True,
            sweeps={n:{g["id"]:dict(contract="bounded-rigid-envelope-v1",source="analytic_test_fixture",
                deviation_bound_m=0.,exact_convex_shape=True) for g in gates} for n in ("item","grip")})
        rows.append(dict(physics_tick=i,simulation_time_s=i/120,detector=frame))
    return manifest,rows


def audit_evidence(tmp_path, manifest, rows):
    root = Path(tmp_path) / 'DR3'
    manifest_receipt = write_manifest(root / 'manifest.json', manifest)
    writer = AppendOnlyTrace(root / 'raw-trace.jsonl')
    for row in rows:
        writer.append(row)
    trace_receipt = writer.receipt()
    writer.close()
    return NativeEvidence('DR3', 'fr3-test', len(rows)-1,
                          dict(difficulty='easy', source_id=SOURCE_ID,
                               condition={'instance_id': 'fr3-test'},
                               manifest=manifest_receipt, raw_trace=trace_receipt))


@pytest.mark.parametrize('extra', ['held', 'no_gripper_pass', 'frame_contact', 'moving'])
def test_final_rules_t0_delivery_has_no_extra_process_gates(extra):
    manifest, rows = fixture(end=30)
    if extra == 'held':
        rows[-1]['detector']['contacts'] += deepcopy(rows[5]['detector']['contacts'])
    elif extra == 'no_gripper_pass':
        for row in rows: row['detector']['poses']['grip']['position'][0] = -.1
    elif extra == 'frame_contact':
        rows[5]['detector']['contacts'].append(dict(
            pair=['/World/RobotRight/Asset/link4/Collider','/World/Task/Rail/Collider'],
            active=True, impulse_ns=.001, item_upward_normals=[]))
    else:
        rows[-1]['detector']['poses']['item']['linear_speed_m_s'] = .03
    result = rebuild_events(rows, manifest)
    assert result['evidence_complete'] and result['score'] == 100


def test_final_rules_center_in_tray_does_not_require_whole_envelope():
    manifest, rows = fixture(end=30)
    rows[-1]['detector']['poses']['item']['position'][0] = .209
    result = rebuild_events(rows, manifest)
    assert result['evidence_complete'] and result['score'] == 100


@pytest.mark.parametrize('normal, impulse, expected', [(1., 0., 100), (0., .001, 5)])
def test_final_rules_real_support_uses_owned_normal_not_impulse(normal, impulse, expected):
    manifest, rows = fixture(end=30)
    rows[-1]['detector']['contacts'] = [dict(
        pair=['/World/Task/Item/Collider','/World/Task/Tray/Collider'],
        active=True, impulse_ns=impulse, item_upward_normals=[normal])]
    assert rebuild_events(rows, manifest)['score'] == expected


def test_final_rules_gripper_suspension_is_not_target_support():
    manifest, rows = fixture(end=30)
    rows[-1]['detector']['contacts'] = deepcopy(rows[5]['detector']['contacts'])
    result = rebuild_events(rows, manifest)
    assert result['score'] == 5 and not result['terminal_tray_support']


def test_final_rules_t0_position_controls_delivery_not_historical_best():
    manifest, rows = fixture()
    rows[-1]['detector']['poses']['item']['position'][1] = .2
    result = rebuild_events(rows, manifest)
    assert result['score'] == 5 and not result['delivered']


def test_actual_floor_extent_replaces_the_old_inset_region():
    manifest, rows = fixture(end=30)
    manifest['tray']['floor_triangles_xy'] = [
        [[-.08,-.1],[.08,-.1],[.08,.1]], [[-.08,-.1],[.08,.1],[-.08,.1]]]
    rows[-1]['detector']['poses']['item']['position'][0] = .225
    assert rebuild_events(rows, manifest)['score'] == 100


def test_real_floor_hole_is_not_filled_by_its_bounding_box():
    manifest, rows = fixture(end=30)
    manifest['tray']['floor_triangles_xy'] = []
    for low, high in [((-.08,-.1),(-.02,.1)), ((.02,-.1),(.08,.1)),
                      ((-.02,-.1),(.02,-.02)), ((-.02,.02),(.02,.1))]:
        x0,y0=low; x1,y1=high
        manifest['tray']['floor_triangles_xy'] += [
            [[x0,y0],[x1,y0],[x1,y1]], [[x0,y0],[x1,y1],[x0,y1]]]
    result = rebuild_events(rows, manifest)
    assert result['score'] == 5 and not result['center_inside_tray']


def test_old_source_inset_without_actual_cavity_geometry_is_unknown():
    manifest, rows = fixture(end=30)
    del manifest['tray']['floor_triangles_xy']
    manifest['tray']['provenance'] = 'whole-object source target region'
    result = rebuild_events(rows, manifest)
    assert not result['evidence_complete'] and result['score'] is None


def test_actual_target_contact_with_missing_normal_is_invalid():
    manifest, rows = fixture(end=30)
    rows[-1]['detector']['contacts'][-1]['item_upward_normals'] = []
    with pytest.raises(ValueError, match='support normal'):
        rebuild_events(rows, manifest)


@pytest.mark.parametrize("hard", [False, True])
def test_complete_ordered_item_pass_and_t0_delivery(hard):
    manifest, rows = fixture(hard)
    value = rebuild_events(rows, manifest)
    assert value['evidence_complete'], value['unknown_reasons']
    assert value['score'] == 100 and value['success']
    assert value['passed_aperture_count'] == (2 if hard else 1)
    assert all(e['status'] == 'OCCURRED' for e in value['events']['passed'])


def test_long_item_may_enter_second_gate_before_clearing_first():
    manifest, rows = fixture(hard=True)
    manifest['bodies']['item']['vertices'] = [[x,y,z] for x in (-.06,.06)
        for y in (-.005,.005) for z in (-.005,.005)]
    result = rebuild_events(rows, manifest)
    assert result['evidence_complete'], result['unknown_reasons']
    assert result['score'] == 100 and not result['violation_latched']
    entry, passed = result['events']['entry'], result['events']['passed']
    assert entry[0]['time_s'] < entry[1]['time_s'] < passed[0]['time_s'] < passed[1]['time_s']


def test_second_gate_cannot_replace_missing_first_gate_entry():
    manifest,rows=fixture(hard=True)
    # Traverse orange before blue: the latter's entry cannot reuse the former.
    manifest['gates'][0]['origin'][0]=.06
    manifest['gates'][1]['origin'][0]=0.
    result=rebuild_events(rows,manifest)
    assert result['evidence_complete']
    assert result['score']==0 and not result['passed_aperture']


@pytest.mark.parametrize('end,expected', [(12, 5), (30, 100), (240, 100)])
def test_historical_pending_gpu_qualification_does_not_override_measured_outcome(end, expected):
    manifest, rows = fixture(end=end)
    if end == 12:
        for row in rows:
            if row['physics_tick'] >= 10:
                row['detector']['contacts'] = deepcopy(rows[5]['detector']['contacts'])
    manifest['native_runtime_validated'] = False
    manifest['qualification_status'] = 'pending_native_geometry_contact_and_sweep_validation'
    result = rebuild_events(rows, manifest)
    assert result['evidence_complete'] and result['score'] == expected
    assert not any('GPU qualification' in reason for reason in result['unknown_reasons'])


def test_expert_fields_and_cached_events_do_not_affect_replay():
    manifest,rows=fixture()
    expected=rebuild_events(rows,manifest)
    for row in rows:
        row.update(expert_release_tick=-999,release_tick=1,entry_tick=0,cached_events={"success":False},old_score=999)
    assert rebuild_events(rows,manifest)==expected


def test_unqualified_native_motion_is_unknown_not_a_model_failure():
    manifest, rows = fixture()
    rows[4]['detector']['sweeps'] = {}
    value = rebuild_events(rows, manifest)
    assert not value['evidence_complete'] and value['score'] is None and value['success'] is None
    assert any('aperture passage' in reason for reason in value['unknown_reasons'])


@pytest.mark.parametrize("problem",["coverage","clock","contacts"])
def test_raw_evidence_defects_are_explicit_unknown(problem):
    manifest,rows=fixture()
    if problem=="coverage":manifest["coverage"]["missing_colliders"]=["right_gripper"]
    elif problem=="clock":rows[8]["simulation_time_s"]+=.001
    else:rows[8]["detector"]["contacts_complete"]=False
    value=rebuild_events(rows,manifest)
    assert value["score"] is None and value["unknown_reasons"]


def test_robot_does_not_need_to_pass_when_the_whole_item_does():
    manifest, rows = fixture(end=9)
    for row in rows: row['detector']['poses']['grip']['position'][0] = -.1
    value = rebuild_events(rows, manifest)
    assert value['score'] == 5 and value['passed_aperture']


def test_complete_pass_without_release_is_valid_five():
    manifest,rows=fixture(end=12)
    for row in rows:
        if row["physics_tick"]>=10:
            row["detector"]["contacts"]=deepcopy(rows[5]["detector"]["contacts"])
    value=rebuild_events(rows,manifest)
    assert value["evidence_complete"] and value["score"]==5 and not value["success"]


def test_delivery_without_waiting_for_close_is_full_success():
    manifest, rows = fixture(end=30)
    value = rebuild_events(rows, manifest)
    assert value['score'] == 100 and value['success']


def test_door_closing_after_a_complete_pass_does_not_invalidate_delivery():
    manifest, rows = fixture()
    for row in rows: row['detector']['gates']['blue'] = 1. if row['physics_tick'] < 11 else 0.
    value = rebuild_events(rows, manifest)
    assert value['score'] == 100 and not value['violation_latched']


def test_harmless_final_robot_contact_does_not_undo_real_tray_support():
    manifest, rows = fixture()
    rows[-1]['detector']['contacts'] += deepcopy(rows[5]['detector']['contacts'])
    value = rebuild_events(rows, manifest)
    assert value['score'] == 100 and value['terminal_tray_support']


@pytest.mark.parametrize('clearance', [.025, .024])
def test_gripper_clearance_is_not_a_score_gate(clearance):
    manifest, rows = fixture()
    for row in rows[20:]: row['detector']['poses']['grip']['position'][0] = -clearance-.005
    assert rebuild_events(rows, manifest)['score'] == 100


def test_robot_reentry_does_not_change_a_completed_item_delivery():
    manifest, rows = fixture()
    rows[-1]['detector']['poses']['grip']['position'][0] = -.01
    assert rebuild_events(rows, manifest)['score'] == 100


def test_past_item_motion_has_no_stability_gate_after_a_legal_pass():
    manifest, rows = fixture()
    rows[130]['detector']['poses']['item']['position'][1] = .002
    rows[140]['detector']['poses']['item']['position'][1] = -.002
    assert rebuild_events(rows, manifest)['score'] == 100


def test_short_terminal_window_needs_no_extra_settling():
    manifest, rows = fixture(end=119)
    assert rebuild_events(rows, manifest)['score'] == 100


def test_exact_convex_bypass_is_violation_but_enclosure_uncertainty_is_unknown():
    manifest,rows=fixture()
    body=manifest["bodies"]["item"];gate=manifest["gates"][0]
    a=deepcopy(rows[2]["detector"]["poses"]["item"])
    b=deepcopy(rows[6]["detector"]["poses"]["item"])
    a["position"][1]=b["position"][1]=.6
    cert=rows[6]["detector"]["sweeps"]["item"]["blue"]
    assert swept_aperture(body["vertices"],a,b,gate,cert,allow_test_certificates=True)["status"]=="VIOLATION"
    cert["deviation_bound_m"]=.001
    assert swept_aperture(body["vertices"],a,b,gate,cert,allow_test_certificates=True)["status"]=="UNKNOWN"


@pytest.mark.parametrize("hard",[False,True])
def test_no_entry_is_not_occurred_and_valid_zero(hard):
    manifest,rows=fixture(hard)
    for row in rows:
        for pose in row["detector"]["poses"].values():pose["position"][0]=-.1
        row["detector"]["contacts"]=[]
    value=rebuild_events(rows,manifest)
    assert value["evidence_complete"] and value["score"]==0
    assert all(e["status"]=="NOT_OCCURRED" for e in value["events"]["entry"])


def test_hard_only_first_opening_is_zero():
    manifest,rows=fixture(True)
    for row in rows:
        if row["physics_tick"]>=5:
            for pose in row["detector"]["poses"].values():pose["position"][0]=.025
            row["detector"]["contacts"]=deepcopy(rows[5]["detector"]["contacts"])
    value=rebuild_events(rows,manifest)
    assert value["score"]==0 and value["attempts"][0]["passed_gate_count"]==1


def test_hard_closed_second_gate_during_its_pass_is_zero():
    manifest,rows=fixture(True)
    for row in rows:
        if 6<=row["physics_tick"]<8:row["detector"]["gates"]["orange"]=0.
    value=rebuild_events(rows,manifest)
    assert value["score"]==0 and value["violation_latched"]


def test_each_gate_can_be_passed_in_its_own_real_open_window():
    manifest, rows = fixture(True)
    for row in rows:
        tick = row['physics_tick']
        row['detector']['gates']['blue'] = 1. if tick < 6 else 0.
        row['detector']['gates']['orange'] = 1. if 6 <= tick < 60 else 0.
    value = rebuild_events(rows, manifest)
    assert value['evidence_complete'] and value['score'] == 100


def test_forward_crossing_a_closed_gate_is_not_a_pass():
    manifest, rows = fixture()
    for row in rows:
        if 3 <= row['physics_tick'] <= 6:
            row['detector']['gates']['blue'] = 0.
    value = rebuild_events(rows, manifest)
    assert value['evidence_complete'] and value['score'] == 0
    assert value['violation_latched']


@pytest.mark.parametrize('hard', [False, True])
def test_late_robot_withdrawal_does_not_gate_item_delivery(hard):
    manifest, rows = fixture(hard)
    for row in rows:
        if 13 <= row['physics_tick'] <= 70: row['detector']['poses']['grip']['position'][0] = .15
    value = rebuild_events(rows, manifest)
    assert value['score'] == 100 and not value['violation_latched']


def test_frame_contact_is_not_an_independent_failure():
    manifest, rows = fixture(True)
    rows[5]['detector']['contacts'].append(dict(pair=['/World/RobotRight/Asset/link4/Collider','/World/Task/Rail/Collider'],
        active=True, impulse_ns=.001, item_upward_normals=[]))
    value = rebuild_events(rows, manifest)
    assert value['score'] == 100 and not value['violation_latched']


def test_item_pass_does_not_require_a_grip_contact_fraction():
    manifest, rows = fixture()
    rows[5]['detector']['contacts'] = rows[5]['detector']['contacts'][:1]
    assert rebuild_events(rows, manifest)['score'] == 100


def test_production_rejects_test_only_sweep_certificates():
    manifest, rows = fixture()
    value = _rebuild_events(rows, manifest)
    assert value['score'] is None and not value['evidence_complete']
    assert any('unresolved item aperture passage' in reason for reason in value['unknown_reasons'])


def test_contact_normals_preserve_collider_ownership_and_lost_contacts():
    from types import SimpleNamespace
    from mind2act_eval.tasks.DR3.capture import contact_measurement
    item="/World/Task/Item/Collider";tray="/World/Task/Tray/Collider"
    point=SimpleNamespace(impulse=[0,0,.01],normal=[0,0,1])
    assert contact_measurement((item,tray),[point])["item_upward_normals"]==[1]
    assert contact_measurement((tray,item),[point])["item_upward_normals"]==[-1]
    assert not contact_measurement((item,tray),[])["active"]
    point.normal=[0,0,float("nan")]
    with pytest.raises(ValueError):contact_measurement((item,tray),[point])


@pytest.mark.parametrize("hard,expected", [(False, 100), (False, 5),
                                            (False, 0), (True, 0)])
def test_public_scorer_uses_official_terminal_bands(hard, expected, tmp_path):
    manifest, rows = fixture(hard=hard, end=240 if expected == 100 else 30)
    for row in rows:
        for certificates in row["detector"]["sweeps"].values():
            for certificate in certificates.values():
                certificate["source"] = "native_pose_full_rotation"
    if expected == 5:
        for row in rows:
            if row["physics_tick"] >= 10:
                row["detector"]["contacts"] = deepcopy(rows[5]["detector"]["contacts"])
    if expected == 0:
        for row in rows:
            for pose in row["detector"]["poses"].values():
                pose["position"][0] = -.1
            row["detector"]["contacts"] = []
    evidence = audit_evidence(tmp_path, manifest, rows)
    result = score(evidence)
    assert result.value == (7.5 if expected == 5 else expected)
    assert result.components["task_success"] is (expected == 100)


def test_public_scorer_rejects_incomplete_physical_evidence_as_invalid(tmp_path):
    manifest, rows = fixture()
    rows[4]["detector"]["sweeps"] = {}
    evidence = audit_evidence(tmp_path, manifest, rows)
    with pytest.raises(ValueError, match="physical evidence is incomplete"):
        score(evidence)


def test_append_only_trace_receipt_rejects_changed_terminal_bytes(tmp_path):
    manifest, rows = fixture(end=3)
    evidence = audit_evidence(tmp_path, manifest, rows)
    receipt = evidence.values['raw_trace']
    with open(receipt['path'], 'ab') as stream:
        stream.write(b'{"physics_tick":4}\n')
    with pytest.raises(ValueError, match='file length mismatch'):
        read_trace(receipt, evidence.tick)


def test_terminal_candidate_evidence_replays_complete_append_audit_at_eof(tmp_path):
    manifest, rows = fixture()
    for row in rows:
        for certificates in row['detector']['sweeps'].values():
            for certificate in certificates.values():
                certificate['source'] = 'native_pose_full_rotation'
    class Capture:
        def __init__(self):
            self.manifest = manifest
        def snapshot(self, contacts, tick, time):
            return rows[tick]['detector']
    world = SimpleNamespace(current_time=rows[0]['simulation_time_s'])
    ctx = SimpleNamespace(world=world, output_dir=tmp_path)
    spec = SimpleNamespace(instance_id='fr3-test', difficulty='easy', protocol_version='v1',
                           clock=SimpleNamespace(physics_hz=120),
                           config={'source_id': SOURCE_ID, 'condition': {'instance_id': 'fr3-test'}})
    scene = GateScene(spec, ctx, None, {}, {}, SimpleNamespace(rows={}),
                      SimpleNamespace(unsubscribe=lambda: None), Capture())
    for tick in range(1, len(rows)):
        world.current_time = rows[tick]['simulation_time_s']
        scene._sample(tick)
    evidence = scene.evidence(len(rows)-1)
    assert evidence.values['raw_trace']['rows'] == len(rows)
    assert not scene.is_complete()
    assert score(evidence).value == 100.
    scene.close()


def measured_fixture(**kwargs):
    """Synthetic actual-step inputs, without any interval trajectory certificate."""
    from mind2act_eval.tasks.DR3.geometry import MEASUREMENT_CONTRACT
    manifest,rows=fixture(**kwargs)
    manifest.update(schema='dr3-collision-pose-evidence-v2',measurement_contract=MEASUREMENT_CONTRACT)
    for body in manifest['bodies'].values():
        body['vertices_are_actual_collision_points']=True
        body['collision_parts']=[dict(vertices=body['vertices'],approximation='convexHull')]
    for i,row in enumerate(rows):
        row['detector']['sweeps']={}
        row['detector']['sampling']=dict(tick=i,before_step_epoch=i-1,subscription_active=True,errors=[])
    return manifest,rows


def measured_scene(tmp_path, manifest, rows):
    class Capture:
        def __init__(self): self.manifest = manifest
        def snapshot(self, contacts, tick, time): return deepcopy(rows[tick]['detector'])
    world = SimpleNamespace(current_time=rows[0]['simulation_time_s'])
    ctx = SimpleNamespace(world=world, output_dir=tmp_path)
    spec = SimpleNamespace(instance_id='fr3-test', difficulty='easy', protocol_version='v1',
                           clock=SimpleNamespace(physics_hz=120),
                           config={'source_id': SOURCE_ID, 'condition': {'instance_id': 'fr3-test'}})
    return GateScene(spec, ctx, None, {}, {}, SimpleNamespace(rows={}),
                     SimpleNamespace(unsubscribe=lambda: None), Capture()), world


def sample_scene(scene, world, rows, end):
    for tick in range(len(scene.rows), end+1):
        world.current_time = rows[tick]['simulation_time_s']
        scene._sample(tick)


def test_action_complete_success_uses_actual_delivery_without_extra_wait(tmp_path):
    manifest, rows = measured_fixture(end=30)
    scene, world = measured_scene(tmp_path, manifest, rows)
    sample_scene(scene, world, rows, 9)
    assert not scene.is_success()  # No actual tray support yet.
    sample_scene(scene, world, rows, 10)
    assert scene.is_success()  # Still held, gate still open; no settling/close wait.
    assert score(scene.evidence(10)).value == 100
    scene.close()


def test_non_delivery_success_probe_does_not_replay_history(tmp_path, monkeypatch):
    manifest, rows = measured_fixture(end=9)
    scene, world = measured_scene(tmp_path, manifest, rows)
    sample_scene(scene, world, rows, 9)
    monkeypatch.setattr('mind2act_eval.tasks.DR3.events.rebuild_events',
                        lambda *args: pytest.fail('no delivery: do not replay'))
    assert not scene.is_success()
    scene.close()


def test_wrong_order_only_latches_score_and_keeps_recording(tmp_path, monkeypatch):
    manifest, rows = measured_fixture(hard=True, end=60)
    manifest['gates'][0]['origin'][0], manifest['gates'][1]['origin'][0] = .06, 0.
    scene, world = measured_scene(tmp_path, manifest, rows)
    sample_scene(scene, world, rows, 30)
    assert not scene.is_success()
    assert score(scene.evidence(30)).value == 0
    sample_scene(scene, world, rows, 60)
    monkeypatch.setattr('mind2act_eval.tasks.DR3.events.rebuild_events',
                        lambda *args: pytest.fail('proven violation already latched'))
    assert not scene.is_success()
    assert not scene.is_complete()
    assert scene.evidence(60).values['raw_trace']['rows'] == 61
    scene.close()


@pytest.mark.parametrize('hard',[False,True])
@pytest.mark.parametrize('end,expected',[(240,100),(30,100),(12,5)])
def test_actual_physics_substeps_preserve_terminal_bands_without_sweep_cache(hard,end,expected):
    manifest,rows=measured_fixture(hard=hard,end=end)
    if end==12:
        for row in rows[10:]:row['detector']['contacts']=deepcopy(rows[5]['detector']['contacts'])
    value=_rebuild_events(rows,manifest)
    assert value['evidence_complete'],value['unknown_reasons']
    assert value['score']==expected
    assert value['intra_step_continuous_trajectory_claimed'] is False


@pytest.mark.parametrize('defect',['clock','pose','contacts','sampling','source','contract','boolean_tick'])
def test_actual_substep_contract_rejects_missing_real_evidence(defect,tmp_path):
    manifest,rows=measured_fixture()
    if defect=='clock':rows[8]['simulation_time_s']+=.001
    elif defect=='pose':del rows[8]['detector']['poses']['grip']
    elif defect=='contacts':rows[8]['detector']['contacts_complete']=False
    elif defect=='sampling':del rows[8]['detector']['sampling']
    elif defect=='source':manifest['bodies']['item']['vertices_are_actual_collision_points']=False
    elif defect=='contract':manifest['measurement_contract']='unknown-contract'
    else:rows[1]['detector']['sampling']['tick']=True
    with pytest.raises(ValueError):score(audit_evidence(tmp_path,manifest,rows))


@pytest.mark.parametrize('violation',['aperture','contact'])
def test_actual_substeps_preserve_measured_bypass_and_forbidden_contacts(violation):
    manifest,rows=measured_fixture()
    if violation=='aperture':
        for name in ('item','grip'):rows[4]['detector']['poses'][name]['position'][1]=.7
    else:
        rows[4]['detector']['contacts'].append(dict(pair=['/World/Task/Item/Collider','/World/Task/Gate/Collider'],
            active=True,impulse_ns=.001,item_upward_normals=[]))
    value=_rebuild_events(rows,manifest)
    assert value['evidence_complete']
    assert value['score'] == (0 if violation == 'aperture' else 100)


@pytest.mark.parametrize('clearance', [.024, .025])
def test_measured_gripper_clearance_does_not_gate_t0_delivery(clearance):
    manifest, rows = measured_fixture()
    for row in rows[20:]: row['detector']['poses']['grip']['position'][0] = -clearance-.005
    value = _rebuild_events(rows, manifest)
    assert value['evidence_complete'] and value['score'] == 100


def test_parked_far_link_sphere_overlap_is_not_closed_window_occupancy():
    manifest,rows=measured_fixture(hard=True)
    vertices=[[x,y,z] for x in (-.01,.01) for y in (-.8,.8) for z in (-.01,.01)]
    manifest['bodies']['parked']=dict(vertices=vertices,role='robot',arm='Right',gripper=False,
        vertices_are_actual_collision_points=True,collision_parts=[dict(vertices=vertices,approximation='convexHull')])
    manifest['coverage']['expected_colliders'].append('parked');manifest['coverage']['represented_colliders'].append('parked')
    for row in rows:
        row['detector']['poses']['parked']=dict(position=[-.3,0,0],quaternion_wxyz=[1,0,0,0],linear_speed_m_s=0.,angular_speed_rad_s=0.)
    value=_rebuild_events(rows,manifest)
    assert value['evidence_complete'] and value['score']==100
    assert not value['violation_latched']


@pytest.mark.parametrize('hard,initial_time_s', [(False, 1.0666667222976685), (True, 1.1083333911374211)])
def test_recorded_native_clock_preserves_complete_terminal_second(hard, initial_time_s, tmp_path):
    # Actual A-random DR3 traces use float32(1/120):120 intervals span
    # 1.0000000521540642s, within the existing100ns clock validation tolerance.
    manifest, rows = measured_fixture(hard=hard)
    for row in rows:
        row['simulation_time_s'] = initial_time_s + row['physics_tick'] * 0.008333333767950535
    result = score(audit_evidence(tmp_path, manifest, rows))
    assert result.value == 100
    facts = result.components['evidence']
    assert facts['terminal_tray_support'] and facts['center_inside_tray']


@pytest.mark.parametrize('defect', ['angular_speed', 'displacement', 'support', 'short_window', 'clock_window', 'forbidden_contact', 'aperture'])
def test_recorded_clock_keeps_only_required_physical_and_path_conditions(defect):
    manifest, rows = measured_fixture(hard=True, end=119 if defect == 'short_window' else 240)
    for row in rows:
        row['simulation_time_s'] = 1.1083333911374211 + row['physics_tick'] * 0.008333333767950535
    if defect == 'angular_speed': rows[-1]['detector']['poses']['item']['angular_speed_rad_s'] = .050001
    elif defect == 'displacement':
        rows[130]['detector']['poses']['item']['position'][1] = .002
        rows[140]['detector']['poses']['item']['position'][1] = -.002
    elif defect == 'support':
        for row in rows[150:170]: row['detector']['contacts'] = []
    elif defect == 'clock_window':
        for row in rows: row['simulation_time_s'] = 1.1083333911374211 + row['physics_tick'] * (1/120 + 2e-9)
    elif defect == 'forbidden_contact':
        rows[4]['detector']['contacts'].append(dict(pair=['/World/Task/Item/Collider','/World/Task/Gate/Collider'],
            active=True, impulse_ns=.001, item_upward_normals=[]))
    elif defect == 'aperture': rows[4]['detector']['poses']['item']['position'][1] = .7
    result = _rebuild_events(rows, manifest)
    assert result['evidence_complete']
    assert result['score'] == (0 if defect == 'aperture' else 100)


@pytest.mark.parametrize('hard', [False, True])
@pytest.mark.parametrize('violation', ['reverse_bypass', 'reverse_closed_gate', 'lateral_bypass'])
def test_illegal_return_latches_even_after_a_later_legal_delivery(tmp_path, hard, violation):
    """SPEC-3: gate legality is bidirectional; progress remains forward-only."""
    manifest, rows = measured_fixture(hard=hard, end=30)
    if violation == 'reverse_bypass':
        poses = [(-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0),
                 (.15, .8), (-.1, .8), (-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0)]
    elif violation == 'reverse_closed_gate':
        poses = [(-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0),
                 (.05, 0), (0, 0), (-.05, 0), (-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0)]
        for row in rows[5:9]:
            row['detector']['gates'] = {gate['id']: 0. for gate in manifest['gates']}
    else:
        poses = [(-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0),
                 (0, 0), (0, .8), (0, 0), (.05, 0), (.15, 0)]
    for row, xy in zip(rows, poses):
        row['detector']['poses']['item']['position'][:2] = xy
    rebuilt = _rebuild_events(rows, manifest)
    assert rebuilt['evidence_complete'], rebuilt['unknown_reasons']
    assert rebuilt['violation_latched'] and rebuilt['score'] == 0.
    assert rebuilt['delivered']  # Later delivery cannot clear the earlier violation.
    assert score(audit_evidence(tmp_path, manifest, rows)).value == 0.


@pytest.mark.parametrize('hard', [False, True])
def test_legal_open_gate_return_does_not_reverse_progress_order(tmp_path, hard):
    manifest, rows = measured_fixture(hard=hard, end=30)
    poses = [(-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0),
             (.05, 0), (0, 0), (-.05, 0), (-.1, 0), (-.05, 0), (0, 0), (.05, 0), (.15, 0)]
    for row, xy in zip(rows, poses):
        row['detector']['poses']['item']['position'][:2] = xy
    rebuilt = _rebuild_events(rows, manifest)
    assert rebuilt['evidence_complete'] and not rebuilt['violation_latched']
    assert rebuilt['score'] == 100.
    assert len(rebuilt['attempts']) == 2  # Reverse travel is not another forward attempt.
    assert all(attempt['passed_gate_count'] == len(manifest['gates']) for attempt in rebuilt['attempts'])
    assert score(audit_evidence(tmp_path, manifest, rows)).value == 100.
