"""Real audit-schema replay with synthetic meshes; no native-run claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from mind2act_eval.contracts import NativeEvidence, to_json
from mind2act_eval.tasks.PR4.evidence import Recorder
from mind2act_eval.tasks.PR4.instances import RULE_VERSION, portable_case
from mind2act_eval.tasks.PR4.scorer import score
from mind2act_eval.tasks.catalog import load_task
from .test_pr4_binding_audit import case, geometry, row
from .support_fixture import support_geometry


def mesh(tier):
    y, x = np.mgrid[:81, :81]
    width, top = .4, .8
    if tier == 'medium':
        p = np.stack([(x-y)*width/(80*np.sqrt(2)), (x+y-80)*width/(80*np.sqrt(2)),
                      np.full_like(x, top, dtype=float)], axis=-1)
        moving = y+x < 80
    elif tier == 'hard':
        yy = np.linspace(-width*np.sqrt(3)/4, width*np.sqrt(3)/4, 81)
        xx = (width/2-np.abs(yy)/np.sqrt(3))[:, None]*np.linspace(-1, 1, 81)
        p = np.stack([xx, np.broadcast_to(yy[:, None], xx.shape), np.full_like(xx, top)], axis=-1)
        moving = y < 40
    else:
        p = np.stack([(x-40)*width/80, (y-40)*width/80, np.full_like(x, top, dtype=float)], axis=-1)
        moving = y < 40
    initial = p.reshape(-1, 3).astype('<f4')
    final = initial.copy().reshape(81, 81, 3)
    final[moving, 1] *= -1
    final[moving, 2] += .003
    return initial, final.reshape(-1, 3)


@pytest.fixture(scope='module')
def traces(tmp_path_factory):
    root = tmp_path_factory.mktemp('pr4-native-schema')
    _, triangles, hulls = geometry()
    result = {}
    for tier in ('easy', 'medium', 'hard'):
        initial, final = mesh(tier)
        c = portable_case(case());c['level'] = tier
        c['initial_points_sha256'] = hashlib.sha256(initial.tobytes()).hexdigest()
        c['origin'].update(initial_points_sha256=c['initial_points_sha256'],
                           fold_target_center=[0., 0.], width=.4, support_top=.8)
        r = Recorder('fixture-'+tier, c, initial, initial, triangles, hulls, root/tier, support_geometry=support_geometry())
        for tick in range(122):
            f = row(tick)
            for body in f['finger_poses'].values():body['position'] = [2., 2., 2.]
            r.record(f, initial if tick == 0 else final)
        result[tier] = r.evidence(121);r.close()
    return result


def read_points(e):
    return np.fromfile(e.values['points']['path'], dtype='<f4').reshape(-1, 6561, 3)


def read_rows(e):
    return [json.loads(s) for s in Path(e.values['raw_trace']['path']).read_text().splitlines()]


def variant(e, tmp_path, *, points=None, rows=None, initial=None):
    v = to_json(e.values)
    if rows is not None:
        raw = b''.join((json.dumps(r, sort_keys=True, separators=(',', ':'))+'\n').encode() for r in rows)
        p = tmp_path/'rows.jsonl';p.write_bytes(raw)
        v['raw_trace'].update(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), frames=len(rows))
    if points is not None:
        p = tmp_path/'points.f32';points.astype('<f4').tofile(p)
        v['points'].update(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, frames=len(points))
    if initial is not None:
        with np.load(v['initial']['path']) as z:tri = z['triangles'].copy()
        p = tmp_path/'initial.npz';np.savez_compressed(p, initial=initial, settled=initial, triangles=tri)
        v['initial'].update(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        pin = hashlib.sha256(initial.astype('<f4').tobytes()).hexdigest()
        v['case']['initial_points_sha256'] = v['case']['origin']['initial_points_sha256'] = pin
    return NativeEvidence('PR4', e.instance_id, v['points']['frames']-1, v)


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_actual_shape_release_and_stability_audit_replay(traces, tier):
    result = load_task('PR4').scorer(traces[tier])
    assert result.value == 100
    assert result.components['rule_version'] == RULE_VERSION
    assert result.components['physical_facts']['hands_and_grasp_regions_scored'] is False


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_surface_usd_readback_rejudges_same_points_without_claiming_tensor_equality(traces, tmp_path, tier):
    evidence=traces[tier];rows=read_rows(evidence)
    for sample in rows:
        sample.update(point_source='native_surface_simulation_mesh_usd',render_physics_error_m=None)
    measured=score(variant(evidence,tmp_path,rows=rows))
    original=score(evidence)
    assert measured.value==original.value
    assert measured.components==original.components


@pytest.mark.parametrize('hands,vertex', [(['Left'], 0), (['Right'], 3280), (['Left', 'Right'], 6560)])
def test_hand_count_and_arbitrary_material_region_are_irrelevant(traces, tmp_path, hands, vertex):
    e = traces['medium'];rows = read_rows(e);p = read_points(e)
    for f in rows:
        for side in ('Left', 'Right'):f['joints'][side][6:] = [0., 0.]
    for tick in (0, 1):
        for side in hands:
            for finger in (7, 8):
                rows[tick]['finger_poses'][side+'/link'+str(finger)]['position'] = p[tick, vertex].tolist()
    assert score(variant(e, tmp_path, rows=rows)).value == 100


def test_untouched_initial_target_coincidence_cannot_score(traces, tmp_path):
    e = traces['easy'];p = read_points(e);p[:] = p[-1]
    result = score(variant(e, tmp_path, points=p, initial=p[0]))
    assert result.value == 100
    assert result.components['physical_facts']['shape']['shape_success']


def test_transport_without_fold_is_zero_not_partial_credit(traces, tmp_path):
    e = traces['easy'];p = read_points(e);p[1:] = p[0];p[1:, :, 2] += .04
    result = score(variant(e, tmp_path, points=p))
    assert result.value == 0
    assert not result.components['physical_facts']['checks']['target_shape']


@pytest.mark.parametrize('frames,expected', [(119, 100), (120, 100)])
def test_release_observation_119_fails_120_passes(traces, tmp_path, frames, expected):
    e = traces['easy'];p = read_points(e)[:frames];r = read_rows(e)[:frames]
    p[:] = read_points(e)[-1]
    assert score(variant(e, tmp_path, points=p, rows=r)).value == expected


@pytest.mark.parametrize('clearance,expected', [(.0149, 100), (.0151, 100)])
def test_all_actual_fingers_clear_not_commanded_openings(traces, tmp_path, clearance, expected):
    e = traces['easy'];r = read_rows(e);p = read_points(e);point = p[-1, 0].astype(float)
    for f in r[-120:]:f['finger_poses']['Right/link8']['position'] = (point+[0., 0., .01+clearance]).tolist()
    assert score(variant(e, tmp_path, rows=r)).value == expected


@pytest.mark.parametrize('motion,expected', [(.0014, 100), (.0016, 100)])
def test_all_particle_final_stability(traces, tmp_path, motion, expected):
    e = traces['easy'];p = read_points(e);p[-2, 3000, 2] += motion
    assert score(variant(e, tmp_path, points=p)).value == expected


def test_expert_stage_cannot_replace_measured_audit(traces, tmp_path):
    e = traces['easy'];r = read_rows(e);r[-1]['expert_phase'] = 'success'
    with pytest.raises(ValueError, match='without phase labels'):score(variant(e, tmp_path, rows=r))


def test_offline_rejudge_is_versioned_and_preserves_history(traces, tmp_path, monkeypatch):
    import sys
    from mind2act_eval.tasks.PR4.rejudge import main
    source = tmp_path/'source-evidence.json'
    raw = json.dumps(to_json(traces['easy'])).encode();source.write_bytes(raw)
    output = tmp_path/'new-score.json'
    monkeypatch.setattr(sys, 'argv', ['rejudge', '--evidence', str(source), '--output', str(output)])
    main()
    result = json.loads(output.read_text())
    assert result['score']['value'] == 100 and result['rule_version'] == RULE_VERSION
    assert result['simulation_rerun'] is False
    assert result['source_evidence_sha256'] == hashlib.sha256(raw).hexdigest()
    assert set(result['implementation_sha256']) >= {'scorer.py', 'evidence.py'}
    assert source.read_bytes() == raw
    with pytest.raises(FileExistsError):main()


def test_only_final_outline_matters_without_release_window_or_material_order(traces, tmp_path):
    e = traces['easy']
    p = read_points(e)[-1:].copy()
    rows = read_rows(e)[-1:]
    rows[0]['tick'] = 0
    for body in rows[0]['finger_poses'].values():
        body['position'] = p[0, 0].tolist()
    assert score(variant(e, tmp_path, points=p, rows=rows)).value == 100.


def test_correct_fold_cannot_be_freely_translated_into_target_during_scoring(traces, tmp_path):
    e = traces['easy']
    p = read_points(e)
    p[-1, :, 0] += .5
    assert score(variant(e, tmp_path, points=p)).value == 0.


@pytest.mark.parametrize('height', [.02, -.02])
def test_correct_outline_without_real_support_is_zero(traces, tmp_path, height):
    e = traces['easy']; points = read_points(e)
    points[-1, :, 2] += height
    result = score(variant(e, tmp_path, points=points))
    assert result.value == 0
    assert not result.components['physical_facts']['checks']['target_support']


def test_legacy_raw_missing_native_support_is_infra_not_model_zero(traces):
    e = traces['easy']; values = to_json(e.values)
    values.pop('support_geometry', None)
    with pytest.raises(ValueError, match='support geometry'):
        score(NativeEvidence('PR4', e.instance_id, e.tick, values))
