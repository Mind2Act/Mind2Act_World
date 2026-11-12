import pytest
from mind2act_eval.tasks.CP08.projection import validate, cue_visible, H5_SHA256
from mind2act_eval.tasks.CP08.audit import Audit, read


def projection(level='easy'):
    return dict(source_physics_hz=120,source_state_tick=72,source_action_tick=76,
        source_h5_row=18,source_cue_hide_tick=360,h5_sha256=H5_SHA256[level])


@pytest.mark.parametrize('level', ['easy','medium','hard'])
def test_source_tick_offset_and_remaining_exposure(level):
    p = validate(projection(level), level)
    assert cue_visible(0,p)
    assert cue_visible(288,p)
    assert not cue_visible(289,p)
    assert 288/120 == 2.4


@pytest.mark.parametrize('key,value', [('source_physics_hz',30),('source_state_tick',18),
    ('source_action_tick',19),('h5_sha256','0'*64)])
def test_reject_row_as_tick_or_unmatched_h5(key,value):
    p = projection();p[key]=value
    with pytest.raises(ValueError):validate(p,'easy')


def test_early_eof_and_tampering(tmp_path):
    a=Audit(tmp_path/'raw.jsonl')
    for tick in range(31):a.append(dict(tick=tick))
    receipt=a.receipt();a.close()
    assert len(read(receipt,30))==31
    with pytest.raises(ValueError):read(receipt,29)
    path=tmp_path/'raw.jsonl';path.write_bytes(path.read_bytes().replace(b'30',b'31'))
    with pytest.raises(ValueError):read(receipt,30)


def test_dropped_tick_rejected(tmp_path):
    a=Audit(tmp_path/'raw.jsonl')
    with pytest.raises(ValueError):a.append(dict(tick=1))
    a.close()
