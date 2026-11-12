"""CP09 section-6 examples, prefix failures, and physical-raster boundaries."""
from copy import deepcopy
import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP09.evidence import CONTRACT, Progress, RasterMetrics
from mind2act_eval.tasks.CP09.scorer import score_progress as score, SCORING_VERSION


def pickup(tick, tool, color=0, lift=.041):
    return dict(tick=tick, tool=tool, color_id=color, two_finger_contact=True,
                lift_m=lift, consecutive_lift_ticks=24,
                attachment_joint_created_before_lift=False)


def state(*, parked=False, board=True, pickups=(), caps=()):
    return dict(contract=CONTRACT, board_retrieved=board, eraser_parked=parked,
                pickup_events=list(pickups), cap_events=list(caps))


def fixture(level, *, missing_colors=None):
    count = {"easy": 1, "medium": 2, "hard": 3}[level]
    target = np.zeros((12, 40), np.uint8)
    for c in range(1, count + 1):
        target[c * 3 - 2, 2:10] = c
    actual = target.copy()
    for c in range(1, count + 1) if missing_colors is None else missing_colors:
        actual[c * 3 - 2, 8:10] = 0
    actual[-1, 20:24] = 1
    p = Progress(target, actual, tolerance_pixels=0, difficulty=level)
    p.annotate(actual)
    return p


def mutate(p, tick, tool, color, operation, *, qualified=True, cap=True):
    before = p.actual.copy()
    operation(p.actual)
    changed = int(np.count_nonzero(before != p.actual))
    p.mutation(tick, dict(tool=tool, color=color, changed_pixels=changed,
                         pickup_verified=qualified, cap_open=cap),
               (0, 0, p.actual.shape[1], p.actual.shape[0]), before)


def erase_and_return(p, *, keep=None):
    s = state(pickups=[pickup(1, "eraser")])
    p.before_contacts(1, s)
    p.after_contacts(1, s)
    s = state()
    p.before_contacts(2, s)

    def erase(a):
        a[-1, 20:24] = 0
        if keep is not None:
            for c in p.raster.colors:
                a[c * 3 - 2, 2:10] = 0
                a[c * 3 - 2, 2:2 + keep] = c

    mutate(p, 2, "eraser", 1, erase)
    p.after_contacts(2, s)
    s = state(parked=True)
    p.before_contacts(3, s)
    p.after_contacts(3, s)


def repair(p, color, tick, *, cap=True, qualified=True):
    caps = [dict(tick=tick, event="cap_pulled_open", physical_grasps=True,
                 axial_travel_m=.004, mechanism="axial_pull_v2", color=color)] if cap else []
    s = state(parked=True, pickups=[pickup(tick, "pen", color)] if qualified else [], caps=caps)
    p.before_contacts(tick, s)
    mutate(p, tick, "pen", color,
           lambda a: a.__setitem__((color * 3 - 2, slice(2, 10)), color),
           qualified=qualified, cap=cap)
    p.after_contacts(tick, s)


def result(p, level, *, success=False, normal=True):
    return score(NativeEvidence("CP09", "synthetic-cpu", 100, dict(
        difficulty=level, normal_termination=normal,
        cp09_progress=p.result(success))))


@pytest.mark.parametrize('level,n',[('easy',1),('medium',2),('hard',3)])
def test_color_units_and_intact_colors(level,n):
    p=fixture(level,missing_colors=())
    mutate(p,1,'eraser',1,lambda a:a.__setitem__((-1,slice(20,24)),0),qualified=False)
    value=result(p,level,success=True,normal=False)
    assert value.value==100 and value.components['completion']==dict(k=n,n=n,ratio=1.)
    assert value.components['scoring_version']=='mind2act-final-scoring-20261105'


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_full_erase_and_redraw_recovers_all_colors(level):
    p=fixture(level)
    mutate(p,1,'eraser',1,lambda a:a.fill(0),qualified=False,cap=False)
    assert result(p,level,success=True).value==0
    mutate(p,2,'pen',1,lambda a:a.__setitem__(slice(None),p.target),qualified=False,cap=False)
    assert result(p,level,success=True,normal=False).value==100


def test_colors_are_independent_not_prefix():
    p=fixture('hard')
    mutate(p,1,'eraser',1,lambda a:a.fill(0))
    mutate(p,2,'pen',3,lambda a:a.__setitem__((7,slice(2,10)),3))
    value=result(p,'hard',success=True)
    assert value.value==pytest.approx(7/3)
    assert value.components['completion']['k']==1


def test_wrong_color_can_be_erased_and_fixed():
    p=fixture('medium')
    mutate(p,1,'pen',1,lambda a:a.fill(1))
    assert result(p,'medium',success=True).value==0
    mutate(p,2,'eraser',1,lambda a:a.fill(0))
    mutate(p,3,'pen',1,lambda a:a.__setitem__(slice(None),p.target))
    assert result(p,'medium',success=True).value==100
    assert p.first_error is None


@pytest.mark.parametrize('missing,accepted',[(5,True),(6,False)])
def test_coverage_boundary(missing,accepted):
    target=np.ones((10,10),dtype=np.uint8);actual=target.copy();actual.flat[:missing]=0
    p=Progress(target,actual,tolerance_pixels=0,difficulty='easy');p.annotate(actual)
    assert (result(p,'easy',success=True).value==100) is accepted


@pytest.mark.parametrize('extra,accepted',[(3,True),(4,False)])
def test_per_color_extra_ink_boundary(extra,accepted):
    target=np.zeros((20,20),np.uint8);target[:10,:10]=1;actual=target.copy();actual[15,:extra]=1
    p=Progress(target,actual,tolerance_pixels=0,difficulty='easy');p.annotate(actual)
    assert (result(p,'easy',success=True).value==100) is accepted


def test_large_good_color_cannot_hide_extras_on_small_color():
    target=np.zeros((20,20),np.uint8);target[:10,:10]=1;target[12,:10]=2
    actual=target.copy();actual[15,0]=2
    p=Progress(target,actual,tolerance_pixels=0,difficulty='medium');p.annotate(actual)
    assert result(p,'medium',success=True).value==5


def test_before_annotation_is_not_a_restored_board():
    p=fixture('easy',missing_colors=())
    mutate(p,1,'eraser',1,lambda a:a.__setitem__((-1,slice(20,24)),0))
    assert result(p,'easy',success=False).value==0


def test_later_erasing_removes_only_affected_current_color():
    p=fixture('hard',missing_colors=())
    mutate(p,1,'eraser',1,lambda a:a.__setitem__((-1,slice(20,24)),0))
    assert result(p,'hard',success=True).value==100
    mutate(p,2,'eraser',1,lambda a:a.__setitem__((1,2),0))
    assert result(p,'hard',success=True).value==pytest.approx(14/3)


def test_changed_pixel_proof_is_still_required():
    p=fixture('easy');before=p.actual.copy()
    with pytest.raises(ValueError,match='changed-pixel'):
        p.mutation(1,dict(tool='pen',color=1,changed_pixels=1),(0,0,40,12),before)


def test_ink_tolerance_is_three_pixels():
    target=np.zeros((15,15),np.uint8);target[7,7]=1
    for distance,expected in [(3,1),(4,0)]:
        actual=np.zeros_like(target);actual[7,7+distance]=1
        assert RasterMetrics(target,actual,3).metrics()['recall']==expected


def test_missing_or_nonfinite_surface_is_invalid():
    with pytest.raises(ValueError,match='incomplete'):
        score(NativeEvidence('CP09','cpu',0,dict(difficulty='easy')))
    p=fixture('easy');progress=p.result(True)
    progress['final_surface']['per_color'][0]['recall']=float('nan')
    with pytest.raises(ValueError,match='finite'):
        score(NativeEvidence('CP09','cpu',100,dict(difficulty='easy',cp09_progress=progress)))


def test_current_final_version_also_requires_color_precision():
    target=np.zeros((20,20),np.uint8);target[:10,:10]=1
    actual=np.zeros_like(target)
    for y in (1,4,7):
        for x in (1,4,7):actual[y,x]=1
    actual[18,:3]=1
    p=Progress(target,actual,tolerance_pixels=3,difficulty='easy');p.annotate(actual)
    metric=p.raster.metrics()['per_color'][0]
    assert metric['recall']==1 and metric['extra_fraction']==.03 and metric['precision']==.75
    assert result(p,'easy',success=True).value==0
