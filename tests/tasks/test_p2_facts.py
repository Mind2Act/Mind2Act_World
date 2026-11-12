"""Independent raw transfer/stability facts; no mat policy or score invented."""
from copy import deepcopy
import pytest
from .test_p2_measurements import source,frame,contact
from mind2act_eval.tasks.P2.facts import PhysicalFacts,landed


class FactsFixture:
    def __init__(self,level='easy'):
        self.source=source(level);self.initial=frame(self.source)
        self.poses=deepcopy(self.initial['poses']);self.facts=PhysicalFacts(self.source,self.initial)
        self.tick=0

    def step(self,count=1,*,held=(),finger_count=2,answer_support=None):
        for _ in range(count):
            self.tick+=1;row=frame(self.source,self.tick);row['poses']=deepcopy(self.poses)
            for name in self.poses:
                if name not in held:
                    other='/World/Task/'+answer_support+'/Collision' if answer_support and name=='Tier1' else '/World/Task/Board/Collision'
                    row['contacts'].append(contact('/World/Task/'+name+'/Collision',other))
                else:
                    for link in ('link7','link8')[:finger_count]:
                        row['contacts'].append(contact('/World/Task/'+name+'/Collision','/World/RobotLeft/Asset/'+link+'/Collision'))
            event=self.facts.sample(row)
        return event


def test_bilateral_measured_lift_and_supported_release_emit_one_transfer():
    p=FactsFixture();p.step(held=('Tier3',))
    p.poses['Tier3']['position'][2]+=.03
    event=p.step(held=('Tier3',))
    assert [e['item'] for e in event['lifts']]==['Tier3'] and not event['releases']
    event=p.step()
    assert [e['item'] for e in event['releases']]==['Tier3']
    assert p.step()['releases']==[]


def test_single_finger_and_two_subthreshold_hops_cannot_qualify_a_transport():
    p=FactsFixture();p.step(held=('Tier3',),finger_count=1)
    p.poses['Tier3']['position'][2]+=.04
    assert not p.step(held=('Tier3',),finger_count=1)['lifts']
    assert not p.step()['releases']
    for _ in range(2):
        p.step(held=('Tier3',));p.poses['Tier3']['position'][2]+=.02
        assert not p.step(held=('Tier3',))['lifts']
        assert not p.step()['releases']


def test_three_seconds_requires_360_actual_samples_not_the_reset():
    p=FactsFixture();p.step(359)
    assert not p.facts.stable
    p.step();assert p.facts.stable
    assert len(p.facts.tail)==360


def test_stability_checks_pairwise_displacement_and_expires_the_oldest_bad_pair():
    p=FactsFixture();p.poses['Tier1']['position'][0]+=.0004;p.step()
    p.poses['Tier1']['position'][0]-=.0008;p.step(359)
    assert not p.facts.stable  # Pair gap .8mm, even though each is within .4mm of reset.
    p.step();assert p.facts.stable


def test_held_or_unsupported_sample_restarts_the_release_support_window():
    p=FactsFixture();p.step(360);assert p.facts.stable
    p.step(held=('Lid',));assert not p.facts.stable
    p.step(359);assert not p.facts.stable
    p.step();assert p.facts.stable


def test_hard_decoy_movement_and_real_grasp_are_only_observed():
    p=FactsFixture('hard');p.step(held=('Decoy1',))
    p.poses['Decoy1']['position'][2]+=.04
    p.step(held=('Decoy1',));p.step()
    p.poses['Decoy1']['position'][0]+=.15;p.step()
    assert p.facts.decoy_motion['Decoy1']>.001


def test_past_decoy_support_does_not_create_a_permanent_violation():
    p=FactsFixture('hard');p.step(answer_support='Decoy1')
    p.step()
    assert not hasattr(p.facts, 'decoy_used')


def test_source_landing_boundaries_use_actual_pose_support_and_release():
    p=FactsFixture();p.step();row=p.facts.last;expected=list(row['poses']['Tier1']['position'])
    row['poses']['Tier1']['position'][0]+=.01
    row['poses']['Tier1']['position'][2]+=.003
    assert landed(row,'Tier1',expected)
    row['poses']['Tier1']['position'][0]+=.000001
    assert not landed(row,'Tier1',expected)


def test_physical_tick_gap_is_rejected_before_progress():
    p=FactsFixture()
    with pytest.raises(ValueError,match='consecutive'):
        p.facts.sample(frame(p.source,2))


def test_physical_stability_cannot_be_claimed_by_zero_duration_tick_labels():
    p=FactsFixture();row=frame(p.source,1)
    row['simulation_time_s']=p.initial['simulation_time_s']
    with pytest.raises(ValueError,match='120Hz'):
        p.facts.sample(row)
