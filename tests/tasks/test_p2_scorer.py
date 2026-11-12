"""Raw native-frame -> sealed evidence -> P2 physical stage integration."""
from copy import deepcopy

import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.P2.binding import DISMANTLING_RULE, VERSION, digest, geometry
from mind2act_eval.tasks.P2.evidence import Rows
from mind2act_eval.tasks.P2.progress import Progress
from mind2act_eval.tasks.P2.scorer import score
from .test_p2_measurements import source, frame, contact


class Episode:
    def __init__(self, tmp_path, tier='easy'):
        self.source = source(tier)
        self.initial = frame(self.source)
        self.poses = deepcopy(self.initial['poses'])
        self.support = {n: 'Board' for n in self.poses}
        self.tick = 0
        self.rows = Rows(tmp_path / 'native.jsonl')
        self.progress = Progress(self.source, self.initial)
        self.current = deepcopy(self.initial)

    def step(self, count=1, *, held=(), fingers=2):
        for _ in range(count):
            self.tick += 1
            row = frame(self.source, self.tick)
            row['poses'] = deepcopy(self.poses)
            for name in self.poses:
                own = '/World/Task/' + name + '/Collision'
                if name in held:
                    for link in ('link7', 'link8')[:fingers]:
                        row['contacts'].append(contact(own, '/World/RobotLeft/Asset/' + link + '/Collision'))
                elif self.support.get(name):
                    value = self.support[name]
                    other = value if value.startswith('/World/') else '/World/Task/' + value + '/Collision'
                    row['contacts'].append(contact(own, other))
            self.rows.append(row)
            self.progress.sample(row)
            self.current = row

    def carry(self, name, position, support='Board', *, fingers=2, lift=.04):
        self.step(held=(name,), fingers=fingers)
        self.poses[name]['position'][2] += lift
        self.step(held=(name,), fingers=fingers)
        self.poses[name]['position'] = list(position)
        self.support[name] = support
        self.step()

    def dismantle(self, *, order=None):
        self.carry('Lid', [.4, .2, .784])
        names = list(reversed(self.progress.targets)) if order is None else order
        for i, name in enumerate(names):
            # Arbitrary tabletop positions, deliberately unrelated to target_drop_xyz.
            self.carry(name, [.52-i*.15, .1, .784])

    def restore(self, count=None):
        names = self.progress.targets[:count]
        for i, name in enumerate(names):
            self.carry(name, geometry(self.source)['final_xyz'][name],
                       'Stove' if i == 0 else self.progress.targets[i-1])

    def finish(self, stable=360):
        self.carry('Lid', geometry(self.source)['final_xyz']['Lid'], self.progress.targets[-1])
        self.step(stable)

    def evidence(self):
        return NativeEvidence('P2', 'test', self.tick, dict(source_instance=self.source,
            source_record_sha256=digest(self.source), initial=self.initial,
            measurements=self.rows.receipt(), input_protocol_version=VERSION,
            dismantling_rule_version=DISMANTLING_RULE))

    def value(self):
        result = score(self.evidence())
        assert result.components['task_success'] == self.progress.result()['task_success']
        return result.value


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_success_all_tiers_uses_physical_order_lid_and_final_stability(tmp_path, tier):
    ep = Episode(tmp_path, tier)
    assert ep.value() == 0
    ep.dismantle()
    assert ep.value() == 0
    ep.restore(1)
    assert ep.value() == {'easy': 10, 'medium': 7.5, 'hard': 5}[ep.source['difficulty']]
    ep.restore()  # includes real re-transports, no synthetic stage flags
    assert ep.value() == {'easy': 30, 'medium': 22, 'hard': 15}[ep.source['difficulty']]
    ep.finish()
    result = score(ep.evidence())
    assert result.value == 100
    assert result.components['dismantling_rule_version'] == DISMANTLING_RULE
    assert result.components['final_stable'] is None
    assert result.components['evaluation_mode'] == 'formal'
    assert result.components['formal_eligible'] is True
    assert result.components['accepted_expert_episode'] is False


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_half_bottom_prefix_released_supported_at_actual_pot_is20(tmp_path, tier):
    ep = Episode(tmp_path, tier)
    ep.dismantle()
    ep.restore(2)
    assert ep.value() == {'easy': 20, 'medium': 15, 'hard': 10}[ep.source['difficulty']]


def test_no_mat_or_designated_drop_coordinates_can_affect_disassembly(tmp_path):
    ep = Episode(tmp_path)
    ep.source.pop('target_drop_xyz')
    ep.dismantle()
    assert ep.value() == 0
    ep.restore()
    ep.finish()
    assert ep.value() == 100


@pytest.mark.parametrize('bad', ['no_lid', 'wrong_order', 'single_finger', 'short_lift', 'floor'])
def test_invalid_dismantling_stays_numeric_zero(tmp_path, bad):
    ep = Episode(tmp_path)
    if bad == 'no_lid':
        for name in reversed(ep.progress.targets):
            ep.carry(name, [.4, .2, .784])
    elif bad == 'wrong_order':
        ep.dismantle(order=ep.progress.targets)
    else:
        ep.carry('Lid', [.4, .2, .784])
        for name in reversed(ep.progress.targets):
            ep.carry(name, [.4, .2, .784], '/World/Floor/Collision' if bad == 'floor' else 'Board',
                     fingers=1 if bad == 'single_finger' else 2, lift=.029 if bad == 'short_lift' else .04)
    assert ep.value() == 0


def test_table_contact_is_enough_without_a_region_or_drop_map(tmp_path):
    ep = Episode(tmp_path)
    ep.carry('Lid', [.4, .2, .784], '/World/CollisionTable')
    for i, name in enumerate(reversed(ep.progress.targets)):
        ep.carry(name, [-.2+i*.15, .4, .784], '/World/CollisionTable')
    assert ep.value() == 0


def test_correct_looking_stack_without_real_rebuild_transport_gets_only_dismantle(tmp_path):
    ep = Episode(tmp_path)
    ep.dismantle()
    for i, name in enumerate(ep.progress.targets):
        ep.poses[name]['position'] = geometry(ep.source)['final_xyz'][name]
        ep.support[name] = 'Stove' if i == 0 else ep.progress.targets[i-1]
    ep.step(360)
    assert ep.value() == {'easy': 30, 'medium': 22, 'hard': 15}[ep.source['difficulty']]


def test_hard_movement_and_misplacement_use_terminal_stage(tmp_path):
    ep = Episode(tmp_path, 'hard')
    ep.dismantle()
    ep.restore()
    ep.finish()
    ep.poses['Decoy1']['position'][0] += .001
    ep.step()
    assert ep.value() == 100
    ep.poses['Decoy1']['position'][0] += .1
    ep.step()
    assert ep.value() == 100
    ep.support['Tier1'] = 'Decoy1'
    ep.step()
    assert ep.value() == 0
    ep.support['Tier1'] = 'Stove'
    ep.step(360)
    assert ep.value() == 100


def test_final_placement_error_and_stability_boundaries_are_measured(tmp_path):
    ep = Episode(tmp_path)
    ep.dismantle()
    ep.restore()
    ep.carry('Lid', geometry(ep.source)['final_xyz']['Lid'], ep.progress.targets[-1])
    ep.step(358)
    assert ep.value() == 100
    ep.step(2)
    assert ep.value() == 100
    ep.poses['Tier1']['position'][0] += .010001
    ep.step()
    assert ep.value() == 0


def test_missing_or_tampered_receipts_are_invalid_not_default_zero(tmp_path):
    ep = Episode(tmp_path)
    ep.step()
    evidence = ep.evidence()
    ep.rows.stream.write(b'{}\n')
    ep.rows.stream.flush()
    with pytest.raises((KeyError, ValueError)):
        score(evidence)


def test_hard_distractor_can_be_carried_aside_without_losing_success(tmp_path):
    ep=Episode(tmp_path,'hard')
    ep.carry('Decoy1',[.6,.3,.784])
    ep.dismantle();ep.restore();ep.finish()
    result=score(ep.evidence())
    assert result.value==100
    assert 'distractor_used' not in result.components
    assert result.components['task_success'] is True
    assert result.components['scoring_version']=='mind2act-final-scoring-20261105'


def test_hard_decoy_in_middle_keeps_only_correct_bottom_prefix(tmp_path):
    ep=Episode(tmp_path,'hard');ep.dismantle();ep.restore();ep.finish()
    ep.support['Tier3']='Decoy1';ep.support['Decoy1']='Tier2';ep.step(360)
    assert ep.value()=={'easy': 20, 'medium': 15, 'hard': 10}[ep.source['difficulty']]
    ep.support['Tier3']='Tier2';ep.support['Decoy1']='Board';ep.step(360)
    assert ep.value()==100


def test_hard_extra_decoy_on_top_prevents_exact_stack_success(tmp_path):
    ep=Episode(tmp_path,'hard');ep.dismantle();ep.restore();ep.finish()
    ep.support['Decoy1']='Lid';ep.step(360)
    assert ep.value()=={'easy': 30, 'medium': 22, 'hard': 15}[ep.source['difficulty']]
    ep.support['Decoy1']='Board';ep.step()
    assert ep.value()==100


def test_terminal_supported_stack_needs_no_dismantling_or_release_history(tmp_path):
    ep = Episode(tmp_path)
    for i, name in enumerate(ep.progress.targets):
        ep.poses[name]['position'] = geometry(ep.source)['final_xyz'][name]
        ep.support[name] = 'Stove' if i == 0 else ep.progress.targets[i-1]
    ep.poses['Lid']['position'] = geometry(ep.source)['final_xyz']['Lid']
    ep.support['Lid'] = ep.progress.targets[-1]
    ep.step()
    assert ep.value() == 100
    ep.support['Tier1'] = 'Board'
    ep.step()
    assert ep.value() == 0


def test_dismantling_itself_has_no_points(tmp_path):
    ep = Episode(tmp_path)
    ep.dismantle()
    assert ep.value() == 0


def test_explicit_final_goal_pose_precision_is_not_removed_with_process_gates(tmp_path):
    ep=Episode(tmp_path);ep.restore();ep.finish(stable=0)
    expected=list(ep.poses['Tier1']['position'])
    ep.poses['Tier1']['position'][0]=expected[0]+.01
    ep.step()
    assert ep.value()==100
    ep.poses['Tier1']['position'][0]=expected[0]+.010001
    ep.step()
    assert ep.value()==0


@pytest.mark.parametrize('axis,boundary,outside', [('z',.003,.003001),('tilt',5.,5.001)])
def test_final_goal_vertical_and_tilt_precision_boundaries(tmp_path, axis, boundary, outside):
    from math import sin,cos,radians
    ep=Episode(tmp_path);ep.restore();ep.finish(stable=0)
    initial=list(ep.poses['Tier1']['position'])
    def set_error(value):
        if axis=='z':ep.poses['Tier1']['position'][2]=initial[2]+value
        else:ep.poses['Tier1']['quaternion_wxyz']=[cos(radians(value)/2),sin(radians(value)/2),0.,0.]
        ep.step()
    set_error(boundary);assert ep.value()==100
    set_error(outside);assert ep.value()==0
