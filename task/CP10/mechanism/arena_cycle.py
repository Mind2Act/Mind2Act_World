"""CP10 source shot sequencing for the native launcher mechanism.

Inputs must come from the eventual native adapter: actual yaw/pitch, muzzle
clearance and a settled-in-net receipt for the active shot. This module never
moves a body, advances physics or declares episode qualification.
"""
from dataclasses import dataclass
import random
import math
from .arena_preparation import PrepareClock


def sample_heading(config, shot_index):
    if type(shot_index) is not int or shot_index < 0:
        raise ValueError('Shot index must be a nonnegative integer')
    a = config['arena']['heading_sampling']
    if a['distribution']=='mobile_inbound_mix':
        from .mobile_launcher import sample_pose
        return sample_pose(config,shot_index)[1]
    if a['distribution']=='quota_sidewall_mix':
        from .rebound_schedule import sample_rebound_heading
        return sample_rebound_heading(config,shot_index)
    seed = f"{a['stream']}/{config['difficulty']}/{config['seed']}/{shot_index}"
    # Independent per-shot randomness; ball color never changes the angle stream.
    return round(random.Random(seed).uniform(*a['range_deg']), 6)


def shot_plan(config, shot_index, ball_color):
    if ball_color not in ('white', 'red'):
        raise ValueError('Unknown ball color')
    plan=dict(shot_index=shot_index, shot_number=shot_index+1,
        practice=shot_index < config['arena']['practice_shots'],
        yaw_deg=sample_heading(config, shot_index), ball_color=ball_color,
        prep_seconds=config['arena']['countdown']['seconds'])
    if 'slide_limits_m' in config['arena']['launcher']:
        from .mobile_launcher import sample_pose
        plan['slide_x_m']=sample_pose(config,shot_index)[0]
    return plan


@dataclass(frozen=True)
class NetReceipt:
    shot_index: int
    net: str
    settled: bool
    paddle_contact: bool


class ShotCycle:
    """Finite caller-owned color schedule; two unscored practice shots first.

No color mixture or acceptance threshold is invented here. The task supplies
its private color schedule; this coordinator controls when a shot may proceed.
"""
    def __init__(self, config, shot_colors):
        self.config = config
        self.colors = tuple(shot_colors)
        if len(self.colors) <= config['arena']['practice_shots']:
            raise ValueError('Need practice shots followed by scored shots')
        if any(color not in ('white', 'red') for color in self.colors):
            raise ValueError('Unknown ball color')
        self.last_tick = None
        self.results = []
        self._begin(0)

    def _begin(self, index):
        self.previous_plan=getattr(self,'plan',None)
        self.plan = shot_plan(self.config, index, self.colors[index])
        self.clock = PrepareClock(self.config, self.plan['yaw_deg'],self.plan.get('slide_x_m',0.))
        self.phase = 'resetting'
        self.reset_stage='raise' if self.previous_plan is not None else 'lower'
        self.fired = False
        self.clear = False

    def _public(self, countdown=None, release_now=False):
        a = self.config['arena']
        pitch = a['launcher']['stowed_pitch_deg'] if self.fired and self.clear else 0.
        yaw=self.plan['yaw_deg'];slide=self.plan.get('slide_x_m',0.)
        if self.phase in ('positioning','resetting_head'):pitch=a['launcher']['stowed_pitch_deg']
        if self.phase=='resetting_head' and self.previous_plan:
            yaw=self.previous_plan['yaw_deg'];slide=self.previous_plan.get('slide_x_m',0.)
        public=dict(phase=self.phase, shot_number=self.plan['shot_number'],
            practice=self.plan['practice'], score_eligible=not self.plan['practice'],
            target_yaw_deg=yaw, target_pitch_deg=pitch,
            countdown=countdown, release_now=release_now,
            ball_color=self.plan['ball_color'] if self.fired else None)
        if 'slide_x_m' in self.plan:public['target_slide_x_m']=slide
        return public

    def sample(self, tick, *, yaw_deg, yaw_speed_deg_s, pitch_deg,
               pitch_speed_deg_s=0., slide_x_m=0., slide_speed_m_s=0., ball_cleared_muzzle=False, collection=None):
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick+1):
            raise ValueError('Shot cycle requires consecutive 120 Hz samples')
        if not all(math.isfinite(v) for v in (yaw_deg,yaw_speed_deg_s,pitch_deg,pitch_speed_deg_s,slide_x_m,slide_speed_m_s)):
            raise ValueError('Invalid measured launcher pose')
        if collection is not None:
            if (not isinstance(collection, NetReceipt) or collection.net not in (None,'near','far')
                    or type(collection.shot_index) is not int or collection.shot_index < 0
                    or type(collection.settled) is not bool or type(collection.paddle_contact) is not bool):
                raise ValueError('Invalid measured net receipt')
            if collection.shot_index == self.plan['shot_index'] and not self.fired:
                raise ValueError('Cannot collect a shot before it is released')
        self.last_tick = tick
        if self.phase == 'complete':
            return self._public()
        if self.phase == 'collected':
            nxt = self.plan['shot_index']+1
            if nxt == len(self.colors):
                self.phase = 'complete'
                return self._public()
            # The only path to lowering / selecting another heading is a
            # measured settled receipt for the previously active ball.
            self._begin(nxt)
        if self.fired:
            self.clear = self.clear or bool(ball_cleared_muzzle)
            self.phase = 'raising' if self.clear else 'muzzle_clearance'
            raised = abs(pitch_deg-self.config['arena']['launcher']['stowed_pitch_deg']) <= .5
            if self.clear and raised and abs(pitch_speed_deg_s) <= 1.:
                self.phase = 'waiting_for_net'
            if (collection is not None and collection.shot_index == self.plan['shot_index']
                    and collection.settled):
                correct = ((self.plan['ball_color']=='white' and collection.net=='far' and collection.paddle_contact)
                    or (self.plan['ball_color']=='red' and collection.net=='near' and not collection.paddle_contact))
                self.results.append(dict(shot_number=self.plan['shot_number'],
                    practice=self.plan['practice'], correct=correct,
                    score_eligible=not self.plan['practice'],
                    scored_correct=correct and not self.plan['practice']))
                self.clear = True
                self.phase = 'collected'
            return self._public()
        if 'slide_x_m' in self.plan:
            c=self.config['arena']['countdown']
            aligned=(abs(yaw_deg-self.plan['yaw_deg'])<=c['heading_tolerance_deg'] and
                abs(yaw_speed_deg_s)<=c['max_yaw_speed_deg_s'] and
                abs(slide_x_m-self.plan['slide_x_m'])<=c['slide_tolerance_m'] and
                abs(slide_speed_m_s)<=c['max_slide_speed_m_s'])
            if self.reset_stage=='raise':
                raised=abs(pitch_deg-self.config['arena']['launcher']['stowed_pitch_deg'])<=.5 and abs(pitch_speed_deg_s)<=1.
                if not raised:
                    self.phase='resetting_head'
                    return self._public()
                self.reset_stage='translate'
            if self.reset_stage=='translate':
                if not aligned:
                    self.phase='positioning'
                    return self._public()
                self.reset_stage='lower'
            # Once the carriage has aligned, lowering must hold the NEW pose.
            # Small coupling drift cancels the countdown; it cannot command
            # the previous shot's pose and restart a raise/move oscillation.
        if abs(pitch_deg) > .5 or abs(pitch_speed_deg_s) > 1.:
            self.phase = 'resetting'
            self.clock = PrepareClock(self.config, self.plan['yaw_deg'],self.plan.get('slide_x_m',0.))
            return self._public()
        result = self.clock.sample(tick,yaw_deg,yaw_speed_deg_s,slide_x_m,slide_speed_m_s)
        self.phase = result['phase']
        if result['release_now']:
            self.fired = True
            self.phase = 'muzzle_clearance'
        return self._public(result['countdown'],result['release_now'])
