"""Measured heading -> configured preparation interval, without physics writes.

The timing boundary comes from measured launcher alignment, never from a
requested yaw command.
"""
from dataclasses import dataclass
import math


def launch_pose(arena, yaw_deg, slide_x_m=0.):
    launcher = arena['launcher']
    low, high = launcher['yaw_limits_deg']
    if not math.isfinite(yaw_deg) or not low <= yaw_deg <= high:
        raise ValueError('Launcher heading is outside its mechanical range')
    if 'slide_limits_m' in launcher:
        from .mobile_launcher import safe_pose
        if not safe_pose(arena,slide_x_m,yaw_deg):raise ValueError('Unsafe launcher slide/heading combination')
    radians = math.radians(yaw_deg)
    direction = [math.sin(radians), -math.cos(radians)]
    origin = [round(launcher['base_xy'][i]+direction[i]*launcher['muzzle_distance_m'], 12)
              for i in range(2)]
    origin[0]=round(origin[0]+slide_x_m,12)
    return origin, direction


@dataclass
class PrepareClock:
    config: dict
    target_yaw_deg: float
    target_slide_x_m: float = 0.
    phase: str = 'aiming'
    stable_ticks: int = 0
    deadline_tick: int | None = None
    last_tick: int | None = None

    def __post_init__(self):
        launch_pose(self.config['arena'], self.target_yaw_deg,self.target_slide_x_m)

    def sample(self, tick, measured_yaw_deg, measured_yaw_speed_deg_s, measured_slide_x_m=0., measured_slide_speed_m_s=0.):
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick+1):
            raise ValueError('Preparation clock requires consecutive 120 Hz samples')
        if not all(math.isfinite(v) for v in (measured_yaw_deg, measured_yaw_speed_deg_s, measured_slide_x_m, measured_slide_speed_m_s)):
            raise ValueError('Invalid measured launcher heading')
        self.last_tick = tick
        if self.phase == 'released':
            return dict(phase='released', countdown=None, release_now=False)
        c = self.config['arena']['countdown']
        error = (measured_yaw_deg-self.target_yaw_deg+180.) % 360.-180.
        settled = abs(error) <= c['heading_tolerance_deg'] and abs(measured_yaw_speed_deg_s) <= c['max_yaw_speed_deg_s']
        if 'slide_limits_m' in self.config['arena']['launcher']:
            settled=settled and abs(measured_slide_x_m-self.target_slide_x_m)<=c['slide_tolerance_m'] and abs(measured_slide_speed_m_s)<=c['max_slide_speed_m_s']
        if not settled:
            self.phase, self.stable_ticks, self.deadline_tick = 'aiming', 0, None
            return dict(phase='aiming', countdown=None, release_now=False)
        if self.phase == 'aiming':
            self.stable_ticks += 1
            if self.stable_ticks < c['settle_ticks']:
                return dict(phase='aiming', countdown=None, release_now=False)
            self.phase = 'prepare'
            self.deadline_tick = tick+round(c['seconds']*self.config['physics_hz'])
        remaining = self.deadline_tick-tick
        if remaining == 0:
            self.phase = 'released'
            return dict(phase='released', countdown=None, release_now=True)
        return dict(phase='prepare', countdown=math.ceil(remaining/self.config['physics_hz']), release_now=False)
