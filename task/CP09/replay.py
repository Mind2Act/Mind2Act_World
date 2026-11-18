"""Incremental physical raster/progress, independently repeated at terminal scoring."""
import numpy as np
from . import CONTRACT
from .evidence import Progress,contact_bounds
from .mechanism.layouts import make_case,from_case,CONFIG,PPM,BW,BH
from .mechanism.surface import InkSurface
from .mechanism.presentation import ANNOTATION_TICK
from .mechanism.retrieval_geometry import BODY_Z,RACK_FRONT_Y,MACHINE_RETRACTION_M


class BoardFacts:
    """Source board-support thresholds, never expert phase or action indices."""
    def __init__(self):
        self.extracted=False;self.placed=False;self.seat_ticks=0
        self.motion_origin=None;self.drift=0.;self.minimum_clearance=float('inf');self.previous_offset=0.

    def sample(self,raw):
        p=np.asarray(raw['position_m'],float);rotation=np.asarray(raw['rotation'],float)
        if p.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(rotation).all():
            raise ValueError('CP09 malformed native board pose')
        clearance=RACK_FRONT_Y+raw['machine_offset_m']-p[1]-float(np.dot(np.abs(rotation[1]),[.240,.165,.016]))
        if raw['storage_clip_removed'] and clearance>=.012 and p[2]>=BODY_Z+.08 and any(raw['latches'].values()):self.extracted=True
        seated=self.extracted and abs(p[1]-.065)<.008 and abs(p[2]-BODY_Z)<.006 and raw['foot_table_contact']
        self.seat_ticks=self.seat_ticks+1 if seated else 0
        if self.seat_ticks>=24:self.placed=True
        if self.placed:
            self.minimum_clearance=min(self.minimum_clearance,float(clearance))
            if self.motion_origin is None:self.motion_origin=p.copy()
            # Original source collects positions during the 180tick retraction.
            if self.previous_offset<MACHINE_RETRACTION_M:
                self.drift=max(self.drift,float(np.linalg.norm(p-self.motion_origin)))
        self.previous_offset=raw['machine_offset_m']
        released=not any(raw['latches'].values()) and not any(raw['release_pending'].values())
        return bool(self.extracted and self.placed and released and raw['machine_offset_m']>=MACHINE_RETRACTION_M-.001
                    and self.drift<.003 and self.minimum_clearance>.012)


class Replay:
    def __init__(self,difficulty,seed,case=None):
        self.case,target,self.work=make_case(difficulty,seed) if case is None else from_case(case)
        self.surface=InkSurface(target);self.annotated=False;self.tick=0;self.board=BoardFacts();self.last_time=None
        self.progress=Progress(target,self.surface.ink,tolerance_pixels=CONFIG['grading']['tolerance_m']*PPM,difficulty=difficulty)

    def append(self,row):
        tick=row['tick']
        if type(tick) is not int or tick!=self.tick+1:raise ValueError('CP09 missing physical tick')
        now=row['simulation_time_s']
        if not np.isfinite(now) or (self.last_time is not None and abs(now-self.last_time-1/120)>1e-7):
            raise ValueError('CP09 native audit clock differs')
        self.last_time=now
        if row['annotation_applied']!=(tick>=ANNOTATION_TICK):raise ValueError('CP09 source presentation clock differs')
        if row['annotation_applied'] and not self.annotated:
            self.surface.ink[:]=self.work;self.annotated=True;self.progress.annotate(self.surface.ink)
        parked=(not row['eraser']['latched'] and row['eraser']['tray_contact']
                and np.linalg.norm(np.asarray(row['eraser']['position_m'])-np.asarray(row['eraser']['initial_position_m']))<.02)
        state=dict(contract=CONTRACT,board_retrieved=self.board.sample(row['board']),eraser_parked=bool(parked),
                   pickup_events=row['pickup_events'],cap_events=row['cap_events'])
        self.progress.before_contacts(tick,state)
        for contact in row['ink_contacts']:
            if contact['physx_contact'] is not True:raise ValueError('CP09 ink requires native physical contact')
            bounds=contact_bounds(contact['tool'],contact['xy_m'],self.surface.ink.shape,PPM,BW,BH)
            x0,y0,x1,y1=bounds;before=self.surface.ink[y0:y1,x0:x1].copy()
            changed=self.surface.contact(contact['tool'],contact['xy_m'],True,tick,contact['angle_rad'],color=contact['color'])
            if changed!=contact['changed_pixels']:raise ValueError('CP09 physical raster replay differs')
            self.progress.mutation(tick,contact,bounds,before)
        self.progress.after_contacts(tick,state);self.tick=tick

    def result(self):return self.progress.result(self.annotated)
