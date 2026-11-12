"""Source: cp03-em-b-20261028 cases_unified/CP03/observable_quality.py (identical in Hard).
(scipy Rotation replaced by the equivalent closed-form z-axis x component.)

Original CP03 tilt windows measured at 120Hz, independent of expert labels.

The world supplies slot-mouth intersection from authored collider geometry and
actual plate/support contacts. Contact loss is accumulated as evidence; a numeric
loss allowance is applied only when it is present in the bound source protocol.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import math


def plate_tilt_deg(quaternion_wxyz):
    # x component of the rotated z axis (third column of the wxyz rotation matrix)
    w,x,y,z=quaternion_wxyz;n=math.sqrt(w*w+x*x+y*y+z*z)
    w,x,y,z=w/n,x/n,y/n,z/n
    return math.degrees(math.acos(min(1.,max(0.,abs(2*(x*z+y*w))))))


def slot_mouth_intersects(plate_bounds,opposing_side_bounds):
    """Entry volume from the two enabled authored side collider structures.

    Bounds are actual world bounds after carriage translation. Use the inner
    side surfaces and their shared y/z range, without guessing slot dimensions.
    """
    left,right=sorted(opposing_side_bounds,key=lambda b:(b[0][0]+b[1][0])/2)
    lo=[left[1][0],max(left[0][1],right[0][1]),max(left[0][2],right[0][2])]
    hi=[right[0][0],min(left[1][1],right[1][1]),min(left[1][2],right[1][2])]
    if any(a>=b for a,b in zip(lo,hi)):raise ValueError('Opposing slot capture geometry is incomplete')
    return all(plate_bounds[1][axis]>=lo[axis] and plate_bounds[0][axis]<=hi[axis] for axis in range(3))


@dataclass
class PlateQuality:
    protocol:dict
    initial_z:float
    tick:int=0
    phase:str='stock'
    transport_start:int|None=None
    insertion_start:int|None=None
    release_tick:int|None=None
    max_transport_tilt:float=0.
    max_insertion_tilt:float=0.
    total_excursion_ticks:int=0
    contiguous_excursion_ticks:int=0
    longest_excursion_ticks:int=0
    contact_loss_ticks:int=0
    contiguous_loss_ticks:int=0
    longest_loss_ticks:int=0
    excursion_contact_loss:bool=False
    violations:set=field(default_factory=set)
    boundaries:list=field(default_factory=list)

    def observe(self,tick,*,z,tilt_deg,fingers,slot_mouth_intersection,slot_support):
        if tick!=self.tick+1:raise ValueError('CP03 quality requires every120Hz tick')
        self.tick=tick
        if not all(math.isfinite(v) for v in (z,tilt_deg)):raise ValueError('Nonfinite CP03 measured state')
        held=any(fingers);bilateral=all(fingers)
        old=self.phase
        # 4cm lift is the original common world physical elevated predicate.
        if self.phase=='stock' and bilateral and z>self.initial_z+.04:
            self.phase='transport';self.transport_start=tick
        if self.phase=='transport' and (slot_mouth_intersection or slot_support):
            self.phase='insertion';self.insertion_start=tick
        if self.phase=='insertion' and not held and slot_support:
            self.phase='released';self.release_tick=tick
        if old!=self.phase:self.boundaries.append(dict(tick=tick,before=old,after=self.phase,source='measured lift, authored slot mouth intersection and actual support/finger contacts'))
        if self.phase in ('transport','insertion'):
            if not bilateral:
                self.contact_loss_ticks+=1;self.contiguous_loss_ticks+=1
                self.longest_loss_ticks=max(self.longest_loss_ticks,self.contiguous_loss_ticks)
            else:self.contiguous_loss_ticks=0
        else:self.contiguous_loss_ticks=0
        if self.phase=='transport':
            self.max_transport_tilt=max(self.max_transport_tilt,tilt_deg)
            excess=tilt_deg>float(self.protocol['max_transport_tilt_deg'])
            if excess:
                self.total_excursion_ticks+=1;self.contiguous_excursion_ticks+=1
                self.longest_excursion_ticks=max(self.longest_excursion_ticks,self.contiguous_excursion_ticks)
                self.excursion_contact_loss|=not bilateral
            else:self.contiguous_excursion_ticks=0
        elif self.phase=='insertion':
            self.max_insertion_tilt=max(self.max_insertion_tilt,tilt_deg)
            self.contiguous_excursion_ticks=0
        else:self.contiguous_excursion_ticks=0

    def result(self):
        P=self.protocol;base=float(P['max_transport_tilt_deg'])
        transient='transient_peak_limit_deg' in P
        if transient:
            transport=(self.max_transport_tilt<=P['transient_peak_limit_deg'] and
                self.longest_excursion_ticks/120<=P['max_contiguous_excursion_s']+1e-12 and
                self.total_excursion_ticks/120<=P['max_total_excursion_s']+1e-12 and not self.excursion_contact_loss)
        else:transport=self.max_transport_tilt<=base
        checks=dict(transport_tilt=transport,insertion_tilt=self.max_insertion_tilt<=base,
            physical_boundaries=self.transport_start is not None and self.insertion_start is not None and self.release_tick is not None)
        allowance=P.get('max_total_contact_loss_s')
        if allowance is not None:checks['contact_loss']=self.contact_loss_ticks/120<=float(allowance)+1e-12
        return dict(checks=checks,tilt_quality_pass=all(checks.values()),phase=self.phase,boundaries=self.boundaries,
            max_transport_tilt_deg=self.max_transport_tilt,max_insertion_tilt_deg=self.max_insertion_tilt,
            total_excursion_s=self.total_excursion_ticks/120,longest_excursion_s=self.longest_excursion_ticks/120,
            contact_loss_s=self.contact_loss_ticks/120,longest_contact_loss_s=self.longest_loss_ticks/120,
            contact_loss_gate_bound=allowance is not None,contact_loss_allowance_s=allowance,
            expert_phase_labels_used=False,physical_hz=120)
