"""Terminal occupancy ledger for P1: continuous 1 s placement windows and residence entry ticks.

Source: old eval cases/P/runtime/current_evidence.py (PlacementWindow, OccupancyLedger; 120 Hz fixed
sample dt). Trusted-evaluator measurement; no expert action index or milestone participates.
"""
from collections import deque
from copy import deepcopy
import math
import numpy as np

SAMPLE_DT = 1 / 120

class PlacementWindow:
    """One continuous second, calculated from measured physics dt × substeps."""
    def __init__(self, sample_dt):
        if not math.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError('P1 positive actual sample dt required')
        self.dt = sample_dt
        self.samples = deque()
        self.last_tick = None
        self.residence = None
        self.fields = dict(inside=False, support=False, released=False)

    def observe(self, tick, position, *, inside, support, released, residence):
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick+1):
            raise ValueError('P1 requires consecutive native control samples')
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('P1 finite measured position required')
        if any(type(x) is not bool for x in (inside, support, released)):
            raise ValueError('P1 measured placement booleans required')
        self.last_tick = tick
        self.fields = dict(inside=inside, support=support, released=released)
        if residence != self.residence or not all(self.fields.values()):
            self.samples.clear()
        self.residence = residence
        if all(self.fields.values()):
            self.samples.append((tick,point.copy()))
            while len(self.samples)>1 and (tick-self.samples[1][0])*self.dt >= 1.:
                self.samples.popleft()

    def snapshot(self):
        duration = (self.samples[-1][0]-self.samples[0][0])*self.dt if self.samples else 0.
        points = np.array([point for _,point in self.samples])
        spread = float(np.linalg.norm(points[:,None]-points[None,:],axis=-1).max()) if len(points) else 0.
        return dict(self.fields,stable_seconds=duration,position_range_m=spread)


class OccupancyLedger:
    def __init__(self, object_ids, sample_dt):
        self.windows = {key:PlacementWindow(sample_dt) for key in object_ids}
        self.residences = {}
        self.uncertain = set()
        self.occupancy = []
        self.ambiguities = []
        self.last_tick = None
        self.demand = {}
        self.all_sent = False

    def observe(self, tick, objects, *, published_demand, all_orders_published):
        if set(objects) != set(self.windows):
            raise ValueError('P1 every fixed food must be sampled')
        if type(all_orders_published) is not bool:
            raise ValueError('P1 measured publication state required')
        self.occupancy = []; self.ambiguities = []; self.uncertain.clear()
        for ident,row in objects.items():
            cups = row['occupied_containers']
            if type(cups) is not list or any(type(c) is not str for c in cups) or len(set(cups)) != len(cups):
                raise ValueError('P1 unique measured container list required')
            if len(cups)>1:
                self.uncertain.add(ident)
                self.ambiguities.append(dict(object_id=ident,containers=list(cups),reason='multiple original predicates true'))
            cid = cups[0] if len(cups)==1 else None
            previous = self.residences.get(ident)
            if cid is None:self.residences.pop(ident,None)
            elif previous is None or previous[0] != cid:self.residences[ident]=(cid,tick)
            if not cups:self.uncertain.discard(ident)
            self.windows[ident].observe(tick,row['position'],inside=row['inside'] and cid is not None,
                support=row['support'],released=row['released'],residence=cid)
            if cid is not None:
                if ident in self.uncertain:
                    self.ambiguities.append(dict(object_id=ident,containers=list(cups),reason='current entry time unresolved after multiple predicates true'))
                self.occupancy.append(dict(self.windows[ident].snapshot(),object_id=ident,container=cid,
                    kind=row['kind'],entered_tick=self.residences[ident][1]))
        self.demand = deepcopy(published_demand)
        self.all_sent = all_orders_published
        self.last_tick = tick

    def snapshot(self):
        return dict(occupancy=deepcopy(self.occupancy),occupancy_complete=not self.ambiguities,
            ambiguous_membership=deepcopy(self.ambiguities),published_demand=deepcopy(self.demand),
            all_orders_published=self.all_sent)


