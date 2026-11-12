"""CP09 chronological contact-raster measurements.

Derived from source_id cp09-replay-contact-raster-v1, source SHA256
fe3d67674830eccd6d721a9c08fe8aff12269b2aa17ae8b04743caa0c8d8a7d3.
Only native contact and raster measurements enter this module.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import CONTRACT


class RasterMetrics:
    """Exact tolerance scores updated only around actual changed pixels."""
    def __init__(self, target, actual, tolerance_pixels):
        self.target = np.asarray(target, dtype=np.uint8)
        self.actual = np.asarray(actual, dtype=np.uint8)
        self.tolerance = float(tolerance_pixels)
        self.colors = sorted(int(c) for c in np.unique(self.target) if c)
        self.target_count, self.actual_count, self.matched_actual, self.matched_target = {}, {}, {}, {}
        self.distance, self.covered = {}, {}
        for c in self.colors:
            wanted, present = self.target == c, self.actual == c
            self.target_count[c] = int(wanted.sum())
            self.actual_count[c] = int(present.sum())
            self.distance[c] = distance_transform_edt(~wanted)
            self.covered[c] = wanted & (distance_transform_edt(~present) <= self.tolerance) if present.any() else np.zeros_like(wanted)
            self.matched_target[c] = int(self.covered[c].sum())
            self.matched_actual[c] = int((present & (self.distance[c] <= self.tolerance)).sum())

    def update(self, bounds, before):
        x0, y0, x1, y1 = bounds
        after = self.actual[y0:y1, x0:x1]
        changed = before != after
        if not changed.any():
            return
        ys, xs = np.nonzero(changed)
        # A target pixel's coverage can change only within the tolerance radius.
        halo = int(math.ceil(self.tolerance)) + 1
        h, w = self.target.shape
        qx0, qy0 = max(0, x0+int(xs.min())-halo), max(0, y0+int(ys.min())-halo)
        qx1, qy1 = min(w, x0+int(xs.max())+halo+1), min(h, y0+int(ys.max())+halo+1)
        ox0, oy0, ox1, oy1 = max(0,qx0-halo), max(0,qy0-halo), min(w,qx1+halo), min(h,qy1+halo)
        affected = set(np.unique(before[changed])) | set(np.unique(after[changed]))
        for c in self.colors:
            if c not in affected:
                continue
            old, new = before == c, after == c
            self.actual_count[c] += int(new.sum())-int(old.sum())
            eligible = self.distance[c][y0:y1,x0:x1] <= self.tolerance
            self.matched_actual[c] += int((new & eligible).sum())-int((old & eligible).sum())
            present = self.actual[oy0:oy1,ox0:ox1] == c
            near = distance_transform_edt(~present) <= self.tolerance if present.any() else np.zeros_like(present)
            covered = (self.target[qy0:qy1,qx0:qx1] == c) & near[qy0-oy0:qy1-oy0,qx0-ox0:qx1-ox0]
            view = self.covered[c][qy0:qy1,qx0:qx1]
            self.matched_target[c] += int(covered.sum())-int(view.sum())
            view[:] = covered

    def metrics(self):
        nt = sum(self.target_count.values()); na = int(np.count_nonzero(self.actual))
        mt = sum(self.matched_target.values()); ma = sum(self.matched_actual.values())
        return dict(recall=mt/max(1,nt), precision=ma/max(1,na), extra_fraction=(na-ma)/max(1,nt),
                    per_color=[dict(color=c, recall=self.matched_target[c]/self.target_count[c],
                                    precision=self.matched_actual[c]/max(1,self.actual_count[c]),
                                    extra_fraction=(self.actual_count[c]-self.matched_actual[c])/self.target_count[c]) for c in self.colors])


def contact_bounds(tool, xy, shape, pixels_per_meter, board_width, board_height):
    px, py = (float(xy[0])+board_width/2)*pixels_per_meter, (board_height/2-float(xy[1]))*pixels_per_meter
    if not np.isfinite([px,py]).all():
        raise ValueError('CP09 nonfinite contact')
    radius = 40 if tool == 'eraser' else 10
    h,w = shape
    return max(0,min(w,int(px)-radius)),max(0,min(h,int(py)-radius)),max(0,min(w,int(px)+radius+1)),max(0,min(h,int(py)+radius+1))


class Progress:
    def __init__(self, target, actual, *, tolerance_pixels, difficulty):
        self.target = np.asarray(target,dtype=np.uint8)
        self.actual = actual
        self.tolerance_pixels = tolerance_pixels
        self.difficulty = difficulty
        self.raster = RasterMetrics(self.target,actual,tolerance_pixels)
        self.original = None
        self.board_tick = None
        self.return_snapshot = None
        self.return_ticks = []
        self.was_parked = True
        self.cap_ticks = {}
        self.repair_ticks = {}
        self.pickup = {'eraser':False}
        self.pen_pickup = {}
        self.cap_events, self.pickup_events = [], []
        self.first_error = None
        self.contact_index = 0
        self.eraser_contacts = 0

    def annotate(self, actual):
        if self.original is not None:
            raise ValueError('CP09 repeated raster annotation')
        self.raster = RasterMetrics(self.target,actual,self.tolerance_pixels)
        self.original = (actual == self.target) & (self.target > 0)

    def error(self,tick,kind,**details):
        if self.first_error is None:
            self.first_error = dict(tick=tick,contact_index=self.contact_index,kind=kind,**details)

    def before_contacts(self,tick,state):
        if state.get('contract') != CONTRACT:
            raise ValueError('CP09 measured state contract missing')
        # Pickup, cap, retrieval and return remain diagnostic, never score gates.
        self.pickup_events.extend(deepcopy(state.get('pickup_events', [])))
        self.cap_events.extend(deepcopy(state.get('cap_events', [])))

    def mutation(self,tick,contact,bounds,before):
        x0,y0,x1,y1 = bounds
        after = self.actual[y0:y1,x0:x1]
        if int(np.count_nonzero(before != after)) != contact['changed_pixels']:
            raise ValueError('CP09 changed-pixel proof mismatch')
        self.contact_index += 1
        self.raster.update(bounds,before)
        self.measure_repairs(tick)

    def measure_repairs(self,tick):
        for metric in self.raster.metrics()['per_color']:
            color = metric['color']
            if metric['precision'] >= .95 and metric['recall'] >= .95 and metric['extra_fraction'] <= .03:
                self.repair_ticks.setdefault(color,tick)
            else:
                self.repair_ticks.pop(color,None)

    def after_contacts(self,tick,state):
        self.measure_repairs(tick)

    def result(self,success):
        surface = self.raster.metrics()
        steps = [bool(success and m['precision'] >= .95 and m['recall'] >= .95 and m['extra_fraction'] <= .03)
                 for m in surface['per_color']]
        return dict(contract=CONTRACT,report_step_source='measured_physical_state',
                    measured_report_steps=steps,step_completion_ticks=[None]*len(steps),
                    color_order=list(self.raster.colors),first_error=None,
                    first_eraser_return=None,eraser_return_ticks=[],success=all(steps),
                    final_surface=surface,
                    evidence_references=['complete contact-modified final colored raster'])
