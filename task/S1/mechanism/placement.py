"""Support-frame placement windows, carousel ledger and release tracker for S1.

Placement, carousel and release measurements use observed physical state;
they never read expert phase labels.
"""
from collections import deque
from copy import deepcopy
import numpy as np


class PlacementWindow:
    def __init__(self):
        self.samples = deque(maxlen=121)
        self.last_tick = None
        self.residence = None
        self.fields = dict(inside=False, support=False, released=False)

    def observe(self, tick, position_local, *, inside, support, released, residence):
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick+1):
            raise ValueError('placement window requires consecutive 120 Hz samples')
        point = np.asarray(position_local, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('finite measured support-local position required')
        if any(type(value) is not bool for value in (inside, support, released)):
            raise ValueError('placement qualifiers must be measured booleans')
        self.last_tick = tick
        self.fields = dict(inside=inside, support=support, released=released)
        if residence != self.residence or not (inside and support and released):
            self.samples.clear()
        self.residence = residence
        if inside and support and released:
            self.samples.append((tick, point.copy()))

    def snapshot(self):
        duration = (self.samples[-1][0]-self.samples[0][0])/120. if self.samples else 0.
        points = np.array([sample[1] for sample in self.samples])
        spread = float(np.linalg.norm(points[:,None]-points[None,:],axis=-1).max()) if len(points) else 0.
        return dict(self.fields, stable_seconds=duration, position_range_m=spread)


class CarouselLedger:
    def __init__(self, object_ids):
        self.windows = {key: PlacementWindow() for key in object_ids}
        if len(self.windows) != 4:
            raise ValueError('S1 requires four fixed physical block identities')
        self.early_placement = False
        self.rotation_samples = []
        self.program_finished = False
        self.last_tick = None
        self.blocks = {}

    def observe(self, tick, blocks, *, program_finished, rotation_state=None, cue_visible=False):
        if set(blocks) != set(self.windows):
            raise ValueError('S1 requires every fixed task block each tick')
        if type(program_finished) is not bool or type(cue_visible) is not bool:
            raise ValueError('S1 measured program completion and cue flags required')
        if self.program_finished and not program_finished:
            raise ValueError('S1 program completion cannot reverse')
        self.program_finished = program_finished
        if rotation_state is not None:
            self.rotation_samples.append(dict(tick=tick,rotors=deepcopy(rotation_state)))
        self.blocks = {}
        for key, value in blocks.items():
            if type(value['carousel_supported']) is not bool:
                raise ValueError('S1 carousel support observation required')
            self.early_placement |= bool(not cue_visible and not program_finished
                                         and value['carousel_supported'] and value['released'])
            self.windows[key].observe(tick, value['position_local'], inside=value['inside'],
                support=value['support'], released=value['released'], residence=value['slot_id'])
            self.blocks[key] = dict(self.windows[key].snapshot(), object_id=key,
                original_entity_slot_match=value['original_entity_slot_match'],
                placed_after_program_finished=value['placed_after_program_finished'])
        self.last_tick = tick

    def snapshot(self):
        return dict(early_placement=self.early_placement,
                    rotation_program_finished=self.program_finished,
                    blocks=deepcopy(list(self.blocks.values())))



class SPlacementReleaseTracker:
    """Track each block's last robot-contact release event.

    The stop boundary is the first 120 Hz simulation step at which the S
    schedule has entered ``execution`` after every carousel completed its final
    scheduled motion and the existing one-second settling interval. A block is
    marked ``placed_after_stop`` only when its latest completed robot-contact
    release occurs on a step strictly after that boundary. Intermediate dwell
    intervals do not count as the final stop. Spatial placement checks remain
    independent scorer fields.
    """

    def __init__(self, block_paths):
        self.block_paths = dict(block_paths)
        self.active_pairs = {key: set() for key in self.block_paths}
        self.last_release = {key: None for key in self.block_paths}

    def observe(self, step, events):
        """Apply contact-report events from one physics tick, in report order."""
        for event in events:
            colliders = tuple(sorted(str(path) for path in event.get('colliders', ())))
            if len(colliders) != 2 or not any('/Robot' in path for path in colliders):
                continue
            block = next((key for key, root in self.block_paths.items()
                          if any(path.startswith(root + '/') for path in colliders)), None)
            if block is None:
                continue
            pair = colliders
            active = self.active_pairs[block]
            if event.get('active'):
                active.add(pair)
            elif pair in active:
                active.remove(pair)
                if not active:
                    self.last_release[block] = int(step)

    def placed_after_stop(self, block, execution_start_step):
        """Return false without a release event or final execution boundary."""
        release_step = self.last_release.get(block)
        return (release_step is not None and execution_start_step is not None
                and release_step > execution_start_step)
