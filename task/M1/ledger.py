"""Actual supported station entries, one event until the object leaves that station."""
import numpy as np


class Ledger:
    def __init__(self, stations, target_dimensions):
        self.stations = stations
        self.target_dimensions = target_dimensions
        self.last_node = None
        self.placements = []

    def station_at(self, center, supports):
        center = np.asarray(center, float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('M1 requires finite measured center')
        matches = []
        for name, spec in self.stations.items():
            if type(supports[name]) is not bool:
                raise ValueError('M1 requires actual station support booleans')
            if supports[name] and np.all(abs(center[:2]-np.asarray(spec['center_xy'])) <= np.asarray(spec['clear_dimensions_xy'])/2):
                matches.append(name)
        if len(matches) > 1:
            raise ValueError('M1 simultaneous station membership is ambiguous')
        return matches[0] if matches else None

    def observe(self, step, center, supports, ready):
        if self.last_node is not None:
            top = self.stations[self.last_node]['support_top_z']
            if center[2] > top + self.target_dimensions[2] / 2 + .025:
                self.last_node = None
        station = self.station_at(center, supports) if ready else None
        if station is not None and station != self.last_node:
            self.last_node = station
            event = dict(station=station, step=step, position=list(center))
            self.placements.append(event)
            return event
        return None
