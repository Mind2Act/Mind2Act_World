"""Actual supported station entries, one event until the object leaves that station."""
import numpy as np


class Ledger:
    def __init__(self, stations, target_dims, lay, dt=1/120):
        self.stations = stations
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

    def observe(self, step, center, linear_velocity, angular_velocity, robot_contact, supports, ready):
        station = self.station_at(center, supports) if ready else None
        previous, self.last_node = self.last_node, station
        if station is not None and station != previous:
            event = dict(station=station, step=step, position=list(center))
            self.placements.append(event)
            return event
        return None
