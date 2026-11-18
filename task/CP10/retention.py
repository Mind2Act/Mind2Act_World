"""Actual net support and center containment, with no extra dwell or height gates."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class BallBounds:
    center: tuple
    lower: tuple
    upper: tuple
    released: bool
    supported_nets: tuple = ()

    def __post_init__(self):
        if (type(self.released) is not bool
                or any(len(v) != 3 or not all(isfinite(x) for x in v)
                       for v in (self.center, self.lower, self.upper))
                or any(not self.lower[i] <= self.center[i] <= self.upper[i] for i in range(3))):
            raise ValueError("invalid CP10 measured ball bounds")


def floor_z(config, net, x=0.):
    arena = config["arena"]
    profile = arena.get("collector_profile", {}).get(net)
    if profile and abs(x) > profile["center_half_width_m"]:
        t = ((abs(x) - profile["center_half_width_m"])
             / (arena["court_size"][0] / 2 - profile["center_half_width_m"]))
        return profile["wing_inner_floor_z"] + t * (profile["wing_outer_floor_z"] - profile["wing_inner_floor_z"])
    return arena["surface_z"] - arena.get("collector_depth_m", {}).get(net, arena["net_depth_m"])


def inside(body, config, net):
    arena = config["arena"]
    width, length = arena["court_size"]
    sign = -1 if net == "near" else 1
    mouth = arena["court_center"][1] + sign * length / 2
    rear = mouth + sign * arena["collector_length_m"][net]
    low, high = sorted((mouth, rear))
    return bool(net in body.supported_nets and -width/2 <= body.center[0] <= width/2
                and low <= body.center[1] <= high)


def contained_net(body, config):
    if not body.released:
        return None
    return next((net for net in ("near", "far") if inside(body, config, net)), None)


class ContinuousRetention:
    def __init__(self):
        self.last_tick = None
        self.net = None
        self.count = 0

    def sample(self, tick, net):
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick + 1):
            raise ValueError("retention requires consecutive measured physics ticks")
        if net not in (None, "near", "far"):
            raise ValueError("invalid CP10 measured collector")
        self.count = self.count + 1 if net is not None and net == self.net else 1 if net is not None else 0
        self.last_tick, self.net = tick, net
        return net
