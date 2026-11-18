"""Observe measured releases, ball bounds and contacts on every physics tick."""
from copy import deepcopy
from mind2act_eval.contracts import to_json

from . import CONTRACT
from .protocol import PROTOCOL_ID, RETENTION, digest, validate
from .retention import BallBounds, ContinuousRetention, contained_net

LAYOUT = {"easy": (5, 3), "medium": (8, 4), "hard": (12, 6)}


def validate_schedule(difficulty, colors):
    if difficulty not in LAYOUT:
        raise ValueError("unknown CP10 difficulty")
    count, white = LAYOUT[difficulty]
    if (len(colors) != count + 2 or any(v not in ("red", "white") for v in colors)
            or colors[2:].count("white") != white):
        raise ValueError("CP10 colour schedule differs from the bound difficulty")


class ShotEvidence:
    """Collection receipts follow the training protocol, including wrong nets.

    `emitted` is the physically recorded one-time release index. This class
    never launches a ball, advances the world, or reads a controller phase.
    """
    def __init__(self, difficulty, colors, config, collection_protocol, rule_version):
        self.colors = tuple(colors)
        validate_schedule(difficulty, self.colors)
        validate(collection_protocol, rule_version)
        self.config = to_json(config)
        if (config["arena"]["net_settle_ticks"] != 60
                or config["arena"]["per_shot_timeout_ticks"] != 2400):
            raise ValueError("CP10 requires the source 60-tick collection/20-second timeout")
        self.last_tick = None
        self.active = None
        self.launch_tick = None
        self.paddle_contact = False
        self.retention = None
        self.shots = []
        self.failure = None
        self.final_nets = {}

    def sample(self, tick, body=None, *, emitted=None, paddle_contact=False,
               robot_contact=False, unreleased_ball_moved=False):
        if (type(tick) is not int or tick < 0 or (self.last_tick is None and tick != 0)
                or (self.last_tick is not None and tick != self.last_tick + 1)):
            raise ValueError("CP10 evidence requires every consecutive physics tick")
        if any(type(v) is not bool for v in (paddle_contact, robot_contact, unreleased_ball_moved)):
            raise ValueError("CP10 contact facts must be booleans")
        self.last_tick = tick
        formal = (emitted if emitted is not None else self.active)
        if formal is not None and formal >= 2 and robot_contact and self.failure is None:
            self.failure = dict(tick=tick,reason='robot_touched_ball')
        if emitted is not None:
            if (type(emitted) is not int or self.active is not None
                    or emitted != len(self.shots) or emitted >= len(self.colors)):
                raise ValueError("CP10 release is out of sequence or precedes collection")
            self.active, self.launch_tick = emitted, tick
            self.paddle_contact = False
            self.retention = ContinuousRetention()
        if self.active is None:
            if body is not None:
                raise ValueError("CP10 ball measurement without an active release")
            return
        if not isinstance(body, BallBounds):
            raise ValueError("CP10 active ball physical bounds missing")
        self.paddle_contact |= paddle_contact
        # Preserve the selected source's strict >2400 timeout boundary.
        if tick - self.launch_tick > 2400:
            self.shots.append(dict(shot_index=self.active,color=self.colors[self.active],
                net=None,paddle_contact=self.paddle_contact,released=True,
                launch_tick=self.launch_tick,collection_tick=tick,retention_ticks=0))
            self.final_nets['ball'+str(self.active)]=None
            self.active = self.launch_tick = self.retention = None
            return
        net = self.retention.sample(tick, contained_net(body, self.config))
        if net is not None:
            self.final_nets['ball'+str(self.active)]=net
            self.shots.append(dict(shot_index=self.active, color=self.colors[self.active],
                net=net, paddle_contact=self.paddle_contact, released=True,
                launch_tick=self.launch_tick, collection_tick=tick,
                retention_ticks=self.retention.count))
            self.active = self.launch_tick = self.retention = None

    def snapshot(self, termination="policy_ended"):
        if self.last_tick is None:
            raise ValueError("CP10 missing initial physical sample")
        reason = self.failure["reason"] if self.failure else termination
        return dict(contract=CONTRACT, source="measured_physical_state", colors=list(self.colors),
                    shots=deepcopy(self.shots), terminal_reason=reason,
                    failure=deepcopy(self.failure), last_tick=self.last_tick,
                    final_nets=deepcopy(self.final_nets),scoring_version='mind2act-final-scoring-20261105',
                    collection_protocol_version=PROTOCOL_ID,
                    collection_protocol_sha256=digest(RETENTION))
