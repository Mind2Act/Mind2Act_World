"""Authenticate the immutable source input protocol as provenance.

The source inputs still carry their historical 60-tick collection fields.
FINAL-VERSION scoring and the live sequence use actual net support immediately;
those source fields do not add a dwell gate to the final scoring algorithm.
"""
from collections.abc import Mapping
from hashlib import sha256
import json
from types import MappingProxyType

PROTOCOL_ID = "cp10-net-retention-v1"
RETENTION = MappingProxyType(dict(
    schema="cp10-collection-protocol-v1", version=PROTOCOL_ID,
    consecutive_ticks=60, physics_hz=120, geometry="original_inside",
    released=True, linear_speed_required=False, angular_speed_required=False,
    support_chain_required=False, receipt_and_next_launch="same_machine",
))


def validate(value, rule_version):
    if not isinstance(value, Mapping) or set(value) != set(RETENTION):
        raise ValueError("CP10 requires the bound training collection_protocol")
    if rule_version != PROTOCOL_ID:
        raise ValueError("CP10 rule version differs from collection protocol")
    if any(type(value[k]) is not type(v) or value[k] != v for k, v in RETENTION.items()):
        raise ValueError("unsupported CP10 collection protocol")
    return dict(value)


def digest(value, rule_version=PROTOCOL_ID):
    return sha256(json.dumps(validate(value, rule_version), sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()
