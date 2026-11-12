"""Synthetic observation smoke against an explicitly selected policy service.

No scene is constructed and no score is computed. This cannot validate task
physics, model quality, checkpoint identity on native G0.5, or episode RNG seeds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mind2act_eval.contracts import Observation
from .xpolicylab import CAMERAS, XPolicyLabAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', default='g05-native', choices=('g05-native', 'xpolicy-rpc'))
    parser.add_argument('--url', required=True)
    parser.add_argument('--action-steps', type=int, help='Required native G0.5 execution chunk')
    parser.add_argument('--checkpoint', help='Native configured checkpoint identifier (not remotely verified)')
    parser.add_argument('--checkpoint-sha256', help='Native configured checkpoint hash (not remotely verified)')
    parser.add_argument('--identity', type=Path, help='Required canonical RPC expected identity JSON')
    parser.add_argument('--source-root', type=Path, help='Pinned official XPolicyLab checkout')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout-s', type=float, default=180.)
    args = parser.parse_args(argv)
    if args.transport == 'g05-native':
        if args.action_steps is None or args.identity is not None or args.source_root is not None:
            parser.error('native G0.5 requires --action-steps and does not have an RPC identity handshake')
        adapter = XPolicyLabAdapter.g05_native(args.url, action_steps=args.action_steps, timeout_s=args.timeout_s,
            checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256)
        horizon = args.action_steps
    else:
        if args.identity is None or args.action_steps is not None or args.checkpoint is not None or args.checkpoint_sha256 is not None:
            parser.error('canonical RPC requires --identity (horizon comes from that identity)')
        identity = json.loads(args.identity.read_text())
        adapter = XPolicyLabAdapter.xpolicy_rpc(args.url, expected_identity=identity,
            source_root=args.source_root, timeout_s=args.timeout_s)
        horizon = identity['action_horizon']
    started = time.monotonic()
    try:
        adapter.reset(args.seed)
        for row in range(horizon + 1):
            obs = Observation(row*4, row/30., (0.,)*6, (0.,)*6, 1., 1.,
                {key: np.zeros((224,224,3), np.uint8) for key in CAMERAS},
                'Move the cube')
            adapter.act(obs)
        # Clear an incomplete second chunk at an episode boundary.
        adapter.reset(args.seed)
        adapter.act(obs)
        print(json.dumps({'transport': args.transport, 'synthetic_input': True,
            'rows_checked': horizon + 2, 'reset_count': 2, 'score': None,
            'capabilities': adapter.transport.capabilities, 'policy_metadata': adapter.result_metadata(),
            'elapsed_wall_s': time.monotonic() - started}))
    finally:
        adapter.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
