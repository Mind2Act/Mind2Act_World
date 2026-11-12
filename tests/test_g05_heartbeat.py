"""CPU fake service with a short heartbeat budget and a GIL-holding native build."""
from contextlib import contextmanager
import ctypes
from dataclasses import replace
import multiprocessing
from threading import Thread
import time

import pytest
import numpy as np

from mind2act_eval.adapters.xpolicylab import XPolicyLabAdapter
from mind2act_eval.contracts import ConsumerKind
from mind2act_eval.runner import EpisodeLimits, run_episode
from test_xpolicy_result import synthetic_episode
from test_xpolicylab import pack, unpack


def _server(pipe, connections):
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed

    def policy(ws):
        with connections.get_lock():
            connections.value += 1
        ws.send(pack({'action_steps': 1}))
        try:
            for blob in ws:
                payload = unpack(blob)
                if payload == {'__reset__': True}:
                    ws.send(pack({'__reset__': True}))
                else:
                    ws.send(pack({'action': {'left_arm': np.zeros(6), 'right_arm': np.zeros(6),
                        'left_gripper': np.ones(1), 'right_gripper': np.ones(1)}, 'need_obs': True}))
        except ConnectionClosed:
            pass

    with serve(policy, '127.0.0.1', 0, ping_interval=.05, ping_timeout=.1) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        pipe.send(f'ws://127.0.0.1:{server.socket.getsockname()[1]}')
        pipe.recv()
        server.shutdown()
        thread.join(5)


@contextmanager
def service():
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    connections = ctx.Value('i', 0)
    proc = ctx.Process(target=_server, args=(child, connections))
    proc.start()
    child.close()
    try:
        assert parent.poll(10), 'fake service did not start'
        yield parent.recv(), connections
    finally:
        if proc.is_alive():
            parent.send('stop')
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        parent.close()
        proc.close()


def native_gil_block(seconds):
    # PyDLL keeps the GIL, unlike CDLL/time.sleep. No simulator or GPU involved.
    libc = ctypes.PyDLL(None)
    libc.usleep.argtypes = [ctypes.c_uint]
    libc.usleep.restype = ctypes.c_int
    assert libc.usleep(round(seconds * 1_000_000)) == 0


@pytest.mark.parametrize('block', [time.sleep, native_gil_block])
def test_slow_scene_build_cannot_expire_first_reset_connection(tmp_path, block):
    spec, task, ctx = synthetic_episode(tmp_path)
    original_build = task.build

    def build(spec, ctx):
        block(.6)  # scaled equivalent of E1's 55–74s native scene build
        return original_build(spec, ctx)

    task = replace(task, build=build)
    with service() as (url, connections):
        adapter = XPolicyLabAdapter.g05_native(url, action_steps=1, timeout_s=2.)
        result = run_episode(spec, task, ctx, adapter, consumer_kind=ConsumerKind.DIAGNOSTIC,
                             limits=EpisodeLimits(4, 5))
        assert result.infra_valid, result.error
        assert result.ticks == 4
        assert result.policy_metadata['reset_confirmed'], result.error
        assert connections.value == 1


def test_transport_close_before_reset_never_opens_a_socket(monkeypatch):
    from mind2act_eval.adapters import g05_wire
    monkeypatch.setattr(g05_wire, 'connect', lambda *a, **k: pytest.fail('no connection before reset'))
    adapter = XPolicyLabAdapter.g05_native('ws://unused.invalid', action_steps=1)
    adapter.close()
    adapter.close()
