"""Public synthetic CLI wiring, never a native CodingAgent qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def worker(tmp_path, config, *, video_hz=6):
    config_path = tmp_path / "adapter.json"
    if config is not None:
        config_path.write_text(json.dumps(config))
    output = tmp_path / "episode"
    output.mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "."), str(ROOT / "tests"), str(ROOT / "examples"))))
    completed = subprocess.run(
        [sys.executable, "-m", "mind2act_eval.cli", "_worker", "--backend", "demo",
         "--spec", str(ROOT / "manifests/demo.json"), "--output", str(output),
         "--adapter", "mind2act_eval.adapters.coding_agent:create",
         *(["--adapter-config", str(config_path)] if config is not None else []),
         "--physics-ticks", "8", "--wall-s", "10", "--video-hz", str(video_hz)],
        env=env, text=True, capture_output=True, timeout=20,
    )
    result = json.loads((output / "result.json").read_text()) if (output / "result.json").exists() else None
    return completed, result


def test_actual_worker_json_demo(tmp_path):
    completed, result = worker(tmp_path, {
        "transport_factory": "coding_agent_cpu_demo:create",
    })
    assert completed.returncode == 0, completed.stderr
    assert result["infra_valid"] is True
    assert result["score"] == 100
    assert result["model_tested"] is False
    assert result["policy_metadata"]["native_coding_agent_verified"] is False
    assert result["video_metadata"]['video_hz'] == 6


def test_worker_recording_rate_does_not_modify_instance_rgb_clock(tmp_path):
    completed, result = worker(tmp_path, {"transport_factory": "coding_agent_cpu_demo:create"}, video_hz=10)
    assert completed.returncode == 0, completed.stderr
    assert result['video_metadata']['video_hz'] == 10
    spec = json.loads((tmp_path / 'episode' / 'spec.json').read_text())
    assert spec['clock']['rgb_hz'] == 30


def configured_fixture(tmp_path, mode="success"):
    return {"transport_factory": "coding_agent_fixture:create",
            "transport_config": {"trace": str(tmp_path / "trace.jsonl"), "mode": mode}}


def test_worker_reset_tool_14d_and_close_once(tmp_path):
    completed, result = worker(tmp_path, configured_fixture(tmp_path))
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["reset", "call_tool", "action", "call_tool", "action", "close"]
    assert events[0]["seed"] == 0
    assert events[1]["name"] == "act"
    assert events[1]["fields"] == sorted([
        "tick", "sim_time_s", "left_arm", "right_arm", "left_gripper", "right_gripper",
        "rgb", "instruction", "protocol_version",
    ])
    assert events[2]["width"] == 14
    assert result["score"] == 100


@pytest.mark.parametrize("mode", ["reset_error", "act_error", "timeout", "width", "nan", "gripper", "extra", "none", "close_error"])
def test_worker_failure_never_becomes_success(tmp_path, mode):
    completed, result = worker(tmp_path, configured_fixture(tmp_path, mode))
    assert completed.returncode == 1, completed.stderr
    assert result["infra_valid"] is False
    assert result["score"] is None
    assert result["model_tested"] is False
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "close" for e in events) == 1
    if mode == "timeout":
        assert result["termination"] == "wall_timeout"
        assert "TimeoutError" in result["error"]
    if mode == "reset_error":
        assert "original reset failure" in result["error"]
        assert "cleanup" not in result["error"]
        assert "synthetic cleanup failure" in result["policy_metadata"]["cleanup_error"]


@pytest.mark.parametrize("config", [None, {}, {"transport_factory": "not-a-factory"},
    {"transport_factory": "coding_agent_fixture:invalid"},
    {"transport_factory": "no_such_coding_agent_module:create"},
    {"transport_factory": "coding_agent_cpu_demo:create", "asset_root": "forbidden"},
    {"transport_factory": "coding_agent_cpu_demo:create", "transport_config": []}])
def test_worker_bad_configuration_fails_before_episode(tmp_path, config):
    completed, result = worker(tmp_path, config)
    assert completed.returncode == 2
    assert result is None
    assert completed.stderr


def test_public_projection_strips_extra_fields_and_array_subclass():
    import numpy as np
    from mind2act_eval.adapters.coding_agent import CodingAgentAdapter
    from mind2act_eval.contracts import Observation

    class PrivateArray(np.ndarray):
        pass

    class PrivateObservation(Observation):
        assets = {"never": "forward"}
        scorer = "private"
        evidence = object()

    class Transport:
        def reset(self, *, seed):
            pass

        def call_tool(self, name, *, observation):
            assert type(observation) is Observation
            assert not hasattr(observation, "assets")
            assert not hasattr(observation, "scorer")
            assert not hasattr(observation, "evidence")
            assert type(observation.rgb["front"]) is np.ndarray
            observation.rgb["front"][:] = 1
            return dict(left_arm=[0.] * 6, right_arm=[0.] * 6, left_gripper=0., right_gripper=1.)

        def close(self):
            pass

    rgb = np.zeros((2, 3, 3), np.uint8).view(PrivateArray)
    rgb.private_path = "never forward array metadata"
    observation = PrivateObservation(0, 0., (0.,)*6, (0.,)*6, 0., 1., rgb={"front": rgb})
    adapter = CodingAgentAdapter(Transport())
    adapter.reset(0)
    action = adapter.act(observation)
    assert len(action.left_arm) + len(action.right_arm) + 2 == 14
    assert not rgb.any()
    adapter.close()
    adapter.close()
    with pytest.raises(RuntimeError, match="reset"):
        adapter.act(observation)


def test_legacy_class_rejects_cli_dict_with_factory_hint():
    from mind2act_eval.adapters.coding_agent import CodingAgentAdapter
    with pytest.raises(TypeError, match="coding_agent:create"):
        CodingAgentAdapter({})
