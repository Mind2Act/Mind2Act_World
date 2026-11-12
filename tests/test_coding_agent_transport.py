"""CPU transport fixtures: no remote API, SDK imports, GPU or paid requests."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from coding_payload_fixture import valid_input

import mind2act_eval.adapters.coding_agent_client as module
from mind2act_eval.adapters.coding_agent_audit import THOUGHT_BYPASS, THOUGHT_MARK
from mind2act_eval.adapters.coding_agent_eef import action_defaults, policy_defaults, DEFAULT_MAX_STEP
from mind2act_eval.adapters.coding_agent_planner_service import exclude_fixed_base
from test_coding_agent_client import ledger


class Failure(BaseException):
    def __init__(self, message, *, retryable, status=None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


def transport(tmp_path, monkeypatch, model, envelopes, limit=3):
    sent, sleeps = [], []
    response = {"id": "accepted", "model": model, "usage": {"total_tokens": 7},
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": "public decision synthetic-key", "tool_calls": [{"function": {
                        "name": "wait", "arguments": '{"steps":1}'}}]}}]}
    joint = NS(responses_base_url=lambda url: url, InfrastructureFailure=Failure,
               responses_to_chat_completion=lambda raw, **kwargs: response)
    monkeypatch.setattr(module, "load", lambda source: (joint, None, None))
    # Backoff belongs to this transport fixture. Mutating the stdlib module
    # would also capture unrelated asynchronous cleanup polling as a retry.
    monkeypatch.setattr(module, "time", NS(time=module.time.time, sleep=sleeps.append))
    class Worker:
        returncode = 0
        def __init__(self, *args, **kwargs):
            assert kwargs["env"]["MEMBENCH_NATIVE_HTTP_KEY"] == "synthetic-key"
        def communicate(self, raw, **kwargs):
            sent.append(json.loads(raw)["request"])
            return json.dumps(envelopes.pop(0)), ""
    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    payload = {"model": model, "input": [
        {"type": "function_call", "call_id": "call" + THOUGHT_MARK + "opaque-signature",
         "name": "wait", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call" + THOUGHT_MARK + "opaque-signature",
         "output": "waited synthetic-key"}]}
    native = NS(model=model, reasoning_effort="medium", api_style="responses", max_retries=12,
                timeout_s=300., hard_timeout_s=0., api_key="synthetic-key", azure_endpoint="https://fixture.invalid",
                reasoning_store={}, _responses_request=lambda messages, tools: payload)
    client = module.LoggedNativeClient(native, ledger=ledger(tmp_path, limit), total_limit=limit,
        episode="one", episode_limit=limit, attempt_dir=tmp_path / "attempts")
    client.source_root = "fixture"
    return client, payload, sent, sleeps


def test_transport_backoff_patch_does_not_replace_global_cleanup_sleep(tmp_path, monkeypatch):
    import time
    original_sleep = time.sleep
    _, _, _, sleeps = transport(tmp_path, monkeypatch, 'local-fixture', [])
    assert time.sleep is original_sleep
    module.time.sleep(30.)
    assert sleeps == [30.]


@pytest.mark.parametrize("model", ["gemini-3-pro", "deepseek", "gpt-6", "local-fixture"])
def test_success_rewrites_only_gemini_http_copy_and_records_metadata(tmp_path, monkeypatch, model):
    raw = {"status": "completed", "output": [{"type": "function_call", "status": "completed"}]}
    client, payload, sent, sleeps = transport(tmp_path, monkeypatch, model, [{"status": 200, "response": raw}])
    before = copy.deepcopy(payload)
    assert client._upstream_once(*valid_input())["id"] == "accepted"
    assert payload == before  # Native replay history is never rewritten.
    if "gemini" in model:
        assert sent[0]["input"][-1] == {"role": "user", "content": [{"type": "input_text", "text": "Continue."}]}
        assert all(row["call_id"] == "call" + THOUGHT_MARK + THOUGHT_BYPASS for row in sent[0]["input"][:-1])
        assert client.result_metadata()["gemini_transport"]["gemini_tool_tail_continues"] == 1
    else:
        assert json.dumps(sent[0]) == json.dumps(before)
        assert client.result_metadata()["gemini_transport"]["thought_signatures_bypassed"] == 0
    assert len(sent) == 1 and sleeps == []  # Accepted responses never back off or repeat.
    log = json.loads(next(client.directory.glob("attempt-*.json")).read_text())
    assert log["raw_shape"]["output"][0]["type"] == "function_call"
    assert log["reply"]["tool_calls"][0]["name"] == "wait"
    assert log["tool_feedback"][0]["output_text"] == "waited <key>"
    assert "synthetic-key" not in json.dumps(log)
    assert "opaque-signature" not in json.dumps(log)


def test_gemini_does_not_append_continue_after_other_turns():
    for tail in ({"role": "user", "content": []}, {"type": "function_call"}):
        payload = {"input": [tail]}
        assert module.gemini_compatibility(payload, "gemini-pro") == (0, False)
        assert payload["input"] == [tail]


def test_cjk_added_by_wire_builder_never_starts_http_worker(tmp_path, monkeypatch):
    client, payload, sent, sleeps = transport(tmp_path, monkeypatch, 'local-fixture', [])
    client.enable_input_protocol('mind2act-common_input-coding-v1')
    payload['input'][-1]['output'] = '\u9519\u8bef'
    with pytest.raises(ValueError, match='CJK'):
        client._upstream_once(*valid_input())
    assert not sent and not sleeps
    log = json.loads(next(client.directory.glob('attempt-*.json')).read_text())
    assert log['forwarded_requests'] == 0 and log['completed'] is False


@pytest.mark.parametrize('field', ['reply_summary', 'raw_shape'])
def test_diagnostic_failure_cannot_retry_an_accepted_reply(tmp_path, monkeypatch, field):
    client, _, sent, sleeps = transport(tmp_path, monkeypatch, 'local-fixture',
                                       [{'status': 200, 'response': {'output': []}}])
    def broken(*args):
        raise ValueError('synthetic-key private error')
    monkeypatch.setattr(module, field, broken)
    assert client._upstream_once(*valid_input())['id'] == 'accepted'
    assert len(sent) == client.used == 1 and sleeps == []
    log = json.loads(next(client.directory.glob('attempt-*.json')).read_text())
    assert log['completed'] is True
    assert log['reply' if field == 'reply_summary' else 'raw_shape'] == {'audit_error_type': 'ValueError'}
    assert 'private error' not in json.dumps(log)


@pytest.mark.parametrize("header,expected", [(None, 30.), ("99", 30.), ("2.5", 2.5),
    ("0", 0.), ("-1", 30.), ("nan", 30.), ("inf", 30.), ("invalid", 30.)])
def test_retry_after_is_finite_nonnegative_and_bounded(header, expected):
    assert module.rate_limit_wait(header) == expected


def test_429_is_one_failed_reservation_then_one_accepted_response(tmp_path, monkeypatch):
    client, _, sent, sleeps = transport(tmp_path, monkeypatch, "gemini-pro", [
        {"status": 429, "error_body": "provider echoed synthetic-key", "retry_after": "1000"},
        {"status": 200, "response": {"output": []}}], limit=2)
    with pytest.raises(Failure) as error:
        client._upstream_once(*valid_input())
    assert error.value.retryable and error.value.status == 429
    assert client._upstream_once(*valid_input())["id"] == "accepted"
    assert sleeps == [30.] and len(sent) == client.used == 2
    with pytest.raises(module.AttemptBudgetExceeded):
        client._upstream_once(*valid_input())
    assert len(sent) == 2 and sleeps == [30.]
    logs = [json.loads(path.read_text()) for path in sorted(client.directory.glob("attempt-*.json"))]
    assert [row["completed"] for row in logs] == [False, True]
    assert logs[0]["error_body"] == "provider echoed <key>"
    assert logs[0]["request_skeleton"]["input"][0]["name"] == "wait"
    assert "synthetic-key" not in json.dumps(logs)


def test_native_defaults_preserve_explicit_limits_and_other_profile_fields():
    env = {"L3_INSPECT_MODEL": "fixture", "L3_INSPECT_MAX_RETRIES": "0", "L3_INSPECT_DEPTH": "0"}
    assert policy_defaults(env) == {**env, "L3_INSPECT_DEPTH": "off", "L3_INSPECT_TIMEOUT_S": "300"}
    assert env["L3_INSPECT_DEPTH"] == "0"
    assert policy_defaults({"L3_INSPECT_TIMEOUT_S": "17"})["L3_INSPECT_TIMEOUT_S"] == "17"
    assert action_defaults({"max_step": [None] * 14})["max_step"] == DEFAULT_MAX_STEP
    explicit = [0.01] * 14
    assert action_defaults({"max_step": explicit})["max_step"] == explicit


def test_only_fixed_base_is_removed_from_planner_collisions():
    robot = {"robot_cfg": {"kinematics": {"base_link": "base", "collision_link_names": ["base", "arm1", "arm2"],
                                            "self_collision_ignore": {"arm1": ["arm2"]}}}}
    exclude_fixed_base(robot)
    kin = robot["robot_cfg"]["kinematics"]
    assert kin["collision_link_names"] == ["arm1", "arm2"]
    assert kin["self_collision_ignore"] == {"arm1": ["arm2"]}


def test_http_error_redacts_before_body_truncation(monkeypatch):
    import io
    import urllib.error
    from mind2act_eval.adapters import coding_agent_http as worker
    body = "x" * 1995 + "synthetic-key" + " tail"
    class Opener:
        def open(self, *args, **kwargs):
            raise urllib.error.HTTPError("https://fixture.invalid", 400, "fixture", {"Retry-After": "9"},
                                         io.BytesIO(body.encode()))
    monkeypatch.setenv("FIXTURE_KEY", "synthetic-key")
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO(json.dumps({"key_env": "FIXTURE_KEY",
        "url": "https://fixture.invalid", "request": {}, "timeout_s": 1})))
    output = io.StringIO()
    monkeypatch.setattr(worker.sys, "stdout", output)
    worker.main()
    envelope = json.loads(output.getvalue())
    assert envelope["error_body"] == "x" * 1995 + "<key>"
    assert envelope["retry_after"] == "9" and envelope["status"] == 400


def test_defaults_reach_both_native_policy_and_client_factory(monkeypatch):
    from mind2act_eval.adapters import coding_agent_eef as module
    seen = []
    monkeypatch.setattr(module, "_factory", lambda name, config: seen.append((name, config)) or object())
    monkeypatch.setattr(module, "EefIntegration", lambda config, *resources: config)
    monkeypatch.setattr(module, "NativeCodingAgentAdapter", lambda integration: integration)
    config = {"source_root": "fixture", "urdf": "fixture", "mounts": {}, "physics_ticks": 600,
              "camera_map": {}, "action_spec": {"max_step": [None] * 14}, "policy_env": {},
              "client_factory": "mind2act_eval.adapters.coding_agent_client:create",
              "client_config": {"policy_env": {}}, "planner_factory": "fixture:planner", "planner_config": {}}
    effective = module.create(config)
    assert effective["action_spec"]["max_step"] == DEFAULT_MAX_STEP
    assert effective["policy_env"] == seen[0][1]["policy_env"] == {
        "L3_INSPECT_DEPTH": "off", "L3_INSPECT_TIMEOUT_S": "300"}
    assert config["policy_env"] == {} and config["action_spec"]["max_step"] == [None] * 14
