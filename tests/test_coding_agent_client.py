import concurrent.futures
import json
from types import SimpleNamespace as NS

import pytest

from coding_payload_fixture import valid_input

from mind2act_eval.adapters.coding_agent_client import AttemptBudgetExceeded, LoggedNativeClient, reserve


def ledger(tmp_path, limit=3):
    path = tmp_path / "existing-ledger.json"
    path.write_text(json.dumps({"schema_version": 1, "limit": limit, "attempts": []}))
    return path


def test_existing_global_and_episode_caps_include_failed_reservations(tmp_path):
    path = ledger(tmp_path)
    assert reserve(path, 3, "a", 1)[0] == 1
    with pytest.raises(AttemptBudgetExceeded):
        reserve(path, 3, "a", 1)
    assert reserve(path, 3, "b", 3)[0] == 2
    assert reserve(path, 3, "b", 3)[0] == 3
    with pytest.raises(AttemptBudgetExceeded):
        reserve(path, 3, "c", 3)
    with pytest.raises(ValueError):
        reserve(path, 4, "c", 3)
    assert json.loads(path.read_text())["limit"] == 3


def test_missing_or_malformed_ledger_never_becomes_a_new_budget(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError):
        reserve(missing, 3, "a", 3)
    assert not missing.exists()
    path = ledger(tmp_path)
    doc = json.loads(path.read_text())
    doc["attempts"] = [{"attempt": 2, "episode": "a"}]
    path.write_text(json.dumps(doc))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        reserve(path, 3, "a", 3)
    assert path.read_bytes() == before


def test_run_bound_ledger_cannot_be_reused_by_another_episode(tmp_path):
    path = ledger(tmp_path)
    data = json.loads(path.read_text())
    data["authorized_episode"] = "authorized-run/PR1-easy"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="another authorized episode"):
        reserve(path, 3, "different-run/PR1-easy", 3)
    assert reserve(path, 3, "authorized-run/PR1-easy", 3)[0] == 1


def test_concurrent_reservations_cannot_exceed_original_hard_cap(tmp_path):
    path = ledger(tmp_path, 7)
    def attempt(index):
        try:
            return reserve(path, 7, "same-episode", 7)[0]
        except AttemptBudgetExceeded:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        numbers = list(pool.map(attempt, range(30)))
    assert sorted(n for n in numbers if n is not None) == list(range(1, 8))
    assert len(json.loads(path.read_text())["attempts"]) == 7


def test_one_upstream_callback_per_reserved_attempt_and_error_text_is_not_logged(tmp_path):
    path = ledger(tmp_path, 2)
    native = NS(model="local-fixture", reasoning_effort="medium", api_style="responses", max_retries=12, timeout_s=1., hard_timeout_s=2.)
    client = LoggedNativeClient(native, ledger=path, total_limit=2, episode="single", episode_limit=2,
                                attempt_dir=tmp_path / "attempts")
    calls = []
    def upstream(messages, tools, audit):
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("sensitive-provider-error-must-not-be-persisted")
        return {"model": "local-fixture", "usage": {"total_tokens": 9}, "id": "stub2"}
    client._request_once = upstream
    with pytest.raises(RuntimeError):
        client._upstream_once(*valid_input())
    assert client._upstream_once(*valid_input())["id"] == "stub2"
    with pytest.raises(AttemptBudgetExceeded):
        client._upstream_once(*valid_input())
    assert len(calls) == 2
    logs = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("attempt-*.json")]
    assert len(logs) == 2
    assert all(row["forwarded_requests"] == 1 for row in logs)
    assert sorted(row["completed"] for row in logs) == [False, True]
    assert "sensitive-provider-error" not in json.dumps(logs)
    assert client.result_metadata()["attempts_this_process"] == 2


def test_pin_loader_rejects_unapproved_manifest_before_any_import(tmp_path):
    from mind2act_eval.adapters.coding_agent_pin import load
    (tmp_path / "source-lock.json").write_text('{"commit": "synthetic"}')
    with pytest.raises(RuntimeError, match="source lock"):
        load(tmp_path)


def test_provider_redirect_is_one_failed_attempt_without_second_http_request(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import mind2act_eval.adapters.coding_agent_client as module
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.path)
            self.send_response(307)
            self.send_header("Location", "/forbidden-second-request")
            self.send_header("Content-Length", "0")
            self.end_headers()
        def log_message(self, *args): pass
    class InfrastructureFailure(BaseException):
        def __init__(self, message, *, retryable, status=None):
            super().__init__(message)
            self.retryable, self.status = retryable, status
    joint = NS(responses_base_url=lambda url: url, InfrastructureFailure=InfrastructureFailure)
    monkeypatch.setattr(module, "load", lambda source: (joint, None, None))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        native = NS(model="local-fixture", reasoning_effort="medium", api_style="responses", max_retries=12,
                    timeout_s=1., hard_timeout_s=2., azure_endpoint=f"http://127.0.0.1:{server.server_port}", api_key="synthetic-test-only",
                    _responses_request=lambda messages, tools: {"model": "local-fixture", "input": []})
        client = LoggedNativeClient(native, ledger=ledger(tmp_path, 1), total_limit=1, episode="one", episode_limit=1,
                                    attempt_dir=tmp_path / "attempts")
        client.source_root = "synthetic-test-pin"
        with pytest.raises(InfrastructureFailure):
            client._upstream_once(*valid_input())
        assert received == ["/responses"]
        log = json.loads(next((tmp_path / "attempts").glob("attempt-*.json")).read_text())
        assert log["http_status"] == 307 and log["completed"] is False
        assert len(json.loads((tmp_path / "existing-ledger.json").read_text())["attempts"]) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
