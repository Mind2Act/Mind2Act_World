"""Native provider client with an existing, immutable-cap attempt ledger.

Credentials come only from the explicitly named environment variable loaded by
the site's existing runtime wrapper. No login, credential discovery or ledger
creation/reset is performed. Every native retry reserves another paid attempt.
"""
import fcntl
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from mind2act_eval.adapters.roboprobe.pin import load
from mind2act_eval.adapters.agent_payload import require_english, validate_input
from mind2act_eval.adapters.roboprobe.audit import (
    THOUGHT_MARK, THOUGHT_BYPASS, redacted, text, tool_feedback,
    reply_summary, payload_skeleton, raw_shape, record,
)

RATE_LIMIT_WAIT_S = 30.0


def gemini_compatibility(payload, model):
    """Rewrite only the Gemini HTTP copy; leave the native conversation intact."""
    bypassed, continued = 0, False
    if "gemini" in model.lower():
        for item in payload.get("input", []):
            if (isinstance(item, dict) and item.get("type") in ("function_call", "function_call_output")
                    and isinstance(item.get("call_id"), str) and THOUGHT_MARK in item["call_id"]):
                item["call_id"] = item["call_id"].split(THOUGHT_MARK, 1)[0] + THOUGHT_MARK + THOUGHT_BYPASS
                bypassed += 1
        items = payload.get("input", [])
        if items and isinstance(items[-1], dict) and items[-1].get("type") == "function_call_output":
            items.append({"role": "user", "content": [{"type": "input_text", "text": "Continue."}]})
            continued = True
    return bypassed, continued


def rate_limit_wait(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return RATE_LIMIT_WAIT_S
    return min(seconds, RATE_LIMIT_WAIT_S) if math.isfinite(seconds) and seconds >= 0 else RATE_LIMIT_WAIT_S


class AttemptBudgetExceeded(RuntimeError):
    pass


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def reserve(path, limit, episode, episode_limit):
    """Interoperate with the repository's original schema-v1 upstream ledger."""
    path = Path(path)
    if (not path.is_file() or path.is_symlink() or type(limit) is not int or limit <= 0
            or episode_limit is not None and (type(episode_limit) is not int or episode_limit <= 0)
            or not isinstance(episode, str) or not episode):
        raise ValueError("an existing explicit ledger, episode identity and positive limits are required")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or data.get("limit") != limit or not isinstance(data.get("attempts"), list):
            raise ValueError("existing attempt ledger schema/cap differs; never reset or change it")
        if data.get("authorized_episode", episode) != episode:
            raise ValueError("attempt ledger is bound to another authorized episode")
        attempts = data["attempts"]
        if any(not isinstance(row, dict) or row.get("attempt") != index + 1 or not isinstance(row.get("episode"), str)
               for index, row in enumerate(attempts)):
            raise ValueError("existing attempt ledger is malformed")
        if len(attempts) >= limit or (episode_limit is not None
                and sum(row["episode"] == episode for row in attempts) >= episode_limit):
            raise AttemptBudgetExceeded("upstream attempt budget exhausted")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        number = len(attempts) + 1
        attempts.append({"attempt": number, "episode": episode, "reserved_at": time.time()})
        _write(path, data)
        return number, before


class LoggedNativeClient:
    def __init__(self, client, *, ledger, total_limit, episode, episode_limit, attempt_dir):
        self.client = client
        self.ledger, self.total_limit = Path(ledger), total_limit
        self.episode, self.episode_limit = episode, episode_limit
        self.directory = Path(attempt_dir)
        # Inspect without reserving/spending during configuration admission.
        if not self.ledger.is_file():
            raise ValueError("existing cumulative upstream attempt ledger is required")
        doc = json.loads(self.ledger.read_text())
        if doc.get("schema_version") != 1 or doc.get("limit") != total_limit or not isinstance(doc.get("attempts"), list):
            raise ValueError("existing ledger cap/schema mismatch")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.used = 0
        self._input_protocol = None
        self._decision_budget = None
        self._native_complete = getattr(client, 'complete', None)
        self.compatibility = {"thought_signatures_bypassed": 0, "gemini_tool_tail_continues": 0}
        self.client._complete_fn = self._upstream_once

    @property
    def native_client(self):
        # Keep native isinstance checks, session resets and reasoning replay.
        return self.client

    def _upstream_once(self, messages, tools):
        validate_input(messages, tools)
        if self._decision_budget is not None:
            self._decision_budget.ensure_available()
        number, before = reserve(self.ledger, self.total_limit, self.episode,
                                 None if self._decision_budget is not None else self.episode_limit)
        self.used += 1
        path = self.directory / f"attempt-{number:05d}.json"
        audit = {"attempt": number, "episode": self.episode, "started_at": time.time(),
                 "ledger_sha256_before_reservation": before, "forwarded_requests": 0,
                 "max_upstream_requests": 1, "requested_model": self.client.model,
                 "requested_effort": self.client.reasoning_effort, "transport": self.client.api_style,
                 "usage": None}
        _write(path, audit)
        try:
            audit["forwarded_requests"] = 1
            _write(path, audit)
            response = self._request_once(messages, tools, audit)
            if self._decision_budget is not None:
                self._decision_budget.decision()
            audit.update(returned_model=response.get("model"), usage=response.get("usage"),
                         response_id=response.get("id"), completed=True)
            record(audit, "reply", reply_summary, response, getattr(self.client, "api_key", ""))
            return response
        except BaseException as error:
            # Provider error text may echo a key. Preserve only typed status.
            audit.update(error_type=type(error).__name__, status=getattr(error, "status", None),
                         retryable=getattr(error, "retryable", None), completed=False)
            raise
        finally:
            audit["finished_at"] = time.time()
            _write(path, redacted(audit, getattr(self.client, "api_key", "")))

    def _request_once(self, messages, tools, audit):
        # Reuse the actual native Responses builder and reasoning replay codec.
        # urllib has no retry, and redirects are explicitly refused. One
        # reservation therefore permits at most one upstream HTTP request.
        joint, _, _ = load(self.source_root)
        payload = copy.deepcopy(self.client._responses_request(messages, tools))
        record(audit, "tool_feedback", tool_feedback, payload, self.client.api_key)
        bypassed, continued = gemini_compatibility(payload, self.client.model)
        audit.update(thought_signatures_bypassed=bypassed, gemini_tool_tail_continue=continued)
        self.compatibility["thought_signatures_bypassed"] += bypassed
        self.compatibility["gemini_tool_tail_continues"] += int(continued)
        try:
            require_english(payload)
            if self._input_protocol is not None:
                self._capture_input(payload, audit)
        except BaseException:
            audit["forwarded_requests"] = 0
            raise
        key_env = "MEMBENCH_NATIVE_HTTP_KEY"
        worker_env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR") if name in os.environ}
        worker_env[key_env] = self.client.api_key
        request = {"url": joint.responses_base_url(self.client.azure_endpoint).rstrip("/") + "/responses",
                   "request": payload, "key_env": key_env, "timeout_s": self.client.timeout_s}
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("http_worker.py"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=worker_env)
            try:
                stdout, _ = process.communicate(json.dumps(request, allow_nan=False),
                    timeout=self.client.hard_timeout_s or None)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise joint.InfrastructureFailure("upstream hard timeout", retryable=True) from None
            if process.returncode:
                raise joint.InfrastructureFailure("upstream HTTP worker failed", retryable=True)
            envelope = json.loads(stdout)
            status = envelope.get("status")
            audit["http_status"] = status
            if status != 200:
                audit["error_body"] = text(envelope.get("error_body", ""), self.client.api_key, 2000)
                record(audit, "request_skeleton", payload_skeleton, payload, self.client.api_key)
                if status == 429:
                    delay = rate_limit_wait(envelope.get("retry_after"))
                    audit["rate_limit_wait_s"] = delay
                    time.sleep(delay)
                raise joint.InfrastructureFailure("upstream HTTP failure", retryable=status in (None, 429, 500, 502, 503, 504), status=status)
            raw = envelope["response"]
            record(audit, "raw_shape", raw_shape, raw, self.client.api_key)
        except (OSError, ValueError, KeyError) as error:
            raise joint.InfrastructureFailure("upstream " + type(error).__name__, retryable=True) from None
        return joint.responses_to_chat_completion(raw, reasoning_store=self.client.reasoning_store)

    def complete(self, messages, tools):
        if self._decision_budget is not None:
            self._decision_budget.ensure_available()
        # The pinned SDK retries retryable infrastructure failures internally.
        # Only a complete provider reply is charged by _upstream_once. Exhausted
        # retries propagate as infrastructure failure, never a task-budget stop.
        return self._native_complete(messages, tools)

    def bind_decision_budget(self, budget):
        from mind2act_eval.budgets import DecisionBudget
        if not isinstance(budget, DecisionBudget) or self._decision_budget is not None:
            raise ValueError('a fresh framework decision budget must be bound exactly once')
        if not callable(self._native_complete):
            raise ValueError('native complete method is required for decision accounting')
        self._decision_budget = budget
        self.client.max_retries = budget.retry_limit
        # EefAgentPolicy retains the native object for SDK isinstance checks.
        # Guard logical calls; count complete provider replies, not retries.
        self.client.complete = self.complete
        self.client._mind2act_decision_budget = budget

    def enable_input_protocol(self, version):
        from ..agent_input.templates import VERSION, CP07_TEST_PROTOCOL, CP07_TRAIN_LAYOUT_PROTOCOL
        if version not in (VERSION, CP07_TEST_PROTOCOL, CP07_TRAIN_LAYOUT_PROTOCOL):
            raise ValueError('Unknown tool-agent input protocol')
        if self.used or self._input_protocol is not None:
            raise ValueError('A fresh client is required for the tool-agent input protocol')
        self._input_protocol = version

    def _capture_input(self, payload, audit):
        """Full sent text/tools and content-addressed public images, never auth."""
        import base64
        raw = json.dumps(payload, allow_nan=False, ensure_ascii=False)
        if self.client.api_key and self.client.api_key in raw:
            raise ValueError('Credentials must not enter model input')
        images = []
        def project(value):
            if isinstance(value,str) and value.startswith('data:image/'):
                header, encoded = value.split(',',1)
                image_bytes = base64.b64decode(encoded,validate=True)
                digest = hashlib.sha256(image_bytes).hexdigest()
                directory = self.directory/'input-images'
                directory.mkdir(exist_ok=True)
                path = directory/digest
                if not path.exists():path.write_bytes(image_bytes)
                images.append(dict(path=str(path),sha256=digest,mime=header[5:].split(';')[0]))
                return 'image-sha256:'+digest
            if isinstance(value,dict):return {k:project(v) for k,v in value.items()}
            if isinstance(value,list):return [project(v) for v in value]
            return value
        document = dict(schema='mind2act.sent_common_input.v1',input_protocol=self._input_protocol,
            payload_sha256=hashlib.sha256(raw.encode()).hexdigest(),payload=project(payload),images=images,
            image_count=len(images),capture_point='after native encoding and provider compatibility; before HTTP',
            credentials_included=False)
        path = self.directory/f"attempt-{audit['attempt']:05d}.input.json"
        _write(path,document)
        audit['input_archive'] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            image_count=len(images),payload_sha256=document['payload_sha256'])

    def result_metadata(self):
        return {"attempt_ledger": str(self.ledger), "total_attempt_limit": self.total_limit,
                "episode_attempt_limit": None if self._decision_budget is not None else self.episode_limit,
                "episode": self.episode,
                "attempt_dir": str(self.directory), "attempts_this_process": self.used,
                "native_request_retries": self.client.max_retries,
                "gemini_transport": {"enabled": "gemini" in self.client.model.lower(),
                                     "thought_signature_policy": THOUGHT_BYPASS,
                                     "tool_tail_user_text": "Continue.", **self.compatibility},
                "provider_request_retries": 0, "max_upstream_requests_per_attempt": 1,
                "http_timeout_s": self.client.timeout_s, "hard_timeout_s": self.client.hard_timeout_s}

    def close(self):
        pass


def create(config):
    if isinstance(config, dict) and 'source_root' not in config:
        from mind2act_eval.adapters.roboprobe.pin import resolve_source_root
        config = dict(config, source_root=str(resolve_source_root()))
    required = {"source_root", "policy_env", "key_env", "attempt_ledger", "total_attempt_limit",
                "episode", "attempt_dir"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"episode_attempt_limit"}:
        raise ValueError("explicit source/env/key-env/existing-ledger/limits/episode/attempt-dir required")
    joint, _, _ = load(config["source_root"])
    from mind2act_eval.adapters.roboprobe.eef import policy_defaults
    env = policy_defaults(config["policy_env"])
    if (env.get("L3_INSPECT_API_STYLE") != "responses"
            or env.get("L3_INSPECT_API_KEY_ENV") != config["key_env"]):
        raise ValueError("native Responses and one explicit runtime key environment are required")
    if not isinstance(config["key_env"], str) or not config["key_env"].isidentifier():
        raise ValueError("key_env must name one existing runtime environment variable")
    key = os.environ.get(config["key_env"], "").strip()
    if not key:
        raise ValueError("explicit runtime credential environment is unset or empty")
    env[config["key_env"]] = key
    client = joint.AzureAgentClient.from_env(env)
    result = LoggedNativeClient(client, ledger=config["attempt_ledger"], total_limit=config["total_attempt_limit"],
        episode=config["episode"], episode_limit=config.get("episode_attempt_limit"), attempt_dir=config["attempt_dir"])
    result.source_root = config["source_root"]
    return result
