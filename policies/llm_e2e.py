"""E2E 路线 LLM 关节空间 adapter。

模型每次决策返回若干绝对关节路点；adapter 按单行步长上限把路点线性展开为
控制行，在 adapter 内排队。act 每次返回一行，peek_next_action 返回同块下一行
（块末返回 None，由 runner 保持）。不依赖 RoboProbe SDK、cuRobo，也不发送
extra_headers。密钥只从 key_env 指定的环境变量读取，不进入任何日志。
具体服务商 profile 见 duclaw.py；这是冒烟/对比用 adapter，不是基准提交。
"""
import base64
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from mind2act_eval.contracts import Action, PolicyStop

ADAPTER_VERSION = "llm-e2e-joint-v1"
LEDGER_SCHEMA = "mind2act-llm-e2e-ledger/v1"
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)

DUCLAW = dict(provider="duclaw", api_style="responses", model="A6A",
              base_url="https://duclaw-gateway.baidubce.com/star/openai/v1")

DEFAULTS = dict(
    key_env="MIND2ACT_API_KEY",
    reasoning_effort=None,
    timeout_s=180,
    request_retries=2,            # 每次决策首次请求之外的基础设施重试，均占用 ledger
    parse_retries=1,              # 回复无法解析时带错误反馈重问的次数，均占用 ledger
    max_decisions=None,           # 单局决策上限；达到后 PolicyStop，按当时物理状态计分
    max_waypoints=8,
    max_rows_per_decision=150,    # 30Hz 下 5 s
    fallback_hold_rows=30,        # 解析重试耗尽时保持当前指令的行数
    max_step_rad=0.05,            # 每控制行关节步长上限，与 roboprobe_eef 默认一致
    max_step_gripper=0.25,
    joint_limits=None,            # 可选 {"lower": [6], "upper": [6]}，两臂共用
    include_images=True,
    image_format="jpeg",          # jpeg | png
    jpeg_quality=90,
    save_images=False,
    history_turns=2,              # 只保留最近若干次决策的文本回复，不重发历史图像
    max_output_tokens=4096,
    base_url=None,                # 覆盖 profile 地址；仅用于本地桩服务测试
    model=None,
)
REQUIRED = ("log_dir", "ledger", "ledger_limit")


class LedgerExhausted(RuntimeError):
    pass


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def reserve(path, limit, run):
    """持久化累计请求账本：每次 HTTP 请求前预留一次，失败请求同样计入，永不重置。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("schema") != LEDGER_SCHEMA or data.get("limit") != limit:
                raise ValueError("existing ledger schema/limit differs; never reset or change it")
        else:
            data = dict(schema=LEDGER_SCHEMA, limit=limit, attempts=[])
        attempts = data["attempts"]
        if len(attempts) >= limit:
            raise LedgerExhausted(f"paid request ledger exhausted ({len(attempts)}/{limit})")
        number = len(attempts) + 1
        attempts.append(dict(attempt=number, run=run, reserved_at=time.time()))
        _write_json(path, data)
        return number


def encode_image(array, image_format, quality):
    import numpy as np
    from PIL import Image
    image = Image.fromarray(np.ascontiguousarray(np.asarray(array, dtype=np.uint8)))
    buffer = io.BytesIO()
    if image_format == "png":
        image.save(buffer, format="PNG")
        mime = "image/png"
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        mime = "image/jpeg"
    data = buffer.getvalue()
    return mime, data


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class HttpFailure(RuntimeError):
    def __init__(self, message, *, status=None, retryable=False, retry_after=None, body=""):
        super().__init__(message)
        self.status, self.retryable = status, retryable
        self.retry_after, self.body = retry_after, body


def post_json(url, body, key, timeout_s):
    request = urllib.request.Request(url, data=json.dumps(body, allow_nan=False).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        text = error.read(65536).decode("utf-8", errors="replace").replace(key, "<key>")[:2000]
        raise HttpFailure(f"HTTP {error.code}", status=error.code,
                          retryable=error.code in RETRYABLE_STATUS,
                          retry_after=error.headers.get("Retry-After"), body=text) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        raise HttpFailure(type(error).__name__, retryable=True) from None


SYSTEM_PROMPT = """You control a dual-arm robot (two 6-DoF X5 arms) in a physics simulation.
Each arm is commanded by six absolute joint angles in radians, ordered joint1..joint6.
Grippers are normalized: 0 = fully closed, 1 = fully open.
The controller runs at {control_hz} Hz. Each control row may move a joint by at most
{max_step_rad} rad and a gripper by at most {max_step_gripper}; larger targets are reached
gradually over several rows. Joint limits: {limits}.
At every decision you receive the task instruction, the current joint state, the
remaining control budget and the latest camera images. Reply with ONE JSON object and
nothing else:
{{"note": "<one short sentence explaining the plan>",
  "waypoints": [{{"left_arm": [6 numbers], "right_arm": [6 numbers],
                 "left_gripper": number, "right_gripper": number, "hold_rows": integer}}]}}
Waypoints (1 to {max_waypoints}) are executed in order. After reaching a waypoint the
command is held for hold_rows extra rows (0 to {max_rows}). One decision executes at most
{max_rows} rows in total; then you are asked again with new images.
Keep an arm still by repeating its current joints."""


def _round(values):
    return [round(float(v), 4) for v in values]


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reply contains no JSON object")
    return json.loads(text[start:end + 1])


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_waypoints(text, max_waypoints, max_rows):
    reply = _extract_json(text)
    if not isinstance(reply, dict) or not isinstance(reply.get("waypoints"), list):
        raise ValueError("reply must be an object with a waypoints list")
    waypoints = reply["waypoints"]
    if not 1 <= len(waypoints) <= max_waypoints:
        raise ValueError(f"waypoints must contain 1..{max_waypoints} items")
    parsed = []
    for index, item in enumerate(waypoints):
        if not isinstance(item, dict):
            raise ValueError(f"waypoint {index} must be an object")
        row = {}
        for arm in ("left_arm", "right_arm"):
            values = item.get(arm)
            if not isinstance(values, list) or len(values) != 6:
                raise ValueError(f"waypoint {index} {arm} must list six joint radians")
            row[arm] = [_finite(v, f"waypoint {index} {arm}") for v in values]
        for gripper in ("left_gripper", "right_gripper"):
            row[gripper] = min(1.0, max(0.0, _finite(item.get(gripper), f"waypoint {index} {gripper}")))
        hold = item.get("hold_rows", 0)
        if isinstance(hold, bool) or not isinstance(hold, int) or not 0 <= hold <= max_rows:
            raise ValueError(f"waypoint {index} hold_rows must be an integer 0..{max_rows}")
        row["hold_rows"] = hold
        parsed.append(row)
    note = reply.get("note")
    return parsed, note if isinstance(note, str) else ""


def _vector(left_arm, right_arm, left_gripper, right_gripper):
    return list(left_arm) + list(right_arm) + [left_gripper, right_gripper]


def _action(vector):
    return Action(vector[0:6], vector[6:12], vector[12], vector[13])


def expand(start, waypoints, *, max_step_rad, max_step_gripper, max_rows, limits):
    """把绝对路点按单行步长上限线性展开为 14 维控制行；超过 max_rows 截断。"""
    rows, current, truncated = [], list(start), False
    for waypoint in waypoints:
        target = _vector(waypoint["left_arm"], waypoint["right_arm"],
                         waypoint["left_gripper"], waypoint["right_gripper"])
        if limits is not None:
            lower, upper = limits
            for offset in (0, 6):
                for j in range(6):
                    target[offset + j] = min(upper[j], max(lower[j], target[offset + j]))
        steps = max(1, max(math.ceil(abs(t - c) / max_step_rad - 1e-9)
                           for t, c in zip(target[:12], current[:12])),
                    max(math.ceil(abs(t - c) / max_step_gripper - 1e-9)
                        for t, c in zip(target[12:], current[12:])))
        for k in range(1, steps):
            row = [c + (t - c) * k / steps for c, t in zip(current, target)]
            # 浮点插值可能把 0/1 夹爪算成 -2.78e-17 等越界值，contracts._gripper 会拒绝
            row[12:] = [min(1.0, max(0.0, v)) for v in row[12:]]
            rows.append(row)
        rows.append(list(target))  # 末行精确落在路点上
        rows.extend(list(target) for _ in range(waypoint["hold_rows"]))
        current = target
        if len(rows) >= max_rows:
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            break
    return rows, truncated


def _host(url):
    return urllib.parse.urlsplit(url).netloc


def _reply_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    texts = []
    for item in response.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    texts.append(part.get("text", ""))
    return "".join(texts)


class LlmJointAdapter:
    requires_rgb = True

    def __init__(self, config, profile):
        if not isinstance(config, dict):
            raise ValueError("adapter config must be a JSON object")
        unknown = set(config) - set(DEFAULTS) - set(REQUIRED)
        missing = [name for name in REQUIRED if name not in config]
        if unknown or missing:
            raise ValueError(f"adapter config unknown={sorted(unknown)} missing={missing}")
        self.cfg = {**DEFAULTS, **config}
        if type(self.cfg["ledger_limit"]) is not int or self.cfg["ledger_limit"] <= 0:
            raise ValueError("ledger_limit must be an explicit positive integer")
        if self.cfg["image_format"] not in ("jpeg", "png"):
            raise ValueError("image_format must be jpeg or png")
        if not str(self.cfg["key_env"]).isidentifier():
            raise ValueError("key_env must name one environment variable")
        self._key = os.environ.get(self.cfg["key_env"], "").strip()
        if not self._key:
            raise ValueError("credential environment variable is unset or empty")
        limits = self.cfg["joint_limits"]
        self.joint_limits = None if limits is None else (
            [float(v) for v in limits["lower"]], [float(v) for v in limits["upper"]])
        self.profile = dict(profile)
        self.model = self.cfg["model"] or profile["model"]
        self.api_style = profile["api_style"]
        base = (self.cfg["base_url"] or profile["base_url"]).rstrip("/")
        self.url = base + "/responses"
        self.run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{os.urandom(3).hex()}"
        self.directory = Path(self.cfg["log_dir"]) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.ledger = Path(self.cfg["ledger"])
        if self.ledger.exists():
            # 准入时只检查不预留：既有账本上限不同即拒绝，永不重置或修改
            data = json.loads(self.ledger.read_text())
            if data.get("schema") != LEDGER_SCHEMA or data.get("limit") != self.cfg["ledger_limit"]:
                raise ValueError("existing ledger schema/limit differs; never reset or change it")
        self.queue, self.history, self.command = [], [], None
        self.stats = dict(decisions=0, requests=0, failed_requests=0, parse_failures=0,
                          fallback_holds=0, truncated_decisions=0, rows_planned=0,
                          rows_executed=0, model_request_s=0.0, input_tokens=0, output_tokens=0)
        self.clock = None
        self.physics_ticks = None

    # ---- 契约钩子 ----
    def reset(self, seed):
        self.seed = int(seed)
        self.queue, self.history, self.command = [], [], None

    def validate_clock(self, clock):
        if clock.control_hz != 30 or clock.physics_hz % clock.control_hz:
            raise ValueError("llm e2e adapter requires control_hz=30 and integer substeps")
        self.clock = clock

    def validate_limits(self, limits):
        self.physics_ticks = limits.physics_ticks
        self.policy_kind = limits.policy_kind
        self.budget_version = limits.budget_version

    def act(self, observation):
        if not self.queue:
            self._decide(observation)
        row = self.queue.pop(0)
        self.command = row
        self.stats["rows_executed"] += 1
        return _action(row)

    def peek_next_action(self):
        return _action(self.queue[0]) if self.queue else None

    def close(self):
        pass

    # ---- 决策 ----
    def _remaining_rows(self, tick):
        if self.physics_ticks is None or self.clock is None:
            return None
        return max(0, (self.physics_ticks - tick) // (self.clock.physics_hz // self.clock.control_hz))

    def _state_text(self, observation, remaining):
        state = dict(left_arm=_round(observation.left_arm), right_arm=_round(observation.right_arm),
                     left_gripper=round(observation.left_gripper, 4),
                     right_gripper=round(observation.right_gripper, 4))
        lines = [f"Task instruction: {observation.instruction or '(none)'}",
                 f"Simulation time: {observation.sim_time_s:.3f} s; decision {self.stats['decisions'] + 1}.",
                 f"Current joint state: {json.dumps(state)}"]
        if remaining is not None:
            lines.append(f"Remaining control rows in the episode: {remaining}.")
        if self.cfg["max_decisions"] is not None:
            lines.append(f"Remaining decisions: {self.cfg['max_decisions'] - self.stats['decisions']}.")
        return "\n".join(lines)

    def _images(self, observation, number):
        images = []
        if not self.cfg["include_images"]:
            return images
        for name in sorted(observation.rgb):
            mime, data = encode_image(observation.rgb[name], self.cfg["image_format"], self.cfg["jpeg_quality"])
            digest = hashlib.sha256(data).hexdigest()
            if self.cfg["save_images"]:
                folder = self.directory / "images"
                folder.mkdir(exist_ok=True)
                (folder / f"d{number:05d}-{name}.{self.cfg['image_format']}").write_bytes(data)
            images.append(dict(camera=name, mime=mime, sha256=digest, bytes=len(data),
                               url=f"data:{mime};base64," + base64.b64encode(data).decode()))
        return images

    def _system(self):
        lower_upper = "not provided" if self.joint_limits is None else json.dumps(
            dict(lower=self.joint_limits[0], upper=self.joint_limits[1]))
        return SYSTEM_PROMPT.format(control_hz=self.clock.control_hz if self.clock else 30,
            max_step_rad=self.cfg["max_step_rad"], max_step_gripper=self.cfg["max_step_gripper"],
            limits=lower_upper, max_waypoints=self.cfg["max_waypoints"],
            max_rows=self.cfg["max_rows_per_decision"])

    def _body(self, text, images, feedback):
        history = self.history[-2 * self.cfg["history_turns"]:] if self.cfg["history_turns"] else []
        content = [dict(type="input_text", text=text)]
        for image in images:
            content += [dict(type="input_text", text=f"Camera {image['camera']}:"),
                        dict(type="input_image", image_url=image["url"])]
        items = []
        for role, value in history:
            kind = "output_text" if role == "assistant" else "input_text"
            items.append(dict(role=role, content=[dict(type=kind, text=value)]))
        items.append(dict(role="user", content=content))
        for reply, error in feedback:
            items += [dict(role="assistant", content=[dict(type="output_text", text=reply)]),
                      dict(role="user", content=[dict(type="input_text", text=error)])]
        body = dict(model=self.model, instructions=self._system(), input=items,
                    max_output_tokens=self.cfg["max_output_tokens"])
        if self.cfg["reasoning_effort"]:
            body["reasoning"] = dict(effort=self.cfg["reasoning_effort"])
        return body

    def _skeleton(self, body):
        """审计用请求体：图像替换为摘要，不含任何认证信息。"""
        def project(value):
            if isinstance(value, str) and value.startswith("data:image/"):
                return "image-sha256:" + hashlib.sha256(base64.b64decode(value.split(",", 1)[1])).hexdigest()
            if isinstance(value, dict):
                return {k: project(v) for k, v in value.items()}
            if isinstance(value, list):
                return [project(v) for v in value]
            return value
        return project(body)

    def _request(self, body, audit):
        """一次决策内的请求，含基础设施重试；每次 HTTP 请求前先在 ledger 预留。"""
        last = None
        for attempt in range(self.cfg["request_retries"] + 1):
            number = reserve(self.ledger, self.cfg["ledger_limit"], self.run_id)
            self.stats["requests"] += 1
            record = dict(ledger_attempt=number, started_at=time.time())
            audit["requests"].append(record)
            before = time.monotonic()
            try:
                status, response = post_json(self.url, body, self._key, self.cfg["timeout_s"])
                record.update(status=status, response_id=response.get("id"),
                              returned_model=response.get("model"), usage=response.get("usage"))
                return response
            except HttpFailure as error:
                self.stats["failed_requests"] += 1
                record.update(status=error.status, error=str(error), retryable=error.retryable,
                              error_body=error.body.replace(self._key, "<key>"))
                last = error
                if not error.retryable:
                    break
                if error.status == 429 and attempt < self.cfg["request_retries"]:
                    try:
                        delay = min(30.0, max(0.0, float(error.retry_after)))
                    except (TypeError, ValueError):
                        delay = 30.0
                    record["rate_limit_wait_s"] = delay
                    time.sleep(delay)
            finally:
                elapsed = time.monotonic() - before
                record["wall_s"] = elapsed
                self.stats["model_request_s"] += elapsed
        raise RuntimeError(f"model request failed after retries: {last}")

    def _decide(self, observation):
        limit = self.cfg["max_decisions"]
        if limit is not None and self.stats["decisions"] >= limit:
            raise PolicyStop("give_up", f"adapter decision limit {limit} reached")
        number = self.stats["decisions"] + 1
        start = self.command or _vector(observation.left_arm, observation.right_arm,
                                        observation.left_gripper, observation.right_gripper)
        remaining = self._remaining_rows(observation.tick)
        max_rows = self.cfg["max_rows_per_decision"]
        if remaining:
            max_rows = min(max_rows, remaining)
        text = self._state_text(observation, remaining)
        images = self._images(observation, number)
        audit = dict(decision=number, tick=observation.tick, sim_time_s=observation.sim_time_s,
                     images=[{k: v for k, v in i.items() if k != "url"} for i in images],
                     requests=[], replies=[], started_at=time.time())
        feedback, rows = [], None
        try:
            for _ in range(self.cfg["parse_retries"] + 1):
                body = self._body(text, images, feedback)
                if not feedback:
                    audit["request"] = self._skeleton(body)
                response = self._request(body, audit)
                usage = response.get("usage") or {}
                self.stats["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                self.stats["output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
                reply = _reply_text(response)
                audit["replies"].append(reply)
                try:
                    # hold_rows 按配置上限校验；超出本局剩余行数的部分由 expand 截断
                    waypoints, note = parse_waypoints(reply, self.cfg["max_waypoints"],
                                                      self.cfg["max_rows_per_decision"])
                except (ValueError, KeyError, TypeError) as error:
                    self.stats["parse_failures"] += 1
                    feedback.append((reply, f"Invalid reply: {error}. Reply again with only the JSON object."))
                    continue
                rows, truncated = expand(start, waypoints, max_step_rad=self.cfg["max_step_rad"],
                    max_step_gripper=self.cfg["max_step_gripper"], max_rows=max_rows,
                    limits=self.joint_limits)
                self.stats["truncated_decisions"] += int(truncated)
                audit.update(note=note, waypoints=waypoints, rows=len(rows), truncated=truncated)
                self.history += [("user", text), ("assistant", reply)]
                break
            if rows is None:
                self.stats["fallback_holds"] += 1
                rows = [list(start) for _ in range(min(self.cfg["fallback_hold_rows"], max_rows) or 1)]
                audit.update(fallback="hold_current_command", rows=len(rows))
        finally:
            audit["finished_at"] = time.time()
            _write_json(self.directory / f"decision-{number:05d}.json", audit)
        self.stats["decisions"] = number
        self.stats["rows_planned"] += len(rows)
        self.queue = rows

    def result_metadata(self):
        return dict(policy=ADAPTER_VERSION, provider=self.profile["provider"], model=self.model,
                    api_style=self.api_style, endpoint_host=_host(self.url),
                    reasoning_effort=self.cfg["reasoning_effort"],
                    ledger=str(self.ledger), ledger_limit=self.cfg["ledger_limit"],
                    log_dir=str(self.directory), request_retries=self.cfg["request_retries"],
                    parse_retries=self.cfg["parse_retries"], max_decisions=self.cfg["max_decisions"],
                    max_rows_per_decision=self.cfg["max_rows_per_decision"],
                    max_step_rad=self.cfg["max_step_rad"], max_step_gripper=self.cfg["max_step_gripper"],
                    include_images=self.cfg["include_images"], image_format=self.cfg["image_format"],
                    history_turns=self.cfg["history_turns"], seed=getattr(self, "seed", None),
                    benchmark_submission=False, **{k: v for k, v in self.stats.items()})


def create_adapter(config, profile):
    return LlmJointAdapter(config, profile)
