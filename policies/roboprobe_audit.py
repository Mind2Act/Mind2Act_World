"""Bounded public reply/tool diagnostics, without images or reasoning traces."""
import json


THOUGHT_MARK = "__thought__"
THOUGHT_BYPASS = "skip_thought_signature_validator"


def record(audit, name, summarize, *args):
    """Diagnostics must never turn an accepted reply into another request."""
    try:
        value = summarize(*args)
        json.dumps(value, allow_nan=False)
    except Exception as error:
        value = {"audit_error_type": type(error).__name__}
    audit[name] = value


def redacted(value, key):
    if isinstance(value, str):
        return value.replace(key, "<key>") if key else value
    if isinstance(value, dict):
        return {redacted(k, key): redacted(v, key) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redacted(v, key) for v in value]
    return value


def text(value, key="", limit=4000):
    value = redacted(value, key)
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return value if len(value) <= limit else value[:limit] + f"... [{len(value)} chars]"


def tool_feedback(payload, key):
    rows = []
    for item in reversed(payload.get("input", [])):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            break
        if item.get("type") == "function_call_output":
            rows.append({"call_id": str(item.get("call_id", "")).split(THOUGHT_MARK, 1)[0],
                         "output_text": text(item.get("output"), key)})
    return list(reversed(rows))


def reply_summary(response, key):
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    return {"content": text(message.get("content"), key),
            "tool_calls": [{"name": call.get("function", {}).get("name"),
                            "arguments_text": text(call.get("function", {}).get("arguments"), key)}
                           for call in message.get("tool_calls") or []],
            "finish_reason": choice.get("finish_reason")}


def payload_skeleton(payload, key):
    rows = []
    for item in payload.get("input", []):
        if not isinstance(item, dict):
            rows.append({"type": type(item).__name__})
            continue
        row = {name: item[name] for name in ("type", "role", "name") if name in item}
        if "call_id" in item:
            row["call_id"] = redacted(str(item["call_id"]).split(THOUGHT_MARK, 1)[0], key)[:40]
        for name, limit in (("arguments", 300), ("output", 200)):
            if name in item:
                row[name + "_text"] = text(item[name], key, limit)
        content = item.get("content")
        if isinstance(content, list):
            row["content"] = [{"type": part.get("type"),
                               **({"text_length": len(part["text"])} if isinstance(part.get("text"), str) else {})}
                              for part in content if isinstance(part, dict)]
        rows.append(row)
    return {"top_keys": sorted(payload), "serialized_bytes": len(json.dumps(payload).encode()), "input": rows}


def raw_shape(raw, key):
    if not isinstance(raw, dict):
        return None
    return {"status": raw.get("status"), "incomplete_details": text(raw.get("incomplete_details"), key, 500),
            "output": [{name: item.get(name) for name in ("type", "status", "role") if name in item}
                       for item in raw.get("output") or [] if isinstance(item, dict)],
            **{name: text(raw[name], key, 500) for name in ("finish_reason", "provider_specific_fields", "error") if name in raw},
            "item_provider_fields": [text(item["provider_specific_fields"], key, 300)
                                     for item in raw.get("output") or [] if isinstance(item, dict) and item.get("provider_specific_fields")]}
