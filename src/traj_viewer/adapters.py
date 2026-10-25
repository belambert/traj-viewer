"""Adapters that convert raw dataset rows into `Trajectory` objects.

Each adapter declares which column names it needs; `detect_adapter` picks the
first one whose required columns are present. To support a new dataset, write
a function `(row, index, source) -> Trajectory` and register it in ADAPTERS.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from traj_viewer.models import Step, ToolCall, Trajectory

Row = dict[str, Any]
Converter = Callable[[Row, int, str], Trajectory]


@dataclass(frozen=True)
class Adapter:
    name: str
    required_columns: frozenset[str]
    convert: Converter
    description: str = ""


def _maybe_json(value: Any) -> Any:
    """Some datasets (or the HF viewer) store nested fields as JSON strings."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


# --- SWE-agent style (nebius/SWE-agent-trajectories) ---------------------------

# SWE-agent puts its action in the last fenced block of the assistant message.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _split_swe_agent_action(text: str) -> tuple[str, list[ToolCall]]:
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text, []
    last = matches[-1]
    thought = (text[: last.start()] + text[last.end() :]).strip()
    command = last.group(1).strip()
    return thought, [ToolCall(name="command", arguments=command)]


def convert_swe_agent(row: Row, index: int, source: str) -> Trajectory:
    steps: list[Step] = []
    seen_assistant = False
    for msg in _maybe_json(row["trajectory"]) or []:
        role = msg.get("role")
        text = msg.get("text") or ""
        if role == "system":
            steps.append(Step(role="system", content=msg.get("system_prompt") or text))
        elif role == "ai":
            seen_assistant = True
            thought, calls = _split_swe_agent_action(text)
            steps.append(
                Step(role="assistant", content=thought, tool_calls=calls, trainable=msg.get("mask"))
            )
        else:
            # In SWE-agent, environment observations come back as user messages.
            steps.append(Step(role="tool" if seen_assistant else "user", content=text))

    return Trajectory(
        id=f"{row['instance_id']}#{index}",
        instance_id=row["instance_id"],
        steps=steps,
        source=source,
        index=index,
        model=row.get("model_name"),
        resolved=row.get("target"),
        exit_status=row.get("exit_status"),
        patch=row.get("generated_patch"),
        eval_logs=row.get("eval_logs"),
    )


# --- OpenAI chat-messages style (nvidia/Open-SWE-Traces and many others) -------


def _to_step(msg: Row) -> Step:
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, indent=2)
        calls.append(ToolCall(name=fn.get("name") or "?", arguments=args or "", id=tc.get("id")))

    content = msg.get("content") or ""
    if isinstance(content, list):  # multi-part content blocks
        content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))

    role = msg.get("role", "user")
    if role not in ("system", "user", "assistant", "tool"):
        role = "tool" if role in ("function", "ipython", "observation") else "user"

    return Step(
        role=role,
        content=content,
        reasoning=msg.get("reasoning_content") or None,
        tool_calls=calls,
        tool_call_id=msg.get("tool_call_id"),
    )


def _patch(meta: Row, key: str) -> str | None:
    value = meta.pop(key, None)
    if isinstance(value, dict):
        return value.get("patch")
    return value


def convert_chat_messages(row: Row, index: int, source: str) -> Trajectory:
    messages = _maybe_json(row["messages"]) or []
    metadata = dict(_maybe_json(row.get("metadata")) or {})
    tools = [t for t in (_maybe_json(t) for t in _maybe_json(row.get("tools")) or []) if isinstance(t, dict)]

    resolved = row.get("resolved")
    if isinstance(resolved, int) and not isinstance(resolved, bool):
        resolved = {1: True, 0: False}.get(resolved)  # -1 means not evaluated

    teacher = metadata.get("teacher_model")
    model = teacher.get("name") if isinstance(teacher, dict) else row.get("model") or row.get("model_name")

    # Keep any other top-level columns as metadata so nothing is hidden.
    consumed = {"messages", "metadata", "tools", "resolved", "instance_id", "trajectory_id"}
    extra = {k: v for k, v in row.items() if k not in consumed}

    instance_id = row.get("instance_id") or str(index)
    return Trajectory(
        id=row.get("trajectory_id") or f"{instance_id}#{index}",
        instance_id=instance_id,
        steps=[_to_step(m) for m in messages],
        source=source,
        index=index,
        model=model,
        resolved=resolved,
        patch=_patch(metadata, "model_patch") or row.get("patch"),
        reference_patch=_patch(metadata, "reference_patch"),
        tools=tools,
        metadata={**extra, **metadata},
    )


# --- opencode sessions (rows built by loader._iter_opencode) -------------------


def _join(parts: list[Row], type_: str) -> str:
    return "\n\n".join(p["text"] for p in parts if p["type"] == type_ and p.get("text"))


def _opencode_steps(messages: list[Row]) -> list[Step]:
    steps: list[Step] = []
    for msg in messages:
        parts = msg["parts"]
        text = _join(parts, "text")
        if msg["role"] == "user":
            steps.append(Step(role="user", content=text))
            continue

        if err := msg.get("error"):
            note = f"[{err.get('name')}] {(err.get('data') or {}).get('message', '')}"
            text = f"{text}\n\n{note}" if text else note
        tools = [p for p in parts if p["type"] == "tool"]
        reasoning = _join(parts, "reasoning") or None
        if not (text or reasoning or tools):
            continue

        calls = [
            ToolCall(name=t["tool"], arguments=json.dumps(t["state"].get("input") or {}, indent=2), id=t["callID"])
            for t in tools
        ]
        steps.append(Step(role="assistant", content=text, reasoning=reasoning, tool_calls=calls))
        steps += [
            Step(role="tool", content=t["state"].get("output") or t["state"].get("error") or "", tool_call_id=t["callID"])
            for t in tools
        ]
    return steps


def convert_opencode(row: Row, index: int, source: str) -> Trajectory:
    s = row["session"]
    assistant = [m for m in row["oc_messages"] if m["role"] == "assistant"]
    last = assistant[-1] if assistant else {}

    created = datetime.fromtimestamp(s["time_created"] / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "session": s["id"],
        "directory": s["directory"],
        "created": created,
        "parent": s["parent_id"],
        "cost": s["cost"],
        "tokens_input": s["tokens_input"],
        "tokens_output": s["tokens_output"],
    }
    return Trajectory(
        id=s["id"],
        instance_id=s["title"] or s["id"],
        steps=_opencode_steps(row["oc_messages"]),
        source=source,
        index=index,
        model=next((m["modelID"] for m in assistant if m.get("modelID")), None),
        exit_status=(last.get("error") or {}).get("name") or last.get("finish"),
        metadata=metadata,
    )


ADAPTERS: list[Adapter] = [
    Adapter(
        name="opencode",
        required_columns=frozenset({"session", "oc_messages"}),
        convert=convert_opencode,
        description="Local opencode sessions from its SQLite database",
    ),
    Adapter(
        name="swe-agent",
        required_columns=frozenset({"instance_id", "trajectory"}),
        convert=convert_swe_agent,
        description="SWE-agent format with `trajectory` list (e.g. nebius/SWE-agent-trajectories)",
    ),
    Adapter(
        name="chat-messages",
        required_columns=frozenset({"messages"}),
        convert=convert_chat_messages,
        description="OpenAI-style `messages` with tool_calls (e.g. nvidia/Open-SWE-Traces)",
    ),
]


def get_adapter(name: str) -> Adapter:
    for adapter in ADAPTERS:
        if adapter.name == name:
            return adapter
    raise ValueError(f"Unknown adapter {name!r}; choose from {[a.name for a in ADAPTERS]}")


def detect_adapter(columns: set[str]) -> Adapter:
    for adapter in ADAPTERS:
        if adapter.required_columns <= columns:
            return adapter
    raise ValueError(
        f"No adapter matches columns {sorted(columns)}. "
        f"Known adapters: {', '.join(a.name for a in ADAPTERS)}"
    )
