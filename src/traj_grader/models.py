"""Normalized, dataset-agnostic representation of an agent trajectory.

Every dataset adapter converts its raw rows into these types, so renderers and
(eventually) graders only ever have to understand one shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    name: str
    arguments: str  # raw argument string (usually JSON) as the model emitted it
    id: str | None = None


@dataclass
class Step:
    role: Role
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    # Whether this step is a training target (e.g. nebius `mask`); None if unknown.
    trainable: bool | None = None


@dataclass
class Trajectory:
    id: str
    instance_id: str
    steps: list[Step]
    source: str  # e.g. "nebius/SWE-agent-trajectories:train"
    index: int  # row index within the source split
    model: str | None = None
    # True = solved, False = not solved, None = unknown/not evaluated.
    resolved: bool | None = None
    exit_status: str | None = None
    patch: str | None = None
    reference_patch: str | None = None
    eval_logs: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def num_turns(self) -> int:
        """Number of assistant steps."""
        return sum(1 for s in self.steps if s.role == "assistant")

    @property
    def task(self) -> str:
        """The first user message, which for SWE datasets is the issue/PR text."""
        return next((s.content for s in self.steps if s.role == "user"), "")
