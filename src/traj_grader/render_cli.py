"""Render trajectories in the terminal with Rich."""

from __future__ import annotations

import json
from collections.abc import Iterable

from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from traj_grader.models import Step, ToolCall, Trajectory

ROLE_STYLES = {"system": "magenta", "user": "green", "assistant": "cyan", "tool": "yellow"}


def outcome_text(resolved: bool | None) -> Text:
    if resolved is True:
        return Text("resolved", style="bold green")
    if resolved is False:
        return Text("unresolved", style="bold red")
    return Text("unknown", style="dim")


def _truncate(text: str, max_lines: int | None) -> str:
    if max_lines is None:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head, tail = lines[: max_lines - 5], lines[-3:]
    hidden = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"… [{hidden} lines hidden; use --full] …", *tail])


def _render_tool_call(call: ToolCall, max_lines: int | None) -> RenderableType:
    title = f"→ {call.name}"
    try:
        args = json.loads(call.arguments)
    except (json.JSONDecodeError, TypeError):
        args = None

    if call.name == "command":  # raw shell command (SWE-agent style)
        body: RenderableType = Syntax(_truncate(call.arguments, max_lines), "bash", word_wrap=True)
    elif isinstance(args, dict) and set(args) == {"command"}:
        body = Syntax(_truncate(str(args["command"]), max_lines), "bash", word_wrap=True)
    elif isinstance(args, dict):
        body = Syntax(_truncate(json.dumps(args, indent=2), max_lines), "json", word_wrap=True)
    else:
        body = Text(_truncate(call.arguments, max_lines))
    return Panel(body, title=title, title_align="left", border_style="blue", padding=(0, 1))


def render_step(step: Step, n: int, max_lines: int | None = 40) -> RenderableType:
    parts: list[RenderableType] = []
    if step.reasoning:
        parts.append(Text(_truncate(step.reasoning, max_lines), style="italic dim"))
    if step.content.strip():
        parts.append(Text(_truncate(step.content, max_lines)))
    parts.extend(_render_tool_call(c, max_lines) for c in step.tool_calls)
    if not parts:
        parts.append(Text("(empty)", style="dim"))

    style = ROLE_STYLES.get(step.role, "white")
    title = f"[bold {style}]#{n} {step.role}[/]"
    if step.trainable is False:
        title += " [dim](masked)[/]"
    return Panel(Group(*parts), title=title, title_align="left", border_style=style)


def render_header(traj: Trajectory) -> RenderableType:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("instance", escape(traj.instance_id))
    grid.add_row("source", f"{escape(traj.source)} [dim]row {traj.index}[/]")
    if traj.model:
        grid.add_row("model", escape(traj.model))
    grid.add_row("outcome", outcome_text(traj.resolved))
    if traj.exit_status:
        grid.add_row("exit status", escape(traj.exit_status))
    grid.add_row("steps", f"{len(traj.steps)} ({traj.num_turns} assistant, {traj.num_tool_calls} tool calls)")
    return Panel(grid, title=f"[bold]{escape(traj.id)}[/]", border_style="bold white")


def render_trajectory(
    traj: Trajectory,
    *,
    max_lines: int | None = 40,
    show_system: bool = False,
    show_patch: bool = True,
) -> Iterable[RenderableType]:
    yield render_header(traj)
    for n, step in enumerate(traj.steps):
        if step.role == "system" and not show_system:
            yield Text(f"#{n} system prompt hidden ({len(step.content):,} chars; use --system)", style="dim")
            continue
        yield render_step(step, n, max_lines)

    if show_patch:
        for label, patch in (("model patch", traj.patch), ("reference patch", traj.reference_patch)):
            if patch and patch.strip():
                yield Rule(label)
                yield Syntax(patch.strip(), "diff", word_wrap=True)


def render_table(trajs: Iterable[Trajectory]) -> Table:
    table = Table(header_style="bold", expand=False)
    table.add_column("row", justify="right", style="dim")
    table.add_column("instance")
    table.add_column("model")
    table.add_column("outcome")
    table.add_column("steps", justify="right")
    table.add_column("tool calls", justify="right")
    table.add_column("exit status")
    for t in trajs:
        table.add_row(
            str(t.index),
            escape(t.instance_id),
            escape(t.model or ""),
            outcome_text(t.resolved),
            str(len(t.steps)),
            str(t.num_tool_calls),
            escape(t.exit_status or ""),
        )
    return table


def print_trajectory(traj: Trajectory, console: Console | None = None, pager: bool = False, **kwargs) -> None:
    console = console or Console()
    renderables = list(render_trajectory(traj, **kwargs))
    if pager:
        with console.pager(styles=True):
            console.print(*renderables)
    else:
        console.print(*renderables)
