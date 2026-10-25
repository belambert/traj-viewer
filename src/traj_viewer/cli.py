"""Command-line interface: `traj-viewer list|show|html|info`."""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from traj_viewer.loader import Source, iter_trajectories, list_configs, list_splits, load_one
from traj_viewer.render_cli import print_trajectory, render_table
from traj_viewer.render_html import write_html

app = typer.Typer(no_args_is_help=True, help="View agent trajectories.")
console = Console()

DatasetArg = Annotated[str, typer.Argument(help="HF dataset id (e.g. nebius/SWE-agent-trajectories) or local .jsonl/.parquet")]
ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Dataset config (default: first)")]
SplitOpt = Annotated[str | None, typer.Option("--split", "-s", help="Split (default: first)")]
AdapterOpt = Annotated[str | None, typer.Option(help="Force an adapter (swe-agent, chat-messages)")]


def _open_html(trajs, source: Source, output: Path | None, open_browser: bool) -> None:
    if not trajs:
        console.print("[red]No trajectories found.[/]")
        raise typer.Exit(1)
    if output is None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source.label)
        suffix = f"_{trajs[0].index}" if len(trajs) == 1 else f"_{trajs[0].index}-{trajs[-1].index}"
        output = Path("out") / f"{slug}{suffix}.html"
    path = write_html(trajs, source.label, output)
    console.print(f"Wrote {len(trajs)} trajectories to [bold]{path}[/]")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


@app.command("list")
def list_cmd(
    dataset: DatasetArg,
    config: ConfigOpt = None,
    split: SplitOpt = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    start: int = 0,
    adapter: AdapterOpt = None,
) -> None:
    """Summarize trajectories in a table."""
    source = Source(dataset, config, split).resolve()
    with console.status(f"Loading {source.label}…"):
        trajs = list(iter_trajectories(source, start=start, limit=limit, adapter=adapter))
    console.print(render_table(trajs))


@app.command()
def show(
    dataset: DatasetArg,
    index: Annotated[int, typer.Argument(help="Row index within the split")] = 0,
    config: ConfigOpt = None,
    split: SplitOpt = None,
    instance: Annotated[str | None, typer.Option("--instance", "-i", help="Show first row with this instance_id instead")] = None,
    full: Annotated[bool, typer.Option(help="Don't truncate long messages")] = False,
    system: Annotated[bool, typer.Option(help="Show the system prompt")] = False,
    patch: Annotated[bool, typer.Option(help="Show model/reference patches")] = True,
    pager: Annotated[bool, typer.Option(help="Page output through $PAGER")] = True,
    html: Annotated[bool, typer.Option("--html", help="Open in the browser instead")] = False,
    adapter: AdapterOpt = None,
) -> None:
    """Show a single trajectory, step by step."""
    source = Source(dataset, config, split).resolve()
    with console.status(f"Loading {source.label}…"):
        if instance:
            traj = next(iter_trajectories(source, instance_id=instance, limit=1, adapter=adapter), None)
            if traj is None:
                console.print(f"[red]No trajectory with instance_id {instance!r}[/]")
                raise typer.Exit(1)
        else:
            traj = load_one(source, index, adapter=adapter)
    if html:
        _open_html([traj], source, None, True)
        return
    print_trajectory(traj, console, pager=pager, max_lines=None if full else 40, show_system=system, show_patch=patch)


@app.command()
def html(
    dataset: DatasetArg,
    config: ConfigOpt = None,
    split: SplitOpt = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    start: int = 0,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    adapter: AdapterOpt = None,
) -> None:
    """Build an HTML viewer for a range of trajectories and open it."""
    source = Source(dataset, config, split).resolve()
    with console.status(f"Loading {limit} trajectories from {source.label}…"):
        trajs = list(iter_trajectories(source, start=start, limit=limit, adapter=adapter))
    _open_html(trajs, source, output, open_browser)


@app.command()
def info(dataset: DatasetArg) -> None:
    """List a Hub dataset's configs and splits."""
    for config in list_configs(dataset):
        splits = list_splits(dataset, config)
        console.print(f"[bold]{config}[/]: {', '.join(splits)}")


if __name__ == "__main__":
    app()
