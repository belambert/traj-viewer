"""Render trajectories as a single self-contained HTML viewer page."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from traj_grader.models import Trajectory


def render_html(trajs: list[Trajectory], title: str) -> str:
    template = files("traj_grader").joinpath("templates/viewer.html").read_text()
    payload = json.dumps({"title": title, "trajectories": [asdict(t) for t in trajs]}, default=str)
    # Prevent a "</script>" inside trajectory text from terminating the data block.
    payload = payload.replace("</", "<\\/")
    return template.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


def write_html(trajs: list[Trajectory], title: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(trajs, title), encoding="utf-8")
    return path
