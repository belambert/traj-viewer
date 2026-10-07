# traj-grader

View (and eventually grade) agent trajectories from Hugging Face datasets or local files.

Supported formats (auto-detected from columns, see `src/traj_grader/adapters.py`):

- **swe-agent** — `trajectory` list, e.g. `nebius/SWE-agent-trajectories`
- **chat-messages** — OpenAI-style `messages` with `tool_calls`, e.g. `nvidia/Open-SWE-Traces`

## Usage

```bash
# Configs/splits of a Hub dataset
uv run traj-grader info nvidia/Open-SWE-Traces

# Summary table
uv run traj-grader list nebius/SWE-agent-trajectories -n 10
uv run traj-grader list nvidia/Open-SWE-Traces -c v1.1 -s sweagent --start 100

# One trajectory in the terminal (--full, --system, --no-pager, --instance <id>)
uv run traj-grader show nebius/SWE-agent-trajectories 0

# HTML viewer (written to out/, opened in the browser; j/k to navigate)
uv run traj-grader html nebius/SWE-agent-trajectories -n 50
uv run traj-grader show nebius/SWE-agent-trajectories 0 --html
```

Local `.jsonl`, `.json` and `.parquet` files work in place of a dataset id.

Hub datasets are read from the Hub's auto-converted parquet branch one row group at a time,
so peeking at a few rows doesn't download the whole dataset.
