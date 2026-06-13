---
name: tinker-training-monitor
description: Monitor active or recent Tinker training runs from the CLI, Rich dashboard, or Streamlit dashboard; inspect local run_outputs metadata; interpret heartbeat freshness; and check cooperative stop-signal state. Use when Codex needs to answer whether a run is active, stale, resumable, visible in the dashboard, or safely stoppable.
---

# Tinker Training Monitor

Use this skill for read-only inspection of run health before deciding whether to resume, stop, or
start anything new.

## Quick Start

Take a one-shot CLI snapshot first:

```powershell
.\tinker_env\Scripts\python.exe .\monitor_tinker_runs.py --workspace . --recent 6 --iterations 1
.\tinker_env\Scripts\python.exe .\tinker_stop_cli.py --workspace . --action status
Get-Content .\run_outputs\latest_active_run.json
```

Track a specific run ID when you already know it:

```powershell
.\tinker_env\Scripts\python.exe .\monitor_tinker_runs.py --workspace . --run-id "<training-run-id>" --iterations 1
```

Use the Rich dashboard when a richer terminal view is useful:

```powershell
.\tinker_env\Scripts\python.exe .\monitor_tinker_dashboard.py --workspace . --recent 6 --iterations 1
```

Use the Streamlit dashboard when the user wants a browser UI:

```powershell
.\launch_streamlit_dashboard.bat
```

## Interpretation Rules

- The API monitor labels a run `ACTIVE` when the last request is within 30 seconds and `IDLE` after that. `IDLE` does not automatically mean failed.
- The local payload treats a `running` run as stale after 5 minutes without a heartbeat. That is the main hint that auto-resume may be appropriate.
- A run is only safely describable as auto-resumable when there is both a stale local `running` payload and a checkpoint path.
- The Streamlit dashboard Training tab is a view over local run payloads plus optional API data; verify with CLI commands before taking destructive or duplicate-run actions.

## Workflow

1. Read the API snapshot first.
2. Compare it with `run_outputs\latest_active_run.json`.
3. Check whether `.tinker_stop_request.json` is pending before recommending a restart or resume.
4. If the API looks active but the local payload is old, assume the API is the fresher source of truth.
5. Use the Streamlit dashboard for operator visibility, not as the only evidence for run state.

## Guardrails

- Prefer `monitor_tinker_runs.py` for automation and concise reporting.
- Prefer `monitor_tinker_dashboard.py` for terminal operator visibility.
- Prefer `streamlit_tinker_dashboard.py` or `launch_streamlit_dashboard.bat` when the user is already working in a browser.
- Do not claim a run is dead only because the last request is older than 30 seconds.
