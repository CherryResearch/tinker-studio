---
name: tinker-notebook-recovery
description: Recover Tinker training state after notebook or run-manager interruptions by extracting saved run artifacts, inspecting local run_outputs metadata, checking resumable checkpoints, and choosing the safest resume path. Use when Codex needs to reconstruct training_run_id, sampler_id, checkpoint context, or reconcile notebook state with the Streamlit dashboard.
---

# Tinker Notebook Recovery

Use this when the notebook lost live state, the kernel restarted, a dashboard session was restarted,
or the user only has saved output and needs help figuring out what can be resumed.

Start read-only. Recover identifiers first, then decide whether a resume action is safe.

## Quick Start

Extract artifacts and saved errors from the notebook:

```powershell
@'
from tinker_notebook_diagnostics import extract_notebook_artifacts, recovered_tracking_state

artifact_df, error_df = extract_notebook_artifacts(r".\tinker_train_and_eval.ipynb")
print(artifact_df.to_string(index=False) if not artifact_df.empty else "no notebook artifacts found")
print(error_df.to_string(index=False) if not error_df.empty else "no saved notebook errors found")
print(recovered_tracking_state(r".\tinker_train_and_eval.ipynb"))
'@ | .\tinker_env\Scripts\python.exe -
```

Then inspect resumable API state:

```powershell
@'
import tinker
from tinker_notebook_env import ensure_tinker_api_key
from tinker_notebook_resume import list_resumable_runs_df, build_resume_selection

ensure_tinker_api_key()
rest_client = tinker.ServiceClient().create_rest_client()
runs_df = list_resumable_runs_df(rest_client, limit=20)
print(runs_df.to_string(index=False) if not runs_df.empty else "no resumable runs found")
selection = build_resume_selection(rest_client, limit=20)
print(selection)
'@ | .\tinker_env\Scripts\python.exe -
```

Use `run_outputs\latest_active_run.json` as the local tie-breaker when notebook output is old or
incomplete. The Streamlit dashboard's Training tab reads this same local payload.

## Recovery Workflow

1. Recover `training_run_id`, `sampler_id`, and `session_id` values from the saved notebook.
2. Compare those IDs with recent API runs and resumable checkpoints.
3. Compare against `run_outputs\latest_active_run.json` and the dashboard Training tab when available.
4. If the latest local payload is still actively updating, avoid starting a second resume path.
5. Prefer `run_tinker_experiment.py --resume` when the predefined run name still matches.
6. Use `--resume-from-checkpoint` only when you need an explicit checkpoint path.

## Guardrails

- Do not call a run resumable if there is no checkpoint.
- If the API still shows fresh activity for the run, treat it as active rather than interrupted.
- Saved notebook output can lag reality, so compare notebook artifacts against API and local run metadata before acting.
- Do not use the dashboard alone as proof that a run is dead; it is a view over local payloads and optional API data.
