---
name: tinker-streamlit-dashboard
description: Launch, validate, inspect, and troubleshoot the Tinker Studio Streamlit dashboard. Use when Codex needs to run launch_streamlit_dashboard.bat, open or verify localhost:8501, explain dashboard tabs, inspect shortform/long-form/interview source data, refresh Bluesky posts, test the local endpoint bridge, or debug Streamlit startup and UI issues.
---

# Tinker Streamlit Dashboard

Use this in the Tinker Studio workspace after `streamlit_tinker_dashboard.py` has been added.
The dashboard is an operator UI over generated shortform posts, long-form/imported/interview sources,
local run metadata, stop control state, the local endpoint bridge, and optional Tinker API telemetry.

## Quick Start

Prefer the launcher for user-facing sessions:

```powershell
.\launch_streamlit_dashboard.bat
```

For captured terminal output, run Streamlit directly:

```powershell
.\tinker_env\Scripts\python.exe -m streamlit run .\streamlit_tinker_dashboard.py --server.headless=true --server.port=8501
```

Verify the dependency and syntax before troubleshooting UI behavior:

```powershell
.\tinker_env\Scripts\python.exe -m streamlit --version
.\tinker_env\Scripts\python.exe -m py_compile .\streamlit_tinker_dashboard.py
```

Check the app health endpoint when the server is running:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8501/_stcore/health
```

## Dashboard Data Sources

- Dataset tab: `driift bluesky fine-tune dataset\processed\posts.csv`, reply-context columns, and `tinker\dataset_manifest.json`.
- Sources tab: long-form seed docs in `processed\rentry_pages.jsonl`, imported sources in `processed\imported_sources.jsonl`, and interview rows in `processed\interview_qa.jsonl` when present.
- Evaluation tab: held-out openings and targets loaded on demand from the dataset helper stack.
- Training tab: `run_outputs\latest_active_run.json`, `.tinker_stop_request.json`, and optional Tinker API recent runs.
- Chat / Endpoint tab: sampler checkpoints discovered from `run_outputs\*.json` and exposed through the local OpenAI-compatible bridge.
- Sidebar refresh: calls `driift bluesky fine-tune dataset\build_bluesky_finetune_dataset.py --handle <handle> --outdir <dataset-root>`.

If the user reports stale dataset counts, validate the manifest and CSV directly:

```powershell
@'
import json
import pandas as pd
from pathlib import Path

root = Path("driift bluesky fine-tune dataset")
manifest = json.loads((root / "tinker" / "dataset_manifest.json").read_text(encoding="utf-8"))
posts = pd.read_csv(root / "processed" / "posts.csv")
print(manifest["collected_at_utc"])
print(manifest["counts"])
print(len(posts))
print(posts["created_at"].max())
'@ | .\tinker_env\Scripts\python.exe -
```

## Refreshing Posts

The dashboard button is appropriate when the user wants an interactive refresh. For automation or
subagent work, use the underlying script:

```powershell
cd "driift bluesky fine-tune dataset"
..\tinker_env\Scripts\python.exe .\build_bluesky_finetune_dataset.py --handle driift.bsky.social --outdir .
```

Public Bluesky fetches do not require a Tinker API key, but they do require network access. The
dataset directory may be ignored by Git even when files are rewritten.

## Troubleshooting

- If `No module named streamlit` appears, install dependencies in the workspace venv with `.\tinker_env\Scripts\python.exe -m pip install -r .\requirements.txt`.
- If API runs are missing but dataset panels load, check whether `TINKER_API_KEY` is available; the dashboard still works without it.
- If Windows `Start-Process` reports duplicate `Path` or `PATH`, prefer the direct foreground Streamlit command or the batch launcher.
- If `localhost:8501` is already in use, pass a different port to Streamlit and open that URL.
