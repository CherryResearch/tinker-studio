---
name: tinker-streamlit-dashboard
description: Launch, validate, inspect, and troubleshoot the Tinker Studio Streamlit dashboard. Use when Codex needs to run launch_streamlit_dashboard.bat, open or verify localhost:8501, explain dashboard tabs, inspect shortform/long-form/interview source data, refresh Bluesky posts, test the local endpoint bridge, or debug Streamlit startup and UI issues.
---

# Tinker Streamlit Dashboard

Use this in the Tinker Studio workspace after `streamlit_tinker_dashboard.py` has been added.
The dashboard is an operator UI over generated shortform posts, long-form/imported/interview sources,
local run metadata, stop control state, the local endpoint bridge, and optional Tinker API telemetry.

## Setup And Safety

- Launch from the Tinker workspace root so relative paths in `.env.example` resolve correctly.
- Keep real `TINKER_API_KEY` values in ignored `.env` or local secret stores; do not print them in summaries or commit them.
- Dataset panels default to `data\training_data`, which is ignored by the main Tinker repo and may be a private nested repo.
- If changing publish or dataset boundaries, use `tinker-publish-safety` before staging or pushing.

## Quick Start

Prefer the launcher for user-facing sessions:

```powershell
.\launch_streamlit_dashboard.bat
```

The standalone endpoint bridge is an OpenAI-compatible API server plus a small
browser chat page. Open the chat UI at:

```text
http://127.0.0.1:8765/chat
```

Use `http://127.0.0.1:8765/v1` as the OpenAI-compatible base URL for clients.
Browser visits to `/v1` render the same local chat page for convenience.
The chat page keeps lightweight local testing history in
`run_outputs\endpoint_chat_history\*.json`, preserves the current session across
reloads, sends on Enter while Shift+Enter inserts a newline, and defaults to
192 max output tokens and temperature `0.4`. It defaults to `chat` mode, which sends structured
system/user/assistant messages with a short conversational-friend system prompt. The bridge strips
Harmony/channel control-token leaks from generated text and drops malformed/repetitive assistant
turns from future prompt history before saving.
The chat page also exposes a run selector backed by `/v1/runs`, showing the last 10 sampler-backed
run records from `run_outputs` with completion/start time, base model, dataset variant, status, and
learning rate metadata. Selecting a run calls `/v1/runs/select` and switches the active sampler.
Use `completion` mode only when intentionally testing the older post/opening
completion behavior or held-out opening evaluations. This is a small local
bridge; future Float integration should treat Float conversations as the durable
backing store rather than expanding this endpoint into a full chat product.

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

- Dataset tab: `data\training_data\processed\posts.csv`, reply-context columns, and `tinker\dataset_manifest.json`.
- Sources tab: long-form seed docs in `processed\rentry_pages.jsonl`, imported sources in `processed\imported_sources.jsonl`, synthetic rows in `processed\synthetic_sources.jsonl`, and interview rows in `processed\interview_qa.jsonl` when present.
- Evaluation tab: held-out openings and targets loaded on demand from the dataset helper stack.
- Training tab: `run_outputs\latest_active_run.json`, `.tinker_stop_request.json`, optional Tinker API recent runs, tag-filter previews for `--include-tag` / `--exclude-tag` training commands, and a training-example inspector that shows exact messages, target text, source metadata, tags, and optional JSONL preview export under `run_outputs\dataset_previews`.
- Chat / Endpoint tab: sampler checkpoints discovered from `run_outputs\*.json` and exposed through the local OpenAI-compatible bridge; the standalone chat page can switch between recent sampler runs without restarting the server.
- Sidebar refresh: calls `data\training_data\build_bluesky_finetune_dataset.py --handle <handle> --outdir <dataset-root>`.

If the user reports stale dataset counts, validate the manifest and CSV directly:

```powershell
@'
import json
import pandas as pd
from pathlib import Path

root = Path("data") / "training_data"
manifest = json.loads((root / "tinker" / "dataset_manifest.json").read_text(encoding="utf-8"))
posts = pd.read_csv(root / "processed" / "posts.csv")
print(manifest["collected_at_utc"])
print(manifest["counts"])
print(len(posts))
print(posts["created_at"].max())
print((root / "processed" / "synthetic_sources.jsonl").exists())
'@ | .\tinker_env\Scripts\python.exe -
```

## Refreshing Posts

The dashboard button is appropriate when the user wants an interactive refresh. For automation or
subagent work, use the underlying script:

```powershell
cd "data\training_data"
..\..\tinker_env\Scripts\python.exe .\build_bluesky_finetune_dataset.py --handle <handle>.bsky.social --outdir .
```

Public Bluesky fetches do not require a Tinker API key, but they do require network access. The
dataset directory may be ignored by Git even when files are rewritten.

## Troubleshooting

- If `No module named streamlit` appears, install dependencies in the workspace venv with `.\tinker_env\Scripts\python.exe -m pip install -r .\requirements.txt`.
- If API runs are missing but dataset panels load, check whether `TINKER_API_KEY` is available; the dashboard still works without it.
- If Windows `Start-Process` reports duplicate `Path` or `PATH`, prefer the direct foreground Streamlit command or the batch launcher.
- If `localhost:8501` is already in use, pass a different port to Streamlit and open that URL.
