---
name: tinker-publish-safety
description: Prepare Tinker Studio changes for safe publication by checking Git branches, GitHub remotes, public/private account split, ignored private data, nested training-data repos, environment-key handling, and draft PR state. Use when Codex needs to publish dashboard/skill/code changes, partition datasets into private storage, verify no secrets or training data will be committed, or update a Tinker draft PR.
---

# Tinker Publish Safety

## Overview

Use this before pushing or opening PRs from the Tinker Studio workspace. The intended split is:

- Public or semi-public code/docs/dashboard/skills: publish through the main Tinker repo under Cherry Research when appropriate.
- Private data/storage: keep ignored locally or in a separate private storage repo, usually as a nested repo inside `data/`.
- Environment keys: keep in ignored `.env` or local secret stores only; never commit real values.

## Setup Checks

Run these from the Tinker workspace root:

```powershell
git status --short --branch
git remote -v
gh auth status
git check-ignore -v .env data/training_data_cerise
git -C data\training_data_cerise status --short --branch
git -C data\training_data_cerise remote -v
```

Expected local layout:

```text
tinker/
  data/
    README.md                  # tracked by main repo
    training_data_cerise/       # ignored by main repo, separate private repo
```

Expected env examples:

```env
TINKER_API_KEY=
TINKER_BASE_URL=
TINKER_STUDIO_DATASET_ROOT=data/training_data_cerise
TINKER_DATASET_ROOT=data/training_data_cerise
```

## Public Repo Workflow

1. Work on a `codex/...` branch, not `main`.
2. Stage public source/docs/config only. Do not use broad staging when the tree contains unexpected files.
3. Confirm `data/training_data_cerise/`, run outputs, logs, checkpoints, env files, and local venvs are ignored.
4. Run a secret-pattern scan over commit candidates before committing:

```powershell
$files = git diff --cached --name-only
if ($files) {
  rg -n --hidden --no-ignore-vcs "(sk-[A-Za-z0-9_-]{20,}|gho_[A-Za-z0-9_]+|ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|password\s*[:=])" -- $files
}
```

5. Run focused validation, usually:

```powershell
python -m compileall tinker_training_utils.py streamlit_tinker_dashboard.py
```

6. Push with upstream tracking and keep the PR draft unless the user explicitly asks otherwise.

## Private Data Repo Workflow

Use `data/training_data_cerise` as a plain nested repo, not a submodule, unless the user explicitly wants a public pointer to the private repo.

```powershell
git -C data\training_data_cerise status --short --branch
git -C data\training_data_cerise add -A
git -C data\training_data_cerise commit -m "Update Cerise training data"
git -C data\training_data_cerise push
```

Before the first push, check size and obvious secrets:

```powershell
$files = Get-ChildItem -LiteralPath data\training_data_cerise -Recurse -File -Force
$files | Sort-Object Length -Descending | Select-Object -First 10 FullName,Length
rg -n --hidden "(sk-[A-Za-z0-9_-]{20,}|gho_[A-Za-z0-9_]+|ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|password\s*[:=])" data\training_data_cerise
```

Use Git LFS or another storage strategy before pushing if files approach GitHub's normal file-size limits.

## Account Split

- `CherryResearch`: ML-related, semi-professional public-facing blog, pseudo-startup, and computer-led research work.
- Keep raw training data and local storage out of the main source repo unless the user explicitly chooses a reviewed storage path.

Default to ignored local storage for raw training data. Move polished public research/code or reviewed private tooling to Cherry Research only after explicit user direction.

## Guardrails

- Do not print real `TINKER_API_KEY` values. Use existing helpers that report presence/status only.
- Do not commit `.env`, `.streamlit/secrets.toml`, local databases, logs, checkpoints, run outputs, or private dataset folders.
- Do not direct-push live/deployment branches for Float or other PR-protected projects; use a PR unless the user explicitly says otherwise.
- If `gh auth status` shows the wrong account or insufficient scopes, stop and ask the user to re-authenticate.
