---
name: tinker-publish-safety
description: Prepare Tinker Studio changes for safe publication by checking Git branches, GitHub remotes, public/private repository boundaries, ignored private data, nested training-data repos, environment-key handling, and draft PR state. Use when Codex needs to publish dashboard/skill/code changes, partition datasets into private storage, verify no secrets or training data will be committed, or update a Tinker draft PR.
---

# Tinker Publish Safety

## Overview

Use this before pushing or opening PRs from the Tinker Studio workspace. The intended split is:

- Public or semi-public code/docs/dashboard/skills: publish through the configured public remote when appropriate.
- Private data/storage: keep ignored locally or in a separate private storage repo, usually as a nested repo inside `data/`.
- Environment keys: keep in ignored `.env` or local secret stores only; never commit real values.

## Setup Checks

Run these from the Tinker workspace root:

```powershell
git status --short --branch
git remote -v
gh auth status
git check-ignore -v .env data/training_data
git -C data\training_data status --short --branch
git -C data\training_data remote -v
```

Expected local layout:

```text
tinker/
  data/
    README.md                  # tracked by main repo
    training_data/       # ignored by main repo, separate private repo
```

Expected env examples:

```env
TINKER_API_KEY=
TINKER_BASE_URL=
TINKER_STUDIO_DATASET_ROOT=data/training_data
TINKER_DATASET_ROOT=data/training_data
```

## Public Repo Workflow

1. Work on a `codex/...` branch, not `main`.
2. Build public candidates from a clean public-base worktree. Treat the whole candidate branch as the review surface, not just the staged diff.
3. Stage public source/docs/config only. Do not use broad staging when the tree contains unexpected files.
4. Confirm `data/training_data/`, run outputs, logs, checkpoints, env files, and local venvs are ignored.
5. Before staging, scrub public code/docs/skills/config for private identifiers. Do not publish:
   - private dataset codenames, private repo/account names, usernames, or absolute local paths;
   - cloud-drive, note-vault, takeout, or source-export folder names;
   - real social handles, source document titles, exclusion filenames, note names, synthetic sample titles, or private tags;
   - run-output payloads, sampler/checkpoint IDs, raw manifest contents, notebook outputs, logs, or generated dataset previews.
6. Public examples should use generic placeholders such as `data/training_data`, `<handle>`, `<source-folder>`, and `%USERPROFILE%`. If a feature needs local defaults, read them from env vars or ignored local config instead of hardcoding personal paths or source names.
7. Run a branch-wide redaction scan for known private identifiers before pushing. If a tainted public ref was already pushed, delete or rewrite that ref before continuing.
8. Run a secret-pattern scan over commit candidates before committing:

```powershell
$files = git diff --cached --name-only
if ($files) {
  rg -n --hidden --no-ignore-vcs "(sk-[A-Za-z0-9_-]{20,}|gho_[A-Za-z0-9_]+|ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|password\s*[:=])" -- $files
}
```

9. Run focused validation, usually:

```powershell
python -m compileall tinker_training_utils.py streamlit_tinker_dashboard.py
```

10. Push with upstream tracking and keep the PR draft unless the user explicitly asks otherwise.

## Private Data Repo Workflow

Use `data/training_data` as a plain nested repo, not a submodule, unless the user explicitly wants a public pointer to the private repo.

```powershell
git -C data\training_data status --short --branch
git -C data\training_data add -A
git -C data\training_data commit -m "Update private training data"
git -C data\training_data push
```

Before the first push, check size and obvious secrets:

```powershell
$files = Get-ChildItem -LiteralPath data\training_data -Recurse -File -Force
$files | Sort-Object Length -Descending | Select-Object -First 10 FullName,Length
rg -n --hidden "(sk-[A-Za-z0-9_-]{20,}|gho_[A-Za-z0-9_]+|ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY|api[_-]?key\s*[:=]|secret\s*[:=]|token\s*[:=]|password\s*[:=])" data\training_data
```

Use Git LFS or another storage strategy before pushing if files approach GitHub's normal file-size limits.

## Publishing Boundary

- Keep raw training data and local storage out of the main source repo unless the user explicitly chooses a reviewed storage path.

Default to ignored local storage for raw training data. Move polished public research/code or reviewed private tooling to the public remote only after explicit user direction.

## Guardrails

- Do not print real `TINKER_API_KEY` values. Use existing helpers that report presence/status only.
- Do not commit `.env`, `.streamlit/secrets.toml`, local databases, logs, checkpoints, run outputs, or private dataset folders.
- Do not direct-push live/deployment branches for Float or other PR-protected projects; use a PR unless the user explicitly says otherwise.
- If `gh auth status` shows the wrong Git identity or insufficient scopes, stop and ask the user to re-authenticate.
