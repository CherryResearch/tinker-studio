# Tinker Studio Codex Skills

Repo-local skills for operating Tinker Studio from Codex. Copy or symlink the folders under `codex_skills/tinker-codex-skills/` into your Codex skills directory when you want them available outside this repo.

Skills included:

- `tinker-streamlit-dashboard`: launch and troubleshoot the Streamlit dashboard.
- `tinker-dataset-planning`: inspect dataset splits, source mixes, and run plans.
- `tinker-interview-collection`: collect interview Q&A into processed training rows.
- `tinker-publish-safety`: partition private data, verify env/secret safety, and publish via draft PRs.
- `tinker-training-runs`: launch, smoke-test, resume, and describe runs.
- `tinker-training-monitor`: inspect active/recent runs and stop state.
- `tinker-notebook-recovery`: recover notebook/run-manager state after interruption.

The interview assistant guide lives at `tinker-codex-skills/tinker-interview-collection/references/interview-assistant-guide.md`.

Install or refresh the package for Codex by copying `codex_skills/tinker-codex-skills/` to:

```text
%USERPROFILE%\.codex\skills\tinker-codex-skills
```

Keep real keys in ignored `.env` or local secret stores only. `.env.example` should contain placeholders and relative paths such as `data/training_data`.
