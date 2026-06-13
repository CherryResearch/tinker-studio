# Tinker Studio

Local tools for building, monitoring, importing, and sampling small Tinker fine-tunes from personal writing datasets.

Run the dashboard:

```powershell
.\launch_streamlit_dashboard.bat
```

Private datasets, imports, run outputs, checkpoints, logs, local env files, and Google Takeout folders are ignored by Git. Strip notebook outputs before publishing.

Interview collection:

- Use [the interview assistant guide](codex_skills/tinker-codex-skills/tinker-interview-collection/references/interview-assistant-guide.md) for template questions and save/review commands.
- Run `.\launch_interview_collect.bat list-rounds` or `.\launch_interview_collect.bat append` to write local interview rows.

Codex skills are packaged in [codex_skills](codex_skills/README.md), including dashboard, dataset planning, interview collection, training, monitoring, and notebook recovery workflows.
