# Unix Launchers

These shims use the same `tinker_launcher.py` dispatcher as the Windows `.bat`
files. They load the project-local `.env` before launching tools.

Examples:

```sh
sh unix/launch_streamlit_dashboard.sh
sh unix/launch_tinker_experiment.sh essay_recent_r16 --smoke-test
sh unix/launch_tinker_monitor.sh --recent 10
```

If your checkout preserves executable bits, `./unix/tinker streamlit` works too.
