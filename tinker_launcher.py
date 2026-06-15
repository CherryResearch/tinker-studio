from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tinker_notebook_env import (
    PROJECT_ROOT,
    TINKER_API_KEY_ENV,
    ensure_tinker_api_key,
    load_project_dotenv,
)


DEFAULT_EXPERIMENT_RUN_NAME = "essay_recent_r16"


@dataclass(frozen=True)
class LaunchContext:
    workspace: Path
    python: str


@dataclass(frozen=True)
class LaunchSpec:
    command: list[str]
    requires_api_key: bool = False
    intro: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Tinker Studio tools.")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT),
        help="Project root containing .env and the launcher targets.",
    )
    parser.add_argument(
        "--python",
        dest="python_exe",
        help="Python executable to use for the launched tool.",
    )
    parser.add_argument("target", choices=sorted(target_names()))
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def target_names() -> set[str]:
    return {
        "dashboard",
        "endpoint",
        "experiment",
        "interview",
        "interview-collect",
        "monitor",
        "notebook",
        "sampler",
        "sampler-test",
        "streamlit",
        "streamlit-dashboard",
        "tinker-dashboard",
    }


def resolve_python(workspace: Path, override: str | None) -> str:
    if override:
        return override

    executable_name = "python.exe" if os.name == "nt" else "python"
    candidates = [
        workspace / "tinker_env" / ("Scripts" if os.name == "nt" else "bin") / executable_name,
        workspace / ".venv" / ("Scripts" if os.name == "nt" else "bin") / executable_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def build_launch_spec(target: str, ctx: LaunchContext, target_args: list[str]) -> LaunchSpec:
    normalized = {
        "dashboard": "tinker-dashboard",
        "interview": "interview-collect",
        "sampler": "sampler-test",
        "streamlit-dashboard": "streamlit",
    }.get(target, target)

    if normalized == "notebook":
        notebook_path = ctx.workspace / "tinker_train_and_eval.ipynb"
        return LaunchSpec(
            command=[
                ctx.python,
                "-m",
                "jupyter",
                "lab",
                str(notebook_path),
                f"--ServerApp.root_dir={ctx.workspace}",
                *target_args,
            ],
            requires_api_key=True,
            intro=f"Launching Jupyter Lab from:\n{ctx.workspace}",
        )

    if normalized == "experiment":
        run_args = normalize_experiment_args(target_args)
        return LaunchSpec(
            command=[
                ctx.python,
                str(ctx.workspace / "run_tinker_experiment.py"),
                "--workspace",
                str(ctx.workspace),
                *run_args,
            ],
            requires_api_key=True,
        )

    if normalized == "monitor":
        return LaunchSpec(
            command=[
                ctx.python,
                str(ctx.workspace / "monitor_tinker_runs.py"),
                "--recent",
                "6",
                "--refresh",
                "15",
                *target_args,
            ],
            requires_api_key=True,
        )

    if normalized == "tinker-dashboard":
        return LaunchSpec(
            command=[
                ctx.python,
                str(ctx.workspace / "monitor_tinker_dashboard.py"),
                "--recent",
                "6",
                "--refresh",
                "15",
                *target_args,
            ],
            requires_api_key=True,
        )

    if normalized == "endpoint":
        return LaunchSpec(
            command=[ctx.python, str(ctx.workspace / "serve_tinker_endpoint.py"), *target_args],
            requires_api_key=True,
        )

    if normalized == "streamlit":
        return LaunchSpec(
            command=[
                ctx.python,
                "-m",
                "streamlit",
                "run",
                str(ctx.workspace / "streamlit_tinker_dashboard.py"),
                *target_args,
            ],
        )

    if normalized == "sampler-test":
        return LaunchSpec(
            command=[ctx.python, str(ctx.workspace / "test_sampler_checkpoint.py"), *target_args],
            requires_api_key=True,
        )

    if normalized == "interview-collect":
        return LaunchSpec(
            command=[ctx.python, str(ctx.workspace / "collect_interview_qa.py"), *target_args],
        )

    raise ValueError(f"Unknown launcher target: {target}")


def normalize_experiment_args(target_args: list[str]) -> list[str]:
    if not target_args:
        return ["--run-name", DEFAULT_EXPERIMENT_RUN_NAME]
    if target_args[0].startswith("--"):
        return target_args
    return ["--run-name", target_args[0], *target_args[1:]]


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    ctx = LaunchContext(workspace=workspace, python=resolve_python(workspace, args.python_exe))
    dotenv_path = workspace / ".env"
    load_project_dotenv(dotenv_path)

    target_args = list(args.target_args)
    if target_args[:1] == ["--"]:
        target_args = target_args[1:]

    try:
        spec = build_launch_spec(args.target, ctx, target_args)
    except ValueError as exc:
        print(exc)
        return 2

    if spec.requires_api_key:
        try:
            ensure_tinker_api_key(required=True, dotenv_path=dotenv_path)
        except RuntimeError:
            print(
                f"Could not find {TINKER_API_KEY_ENV} in the process environment, "
                f"{dotenv_path}, or Windows User/Machine environment."
            )
            return 1

    if spec.intro:
        print(spec.intro)
        print()

    return subprocess.call(spec.command, cwd=str(workspace))


if __name__ == "__main__":
    raise SystemExit(main())
