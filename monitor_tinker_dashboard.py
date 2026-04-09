from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tinker

from monitor_tinker_runs import (
    ACTIVE_SECONDS_THRESHOLD,
    describe_payload_status,
    extract_resume_checkpoint,
    format_elapsed_seconds,
    get_run_monitor_df,
    is_payload_auto_resumable,
    last_event_age_seconds,
    list_recent_training_runs_df,
    load_latest_local_run_payload,
    payload_status,
    running_payload_appears_active,
)
from tinker_notebook_env import ensure_tinker_api_key
from tinker_stop_control import default_stop_signal_path, format_stop_request

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Rich dashboard for Tinker training runs.")
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Training run ID to track. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=6,
        help="Number of recent runs to show when --run-id is not provided.",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=15,
        help="Seconds between refreshes.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of refreshes to perform. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Workspace root used to locate the stop signal file.",
    )
    return parser.parse_args()


def truncate_middle(value: Any, max_length: int) -> str:
    text = str(value or "-")
    if len(text) <= max_length:
        return text
    if max_length <= 7:
        return text[:max_length]
    head = max_length // 2 - 1
    tail = max_length - head - 3
    return f"{text[:head]}...{text[-tail:]}"


def compact_model_name(value: Any) -> str:
    text = str(value or "-")
    return text.split("/")[-1] if "/" in text else text


def compact_stop_message(stop_signal_path: str) -> str:
    message = format_stop_request(stop_signal_path)
    if message == "No stop request is currently pending.":
        return "none pending"
    return message


def compact_status_detail(payload: dict[str, Any]) -> str:
    status = payload_status(payload)
    if status == "ok":
        return "completed cleanly"
    if status == "error":
        return "interrupted with an error"
    if status == "stopped":
        return "stopped on request"
    if status == "completed_with_followup_errors":
        return "training done, follow-up had an issue"
    return describe_payload_status(payload)


def compact_api_timestamp(value: Any) -> str:
    if value is None:
        return "-"
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return truncate_middle(value, 18)
    return timestamp.strftime("%m-%d %H:%MZ")


def make_badge(label: str, style: str) -> Text:
    return Text(f" {label} ", style=style)


def local_status_style(status: str) -> str:
    return {
        "running": "bold black on cyan",
        "ok": "bold black on green3",
        "stopped": "bold black on yellow",
        "error": "bold white on red3",
        "completed_with_followup_errors": "bold black on dark_orange",
    }.get(status, "bold black on grey70")


def service_status_style(status: str) -> str:
    return {
        "ACTIVE": "bold black on green3",
        "IDLE": "bold black on yellow",
        "UNKNOWN": "bold black on grey70",
    }.get(status, "bold black on grey70")


def local_panel_border_style(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "grey50"
    status = payload_status(payload)
    if status == "running":
        if running_payload_appears_active(payload):
            return "cyan"
        return "yellow"
    return {
        "ok": "green3",
        "stopped": "yellow",
        "error": "red3",
        "completed_with_followup_errors": "dark_orange",
    }.get(status, "grey50")


def api_status_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "status" not in frame.columns:
        return {"ACTIVE": 0, "IDLE": 0, "UNKNOWN": 0}
    return {
        "ACTIVE": int((frame["status"] == "ACTIVE").sum()),
        "IDLE": int((frame["status"] == "IDLE").sum()),
        "UNKNOWN": int((frame["status"] == "UNKNOWN").sum()),
    }


def render_metric_card(label: str, value: str, *, border_style: str) -> Any:
    body = Group(
        Align.center(Text(value, style="bold bright_white"), vertical="middle"),
        Align.center(Text(label.upper(), style="dim")),
    )
    return Panel(body, box=box.ROUNDED, border_style=border_style, padding=(0, 1))


def inline_line(left: Any, right: str, *, right_style: str = "bright_white") -> Any:
    table = Table.grid(expand=True)
    table.add_column(no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(left, Text(right, style=right_style))
    return table


def build_progress_renderable(completed_steps: Any, planned_steps: Any) -> Any:
    if not isinstance(planned_steps, int) or planned_steps <= 0:
        if isinstance(completed_steps, int):
            return Text(f"{completed_steps} steps completed", style="bright_white")
        return Text("No step plan recorded yet", style="dim")

    completed = 0 if not isinstance(completed_steps, int) else max(0, min(completed_steps, planned_steps))
    percent = (completed / planned_steps) * 100.0

    progress_grid = Table.grid(expand=True)
    progress_grid.add_column(ratio=1)
    progress_grid.add_column(justify="right", no_wrap=True)
    progress_grid.add_row(
        ProgressBar(
            total=planned_steps,
            completed=completed,
            complete_style="cyan",
            finished_style="green3",
            pulse_style="yellow",
        ),
        Text(f"{completed}/{planned_steps}  {percent:5.1f}%", style="bold bright_white"),
    )
    return progress_grid


def build_header_panel(*, now_utc: datetime, args: argparse.Namespace, workspace_root: Path) -> Any:
    header_grid = Table.grid(expand=True)
    header_grid.add_column(ratio=2)
    header_grid.add_column(ratio=1, justify="center")
    header_grid.add_column(ratio=1, justify="right")
    header_grid.add_row(
        Text("Tinker Control Deck", style="bold bright_white"),
        Text("dashboard monitor", style="bold cyan"),
        Text(now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"), style="bold bright_white"),
    )
    mode_label = (
        f"tracking {len(args.run_id)} run(s)"
        if args.run_id
        else f"watching {args.recent} recent API runs"
    )
    header_grid.add_row(
        Text(truncate_middle(str(workspace_root), 54), style="dim"),
        Text(mode_label, style="bright_white"),
        Text(f"refresh {max(1, args.refresh)}s", style="bright_white"),
    )
    return Panel(header_grid, box=box.ROUNDED, border_style="cyan", padding=(0, 1))


def build_local_run_panel(payload: dict[str, Any] | None) -> Any:
    if payload is None:
        empty = Group(
            Text("No local run record found yet.", style="bright_white"),
            Text("Start a run to populate the local card.", style="dim"),
        )
        return Panel(empty, title="Local Run", box=box.ROUNDED, border_style="grey50")

    status = payload_status(payload)
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(style="bold bright_white", width=13)
    grid.add_column(ratio=1)

    run_name = str(payload.get("run_name") or "unknown")
    model_name = compact_model_name(payload.get("resolved_name") or payload.get("model_alias"))
    dataset_variant = str(payload.get("dataset_variant") or "-")
    profile_text = truncate_middle(f"{model_name}  |  {dataset_variant}", 44)
    status_line = inline_line(
        make_badge(status.upper(), local_status_style(status)),
        compact_status_detail(payload),
        right_style="dim",
    )

    grid.add_row("Run", Text(run_name, style="bold bright_white"))
    grid.add_row("Status", status_line)
    grid.add_row("Profile", Text(truncate_middle(profile_text, 58), style="bright_white"))
    grid.add_row(
        "Progress",
        build_progress_renderable(payload.get("completed_steps"), payload.get("planned_steps")),
    )

    last_event = payload.get("last_event")
    if isinstance(last_event, str) and last_event.strip():
        age_text = format_elapsed_seconds(last_event_age_seconds(payload))
        grid.add_row(
            "Heartbeat",
            Text(f"{last_event}  |  {age_text} ago", style="bright_white"),
        )

    waiting_on = payload.get("waiting_on")
    if isinstance(waiting_on, str) and waiting_on.strip():
        wait_elapsed = format_elapsed_seconds(payload.get("wait_elapsed_seconds"))
        in_flight_step = payload.get("in_flight_step")
        suffix = ""
        if isinstance(in_flight_step, int):
            suffix = f"  |  step {in_flight_step}"
        grid.add_row(
            "Waiting",
            Text(f"{waiting_on}  |  {wait_elapsed}{suffix}", style="bright_white"),
        )

    checkpoint_path = extract_resume_checkpoint(payload)
    if checkpoint_path:
        grid.add_row(
            "Checkpoint",
            Text(truncate_middle(checkpoint_path, 60), style="bright_white"),
        )

    if is_payload_auto_resumable(payload):
        resume_hint = payload.get("auto_resume_hint") or payload.get("resume_hint") or "-"
        resume_line = inline_line(
            make_badge("AUTO-RESUME", "bold black on yellow"),
            truncate_middle(resume_hint, 54),
        )
    elif running_payload_appears_active(payload):
        resume_line = inline_line(
            make_badge("ACTIVE", "bold black on cyan"),
            "Looks alive. Do not start a duplicate.",
        )
    else:
        resume_line = inline_line(
            make_badge("NO ACTION", "bold black on green3"),
            "Latest record completed cleanly.",
        )
    grid.add_row("Recovery", resume_line)

    return Panel(
        grid,
        title="Local Run",
        box=box.ROUNDED,
        border_style=local_panel_border_style(payload),
        padding=(0, 1),
    )


def build_service_panel(
    *,
    api_frame: pd.DataFrame,
    api_title: str,
    api_error: str | None,
    stop_signal_path: str,
    args: argparse.Namespace,
    workspace_root: Path,
) -> Any:
    counts = api_status_counts(api_frame)
    cards = Table.grid(expand=True)
    cards.add_column(ratio=1)
    cards.add_column(ratio=1)
    cards.add_column(ratio=1)
    cards.add_row(
        render_metric_card("active", str(counts["ACTIVE"]), border_style="green3"),
        render_metric_card("idle", str(counts["IDLE"]), border_style="yellow"),
        render_metric_card("unk", str(counts["UNKNOWN"]), border_style="grey50"),
    )

    pulse = Text()
    if api_frame.empty:
        pulse.append("No API rows available yet", style="dim")
    else:
        for status in api_frame["status"].tolist():
            pulse.append("||| ", style=service_status_style(str(status)))

    details = Table.grid(padding=(0, 1), expand=True)
    details.add_column(style="bold bright_white", width=12)
    details.add_column(ratio=1)
    details.add_row("Scope", Text(api_title, style="bright_white"))
    details.add_row("Pulse", pulse)
    details.add_row("Stop", Text(compact_stop_message(stop_signal_path), style="bright_white"))
    details.add_row(
        "Commands",
        Text(
            "dashboard  |  plain monitor",
            style="bright_white",
        ),
    )
    if api_error:
        details.add_row("API error", Text(truncate_middle(api_error, 72), style="bold red"))
    details.add_row(
        "Refresh",
        Text(
            f"every {max(1, args.refresh)}s"
            + ("" if args.iterations == 0 else f"  |  {args.iterations} iteration(s)"),
            style="bright_white",
        ),
    )

    return Panel(
        Group(cards, Text(""), details),
        title="Service Pulse",
        box=box.ROUNDED,
        border_style="cyan",
        padding=(0, 1),
    )


def build_api_runs_panel(api_frame: pd.DataFrame, *, title: str, api_error: str | None) -> Any:
    if api_error:
        body = Group(
            Text("Could not load API run data.", style="bold red"),
            Text(api_error, style="bright_white"),
        )
        return Panel(body, title=title, box=box.ROUNDED, border_style="red3")

    if api_frame.empty:
        body = Group(
            Text("No runs returned by the API.", style="bright_white"),
            Text("Try increasing --recent or tracking an explicit --run-id.", style="dim"),
        )
        return Panel(body, title=title, box=box.ROUNDED, border_style="grey50")

    has_checkpoint_columns = "num_checkpoints" in api_frame.columns
    table = Table(
        expand=True,
        box=box.SIMPLE_HEAVY,
        show_edge=False,
        header_style="bold bright_white",
        row_styles=["none", "dim"],
        pad_edge=False,
    )
    table.add_column("State", no_wrap=True)
    table.add_column("Run", ratio=2)
    table.add_column("Model", ratio=1)
    table.add_column("Rank", justify="right", no_wrap=True)
    table.add_column("API age", justify="right", no_wrap=True)
    if has_checkpoint_columns:
        table.add_column("Ckpts", justify="right", no_wrap=True)
        table.add_column("Last checkpoint", ratio=2)
    else:
        table.add_column("Last request", ratio=2)

    for row in api_frame.itertuples(index=False):
        row_dict = row._asdict()
        status = str(row_dict.get("status") or "UNKNOWN")
        state_cell = make_badge(status, service_status_style(status))
        run_cell = Text(truncate_middle(row_dict.get("training_run_id"), 32), style="bright_white")
        model_cell = Text(compact_model_name(row_dict.get("base_model")), style="bright_white")
        rank_value = row_dict.get("lora_rank")
        rank_cell = "-" if rank_value in (None, "") else str(rank_value)
        age_cell = format_elapsed_seconds(row_dict.get("seconds_since_last_request"))

        if has_checkpoint_columns:
            last_checkpoint = row_dict.get("last_checkpoint") or row_dict.get("last_sampler_checkpoint") or "-"
            checkpoint_count = row_dict.get("num_checkpoints")
            table.add_row(
                state_cell,
                run_cell,
                model_cell,
                rank_cell,
                age_cell,
                "-" if checkpoint_count is None else str(checkpoint_count),
                Text(truncate_middle(last_checkpoint, 42), style="bright_white"),
            )
        else:
            last_request = compact_api_timestamp(row_dict.get("last_request_time_utc"))
            table.add_row(
                state_cell,
                run_cell,
                model_cell,
                rank_cell,
                age_cell,
                Text(last_request, style="bright_white"),
            )

    subtitle = Text(
        f"ACTIVE means the API saw a request within {int(ACTIVE_SECONDS_THRESHOLD)}s.",
        style="dim",
    )
    return Panel(Group(table, Text(""), subtitle), title=title, box=box.ROUNDED, border_style="cyan")


def build_footer_panel(*, now_utc: datetime) -> Any:
    footer = Align.left(
        Text(
            "Ctrl+C exits  |  Plain monitor stays available  |  "
            f"Refreshed {now_utc.strftime('%H:%M:%S UTC')}",
            style="bright_white",
        )
    )
    return Panel(footer, box=box.ROUNDED, border_style="grey50", padding=(0, 1))


def fetch_api_snapshot(rest_client: Any, *, run_ids: list[str], recent: int) -> tuple[str, pd.DataFrame, str | None]:
    try:
        if run_ids:
            return "Tracked API Runs", get_run_monitor_df(rest_client, run_ids), None
        return "Recent API Runs", list_recent_training_runs_df(rest_client, limit=recent), None
    except Exception as exc:
        title = "Tracked API Runs" if run_ids else "Recent API Runs"
        return title, pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def build_dashboard(
    *,
    payload: dict[str, Any] | None,
    api_title: str,
    api_frame: pd.DataFrame,
    api_error: str | None,
    stop_signal_path: str,
    args: argparse.Namespace,
    workspace_root: Path,
    now_utc: datetime,
) -> Any:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="summary", size=11),
        Layout(name="api"),
        Layout(name="footer", size=3),
    )
    layout["summary"].split_row(
        Layout(name="local", ratio=5),
        Layout(name="service", ratio=4),
    )

    layout["header"].update(build_header_panel(now_utc=now_utc, args=args, workspace_root=workspace_root))
    layout["local"].update(build_local_run_panel(payload))
    layout["service"].update(
        build_service_panel(
            api_frame=api_frame,
            api_title=api_title,
            api_error=api_error,
            stop_signal_path=stop_signal_path,
            args=args,
            workspace_root=workspace_root,
        )
    )
    layout["api"].update(build_api_runs_panel(api_frame, title=api_title, api_error=api_error))
    layout["footer"].update(build_footer_panel(now_utc=now_utc))
    return layout


def main() -> int:
    if not RICH_AVAILABLE:
        from monitor_tinker_runs import main as plain_main

        print("[INFO] rich is not installed here, so the plain monitor is running instead.")
        return plain_main()

    ensure_tinker_api_key(required=True)
    args = parse_args()
    workspace_root = Path(args.workspace).resolve()
    stop_signal_path = str(default_stop_signal_path(workspace_root))
    rest_client = tinker.ServiceClient().create_rest_client()
    console = Console()

    iteration = 0
    while True:
        now_utc = datetime.now(timezone.utc)
        payload = load_latest_local_run_payload(workspace_root)
        api_title, api_frame, api_error = fetch_api_snapshot(
            rest_client,
            run_ids=args.run_id,
            recent=args.recent,
        )

        console.clear(home=True)
        console.print(
            build_dashboard(
                payload=payload,
                api_title=api_title,
                api_frame=api_frame,
                api_error=api_error,
                stop_signal_path=stop_signal_path,
                args=args,
                workspace_root=workspace_root,
                now_utc=now_utc,
            )
        )

        iteration += 1
        if args.iterations and iteration >= args.iterations:
            break
        time.sleep(max(1, args.refresh))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
