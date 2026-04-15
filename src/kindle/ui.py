"""Rich terminal UI — streams live progress so the user always knows what's happening."""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

MAX_QUESTIONS = 25  # Matches grill.py

STAGE_LABELS = {
    "grill": "🔥 Grill",
    "research": "🔬 Research",
    "architect": "🏗️  Architect",
    "dev": "⚡ Dev",
    "qa": "✅ QA",
    "package": "📦 Package",
}

STAGE_ORDER = list(STAGE_LABELS.keys())

MAX_DISPLAY_CHARS = 3000


class UI:
    def __init__(self, *, auto_approve: bool = False, verbose: bool = False) -> None:
        self.console = Console()
        self._current_stage: str | None = None
        self.auto_approve = auto_approve or not sys.stdin.isatty()
        self.verbose = verbose

    def _safe_print(self, *args: Any, **kwargs: Any) -> None:
        """Print that gracefully handles non-interactive/pipe scenarios."""
        with contextlib.suppress(BlockingIOError, BrokenPipeError, OSError):
            self.console.print(*args, **kwargs)

    # ------------------------------------------------------------------
    # High-level lifecycle
    # ------------------------------------------------------------------

    def banner(self, idea: str, project_id: str) -> None:
        self._safe_print()
        self._safe_print(
            Panel(
                f"[bold white]{idea}[/bold white]\n[dim]Session: {project_id}[/dim]",
                title="[bold cyan]🔥 Kindle — Application Factory[/bold cyan]",
                border_style="cyan",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def stage_start(self, stage: str) -> None:
        self._current_stage = stage
        label = STAGE_LABELS.get(stage, stage)
        with contextlib.suppress(BlockingIOError, BrokenPipeError, OSError):
            self.console.rule(f"[bold yellow] {label} ", style="yellow")

    def stage_done(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        self._safe_print(f"  [green]✓ {label} complete[/green]")
        self._safe_print()

    def stage_log(self, stage: str, message: str) -> None:
        if not self.verbose:
            return
        label = STAGE_LABELS.get(stage, stage)
        self._safe_print(f"  [dim]{label}:[/dim] {message}")

    # ------------------------------------------------------------------
    # Human gates
    # ------------------------------------------------------------------

    def show_artifact(self, title: str, content: str) -> None:
        display = content
        if len(content) > MAX_DISPLAY_CHARS:
            display = content[:MAX_DISPLAY_CHARS] + "\n\n[dim]… (truncated — see full artifact on disk)[/dim]"

        self._safe_print()
        self._safe_print(
            Panel(
                display,
                title=f"[bold]{title}[/bold]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def grill_question(
        self,
        question: str,
        recommended: str,
        category: str,
        number: int,
        why_asking: str = "",
    ) -> str:
        """Present a Grill question and return the human's answer."""
        self._safe_print()
        body = f"[bold]{question}[/bold]\n"
        if why_asking:
            body += f"\n[italic dim]Why I'm asking: {why_asking}[/italic dim]\n"
        body += f"\n[cyan]My recommendation:[/cyan] {recommended}"
        self._safe_print(
            Panel(
                body,
                title=f"[bold magenta]Grill ({number}/~{MAX_QUESTIONS})[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

        if self.auto_approve:
            self._safe_print(f"  [dim](auto-approve) Using recommended: {recommended}[/dim]")
            return recommended

        try:
            response = self.console.input(
                "  [bold]Your answer[/bold] [dim](Enter to accept recommended)[/dim]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            response = ""
        return response if response else recommended

    def prompt_arch_review(self, arch_summary: str) -> tuple[bool, str]:
        """Show the architecture and ask for approval.
        Returns (approved, feedback).
        """
        self.show_artifact("Architecture", arch_summary)

        if self.auto_approve:
            self._safe_print("[bold magenta]Architecture review[/bold magenta] [dim](auto-approved)[/dim]")
            return True, ""

        self._safe_print("[bold magenta]Architecture Review[/bold magenta]")
        self._safe_print("  Type [bold]approve[/bold] to proceed, or enter feedback to adjust:")
        try:
            response = self.console.input("  [bold]> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            self._safe_print("  [dim](no input — auto-approving)[/dim]")
            return True, ""
        if response.lower() in ("approve", "yes", "y", "lgtm", ""):
            return True, ""
        return False, response

    # ------------------------------------------------------------------
    # Build progress
    # ------------------------------------------------------------------

    def task_start(self, task_id: str, title: str, index: int, total: int) -> None:
        self._safe_print(f"  [cyan]({index}/{total})[/cyan] {task_id}: {title}")

    def task_done(self, task_id: str) -> None:
        self._safe_print(f"    [green]✓ {task_id} complete[/green]")

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    def deploy_complete(self, project_path: str) -> None:
        self._safe_print()
        self._safe_print(
            Panel(
                f"[bold green]{project_path}[/bold green]",
                title="[bold green]🎉 Project Ready[/bold green]",
                border_style="green",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def metrics_display(self, metrics: dict) -> None:
        """Display build metrics."""
        table = Table(
            title="Build Metrics",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in metrics.items():
            table.add_row(str(key), str(value))

        self._safe_print()
        self._safe_print(table)
        self._safe_print()

    def error(self, message: str) -> None:
        self._safe_print(f"[bold red]Error:[/bold red] {message}")

    def info(self, message: str) -> None:
        self._safe_print(f"  [cyan]>[/cyan] {message}")

    # ------------------------------------------------------------------
    # Project listing
    # ------------------------------------------------------------------

    def show_projects(self, projects: list[dict]) -> None:
        if not projects:
            self.console.print("[dim]No Kindle build sessions found.[/dim]")
            return

        table = Table(title="Kindle — Build Sessions", box=box.SIMPLE_HEAVY)
        table.add_column("ID", style="cyan")
        table.add_column("Idea", max_width=50)
        table.add_column("Status", style="green")
        table.add_column("Stages")
        table.add_column("Created")

        for p in projects:
            stages = ", ".join(p.get("stages_completed", []))
            table.add_row(
                p["project_id"],
                p.get("idea", "")[:50],
                p.get("status", "unknown"),
                stages or "—",
                p.get("created_at", "")[:19],
            )
        self.console.print(table)
