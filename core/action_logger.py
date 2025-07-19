"""
Action Logger for tracking all autonomous agent actions and decisions
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()
logger = logging.getLogger("codegen.actions")


@dataclass
class ActionLog:
    """Single action log entry"""

    timestamp: datetime
    action_type: (
        str  # 'plan', 'analyze', 'generate', 'test', 'fix', 'validate', 'decision'
    )
    description: str
    details: Dict[str, Any]
    status: str  # 'started', 'completed', 'failed', 'skipped'
    duration_ms: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionLog":
        """Create from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ActionLogger:
    """Logger for tracking all autonomous agent actions"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.actions: List[ActionLog] = []
        self.log_file = Path(f"logs/actions_{self.session_id}.json")
        self.log_file.parent.mkdir(exist_ok=True)

        # Current action tracking
        self.current_action: Optional[ActionLog] = None
        self.action_stack: List[ActionLog] = []

        # Statistics
        self.stats = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "total_duration_ms": 0,
            "actions_by_type": {},
            "session_start": datetime.now(),
        }

    def start_action(
        self, action_type: str, description: str, details: Dict[str, Any] = None
    ) -> ActionLog:
        """Start tracking a new action"""

        action = ActionLog(
            timestamp=datetime.now(),
            action_type=action_type,
            description=description,
            details=details or {},
            status="started",
        )

        # If there's a current action, push it to the stack (nested actions)
        if self.current_action:
            self.action_stack.append(self.current_action)

        self.current_action = action
        self.actions.append(action)

        # Update stats
        self.stats["total_actions"] += 1
        self.stats["actions_by_type"][action_type] = (
            self.stats["actions_by_type"].get(action_type, 0) + 1
        )

        # Display action start
        self._display_action_start(action)

        # Log to file
        logger.info(f"Started {action_type}: {description}")

        return action

    def complete_action(
        self, status: str = "completed", error: str = None, result: Any = None
    ):
        """Complete the current action"""

        if not self.current_action:
            logger.warning("No current action to complete")
            return

        # Calculate duration
        duration_ms = int(
            (datetime.now() - self.current_action.timestamp).total_seconds() * 1000
        )

        # Update action
        self.current_action.status = status
        self.current_action.duration_ms = duration_ms
        self.current_action.error = error

        if result is not None:
            self.current_action.details["result"] = str(result)[
                :500
            ]  # Truncate long results

        # Update stats
        if status == "completed":
            self.stats["successful_actions"] += 1
        elif status == "failed":
            self.stats["failed_actions"] += 1

        self.stats["total_duration_ms"] += duration_ms

        # Display action completion
        self._display_action_complete(self.current_action)

        # Log to file
        logger.info(
            f"Completed {self.current_action.action_type}: {status} ({duration_ms}ms)"
        )

        # Pop from stack if nested
        if self.action_stack:
            self.current_action = self.action_stack.pop()
        else:
            self.current_action = None

        # Save to file
        self._save_to_file()

    def log_decision(
        self,
        decision: str,
        reasoning: str,
        options: List[str] = None,
        chosen: str = None,
    ):
        """Log a decision made by the agent"""

        details = {"reasoning": reasoning, "options": options or [], "chosen": chosen}

        action = self.start_action("decision", decision, details)
        self.complete_action("completed")

        # Special display for decisions
        self._display_decision(decision, reasoning, chosen)

    def log_planning_phase(self, phase: str, result: Any):
        """Log a planning phase completion"""

        action = self.start_action("plan", f"Planning phase: {phase}", {"phase": phase})
        self.complete_action("completed", result=result)

    def log_code_generation(
        self, prompt: str, model: str, code_length: int, reasoning_used: bool
    ):
        """Log code generation"""

        details = {
            "prompt": prompt[:200],  # Truncate long prompts
            "model": model,
            "code_length": code_length,
            "reasoning_used": reasoning_used,
        }

        action = self.start_action(
            "generate", f"Generate code ({code_length} chars)", details
        )
        self.complete_action("completed")

    def log_test_result(self, test_type: str, result: str, error: str = None):
        """Log test execution result"""

        details = {"test_type": test_type, "result": result}

        action = self.start_action("test", f"Test: {test_type}", details)
        status = "completed" if result == "pass" else "failed"
        self.complete_action(status, error)

    def log_fix_attempt(self, strategy: str, success: bool, error_fixed: str = None):
        """Log a fix attempt"""

        details = {"strategy": strategy, "error_fixed": error_fixed}

        action = self.start_action("fix", f"Fix attempt: {strategy}", details)
        status = "completed" if success else "failed"
        self.complete_action(status)

    def _display_action_start(self, action: ActionLog):
        """Display action start in console"""

        # Action type emoji mapping
        emoji_map = {
            "plan": "📋",
            "analyze": "🔍",
            "generate": "⚡",
            "test": "🧪",
            "fix": "🔧",
            "validate": "✅",
            "decision": "🤔",
        }

        emoji = emoji_map.get(action.action_type, "📝")
        indent = "  " * len(self.action_stack)  # Indent nested actions

        console.print(f"{indent}[dim]{emoji} {action.description}...[/dim]")

    def _display_action_complete(self, action: ActionLog):
        """Display action completion in console"""

        status_map = {
            "completed": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
            "skipped": "[yellow]⊘[/yellow]",
        }

        status_symbol = status_map.get(action.status, "?")
        duration_text = f"({action.duration_ms}ms)" if action.duration_ms else ""
        indent = "  " * len(self.action_stack)

        if action.status == "failed" and action.error:
            console.print(
                f"{indent}[dim]{status_symbol} {action.description} {duration_text} - {action.error}[/dim]"
            )
        else:
            console.print(
                f"{indent}[dim]{status_symbol} {action.description} {duration_text}[/dim]"
            )

    def _display_decision(self, decision: str, reasoning: str, chosen: str = None):
        """Display a decision in a special format"""

        decision_text = f"[bold yellow]🤔 Decision:[/bold yellow] {decision}"
        if chosen:
            decision_text += f"\n[bold green]→ Chosen:[/bold green] {chosen}"
        decision_text += f"\n[dim]Reasoning: {reasoning}[/dim]"

        console.print(Panel(decision_text, border_style="yellow", padding=(0, 1)))

    def display_session_summary(self):
        """Display a summary of the current session"""

        console.print(f"\n[bold blue]📊 Session Summary[/bold blue]")

        # Basic stats
        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Value", style="white")

        session_duration = datetime.now() - self.stats["session_start"]
        success_rate = (
            (self.stats["successful_actions"] / self.stats["total_actions"] * 100)
            if self.stats["total_actions"] > 0
            else 0
        )
        avg_duration = (
            self.stats["total_duration_ms"] / self.stats["total_actions"]
            if self.stats["total_actions"] > 0
            else 0
        )

        summary_table.add_row("Session Duration:", str(session_duration).split(".")[0])
        summary_table.add_row("Total Actions:", str(self.stats["total_actions"]))
        summary_table.add_row(
            "Successful Actions:", str(self.stats["successful_actions"])
        )
        summary_table.add_row("Failed Actions:", str(self.stats["failed_actions"]))
        summary_table.add_row("Success Rate:", f"{success_rate:.1f}%")
        summary_table.add_row("Avg Action Duration:", f"{avg_duration:.0f}ms")

        console.print(summary_table)

        # Actions by type
        if self.stats["actions_by_type"]:
            console.print(f"\n[bold blue]📈 Actions by Type:[/bold blue]")
            type_table = Table()
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="white")
            type_table.add_column("Percentage", style="dim")

            for action_type, count in sorted(self.stats["actions_by_type"].items()):
                percentage = count / self.stats["total_actions"] * 100
                type_table.add_row(action_type, str(count), f"{percentage:.1f}%")

            console.print(type_table)

    def display_recent_actions(self, limit: int = 10):
        """Display recent actions"""

        console.print(f"\n[bold blue]📝 Recent Actions (last {limit}):[/bold blue]")

        recent_actions = (
            self.actions[-limit:] if len(self.actions) > limit else self.actions
        )

        actions_table = Table()
        actions_table.add_column("Time", style="dim", width=8)
        actions_table.add_column("Type", style="cyan", width=10)
        actions_table.add_column("Description", style="white")
        actions_table.add_column("Status", style="white", width=10)
        actions_table.add_column("Duration", style="dim", width=8)

        for action in recent_actions:
            time_str = action.timestamp.strftime("%H:%M:%S")

            status_color = {
                "completed": "green",
                "failed": "red",
                "started": "yellow",
                "skipped": "dim",
            }.get(action.status, "white")

            status_text = f"[{status_color}]{action.status}[/{status_color}]"
            duration_text = f"{action.duration_ms}ms" if action.duration_ms else "-"

            actions_table.add_row(
                time_str,
                action.action_type,
                action.description[:50] + "..."
                if len(action.description) > 50
                else action.description,
                status_text,
                duration_text,
            )

        console.print(actions_table)

    def get_action_timeline(self) -> List[Dict[str, Any]]:
        """Get a timeline of all actions"""

        timeline = []
        for action in self.actions:
            timeline.append(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "type": action.action_type,
                    "description": action.description,
                    "status": action.status,
                    "duration_ms": action.duration_ms,
                    "error": action.error,
                }
            )

        return timeline

    def _save_to_file(self):
        """Save actions to JSON file"""

        try:
            data = {
                "session_id": self.session_id,
                "stats": {
                    **self.stats,
                    "session_start": self.stats["session_start"].isoformat(),
                },
                "actions": [action.to_dict() for action in self.actions],
            }

            with open(self.log_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save action log: {e}")

    def load_from_file(self, file_path: Path):
        """Load actions from JSON file"""

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self.session_id = data["session_id"]
            self.stats = data["stats"]
            self.stats["session_start"] = datetime.fromisoformat(
                self.stats["session_start"]
            )

            self.actions = [
                ActionLog.from_dict(action_data) for action_data in data["actions"]
            ]

            logger.info(f"Loaded {len(self.actions)} actions from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load action log: {e}")

    def export_report(self, output_path: Path):
        """Export a detailed report"""

        report = {
            "session_summary": {
                "session_id": self.session_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration": str(datetime.now() - self.stats["session_start"]),
                "total_actions": self.stats["total_actions"],
                "success_rate": (
                    self.stats["successful_actions"] / self.stats["total_actions"] * 100
                )
                if self.stats["total_actions"] > 0
                else 0,
                "actions_by_type": self.stats["actions_by_type"],
            },
            "timeline": self.get_action_timeline(),
            "detailed_actions": [action.to_dict() for action in self.actions],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"[green]✓[/green] Report exported to {output_path}")


# Global action logger instance
_global_logger: Optional[ActionLogger] = None


def get_action_logger() -> ActionLogger:
    """Get the global action logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = ActionLogger()
    return _global_logger


def set_action_logger(logger: ActionLogger):
    """Set the global action logger instance"""
    global _global_logger
    _global_logger = logger
