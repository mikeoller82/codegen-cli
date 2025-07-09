"""
Task Manager for continuous development workflow
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text

console = Console()
logger = logging.getLogger("codegen.tasks")

class TaskStatus(Enum):
    """Task status types"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"

@dataclass
class Task:
    """A development task"""
    id: int
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    completed_at: datetime = None
    code_generated: str = ""
    fix_attempts: List = None
    test_result: Any = None
    notes: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.fix_attempts is None:
            self.fix_attempts = []

class TaskManager:
    """Manages development tasks and workflow"""
    
    def __init__(self):
        self.tasks: List[Task] = []
        self.current_task_id = 0
        self.session_start = datetime.now()
    
    def add_task(self, description: str) -> Task:
        """Add a new task"""
        self.current_task_id += 1
        task = Task(
            id=self.current_task_id,
            description=description
        )
        self.tasks.append(task)
        logger.info(f"Added task {task.id}: {description}")
        return task
    
    def get_current_task(self) -> Optional[Task]:
        """Get the current active task"""
        for task in reversed(self.tasks):
            if task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                return task
        return None
    
    def complete_task(self, task_id: int, code: str = "", notes: str = "") -> bool:
        """Mark a task as completed"""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.code_generated = code
            task.notes = notes
            logger.info(f"Completed task {task_id}")
            return True
        return False
    
    def fail_task(self, task_id: int, reason: str = "") -> bool:
        """Mark a task as failed"""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.notes = reason
            logger.info(f"Failed task {task_id}: {reason}")
            return True
        return False
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def display_tasks(self, show_all: bool = False):
        """Display current tasks"""
        if not self.tasks:
            console.print("[dim]No tasks yet[/dim]")
            return
        
        table = Table(title="Development Tasks")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Status", style="white", width=12)
        table.add_column("Description", style="white")
        table.add_column("Created", style="dim", width=10)
        
        tasks_to_show = self.tasks if show_all else [t for t in self.tasks if t.status != TaskStatus.COMPLETED]
        
        for task in tasks_to_show:
            status_color = {
                TaskStatus.PENDING: "yellow",
                TaskStatus.IN_PROGRESS: "blue",
                TaskStatus.COMPLETED: "green",
                TaskStatus.FAILED: "red",
                TaskStatus.NEEDS_REVISION: "orange"
            }.get(task.status, "white")
            
            status_text = f"[{status_color}]{task.status.value}[/{status_color}]"
            created_str = task.created_at.strftime("%H:%M:%S")
            
            table.add_row(
                str(task.id),
                status_text,
                task.description,
                created_str
            )
        
        console.print(table)
    
    def display_task_details(self, task_id: int):
        """Display detailed information about a task"""
        task = self.get_task(task_id)
        if not task:
            console.print(f"[red]Task {task_id} not found[/red]")
            return
        
        # Task info panel
        info_content = f"[bold]Description:[/bold] {task.description}\n"
        info_content += f"[bold]Status:[/bold] {task.status.value}\n"
        info_content += f"[bold]Created:[/bold] {task.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if task.completed_at:
            duration = task.completed_at - task.created_at
            info_content += f"[bold]Completed:[/bold] {task.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            info_content += f"[bold]Duration:[/bold] {duration}\n"
        
        if task.notes:
            info_content += f"[bold]Notes:[/bold] {task.notes}\n"
        
        if task.fix_attempts:
            info_content += f"[bold]Fix Attempts:[/bold] {len(task.fix_attempts)}\n"
        
        console.print(Panel(info_content, title=f"Task {task.id}", border_style="blue"))
        
        # Show generated code if available
        if task.code_generated:
            console.print("\n[bold]Generated Code:[/bold]")
            from .fs import FileSystemManager
            fs_manager = FileSystemManager()
            fs_manager.display_code(task.code_generated)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks if t.status == TaskStatus.FAILED])
        pending_tasks = len([t for t in self.tasks if t.status == TaskStatus.PENDING])
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'pending_tasks': pending_tasks,
            'success_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'session_duration': datetime.now() - self.session_start
        }
    
    def ask_for_next_task(self) -> Optional[str]:
        """Interactive prompt for next task"""
        console.print("\n" + "="*60)
        console.print("[bold green]🎯 Ready for next task![/bold green]")
        
        # Show current session stats
        stats = self.get_session_stats()
        console.print(f"[dim]Session stats: {stats['completed_tasks']} completed, "
                     f"{stats['pending_tasks']} pending, "
                     f"{stats['success_rate']:.1f}% success rate[/dim]")
        
        console.print("\n[bold]What would you like me to work on next?[/bold]")
        console.print("[dim]Examples:[/dim]")
        console.print("  • Create a web scraper for news articles")
        console.print("  • Add authentication to the existing API")
        console.print("  • Optimize the database queries")
        console.print("  • Write unit tests for the calculator")
        console.print("  • [yellow]Type 'quit' to exit[/yellow]")
        
        task_description = Prompt.ask("\n[cyan]Next task")
        
        if task_description.lower() in ['quit', 'exit', 'q']:
            return None
        
        return task_description
    
    def recap_last_task(self, task: Task):
        """Provide a recap of the last completed task"""
        if not task:
            return
        
        console.print(f"\n[bold blue]📋 Task Recap: #{task.id}[/bold blue]")
        
        recap_table = Table(show_header=False, box=None)
        recap_table.add_column("Label", style="cyan", width=15)
        recap_table.add_column("Value", style="white")
        
        recap_table.add_row("Task:", task.description)
        recap_table.add_row("Status:", f"[green]{task.status.value}[/green]" if task.status == TaskStatus.COMPLETED else f"[red]{task.status.value}[/red]")
        
        if task.completed_at:
            duration = task.completed_at - task.created_at
            recap_table.add_row("Duration:", str(duration).split('.')[0])
        
        if task.fix_attempts:
            recap_table.add_row("Fixes Applied:", str(len(task.fix_attempts)))
        
        if task.notes:
            recap_table.add_row("Notes:", task.notes)
        
        console.print(recap_table)
