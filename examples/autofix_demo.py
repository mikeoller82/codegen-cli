"""
Demo script showing auto-fix and task management capabilities
Run with: python examples/autofix_demo.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ai import AIEngine
from core.task_manager import TaskManager
from rich.console import Console

console = Console()

def demo_autofix_workflow():
    """Demonstrate the auto-fix workflow"""
    
    console.print("[bold blue]🔧 CodeGen Auto-Fix Demo[/bold blue]\n")
    
    # Initialize engines
    ai_engine = AIEngine()
    task_manager = TaskManager()
    
    # Demo tasks with intentional issues
    demo_tasks = [
        "Create a function that calculates fibonacci numbers",
        "Write a simple web scraper using requests",
        "Create a class for managing a todo list",
        "Implement a binary search algorithm"
    ]
    
    for i, task_desc in enumerate(demo_tasks, 1):
        console.print(f"[bold yellow]Demo {i}/4: {task_desc}[/bold yellow]")
        
        # Create task
        task = task_manager.add_task(task_desc)
        
        try:
            # Generate with auto-fix
            result, reasoning_chain, fix_attempts, final_test = ai_engine.generate_code_with_auto_fix(
                task_desc, show_progress=True
            )
            
            # Update task
            task.code_generated = result
            task.fix_attempts = fix_attempts
            task.test_result = final_test
            
            if final_test.result.value == 'pass':
                task_manager.complete_task(task.id, result, "Auto-generated and tested")
                console.print(f"[green]✅ Task {task.id} completed successfully![/green]")
            else:
                task_manager.fail_task(task.id, f"Final test failed: {final_test.result.value}")
                console.print(f"[red]❌ Task {task.id} failed[/red]")
            
            # Show recap
            task_manager.recap_last_task(task)
            
            console.print("\n" + "="*60 + "\n")
            
        except Exception as e:
            console.print(f"[red]Error in task {task.id}: {e}[/red]")
            task_manager.fail_task(task.id, str(e))
    
    # Final session stats
    console.print("[bold blue]📊 Final Session Statistics[/bold blue]")
    stats = task_manager.get_session_stats()
    
    from rich.table import Table
    stats_table = Table(title="Demo Results")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Tasks", str(stats['total_tasks']))
    stats_table.add_row("Completed", str(stats['completed_tasks']))
    stats_table.add_row("Failed", str(stats['failed_tasks']))
    stats_table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
    stats_table.add_row("Session Duration", str(stats['session_duration']).split('.')[0])
    
    console.print(stats_table)
    
    # Show all tasks
    console.print("\n[bold blue]📋 All Tasks[/bold blue]")
    task_manager.display_tasks(show_all=True)

if __name__ == '__main__':
    demo_autofix_workflow()
