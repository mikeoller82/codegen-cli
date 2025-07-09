"""
Demo script showing AI learning capabilities
Run with: python examples/learning_demo.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ai import AIEngine
from core.task_manager import TaskManager
from rich.console import Console
import time

console = Console()

def demo_learning_workflow():
    """Demonstrate the AI learning workflow"""
    
    console.print("[bold blue]🧠 CodeGen AI Learning Demo[/bold blue]\n")
    
    # Initialize engines
    ai_engine = AIEngine()
    task_manager = TaskManager()
    
    console.print("This demo shows how the AI learns from successful fixes and improves over time.\n")
    
    # Show initial learning state
    console.print("[bold yellow]📊 Initial Learning State:[/bold yellow]")
    ai_engine.display_learning_insights()
    
    # Demo tasks that will trigger learning
    learning_tasks = [
        "Create a function that reads a CSV file and returns data",
        "Write a web scraper for a simple website", 
        "Create a class for managing user authentication",
        "Implement a binary search algorithm",
        "Create a Flask API with error handling"
    ]
    
    console.print(f"\n[bold blue]🚀 Running {len(learning_tasks)} learning tasks...[/bold blue]\n")
    
    for i, task_desc in enumerate(learning_tasks, 1):
        console.print(f"[bold yellow]Task {i}/{len(learning_tasks)}: {task_desc}[/bold yellow]")
        
        # Create task
        task = task_manager.add_task(task_desc)
        
        try:
            # Generate with auto-fix (this will trigger learning)
            result, reasoning_chain, fix_attempts, final_test = ai_engine.generate_code_with_auto_fix(
                task_desc, show_progress=True
            )
            
            # Update task
            if final_test.result.value == 'pass':
                task_manager.complete_task(task.id, result, "Generated with learning")
                console.print(f"[green]✅ Task {task.id} completed - AI learned from this session![/green]")
            else:
                task_manager.fail_task(task.id, f"Final test failed: {final_test.result.value}")
                console.print(f"[yellow]⚠️  Task {task.id} completed with issues - AI still learned from attempts[/yellow]")
            
            console.print("\n" + "="*60 + "\n")
            
            # Small delay to make demo more readable
            time.sleep(1)
            
        except Exception as e:
            console.print(f"[red]Error in task {task.id}: {e}[/red]")
            task_manager.fail_task(task.id, str(e))
    
    # Show final learning state
    console.print("[bold green]🎓 Final Learning State:[/bold green]")
    ai_engine.display_learning_insights()
    
    # Show session statistics
    console.print("\n[bold blue]📈 Session Impact:[/bold blue]")
    stats = task_manager.get_session_stats()
    
    from rich.table import Table
    impact_table = Table(title="Learning Session Results")
    impact_table.add_column("Metric", style="cyan")
    impact_table.add_column("Value", style="white")
    
    impact_table.add_row("Tasks Completed", str(stats['completed_tasks']))
    impact_table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
    impact_table.add_row("Session Duration", str(stats['session_duration']).split('.')[0])
    
    # Show learning improvements
    learning_status = ai_engine.get_learning_status()
    if learning_status["status"] == "active":
        impact_table.add_row("Patterns Learned", str(learning_status["total_patterns"]))
        impact_table.add_row("Learning Rate", f"{learning_status['learning_rate']:.2f}/hour")
    
    console.print(impact_table)
    
    console.print(f"\n[bold green]🎉 Demo Complete![/bold green]")
    console.print("[dim]The AI has learned from this session and will perform better on similar tasks in the future.[/dim]")
    
    # Show some learned patterns
    if hasattr(ai_engine, 'learning_engine') and ai_engine.learning_engine:
        patterns = ai_engine.learning_engine.db.get_fix_patterns(min_confidence=0.6)
        if patterns:
            console.print(f"\n[bold blue]🎯 Example Learned Patterns:[/bold blue]")
            for i, pattern in enumerate(patterns[:3], 1):
                console.print(f"{i}. When encountering '{pattern.error_type}', apply '{pattern.fix_strategy}' "
                            f"(success rate: {pattern.confidence:.1%})")

def demo_prompt_improvement():
    """Demonstrate how prompts are improved based on learning"""
    
    console.print("\n[bold blue]💡 Prompt Improvement Demo[/bold blue]\n")
    
    ai_engine = AIEngine()
    
    # Example prompts that could be improved
    test_prompts = [
        "create a web scraper",
        "make an API server", 
        "write a data processor",
        "implement sorting algorithm"
    ]
    
    console.print("Showing how the AI improves prompts based on learned patterns:\n")
    
    for prompt in test_prompts:
        console.print(f"[cyan]Original:[/cyan] {prompt}")
        
        if hasattr(ai_engine, 'learning_engine') and ai_engine.learning_engine:
            improved_prompt, improvement_note = ai_engine.learning_engine.improve_prompt(prompt)
            
            if improvement_note:
                console.print(f"[green]Improved:[/green] {improved_prompt}")
                console.print(f"[dim]Applied: {improvement_note}[/dim]")
            else:
                console.print(f"[yellow]No improvements available yet for this type of prompt[/yellow]")
        else:
            console.print(f"[yellow]Learning system not available[/yellow]")
        
        console.print()

if __name__ == '__main__':
    try:
        demo_learning_workflow()
        demo_prompt_improvement()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
