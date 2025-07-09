"""
Demo script showing reasoning capabilities
Run with: python examples/reasoning_demo.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reasoning import ReasoningEngine
from core.ai import AIEngine
from rich.console import Console

console = Console()

def demo_reasoning():
    """Demonstrate reasoning capabilities"""
    
    console.print("[bold blue]🧠 CodeGen Reasoning Engine Demo[/bold blue]\n")
    
    # Initialize engines
    ai_engine = AIEngine()
    reasoning_engine = ReasoningEngine(ai_engine)
    
    # Demo problems
    problems = [
        {
            'problem': 'Create a web scraper that handles rate limiting and retries',
            'strategy': 'step_by_step'
        },
        {
            'problem': 'Optimize a slow database query with millions of records',
            'strategy': 'critical_thinking'
        },
        {
            'problem': 'Design a distributed caching system',
            'strategy': 'problem_decomposition'
        },
        {
            'problem': 'Implement a recommendation algorithm',
            'strategy': 'first_principles'
        }
    ]
    
    for i, demo in enumerate(problems, 1):
        console.print(f"[bold yellow]Demo {i}: {demo['problem']}[/bold yellow]")
        console.print(f"[dim]Strategy: {demo['strategy']}[/dim]\n")
        
        try:
            reasoning_chain = reasoning_engine.reason_through_problem(
                demo['problem'], 
                strategy=demo['strategy'], 
                show_thinking=True
            )
            
            console.print(f"[green]✓[/green] Reasoning completed with {reasoning_chain.confidence:.1%} confidence")
            console.print("=" * 80)
            console.print()
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            console.print("=" * 80)
            console.print()
    
    # Show reasoning history
    console.print("[bold blue]📊 Reasoning History Summary[/bold blue]")
    history = reasoning_engine.get_reasoning_history()
    
    if history:
        from rich.table import Table
        
        summary_table = Table(title="Session Summary")
        summary_table.add_column("Problem", style="white")
        summary_table.add_column("Strategy", style="cyan")
        summary_table.add_column("Confidence", style="green")
        summary_table.add_column("Time", style="yellow")
        
        for chain in history:
            problem_short = chain.problem[:40] + "..." if len(chain.problem) > 40 else chain.problem
            summary_table.add_row(
                problem_short,
                chain.metadata.get('strategy', 'unknown'),
                f"{chain.confidence:.1%}",
                f"{chain.reasoning_time:.2f}s"
            )
        
        console.print(summary_table)
    else:
        console.print("[dim]No reasoning history available[/dim]")

if __name__ == '__main__':
    demo_reasoning()
