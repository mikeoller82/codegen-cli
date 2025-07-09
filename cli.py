from core.repl import start_repl

import click

@click.group()
def cli():
    """Example CLI tool"""
    pass

@cli.command()
def hello():
    """Says hello"""
    click.echo("Hello, world!")

@cli.command()
def goodbye():
    """Says goodbye"""
    click.echo("Goodbye, world!")

@cli.command()
def repl():
    """Start interactive REPL mode"""
    start_repl()

@cli.command()
@click.argument('problem')
@click.option('--strategy', default='step_by_step', help='Reasoning strategy to use')
@click.option('--no-thinking', is_flag=True, help='Hide thinking process')
def think(problem, strategy, no_thinking):
    """Think through a problem step by step"""
    from core.reasoning import ReasoningEngine
    from core.ai import AIEngine
    from rich import console
    console = console.Console()
    
    ai_engine = AIEngine()
    reasoning_engine = ReasoningEngine(ai_engine)
    
    try:
        reasoning_chain = reasoning_engine.reason_through_problem(
            problem, strategy=strategy, show_thinking=not no_thinking
        )
        
        console.print(f"\n[bold green]✨ Reasoning complete![/bold green]")
        console.print(f"[dim]Confidence: {reasoning_chain.confidence:.1%}[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error in reasoning: {e}")

@cli.command()
@click.argument('prompt')
@click.option('--strategy', default='step_by_step', help='Reasoning strategy')
@click.option('--output', '-o', help='Output file to save generated code')
@click.option('--model', default='gpt-3.5-turbo', help='AI model to use')
def generate_with_reasoning(prompt, strategy, output, model):
    """Generate code with explicit reasoning process"""
    from core.ai import AIEngine
    from rich import console
    console = console.Console()
    
    ai_engine = AIEngine()
    
    with console.status("[bold green]Reasoning and generating..."):
        try:
            result, reasoning_chain = ai_engine.generate_code_with_reasoning(
                prompt, model, strategy, show_thinking=True
            )
            
            if output:
                with open(output, 'w') as f:
                    f.write(result)
                console.print(f"[green]✓[/green] Code saved to {output}")
                
                # Also save reasoning
                reasoning_file = output.replace('.py', '_reasoning.json')
                import json
                with open(reasoning_file, 'w') as f:
                    json.dump(ai_engine.reasoning_engine.export_reasoning(reasoning_chain), f, indent=2)
                console.print(f"[green]✓[/green] Reasoning saved to {reasoning_file}")
            else:
                console.print("\n[bold blue]Generated Code:[/bold blue]")
                from core.fs import FileSystemManager
                fs_manager = FileSystemManager()
                fs_manager.display_code(result)
                
        except Exception as e:
            console.print(f"[red]✗[/red] Error generating with reasoning: {e}")

@cli.command()
@click.argument('prompt')
@click.option('--strategy', default='step_by_step', help='Reasoning strategy')
@click.option('--output', '-o', help='Output file to save generated code')
@click.option('--model', default='gpt-3.5-turbo', help='AI model to use')
@click.option('--no-reasoning', is_flag=True, help='Skip reasoning process')
@click.option('--no-autotest', is_flag=True, help='Skip automatic testing')
def generate_auto(prompt, strategy, output, model, no_reasoning, no_autotest):
    """Generate code with automatic testing and fixing"""
    from core.ai import AIEngine
    from core.task_manager import TaskManager
    from rich.console import Console
    
    console = Console()
    ai_engine = AIEngine()
    task_manager = TaskManager()
    
    # Create task
    task = task_manager.add_task(prompt)
    
    console.print(f"\n[bold blue]🚀 Starting Task: {prompt}[/bold blue]")
    
    try:
        if not no_autotest:
            # Generate with auto-fix
            result, reasoning_chain, fix_attempts, final_test = ai_engine.generate_code_with_auto_fix(
                prompt, model, 
                use_reasoning=not no_reasoning,
                reasoning_strategy=strategy,
                show_progress=True
            )
            
            # Show results
            if final_test.result.value == 'pass':
                console.print(f"\n[bold green]✅ Task completed successfully![/bold green]")
                task_manager.complete_task(task.id, result)
            else:
                console.print(f"\n[bold yellow]⚠️  Task completed with issues[/bold yellow]")
                task_manager.fail_task(task.id, f"Final test: {final_test.result.value}")
        else:
            # Generate without auto-fix
            if not no_reasoning:
                result, reasoning_chain = ai_engine.generate_code_with_reasoning(
                    prompt, model, strategy, show_thinking=True
                )
            else:
                result = ai_engine.generate_code(prompt, model)
            
            task_manager.complete_task(task.id, result)
        
        # Save if requested
        if output:
            with open(output, 'w') as f:
                f.write(result)
            console.print(f"[green]✓[/green] Code saved to {output}")
        else:
            console.print("\n[bold blue]Generated Code:[/bold blue]")
            from core.fs import FileSystemManager
            fs_manager = FileSystemManager()
            fs_manager.display_code(result)
        
        # Show task recap
        task_manager.recap_last_task(task)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        task_manager.fail_task(task.id, str(e))

@cli.command()
def learning():
    """Show AI learning system status and insights"""
    from core.ai import AIEngine
    from rich.console import Console
    
    console = Console()
    ai_engine = AIEngine()
    
    console.print("[bold blue]🧠 CodeGen AI Learning System[/bold blue]\n")
    
    # Show learning status
    status = ai_engine.get_learning_status()
    if status["status"] == "disabled":
        console.print("[yellow]Learning system not available[/yellow]")
        return
    
    # Display insights
    ai_engine.display_learning_insights()
    
    # Show some example patterns if available
    if hasattr(ai_engine, 'learning_engine') and ai_engine.learning_engine:
        patterns = ai_engine.learning_engine.db.get_fix_patterns(min_confidence=0.7)
        if patterns:
            console.print(f"\n[bold green]🎯 Top Learned Patterns:[/bold green]")
            for i, pattern in enumerate(patterns[:3], 1):
                console.print(f"{i}. {pattern.error_type} → {pattern.fix_strategy} ({pattern.confidence:.1%} confidence)")

if __name__ == '__main__':
    cli()
