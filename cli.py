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

if __name__ == '__main__':
    cli()
