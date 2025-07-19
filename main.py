#!/usr/bin/env python3
"""
Enhanced Autonomous Code Generation CLI
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from core.autonomous_agent import AutonomousAgent
from core.ai import AIEngine
from core.task_manager import TaskManager

console = Console()
logger = logging.getLogger("codegen")


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("codegen.log"),
            logging.StreamHandler() if verbose else logging.NullHandler(),
        ],
    )


def display_banner():
    """Display application banner"""
    banner = """
🤖 Autonomous Code Generation CLI
Enhanced with Planning, Action Display & Iterative Improvement
    """
    console.print(Panel(banner, style="bold blue"))


def run_autonomous_mode(args):
    """Run in autonomous mode"""
    console.print("\n[bold blue]🚀 Starting Autonomous Mode[/bold blue]")

    # Initialize the autonomous agent
    ai_engine = AIEngine()
    agent = AutonomousAgent(ai_engine)

    # Configure agent based on arguments
    agent.max_iterations = args.max_iterations
    agent.verbose_logging = args.verbose
    agent.auto_continue = args.auto_continue

    if args.task:
        # Single task mode
        console.print(f"[dim]Executing single task: {args.task}[/dim]")
        final_code, success = agent.execute_autonomous_task(
            args.task, args.auto_continue
        )

        if success:
            console.print("\n[bold green]✅ Task completed successfully![/bold green]")
            if args.output:
                # Save to file
                output_path = Path(args.output)
                output_path.write_text(final_code)
                console.print(f"[dim]Code saved to: {output_path}[/dim]")
        else:
            console.print("\n[bold red]❌ Task failed[/bold red]")
            sys.exit(1)
    else:
        # Continuous mode
        console.print(
            "[dim]Starting continuous mode - the agent will ask for tasks[/dim]"
        )
        agent.run_continuous_mode()


def run_interactive_mode(args):
    """Run in interactive mode"""
    console.print("\n[bold blue]💬 Interactive Mode[/bold blue]")

    ai_engine = AIEngine()
    task_manager = TaskManager()

    console.print("[dim]Available commands:[/dim]")
    console.print("  • [cyan]generate[/cyan] - Generate code with auto-fix")
    console.print("  • [cyan]autonomous[/cyan] - Switch to autonomous mode")
    console.print("  • [cyan]tasks[/cyan] - View task history")
    console.print("  • [cyan]stats[/cyan] - View learning statistics")
    console.print("  • [cyan]quit[/cyan] - Exit")

    while True:
        try:
            command = Prompt.ask("\n[cyan]Command")

            if command.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break

            elif command.lower() == "generate":
                prompt = Prompt.ask("Enter your code generation request")

                console.print(
                    f"\n[bold blue]🔧 Generating Code with Auto-Fix[/bold blue]"
                )
                final_code, reasoning, fix_attempts, test_result = (
                    ai_engine.generate_code_with_auto_fix(prompt, show_progress=True)
                )

                # Add to task manager
                task = task_manager.add_task(prompt)
                task_manager.complete_task(task.id, final_code)

                # Ask if user wants to save
                if Confirm.ask("\n[yellow]Save generated code to file?[/yellow]"):
                    filename = Prompt.ask("Filename", default="generated_code.py")
                    Path(filename).write_text(final_code)
                    console.print(f"[green]✅ Saved to {filename}[/green]")

            elif command.lower() == "autonomous":
                console.print("[yellow]Switching to autonomous mode...[/yellow]")
                agent = AutonomousAgent(ai_engine)
                agent.run_continuous_mode()

            elif command.lower() == "tasks":
                task_manager.display_tasks(show_all=True)

            elif command.lower() == "stats":
                ai_engine.display_learning_insights()

            else:
                console.print(f"[red]Unknown command: {command}[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️  Interrupted[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]💥 Error: {e}[/red]")
            logger.error(f"Interactive mode error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Code Generation CLI with Planning and Iterative Improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --autonomous --task "Create a web scraper for news articles"
  %(prog)s --autonomous --continuous
  %(prog)s --interactive
  %(prog)s --autonomous --task "Build a REST API" --auto-continue --max-iterations 15
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--autonomous",
        action="store_true",
        help="Run in autonomous mode with planning and iteration",
    )
    mode_group.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    # Autonomous mode options
    parser.add_argument(
        "--task", type=str, help="Specific task to execute (autonomous mode)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously asking for tasks (autonomous mode)",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Automatically continue iterations without asking",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per task (default: 10)",
    )

    # General options
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for generated code"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="AI model to use (gpt-4, gpt-3.5-turbo, gemini-1.5-pro-latest)",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    display_banner()

    # Validate arguments
    if args.autonomous and not args.task and not args.continuous:
        console.print(
            "[red]Error: Autonomous mode requires either --task or --continuous[/red]"
        )
        sys.exit(1)

    try:
        if args.autonomous:
            run_autonomous_mode(args)
        elif args.interactive:
            run_interactive_mode(args)

    except KeyboardInterrupt:
        console.print("\n[yellow]⏸️  Application interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]💥 Fatal error: {e}[/red]")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
