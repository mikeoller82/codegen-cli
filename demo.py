#!/usr/bin/env python3
"""
Demo script showing the enhanced autonomous code generation system
"""

import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from core.autonomous_agent import AutonomousAgent
from core.ai import AIEngine

console = Console()


def demo_autonomous_system():
    """Demonstrate the autonomous system capabilities"""

    console.print(
        Panel(
            "[bold blue]🤖 Autonomous Code Generation Demo[/bold blue]\n\n"
            "This demo shows the enhanced autonomous system with:\n"
            "• [green]Planning phase[/green] - Analyzes tasks and creates execution plans\n"
            "• [green]Action display[/green] - Shows all actions and decisions in real-time\n"
            "• [green]Iterative improvement[/green] - Tests code and fixes errors automatically\n"
            "• [green]Learning system[/green] - Learns from fixes and improves over time\n"
            "• [green]Action logging[/green] - Tracks all actions for analysis\n\n"
            "[dim]Note: This demo uses mock AI responses since no API keys are configured[/dim]",
            title="Demo Overview",
            border_style="blue",
        )
    )

    # Initialize the system
    console.print("\n[bold yellow]🔧 Initializing System...[/bold yellow]")
    ai_engine = AIEngine()
    agent = AutonomousAgent(ai_engine)

    # Configure for demo
    agent.max_iterations = 3
    agent.auto_continue = True
    agent.verbose_logging = True

    console.print("[green]✅ System initialized successfully[/green]")
    console.print(f"[dim]Available models: {ai_engine.get_available_models()}[/dim]")

    # Demo task
    demo_task = "Create a simple calculator function that adds two numbers"

    console.print(f"\n[bold blue]🎯 Demo Task:[/bold blue] {demo_task}")

    try:
        # Execute the task
        final_code, success = agent.execute_autonomous_task(
            demo_task, auto_continue=True
        )

        if success:
            console.print("\n[bold green]🎉 Demo completed successfully![/bold green]")
        else:
            console.print(
                "\n[bold yellow]⚠️  Demo completed with limitations (no API keys)[/bold yellow]"
            )

        # Show session statistics
        console.print("\n[bold blue]📈 Session Statistics:[/bold blue]")
        agent.display_session_stats()

        # Show action logs
        console.print("\n[bold blue]📋 Action Log Summary:[/bold blue]")
        agent.action_logger.display_session_summary()

    except Exception as e:
        console.print(f"\n[red]❌ Demo failed: {e}[/red]")

    console.print("\n[bold blue]💡 Key Features Demonstrated:[/bold blue]")
    console.print(
        "• [green]Autonomous planning[/green] - System analyzed the task and created a plan"
    )
    console.print(
        "• [green]Action tracking[/green] - All actions were logged and displayed"
    )
    console.print(
        "• [green]Iterative approach[/green] - System would iterate until success"
    )
    console.print(
        "• [green]Error handling[/green] - Graceful handling of missing API keys"
    )
    console.print(
        "• [green]Rich display[/green] - Beautiful console output with progress tracking"
    )

    console.print(
        f"\n[dim]To use with real AI models, set OPENAI_API_KEY or GOOGLE_API_KEY environment variables[/dim]"
    )


def show_system_architecture():
    """Show the system architecture"""

    console.print("\n[bold blue]🏗️  System Architecture:[/bold blue]")

    architecture = """
[bold cyan]AutonomousAgent[/bold cyan]
├── [yellow]Planning Phase[/yellow]
│   ├── Task Analysis
│   ├── Decomposition into subtasks  
│   ├── Strategy Development
│   └── Risk Assessment
├── [yellow]Execution Phase[/yellow]
│   ├── Iterative Code Generation
│   ├── Automatic Testing
│   ├── Error Detection & Fixing
│   └── Validation
├── [yellow]Learning System[/yellow]
│   ├── Pattern Recognition
│   ├── Fix Strategy Learning
│   └── Continuous Improvement
└── [yellow]Action Logging[/yellow]
    ├── Real-time Action Display
    ├── Decision Tracking
    ├── Performance Metrics
    └── Session Analytics
    """

    console.print(Panel(architecture, title="System Components", border_style="cyan"))


def main():
    """Main demo function"""

    console.print(
        "[bold blue]🚀 Starting Enhanced Autonomous Code Generation Demo[/bold blue]\n"
    )

    # Show architecture
    show_system_architecture()

    # Run demo
    demo_autonomous_system()

    console.print("\n[bold green]✨ Demo Complete![/bold green]")
    console.print("[dim]Check the logs/ directory for detailed action logs[/dim]")


if __name__ == "__main__":
    main()
