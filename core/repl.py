"""
REPL (Read-Eval-Print Loop) interactive mode for CodeGen CLI
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter, PathCompleter, Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import get_app

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt

from .ai import AIEngine
from .fs import FileSystemManager
from .web import WebManager
from .sandbox import CodeSandbox
from .memory import MemoryManager
from .reasoning import ReasoningEngine, ThinkingStep
from .task_manager import TaskManager, TaskStatus
from .auto_fix import AutoFixEngine

console = Console()
logger = logging.getLogger("codegen.repl")

class CodeGenCompleter(Completer):
    """Custom completer for CodeGen REPL commands"""
    
    def __init__(self):
        self.commands = [
            'generate', 'gen', 'read', 'write', 'append', 'edit', 'run', 'exec',
            'fetch', 'web', 'memory', 'mem', 'clear', 'cls', 'help', 'exit', 'quit',
            'ls', 'dir', 'cd', 'pwd', 'cat', 'save', 'load', 'history', 'models',
            'context', 'vars', 'env', 'config', 'debug', 'verbose',
            'think', 'reason', 'analyze', 'strategies', 'reasoning',
            'generate-auto', 'gen-auto', 'tasks', 'autotest', 'next',
            'learning', 'patterns', 'insights'  # New learning commands
        ]
        self.path_completer = PathCompleter()
        self.word_completer = WordCompleter(self.commands, ignore_case=True)
    
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        
        if not words or (len(words) == 1 and not text.endswith(' ')):
            # Complete command names
            yield from self.word_completer.get_completions(document, complete_event)
        else:
            # Complete file paths for file-related commands
            file_commands = ['read', 'write', 'append', 'edit', 'run', 'exec', 'save', 'load']
            if words[0] in file_commands:
                yield from self.path_completer.get_completions(document, complete_event)

class REPLSession:
    """Interactive REPL session for CodeGen CLI"""
    
    def __init__(self):
        self.ai_engine = AIEngine()
        self.fs_manager = FileSystemManager()
        self.web_manager = WebManager()
        self.sandbox = CodeSandbox()
        self.memory = MemoryManager()
        self.task_manager = TaskManager()
        self.auto_test_mode = True  # Enable auto-testing by default
        
        self.history = InMemoryHistory()
        self.completer = CodeGenCompleter()
        self.current_dir = Path.cwd()
        self.variables = {}
        self.config = {
            'verbose': False,
            'debug': False,
            'auto_save': False,
            'default_model': 'gpt-3.5-turbo',
            'timeout': 30
        }
        
        # Command aliases
        self.aliases = {
            'gen': 'generate',
            'mem': 'memory',
            'cls': 'clear',
            'cat': 'read',
            'dir': 'ls',
            'exec': 'run',
            'web': 'fetch',
            'q': 'quit',
            'exit': 'quit',
            'gen-auto': 'generate-auto'
        }
        
        self.running = True
        self.setup_key_bindings()
    
    def setup_key_bindings(self):
        """Setup custom key bindings"""
        self.bindings = KeyBindings()
        
        @self.bindings.add('c-c')
        def _(event):
            """Handle Ctrl+C"""
            console.print("\n[yellow]Use 'quit' or 'exit' to leave REPL[/yellow]")
            event.app.output.write('\n')
            event.app.output.flush()
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = Text("CodeGen Interactive REPL", style="bold magenta")
        welcome_panel = Panel(
            welcome_text,
            subtitle="Type 'help' for commands, 'quit' to exit",
            border_style="blue"
        )
        console.print(welcome_panel)
        console.print()
        
        # Show available models
        models = self.ai_engine.get_available_models()
        if models and 'mock' not in models:
            console.print(f"[green]✓[/green] AI models available: {', '.join(models)}")
        else:
            console.print("[yellow]⚠️[/yellow] Using mock AI (configure API keys for full functionality)")
        
        # Show learning status
        learning_status = self.ai_engine.get_learning_status()
        if learning_status["status"] == "active":
            console.print(f"[green]🧠[/green] AI Learning: {learning_status['total_patterns']} patterns learned")
        
        console.print(f"[dim]Working directory: {self.current_dir}[/dim]")
        console.print()
    
    def get_prompt_text(self):
        """Get the prompt text with context"""
        cwd = str(self.current_dir.name) if self.current_dir.name else str(self.current_dir)
        model = self.config['default_model']
        
        return HTML(f'<ansigreen>codegen</ansigreen>:<ansiblue>{cwd}</ansiblue> <ansiyellow>[{model}]</ansiyellow> $ ')
    
    def run(self):
        """Run the REPL session"""
        self.display_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = prompt(
                        self.get_prompt_text(),
                        history=self.history,
                        completer=self.completer,
                        key_bindings=self.bindings,
                        complete_while_typing=True
                    ).strip()
                    
                    if not user_input:
                        continue
                    
                    # Process command
                    self.process_command(user_input)
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Use 'quit' or 'exit' to leave REPL[/yellow]")
                    continue
                except EOFError:
                    break
                    
        except Exception as e:
            console.print(f"[red]REPL error: {e}[/red]")
            logger.error(f"REPL session error: {e}")
        
        console.print("\n[dim]Goodbye![/dim]")
    
    def process_command(self, command_line: str):
        """Process a command line input"""
        parts = command_line.split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:]
        
        # Handle aliases
        command = self.aliases.get(command, command)
        
        try:
            # Route to appropriate handler
            if command == 'help':
                self.cmd_help(args)
            elif command == 'quit':
                self.running = False
            elif command == 'generate':
                self.cmd_generate(args)
            elif command == 'read':
                self.cmd_read(args)
            elif command == 'write':
                self.cmd_write(args)
            elif command == 'append':
                self.cmd_append(args)
            elif command == 'edit':
                self.cmd_edit(args)
            elif command == 'run':
                self.cmd_run(args)
            elif command == 'fetch':
                self.cmd_fetch(args)
            elif command == 'memory':
                self.cmd_memory(args)
            elif command == 'clear':
                self.cmd_clear(args)
            elif command == 'ls':
                self.cmd_ls(args)
            elif command == 'cd':
                self.cmd_cd(args)
            elif command == 'pwd':
                self.cmd_pwd(args)
            elif command == 'history':
                self.cmd_history(args)
            elif command == 'models':
                self.cmd_models(args)
            elif command == 'context':
                self.cmd_context(args)
            elif command == 'vars':
                self.cmd_vars(args)
            elif command == 'config':
                self.cmd_config(args)
            elif command == 'save':
                self.cmd_save(args)
            elif command == 'load':
                self.cmd_load(args)
            elif command == 'env':
                self.cmd_env(args)
            elif command == 'think':
                self.cmd_think(args)
            elif command == 'reason':
                self.cmd_reason(args)
            elif command == 'analyze':
                self.cmd_analyze(args)
            elif command == 'strategies':
                self.cmd_strategies(args)
            elif command == 'reasoning':
                self.cmd_reasoning_history(args)
            elif command == 'generate-auto' or command == 'gen-auto':
                self.cmd_generate_with_autofix(args)
            elif command == 'tasks':
                self.cmd_tasks(args)
            elif command == 'autotest':
                self.cmd_autotest(args)
            elif command == 'next':
                self.cmd_next(args)
            elif command == 'learning':
                self.cmd_learning(args)
            elif command == 'patterns':
                self.cmd_patterns(args)
            elif command == 'insights':
                self.cmd_insights(args)
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands")
                
        except Exception as e:
            console.print(f"[red]Error executing command: {e}[/red]")
            if self.config['debug']:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    def cmd_help(self, args: List[str]):
        """Show help information"""
        if args:
            # Show help for specific command
            command = args[0]
            help_text = self.get_command_help(command)
            console.print(help_text)
        else:
            # Show general help
            help_table = Table(title="CodeGen REPL Commands")
            help_table.add_column("Command", style="cyan")
            help_table.add_column("Description", style="white")
            help_table.add_column("Example", style="green")
            
            commands = [
                ("generate <prompt>", "Generate code using AI", "generate 'create a Flask app'"),
                ("read <file>", "Read and display file", "read app.py"),
                ("write <file> <content>", "Write content to file", "write test.py 'print(\"hello\")'"),
                ("append <file> <content>", "Append to file", "append app.py '# comment'"),
                ("edit <file> <prompt>", "Edit file with AI", "edit app.py 'add error handling'"),
                ("run <file>", "Execute Python file", "run script.py"),
                ("fetch <url>", "Fetch web content", "fetch https://api.github.com/users/octocat"),
                ("memory", "Show session memory", "memory"),
                ("clear", "Clear screen/memory", "clear"),
                ("ls [dir]", "List files", "ls"),
                ("cd <dir>", "Change directory", "cd /home/user"),
                ("pwd", "Show current directory", "pwd"),
                ("history", "Show command history", "history"),
                ("models", "Show available AI models", "models"),
                ("config [key] [value]", "Show/set configuration", "config verbose true"),
                ("vars", "Show session variables", "vars"),
                ("save <file>", "Save session to file", "save session.json"),
                ("load <file>", "Load session from file", "load session.json"),
                ("quit/exit", "Exit REPL", "quit"),
                ("think <problem>", "Think through a problem step by step", "think 'how to implement a search algorithm'"),
                ("reason <strategy> <problem>", "Reason through a problem with a strategy", "reason first_principles 'what is the best way to learn a new language'"),
                ("analyze <problem|file>", "Analyze a problem or code file", "analyze 'how to improve this code'"),
                ("strategies", "Show available reasoning strategies", "strategies"),
                ("reasoning", "Show reasoning history", "reasoning"),
                ("generate-auto <prompt>", "Generate code with auto-testing", "generate-auto 'create a calculator'"),
                ("tasks [all|stats|id]", "Show tasks and statistics", "tasks stats"),
                ("autotest [on|off]", "Toggle auto-testing", "autotest on"),
                ("next", "Ask for next task", "next"),
                ("learning", "Show AI learning status", "learning"),
                ("patterns", "Show learned fix patterns", "patterns"),
                ("insights", "Show learning insights", "insights"),
            ]
            
            for cmd, desc, example in commands:
                help_table.add_row(cmd, desc, example)
            
            console.print(help_table)
            console.print("\n[dim]Aliases: gen=generate, mem=memory, cls=clear, cat=read, q=quit[/dim]")
    
    def cmd_generate(self, args: List[str]):
        """Generate code using AI with optional reasoning"""
        if not args:
            console.print("[red]Usage: generate <prompt> [--reasoning] [--strategy <strategy>][/red]")
            return
        
        # Parse arguments
        use_reasoning = '--reasoning' in args
        strategy = 'step_by_step'
        
        if '--strategy' in args:
            strategy_idx = args.index('--strategy')
            if strategy_idx + 1 < len(args):
                strategy = args[strategy_idx + 1]
                args.remove('--strategy')
                args.remove(strategy)
        
        if use_reasoning:
            args.remove('--reasoning')
        
        prompt_text = ' '.join(args)
        
        with console.status("[bold green]Generating code..."):
            try:
                if use_reasoning:
                    result, reasoning_chain = self.ai_engine.generate_code_with_reasoning(
                        prompt_text, self.config['default_model'], strategy, show_thinking=True
                    )
                    self.variables['last_reasoning'] = reasoning_chain
                else:
                    result = self.ai_engine.generate_code(prompt_text, self.config['default_model'])
            
                console.print("\n[bold blue]Generated Code:[/bold blue]")
                self.fs_manager.display_code(result)
                
                # Store in memory and variables
                self.memory.add_generation(prompt_text, result)
                self.variables['last_generated'] = result
                
                # Ask if user wants to save
                if confirm("Save generated code to file?"):
                    filename = prompt("Enter filename: ")
                    if filename:
                        self.fs_manager.write_file(filename, result)
                        console.print(f"[green]✓[/green] Saved to {filename}")
            
            except Exception as e:
                console.print(f"[red]Generation failed: {e}[/red]")
    
    def cmd_read(self, args: List[str]):
        """Read and display a file"""
        if not args:
            console.print("[red]Usage: read <filename>[/red]")
            return
        
        filename = args[0]
        try:
            content = self.fs_manager.read_file(filename)
            console.print(f"\n[bold blue]Contents of {filename}:[/bold blue]")
            self.fs_manager.display_code(content, filename)
            
            self.memory.add_file_access(filename, 'read')
            self.variables['last_read'] = content
            
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
    
    def cmd_write(self, args: List[str]):
        """Write content to a file"""
        if len(args) < 2:
            console.print("[red]Usage: write <filename> <content>[/red]")
            return
        
        filename = args[0]
        content = ' '.join(args[1:])
        
        try:
            self.fs_manager.write_file(filename, content)
            console.print(f"[green]✓[/green] Content written to {filename}")
            self.memory.add_file_access(filename, 'write')
            
        except Exception as e:
            console.print(f"[red]Error writing file: {e}[/red]")
    
    def cmd_append(self, args: List[str]):
        """Append content to a file"""
        if len(args) < 2:
            console.print("[red]Usage: append <filename> <content>[/red]")
            return
        
        filename = args[0]
        content = ' '.join(args[1:])
        
        try:
            self.fs_manager.append_file(filename, content)
            console.print(f"[green]✓[/green] Content appended to {filename}")
            self.memory.add_file_access(filename, 'append')
            
        except Exception as e:
            console.print(f"[red]Error appending to file: {e}[/red]")
    
    def cmd_edit(self, args: List[str]):
        """Edit a file using AI"""
        if len(args) < 2:
            console.print("[red]Usage: edit <filename> <prompt>[/red]")
            return
        
        filename = args[0]
        edit_prompt = ' '.join(args[1:])
        
        try:
            existing_content = self.fs_manager.read_file(filename)
            
            with console.status("[bold green]Editing file with AI..."):
                full_prompt = f"Edit this code based on the request: '{edit_prompt}'\n\nExisting code:\n{existing_content}"
                edited_content = self.ai_engine.generate_code(full_prompt, self.config['default_model'])
                
                self.fs_manager.write_file(filename, edited_content)
                console.print(f"[green]✓[/green] File {filename} edited successfully")
                
                console.print(f"\n[bold blue]Edited {filename}:[/bold blue]")
                self.fs_manager.display_code(edited_content, filename)
                
                self.memory.add_generation(f"Edit {filename}: {edit_prompt}", edited_content)
                self.variables['last_edited'] = edited_content
                
        except Exception as e:
            console.print(f"[red]Error editing file: {e}[/red]")
    
    def cmd_run(self, args: List[str]):
        """Execute a Python file"""
        if not args:
            console.print("[red]Usage: run <filename>[/red]")
            return
        
        filename = args[0]
        timeout = self.config['timeout']
        
        with console.status(f"[bold green]Executing {filename}..."):
            try:
                result = self.sandbox.execute_file(filename, timeout)
                console.print(f"\n[bold blue]Execution Results for {filename}:[/bold blue]")
                self.sandbox.display_results(result)
                
                self.variables['last_execution'] = result
                
            except Exception as e:
                console.print(f"[red]Error executing file: {e}[/red]")
    
    def cmd_fetch(self, args: List[str]):
        """Fetch content from a URL"""
        if not args:
            console.print("[red]Usage: fetch <url> [format][/red]")
            return
        
        url = args[0]
        format_type = args[1] if len(args) > 1 else 'auto'
        
        with console.status(f"[bold green]Fetching {url}..."):
            try:
                content = self.web_manager.fetch_url(url, format_type)
                console.print(f"\n[bold blue]Content from {url}:[/bold blue]")
                self.web_manager.display_content(content, format_type)
                
                self.variables['last_fetch'] = content
                
            except Exception as e:
                console.print(f"[red]Error fetching URL: {e}[/red]")
    
    def cmd_memory(self, args: List[str]):
        """Display session memory"""
        if args and args[0] == 'clear':
            self.memory.clear_memory()
            console.print("[green]✓[/green] Memory cleared")
        else:
            self.memory.display_memory()
    
    def cmd_clear(self, args: List[str]):
        """Clear screen or memory"""
        if args and args[0] == 'memory':
            self.memory.clear_memory()
            console.print("[green]✓[/green] Memory cleared")
        else:
            os.system('clear' if os.name == 'posix' else 'cls')
    
    def cmd_ls(self, args: List[str]):
        """List files in directory"""
        directory = args[0] if args else "."
        try:
            files = self.fs_manager.list_files(directory)
            
            if not files:
                console.print(f"[dim]No files found in {directory}[/dim]")
                return
            
            table = Table(title=f"Files in {directory}")
            table.add_column("Name", style="white")
            table.add_column("Type", style="cyan")
            
            for file_path in sorted(files):
                path_obj = Path(file_path)
                file_type = "Directory" if path_obj.is_dir() else "File"
                table.add_row(path_obj.name, file_type)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error listing files: {e}[/red]")
    
    def cmd_cd(self, args: List[str]):
        """Change current directory"""
        if not args:
            console.print("[red]Usage: cd <directory>[/red]")
            return
        
        new_dir = Path(args[0]).expanduser().resolve()
        
        if not new_dir.exists():
            console.print(f"[red]Directory does not exist: {new_dir}[/red]")
            return
        
        if not new_dir.is_dir():
            console.print(f"[red]Not a directory: {new_dir}[/red]")
            return
        
        self.current_dir = new_dir
        os.chdir(new_dir)
        console.print(f"[green]✓[/green] Changed to {new_dir}")
    
    def cmd_pwd(self, args: List[str]):
        """Show current directory"""
        console.print(f"[blue]{self.current_dir}[/blue]")
    
    def cmd_history(self, args: List[str]):
        """Show command history"""
        history_entries = list(self.history.get_strings())
        
        if not history_entries:
            console.print("[dim]No command history[/dim]")
            return
        
        table = Table(title="Command History")
        table.add_column("#", style="dim")
        table.add_column("Command", style="white")
        
        for i, cmd in enumerate(history_entries[-20:], 1):  # Show last 20
            table.add_row(str(i), cmd)
        
        console.print(table)
    
    def cmd_models(self, args: List[str]):
        """Show available AI models"""
        models = self.ai_engine.get_available_models()
        
        table = Table(title="Available AI Models")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="white")
        
        for model in models:
            status = "✓ Available" if model != 'mock' else "⚠️ Mock only"
            table.add_row(model, status)
        
        console.print(table)
        console.print(f"\n[dim]Current model: {self.config['default_model']}[/dim]")
    
    def cmd_context(self, args: List[str]):
        """Show or manage context"""
        if args and args[0] == 'clear':
            self.variables.clear()
            console.print("[green]✓[/green] Context cleared")
        else:
            stats = self.memory.get_memory_stats()
            
            table = Table(title="Session Context")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Working Directory", str(self.current_dir))
            table.add_row("AI Model", self.config['default_model'])
            table.add_row("Session Duration", str(stats['session_duration']).split('.')[0])
            table.add_row("Generations", str(stats['total_generations']))
            table.add_row("File Operations", str(stats['total_file_operations']))
            table.add_row("Variables", str(len(self.variables)))
            
            console.print(table)
    
    def cmd_vars(self, args: List[str]):
        """Show session variables"""
        if not self.variables:
            console.print("[dim]No session variables[/dim]")
            return
        
        table = Table(title="Session Variables")
        table.add_column("Variable", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Preview", style="white")
        
        for name, value in self.variables.items():
            value_type = type(value).__name__
            preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            table.add_row(name, value_type, preview)
        
        console.print(table)
    
    def cmd_config(self, args: List[str]):
        """Show or set configuration"""
        if not args:
            # Show all config
            table = Table(title="Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in self.config.items():
                table.add_row(key, str(value))
            
            console.print(table)
        
        elif len(args) == 1:
            # Show specific config
            key = args[0]
            if key in self.config:
                console.print(f"[cyan]{key}[/cyan]: {self.config[key]}")
            else:
                console.print(f"[red]Unknown config key: {key}[/red]")
        
        elif len(args) == 2:
            # Set config
            key, value = args
            if key in self.config:
                # Convert value to appropriate type
                if isinstance(self.config[key], bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(self.config[key], int):
                    value = int(value)
                
                self.config[key] = value
                console.print(f"[green]✓[/green] Set {key} = {value}")
            else:
                console.print(f"[red]Unknown config key: {key}[/red]")
    
    def cmd_save(self, args: List[str]):
        """Save session to file"""
        if not args:
            console.print("[red]Usage: save <filename>[/red]")
            return
        
        filename = args[0]
        try:
            import json
            session_data = {
                'memory': self.memory.export_memory(),
                'variables': {k: str(v) for k, v in self.variables.items()},
                'config': self.config,
                'current_dir': str(self.current_dir)
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            console.print(f"[green]✓[/green] Session saved to {filename}")
            
        except Exception as e:
            console.print(f"[red]Error saving session: {e}[/red]")
    
    def cmd_load(self, args: List[str]):
        """Load session from file"""
        if not args:
            console.print("[red]Usage: load <filename>[/red]")
            return
        
        filename = args[0]
        try:
            import json
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            # Restore config
            if 'config' in session_data:
                self.config.update(session_data['config'])
            
            # Restore variables (as strings)
            if 'variables' in session_data:
                self.variables.update(session_data['variables'])
            
            console.print(f"[green]✓[/green] Session loaded from {filename}")
            
        except Exception as e:
            console.print(f"[red]Error loading session: {e}[/red]")
    
    def cmd_env(self, args: List[str]):
        """Show environment information"""
        import sys
        
        table = Table(title="Environment Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Python Version", sys.version.split()[0])
        table.add_row("Platform", sys.platform)
        table.add_row("Working Directory", str(self.current_dir))
        table.add_row("Python Path", sys.executable)
        
        # Check API keys
        openai_key = "✓ Configured" if os.getenv('OPENAI_API_KEY') else "✗ Not set"
        gemini_key = "✓ Configured" if os.getenv('GEMINI_API_KEY') else "✗ Not set"
        
        table.add_row("OpenAI API Key", openai_key)
        table.add_row("Gemini API Key", gemini_key)
        
        console.print(table)
    
    def cmd_think(self, args: List[str]):
        """Think through a problem step by step"""
        if not args:
            console.print("[red]Usage: think <problem description>[/red]")
            return
        
        problem = ' '.join(args)
        
        try:
            reasoning_chain = self.ai_engine.reasoning_engine.reason_through_problem(
                problem, strategy='step_by_step', show_thinking=True
            )
        
            # Store in variables
            self.variables['last_reasoning'] = reasoning_chain
        
        except Exception as e:
            console.print(f"[red]Error in thinking process: {e}[/red]")

    def cmd_reason(self, args: List[str]):
        """Reason through a problem with specified strategy"""
        if len(args) < 2:
            console.print("[red]Usage: reason <strategy> <problem description>[/red]")
            console.print("Available strategies: step_by_step, problem_decomposition, first_principles, analogical, critical_thinking")
            return
        
        strategy = args[0]
        problem = ' '.join(args[1:])
        
        try:
            reasoning_chain = self.ai_engine.reasoning_engine.reason_through_problem(
                problem, strategy=strategy, show_thinking=True
            )
        
            # Store in variables
            self.variables['last_reasoning'] = reasoning_chain
        
        except Exception as e:
            console.print(f"[red]Error in reasoning process: {e}[/red]")

    def cmd_analyze(self, args: List[str]):
        """Analyze a problem or code file"""
        if not args:
            console.print("[red]Usage: analyze <problem|filename>[/red]")
            return
        
        target = ' '.join(args)
        
        try:
            # Check if it's a file
            if Path(target).exists():
                content = self.fs_manager.read_file(target)
                problem = f"Analyze this code and suggest improvements:\n\n{content}"
            else:
                problem = f"Analyze this problem: {target}"
        
            reasoning_chain = self.ai_engine.reasoning_engine.reason_through_problem(
                problem, strategy='critical_thinking', show_thinking=True
            )
        
            self.variables['last_analysis'] = reasoning_chain
        
        except Exception as e:
            console.print(f"[red]Error in analysis: {e}[/red]")

    def cmd_strategies(self, args: List[str]):
        """Show available reasoning strategies"""
        strategies_table = Table(title="Reasoning Strategies")
        strategies_table.add_column("Strategy", style="cyan")
        strategies_table.add_column("Description", style="white")
        strategies_table.add_column("Best For", style="green")
        
        strategies = [
            ("step_by_step", "Systematic step-by-step analysis", "General problem solving"),
            ("problem_decomposition", "Break complex problems into parts", "Complex, multi-faceted problems"),
            ("first_principles", "Reason from fundamental concepts", "Novel or unfamiliar problems"),
            ("analogical", "Use analogies and similar problems", "Problems similar to known solutions"),
            ("critical_thinking", "Question assumptions and evaluate evidence", "Debugging and optimization")
        ]
        
        for strategy, desc, best_for in strategies:
            strategies_table.add_row(strategy, desc, best_for)
        
        console.print(strategies_table)

    def cmd_reasoning_history(self, args: List[str]):
        """Show reasoning history"""
        history = self.ai_engine.reasoning_engine.get_reasoning_history()
        
        if not history:
            console.print("[dim]No reasoning history[/dim]")
            return
        
        if args and args[0] == 'clear':
            self.ai_engine.reasoning_engine.reasoning_history.clear()
            console.print("[green]✓[/green] Reasoning history cleared")
            return
        
        history_table = Table(title="Reasoning History")
        history_table.add_column("Time", style="dim")
        history_table.add_column("Problem", style="white")
        history_table.add_column("Strategy", style="cyan")
        history_table.add_column("Steps", style="yellow")
        history_table.add_column("Confidence", style="green")
    
        for chain in history[-10:]:  # Show last 10
            problem_preview = chain.problem[:50] + "..." if len(chain.problem) > 50 else chain.problem
            confidence_bar = "█" * int(chain.confidence * 5) + "░" * (5 - int(chain.confidence * 5))
        
            history_table.add_row(
                f"{chain.reasoning_time:.1f}s",
                problem_preview,
                chain.metadata.get('strategy', 'unknown'),
                str(len(chain.steps)),
                f"{confidence_bar} {chain.confidence:.1%}"
            )
    
        console.print(history_table)

    def cmd_generate_with_autofix(self, args: List[str]):
        """Generate code with automatic testing and fixing"""
        if not args:
            console.print("[red]Usage: generate-auto <prompt> [--no-reasoning] [--no-autotest][/red]")
            return
        
        # Parse arguments
        use_reasoning = '--no-reasoning' not in args
        use_autotest = '--no-autotest' not in args and self.auto_test_mode
        
        # Remove flags from args
        args = [arg for arg in args if not arg.startswith('--')]
        prompt_text = ' '.join(args)
        
        # Create a new task
        task = self.task_manager.add_task(prompt_text)
        task.status = TaskStatus.IN_PROGRESS
        
        console.print(f"\n[bold blue]🚀 Starting Task #{task.id}[/bold blue]")
        console.print(f"[dim]{prompt_text}[/dim]")
        
        try:
            if use_autotest:
                # Generate with auto-fix
                result, reasoning_chain, fix_attempts, final_test = self.ai_engine.generate_code_with_auto_fix(
                    prompt_text, 
                    self.config['default_model'],
                    use_reasoning=use_reasoning,
                    show_progress=True
                )
                
                # Update task with results
                task.code_generated = result
                task.fix_attempts = fix_attempts
                task.test_result = final_test
                
                if final_test.result.value == 'pass':
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    console.print(f"\n[bold green]✅ Task #{task.id} completed successfully![/bold green]")
                else:
                    task.status = TaskStatus.NEEDS_REVISION
                    console.print(f"\n[bold yellow]⚠️  Task #{task.id} needs revision[/bold yellow]")
                
            else:
                # Generate without auto-fix
                if use_reasoning:
                    result, reasoning_chain = self.ai_engine.generate_code_with_reasoning(
                        prompt_text, self.config['default_model'], show_thinking=True
                    )
                else:
                    result = self.ai_engine.generate_code(prompt_text, self.config['default_model'])
                    reasoning_chain = None
                
                task.code_generated = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
            
            # Display the final code
            console.print("\n[bold blue]📄 Final Code:[/bold blue]")
            self.fs_manager.display_code(result)
            
            # Store in memory and variables
            self.memory.add_generation(prompt_text, result)
            self.variables['last_generated'] = result
            self.variables['current_task'] = task
            
            # Ask if user wants to save
            if Confirm.ask("Save generated code to file?"):
                filename = Prompt.ask("Enter filename")
                if filename:
                    self.fs_manager.write_file(filename, result)
                    console.print(f"[green]✓[/green] Saved to {filename}")
                    task.notes += f" Saved to {filename}."
            
            # Provide recap and ask for next task
            self.task_manager.recap_last_task(task)
            
            # Ask for next task
            next_task_desc = self.task_manager.ask_for_next_task()
            if next_task_desc:
                # Automatically start the next task
                self.cmd_generate_with_autofix(next_task_desc.split())
            else:
                console.print("\n[dim]Session ended. Type 'help' for commands or 'quit' to exit.[/dim]")
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.notes = str(e)
            console.print(f"[red]❌ Task #{task.id} failed: {e}[/red]")

    def cmd_tasks(self, args: List[str]):
        """Show and manage tasks"""
        if not args:
            self.task_manager.display_tasks()
        elif args[0] == 'all':
            self.task_manager.display_tasks(show_all=True)
        elif args[0] == 'stats':
            stats = self.task_manager.get_session_stats()
            
            stats_table = Table(title="Session Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Total Tasks", str(stats['total_tasks']))
            stats_table.add_row("Completed", str(stats['completed_tasks']))
            stats_table.add_row("Failed", str(stats['failed_tasks']))
            stats_table.add_row("Pending", str(stats['pending_tasks']))
            stats_table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
            stats_table.add_row("Session Duration", str(stats['session_duration']).split('.')[0])
            
            console.print(stats_table)
        elif args[0].isdigit():
            task_id = int(args[0])
            self.task_manager.display_task_details(task_id)
        else:
            console.print("[red]Usage: tasks [all|stats|<task_id>][/red]")

    def cmd_autotest(self, args: List[str]):
        """Toggle or configure auto-testing"""
        if not args:
            status = "enabled" if self.auto_test_mode else "disabled"
            console.print(f"Auto-testing is currently [bold]{status}[/bold]")
        elif args[0] in ['on', 'enable', 'true']:
            self.auto_test_mode = True
            console.print("[green]✓[/green] Auto-testing enabled")
        elif args[0] in ['off', 'disable', 'false']:
            self.auto_test_mode = False
            console.print("[yellow]⚠️[/yellow] Auto-testing disabled")
        else:
            console.print("[red]Usage: autotest [on|off][/red]")

    def cmd_next(self, args: List[str]):
        """Ask for next task"""
        next_task_desc = self.task_manager.ask_for_next_task()
        if next_task_desc:
            self.cmd_generate_with_autofix(next_task_desc.split())

    def cmd_learning(self, args: List[str]):
        """Show AI learning system status"""
        if args and args[0] == 'clear':
            if Confirm.ask("Are you sure you want to clear all learned patterns?"):
                # Clear learning database
                if hasattr(self.ai_engine, 'learning_engine') and self.ai_engine.learning_engine:
                    # This would need to be implemented in the learning engine
                    console.print("[yellow]Learning clear not implemented yet[/yellow]")
                else:
                    console.print("[yellow]Learning system not available[/yellow]")
            return
        
        if hasattr(self.ai_engine, 'learning_engine') and self.ai_engine.learning_engine:
            self.ai_engine.learning_engine.display_learning_status()
        else:
            console.print("[yellow]Learning system not available[/yellow]")

    def cmd_patterns(self, args: List[str]):
        """Show learned fix patterns"""
        if not hasattr(self.ai_engine, 'learning_engine') or not self.ai_engine.learning_engine:
            console.print("[yellow]Learning system not available[/yellow]")
            return
        
        patterns = self.ai_engine.learning_engine.db.get_fix_patterns(min_confidence=0.5)
        
        if not patterns:
            console.print("[dim]No learned patterns yet[/dim]")
            return
        
        patterns_table = Table(title="Learned Fix Patterns")
        patterns_table.add_column("Error Type", style="cyan")
        patterns_table.add_column("Fix Strategy", style="yellow")
        patterns_table.add_column("Confidence", style="green")
        patterns_table.add_column("Success Count", style="white")
        patterns_table.add_column("Last Used", style="dim")
        
        for pattern in patterns[:10]:  # Show top 10
            confidence_bar = "█" * int(pattern.confidence * 10) + "░" * (10 - int(pattern.confidence * 10))
            last_used = pattern.last_used.strftime("%m/%d %H:%M") if pattern.last_used else "Never"
            
            patterns_table.add_row(
                pattern.error_type,
                pattern.fix_strategy.replace('_', ' ').title(),
                f"{confidence_bar} {pattern.confidence:.1%}",
                str(pattern.success_count),
                last_used
            )
        
        console.print(patterns_table)

    def cmd_insights(self, args: List[str]):
        """Show detailed learning insights and recommendations"""
        if hasattr(self.ai_engine, 'learning_engine') and self.ai_engine.learning_engine:
            self.ai_engine.display_learning_insights()
        else:
            console.print("[yellow]Learning system not available[/yellow]")
    
    def get_command_help(self, command: str) -> str:
        """Get detailed help for a specific command"""
        help_texts = {
            'generate': """
Generate code using AI

Usage: generate <prompt> [--reasoning] [--strategy <strategy>]
Aliases: gen

Options:
  --reasoning     Use reasoning process before generation
  --strategy      Specify reasoning strategy (default: step_by_step)

Examples:
  generate create a Flask web server
  generate implement bubble sort algorithm --reasoning
  gen "create a class for handling user authentication" --strategy first_principles

The generated code will be displayed with syntax highlighting.
You'll be prompted to save it to a file if desired.
""",
            'learning': """
Show AI learning system status and insights

Usage: learning [clear]

Examples:
  learning        # Show learning status and insights
  learning clear  # Clear all learned patterns (with confirmation)

The learning system automatically improves code generation by learning from
successful fixes and common patterns. It shows:
- Total patterns learned
- Confidence levels
- Recent improvements
- Learning rate and recommendations
""",
            'patterns': """
Show learned fix patterns

Usage: patterns

Displays a table of learned fix patterns including:
- Error types and fix strategies
- Confidence levels based on success rates
- Usage counts and last used timestamps

These patterns are automatically applied when similar errors occur,
improving the success rate of automatic fixes.
""",
            'insights': """
Show detailed learning insights and recommendations

Usage: insights

Provides comprehensive information about the AI learning system:
- Learning metrics and statistics
- Pattern distribution by error type
- Top performing learned patterns
- Personalized recommendations for improving learning

Use this to understand how the AI is learning from your coding sessions.
"""
        }
        
        return help_texts.get(command, f"No detailed help available for '{command}'")

def start_repl():
    """Start the REPL session"""
    session = REPLSession()
    session.run()
