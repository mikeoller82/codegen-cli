"""
Simplified REPL without external dependencies
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from .ai import AIEngine
from .fs import FileSystemManager
from .task_manager import TaskManager

try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

class SimpleREPL:
    """Simplified REPL using only built-in modules"""
    
    def __init__(self):
        self.ai_engine = AIEngine()
        self.fs_manager = FileSystemManager()
        self.task_manager = TaskManager()
        self.current_dir = Path.cwd()
        self.variables = {}
        self.config = {
            'verbose': False,
            'debug': False,
            'default_model': 'gpt-3.5-turbo',
            'timeout': 30
        }
        self.running = True
        self.history = []
    
    def print_message(self, message, style=None):
        """Print message with optional styling"""
        if HAS_RICH and console:
            if style:
                console.print(f"[{style}]{message}[/{style}]")
            else:
                console.print(message)
        else:
            print(message)
    
    def display_welcome(self):
        """Display welcome message"""
        self.print_message("=" * 60)
        self.print_message("CodeGen Interactive REPL", "bold blue")
        self.print_message("Type 'help' for commands, 'quit' to exit", "dim")
        self.print_message("=" * 60)
        
        # Show available models
        models = self.ai_engine.get_available_models()
        if models and 'mock' not in models:
            self.print_message(f"✓ AI models available: {', '.join(models)}", "green")
        else:
            self.print_message("⚠️ Using mock AI (configure API keys for full functionality)", "yellow")
        
        self.print_message(f"Working directory: {self.current_dir}")
        print()
    
    def get_prompt(self):
        """Get the prompt string"""
        cwd = str(self.current_dir.name) if self.current_dir.name else str(self.current_dir)
        model = self.config['default_model']
        return f"codegen:{cwd} [{model}] $ "
    
    def run(self):
        """Run the REPL session"""
        self.display_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input(self.get_prompt()).strip()
                    
                    if not user_input:
                        continue
                    
                    # Add to history
                    self.history.append(user_input)
                    
                    # Process command
                    self.process_command(user_input)
                    
                except KeyboardInterrupt:
                    self.print_message("\nUse 'quit' or 'exit' to leave REPL", "yellow")
                    continue
                except EOFError:
                    break
                    
        except Exception as e:
            self.print_message(f"REPL error: {e}", "red")
        
        self.print_message("Goodbye!", "dim")
    
    def process_command(self, command_line: str):
        """Process a command line input"""
        parts = command_line.split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:]
        
        # Handle aliases
        aliases = {
            'gen': 'generate',
            'mem': 'memory',
            'cls': 'clear',
            'cat': 'read',
            'dir': 'ls',
            'q': 'quit',
            'exit': 'quit'
        }
        command = aliases.get(command, command)
        
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
            elif command == 'run':
                self.cmd_run(args)
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
            elif command == 'config':
                self.cmd_config(args)
            elif command == 'tasks':
                self.cmd_tasks(args)
            elif command == 'learning':
                self.cmd_learning(args)
            else:
                self.print_message(f"Unknown command: {command}", "red")
                self.print_message("Type 'help' for available commands")
                
        except Exception as e:
            self.print_message(f"Error executing command: {e}", "red")
            if self.config['debug']:
                import traceback
                print(traceback.format_exc())
    
    def cmd_help(self, args):
        """Show help information"""
        print("\nAvailable Commands:")
        print("==================")
        commands = [
            ("generate <prompt>", "Generate code using AI"),
            ("read <file>", "Read and display file"),
            ("write <file> <content>", "Write content to file"),
            ("run <file>", "Execute Python file"),
            ("ls [dir]", "List files"),
            ("cd <dir>", "Change directory"),
            ("pwd", "Show current directory"),
            ("history", "Show command history"),
            ("models", "Show available AI models"),
            ("config [key] [value]", "Show/set configuration"),
            ("tasks [all|stats|id]", "Show tasks and statistics"),
            ("learning", "Show AI learning status"),
            ("clear", "Clear screen"),
            ("quit/exit", "Exit REPL"),
        ]
        
        for cmd, desc in commands:
            print(f"  {cmd:<25} {desc}")
        print()
    
    def cmd_generate(self, args):
        """Generate code using AI"""
        if not args:
            self.print_message("Usage: generate <prompt>", "red")
            return
        
        prompt_text = ' '.join(args)
        
        print("Generating code...")
        try:
            result = self.ai_engine.generate_code(prompt_text, self.config['default_model'])
            
            print("\nGenerated Code:")
            print("-" * 40)
            print(result)
            print("-" * 40)
            
            # Store in variables
            self.variables['last_generated'] = result
            
            # Ask if user wants to save
            save = input("Save generated code to file? (y/n): ").lower().startswith('y')
            if save:
                filename = input("Enter filename: ")
                if filename:
                    self.fs_manager.write_file(filename, result)
                    self.print_message(f"✓ Saved to {filename}", "green")
        
        except Exception as e:
            self.print_message(f"Generation failed: {e}", "red")
    
    def cmd_read(self, args):
        """Read and display a file"""
        if not args:
            self.print_message("Usage: read <filename>", "red")
            return
        
        filename = args[0]
        try:
            content = self.fs_manager.read_file(filename)
            print(f"\nContents of {filename}:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            self.variables['last_read'] = content
            
        except Exception as e:
            self.print_message(f"Error reading file: {e}", "red")
    
    def cmd_write(self, args):
        """Write content to a file"""
        if len(args) < 2:
            self.print_message("Usage: write <filename> <content>", "red")
            return
        
        filename = args[0]
        content = ' '.join(args[1:])
        
        try:
            self.fs_manager.write_file(filename, content)
            self.print_message(f"✓ Content written to {filename}", "green")
            
        except Exception as e:
            self.print_message(f"Error writing file: {e}", "red")
    
    def cmd_run(self, args):
        """Execute a Python file"""
        if not args:
            self.print_message("Usage: run <filename>", "red")
            return
        
        filename = args[0]
        
        print(f"Executing {filename}...")
        try:
            from .sandbox import CodeSandbox
            sandbox = CodeSandbox()
            result = sandbox.execute_file(filename, self.config['timeout'])
            
            print(f"\nExecution Results for {filename}:")
            print("-" * 40)
            if result['success']:
                print("✓ Success")
                if result['stdout']:
                    print("Output:", result['stdout'])
            else:
                print("✗ Failed")
                if result['stderr']:
                    print("Error:", result['stderr'])
            print("-" * 40)
            
            self.variables['last_execution'] = result
            
        except Exception as e:
            self.print_message(f"Error executing file: {e}", "red")
    
    def cmd_clear(self, args):
        """Clear screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def cmd_ls(self, args):
        """List files in directory"""
        directory = args[0] if args else "."
        try:
            files = self.fs_manager.list_files(directory)
            
            if not files:
                print(f"No files found in {directory}")
                return
            
            print(f"\nFiles in {directory}:")
            print("-" * 40)
            for file_path in sorted(files):
                path_obj = Path(file_path)
                file_type = "DIR" if path_obj.is_dir() else "FILE"
                print(f"{file_type:<6} {path_obj.name}")
            print("-" * 40)
            
        except Exception as e:
            self.print_message(f"Error listing files: {e}", "red")
    
    def cmd_cd(self, args):
        """Change current directory"""
        if not args:
            self.print_message("Usage: cd <directory>", "red")
            return
        
        new_dir = Path(args[0]).expanduser().resolve()
        
        if not new_dir.exists():
            self.print_message(f"Directory does not exist: {new_dir}", "red")
            return
        
        if not new_dir.is_dir():
            self.print_message(f"Not a directory: {new_dir}", "red")
            return
        
        self.current_dir = new_dir
        os.chdir(new_dir)
        self.print_message(f"✓ Changed to {new_dir}", "green")
    
    def cmd_pwd(self, args):
        """Show current directory"""
        print(self.current_dir)
    
    def cmd_history(self, args):
        """Show command history"""
        if not self.history:
            print("No command history")
            return
        
        print("\nCommand History:")
        print("-" * 40)
        for i, cmd in enumerate(self.history[-20:], 1):  # Show last 20
            print(f"{i:2d}. {cmd}")
        print("-" * 40)
    
    def cmd_models(self, args):
        """Show available AI models"""
        models = self.ai_engine.get_available_models()
        
        print("\nAvailable AI Models:")
        print("-" * 40)
        for model in models:
            status = "✓ Available" if model != 'mock' else "⚠️ Mock only"
            print(f"{model:<20} {status}")
        print("-" * 40)
        print(f"Current model: {self.config['default_model']}")
    
    def cmd_config(self, args):
        """Show or set configuration"""
        if not args:
            # Show all config
            print("\nConfiguration:")
            print("-" * 40)
            for key, value in self.config.items():
                print(f"{key:<15} {value}")
            print("-" * 40)
        
        elif len(args) == 1:
            # Show specific config
            key = args[0]
            if key in self.config:
                print(f"{key}: {self.config[key]}")
            else:
                self.print_message(f"Unknown config key: {key}", "red")
        
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
                self.print_message(f"✓ Set {key} = {value}", "green")
            else:
                self.print_message(f"Unknown config key: {key}", "red")
    
    def cmd_tasks(self, args):
        """Show and manage tasks"""
        if not args:
            tasks = self.task_manager.tasks
            if not tasks:
                print("No tasks yet")
                return
            
            print("\nCurrent Tasks:")
            print("-" * 60)
            for task in tasks[-10:]:  # Show last 10
                status_symbol = {
                    'pending': '⏳',
                    'in_progress': '🔄',
                    'completed': '✅',
                    'failed': '❌',
                    'needs_revision': '⚠️'
                }.get(task.status.value, '❓')
                
                print(f"{task.id:2d}. {status_symbol} {task.description[:50]}")
            print("-" * 60)
        
        elif args[0] == 'stats':
            stats = self.task_manager.get_session_stats()
            
            print("\nSession Statistics:")
            print("-" * 40)
            print(f"Total Tasks:     {stats['total_tasks']}")
            print(f"Completed:       {stats['completed_tasks']}")
            print(f"Failed:          {stats['failed_tasks']}")
            print(f"Pending:         {stats['pending_tasks']}")
            print(f"Success Rate:    {stats['success_rate']:.1f}%")
            print(f"Session Duration: {str(stats['session_duration']).split('.')[0]}")
            print("-" * 40)
    
    def cmd_learning(self, args):
        """Show AI learning system status"""
        if hasattr(self.ai_engine, 'learning_engine') and self.ai_engine.learning_engine:
            metrics = self.ai_engine.learning_engine.get_learning_metrics()
            
            print("\nAI Learning Status:")
            print("-" * 40)
            print(f"Total Patterns:      {metrics.total_patterns}")
            print(f"Active Patterns:     {metrics.active_patterns}")
            print(f"Average Confidence:  {metrics.avg_confidence:.1%}")
            print(f"Learning Rate:       {metrics.learning_rate:.2f}/hour")
            print(f"Recent Improvements: {metrics.recent_improvements}")
            print("-" * 40)
            
            if metrics.patterns_by_type:
                print("\nPatterns by Type:")
                for error_type, count in metrics.patterns_by_type.items():
                    print(f"  {error_type:<15} {count}")
        else:
            self.print_message("Learning system not available", "yellow")

def start_simple_repl():
    """Start the simplified REPL session"""
    repl = SimpleREPL()
    repl.run()
