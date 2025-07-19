#!/usr/bin/env python3
"""
Enhanced Autonomous LLM Coding Agent
OpenRouter-optimized autonomous coding agent with rich CLI experience
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import argparse
from openai import OpenAI
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dotenv import load_dotenv

load_dotenv()

# Rich CLI dependencies
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.status import Status
    from rich.traceback import install as install_rich_traceback
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    
    # Install rich traceback handling
    install_rich_traceback()
    
    # Create console with color support
    console = Console(force_terminal=True, color_system="truecolor")
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False
    print("⚠️  Rich not available. Install with: pip install rich")

# HTTP client for OpenRouter
try:
    import httpx
    HTTP_CLIENT = httpx.Client(timeout=120.0)
except ImportError:
    try:
        import requests
        HTTP_CLIENT = requests.Session()
        HTTP_CLIENT.timeout = 120.0 # type: ignore
    except ImportError:
        HTTP_CLIENT = None
        print("⚠️  No HTTP client available. Install httpx or requests")

# Import our enhanced memory system
try:
    from simple_memory import SimpleMemory
except ImportError:
    # Fallback simple memory implementation
    class SimpleMemory:
        def __init__(self):
            self.memories = []
        
        def add_memory(self, **kwargs):
            self.memories.append(kwargs)
        
        def find_similar_tasks(self, task):
            return []
        
        def get_memory_stats(self):
            return {"total": 0, "successful": 0, "success_rate": 0}
        
        def get_successful_patterns(self):
            return {}

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(sys.stdout) if not RICH_AVAILABLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenRouter model configurations
OPENROUTER_MODELS = {
    # High-performance models
    "kimi-k2:free": {
        "name": "moonshotai/kimi-k2:free",
        "max_tokens": 4096,
        "context_window": 65000,
        "cost_per_1k": {"input": 0.000, "output": 0.000},
        "reasoning": "excellent",
        "coding": "excellent"
    },
    "Venice Uncensored": {
        "name": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        "max_tokens": 4096,
        "context_window": 33000,
        "cost_per_1k": {"input": 0.000, "output": 0.000},
        "reasoning": "excellent",
        "coding": "excellent"
    },
    "TNG: DeepSeek R1T2 Chimera": {
        "name": "tngtech/deepseek-r1t2-chimera:free",
        "max_tokens": 4096,
        "context_window": 164000,
        "cost_per_1k": {"input": 0.000, "output": 0.000},
        "reasoning": "good",
        "coding": "excellent"
    },
    # Cost-effective models
    "Mistral: Devstral Small 2505": {
        "name": "mistralai/devstral-small-2505:free",
        "max_tokens": 4096,
        "context_window": 33000,
        "cost_per_1k": {"input": 0.000, "output": 0.000},
        "reasoning": "very-good",
        "coding": "very-good"
    },
    "Qwen: Qwen3 235B A22B": {
        "name": "qwen/qwen3-235b-a22b:free",
        "max_tokens": 4096,
        "context_window": 131072,
        "cost_per_1k": {"input": 0.000, "output": 0.000},
        "reasoning": "very-good",
        "coding": "good"
    },
    "qwen-2.5-coder-32b": {
        "name": "qwen/qwen-2.5-coder-32b-instruct:free",
        "max_tokens": 4096,
        "context_window": 32768,
        "cost_per_1k": {"input": 0.0002, "output": 0.0006},
        "reasoning": "good",
        "coding": "excellent"
    }
}

# Default model priority (cost-effective to premium)
DEFAULT_MODEL_PRIORITY = [
    "kimi-k2:free",
    "Venice Uncensored", 
    "TNG: DeepSeek R1T2 Chimera",
    "Qwen: Qwen3 235B A22B",
    "qwen-2.5-coder-32b",
    "Mistral: Devstral Small 2505"
]


@dataclass
class AgentMessage:
    """Enhanced message structure for agent communication"""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    token_count: Optional[int] = None
    cost: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToolCall:
    """Enhanced tool call structure"""
    name: str
    args: Dict[str, Any]
    result: Any = None
    error: str = None
    execution_time: float = None
    success: bool = None


@dataclass
class AgentContext:
    """Enhanced context management with rich state tracking"""
    messages: List[AgentMessage]
    current_task: str = ""
    working_directory: str = ""
    files_created: List[str] = None
    tools_used: List[str] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    session_start: datetime = None
    iterations_count: int = 0

    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.tools_used is None:
            self.tools_used = []
        if self.session_start is None:
            self.session_start = datetime.now()


class RichLogger:
    """Rich console output manager"""
    
    def __init__(self):
        self.console = console if RICH_AVAILABLE else None
        
    def print(self, *args, **kwargs):
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    def success(self, message: str):
        self.print(f"✅ {message}", style="bold green")
    
    def error(self, message: str):
        self.print(f"❌ {message}", style="bold red")
    
    def warning(self, message: str):
        self.print(f"⚠️  {message}", style="bold yellow")
    
    def info(self, message: str):
        self.print(f"ℹ️  {message}", style="bold blue")
    
    def thinking(self, message: str):
        self.print(f"🤔 {message}", style="bold magenta")
    
    def tool(self, tool_name: str, message: str):
        self.print(f"🔧 [bold cyan]{tool_name}[/bold cyan]: {message}")
    
    def code(self, code: str, language: str = "python"):
        if self.console:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.print(Panel(syntax, title=f"[bold blue]{language.title()} Code[/bold blue]"))
        else:
            print(f"Code ({language}):\n{code}")
    
    def panel(self, content: str, title: str = "", style: str = ""):
        if self.console:
            self.print(Panel(content, title=title, border_style=style))
        else:
            print(f"\n{title}\n{content}\n")
    
    def table(self, data: Dict[str, List], title: str = ""):
        if not self.console:
            print(f"\n{title}")
            for key, values in data.items():
                print(f"{key}: {', '.join(map(str, values))}")
            return
            
        table = Table(title=title)
        if data:
            # Add columns
            for key in data.keys():
                table.add_column(key, style="cyan")
            
            # Add rows
            max_len = max(len(values) for values in data.values()) if data else 0
            for i in range(max_len):
                row = []
                for values in data.values():
                    row.append(str(values[i]) if i < len(values) else "")
                table.add_row(*row)
        
        self.print(table)
    
    def progress_bar(self):
        if self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
        else:
            return None
    
    def status(self, message: str):
        if self.console:
            return self.console.status(message)
        else:
            print(f"Status: {message}")
            # a simple dummy context manager
            class DummyStatus:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return DummyStatus()


class EnhancedAgenticCoder:
    """Enhanced autonomous coding agent with OpenRouter integration"""

    def __init__(self, model: str = "auto", openrouter_api_key: str = None):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model
        self.model_config = None
        self.max_iterations = 15
        self.working_dir = Path.cwd() / "agent_workspace"
        self.working_dir.mkdir(exist_ok=True)
        self.memory = SimpleMemory()
        self.rich = RichLogger()
        
        # Initialize context after setting working directory
        self.context = AgentContext(messages=[], working_directory=str(self.working_dir))
        self.tools = self._init_tools()
        
        # Initialize model
        self._init_model()
        
        # Enhanced system prompt with better reasoning
        memory_stats = self.memory.get_memory_stats()
        self.system_prompt = self._create_system_prompt(memory_stats)
        self._add_message("system", self.system_prompt)
        
        # Display initialization info
        self._display_startup_info()

    def _display_startup_info(self):
        """Display rich startup information"""
        if not RICH_AVAILABLE:
            print(f"🤖 Enhanced Autonomous Coding Agent")
            print(f"📋 Model: {self.model_name}")
            print(f"📁 Working Directory: {self.working_dir}")
            return
        
        startup_table = Table(title="🤖 Enhanced Autonomous Coding Agent", show_header=False, border_style="blue")
        startup_table.add_column("Setting", style="bold cyan")
        startup_table.add_column("Value", style="green")
        
        startup_table.add_row("Model", self.model_name)
        startup_table.add_row("Provider", "OpenRouter" if self.openrouter_api_key else "Mock")
        startup_table.add_row("Working Directory", str(self.working_dir))
        startup_table.add_row("Max Iterations", str(self.max_iterations))
        
        if self.model_config:
            startup_table.add_row("Context Window", f"{self.model_config['context_window']:,} tokens")
            startup_table.add_row("Max Output", f"{self.model_config['max_tokens']:,} tokens")
            startup_table.add_row("Input Cost", f"${self.model_config['cost_per_1k']['input']}/1k tokens")
            startup_table.add_row("Output Cost", f"${self.model_config['cost_per_1k']['output']}/1k tokens")
        
        self.rich.print(startup_table)

    def _init_model(self):
        """Initialize model with OpenRouter or auto-selection"""
        if not self.openrouter_api_key:
            self.rich.warning("No OpenRouter API key found. Using mock mode.")
            self.model_name = "moonshotai/kimi-k2:free"
            return

        if self.model_name == "auto":
            # Auto-select best available model
            self.model_name = self._auto_select_model()
        
        if self.model_name in OPENROUTER_MODELS:
            self.model_config = OPENROUTER_MODELS[self.model_name]
        else:
            # Custom model
            self.model_config = {
                "name": self.model_name,
                "max_tokens": 4096,
                "context_window": 32000,
                "cost_per_1k": {"input": 0.001, "output": 0.002}
            }
        
        self.rich.info(f"Initialized model: {self.model_config['name']}")

    def _auto_select_model(self) -> str:
        """Auto-select the best available model from OpenRouter"""
        self.rich.info("Auto-selecting best model...")
        for model_key in DEFAULT_MODEL_PRIORITY:
            with self.rich.status(f"Testing {model_key}..."):
                if self._test_model_availability(model_key):
                    self.rich.success(f"Selected model: {model_key}")
                    return model_key
        
        # Fallback
        self.rich.warning("Could not connect to priority OpenRouter models, using fallback.")
        return "moonshotai/kimi-k2:free"

    def _test_model_availability(self, model_key: str) -> bool:
        """Test if a model is available via OpenRouter"""
        if not HTTP_CLIENT: return False
            
        try:
            model_config = OPENROUTER_MODELS.get(model_key, {})
            test_payload = {
                "model": model_config.get("name", model_key),
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            headers = { "Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json" }
            
            if hasattr(HTTP_CLIENT, 'post'):
                # httpx or requests
                response = HTTP_CLIENT.post( "https://openrouter.ai/api/v1/chat/completions", json=test_payload, headers=headers, timeout=10)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Model {model_key} test failed: {e}")
            
        return False

    def _create_system_prompt(self, memory_stats: Dict) -> str:
        """Create an enhanced system prompt"""
        return f"""You are an elite autonomous coding agent with advanced problem-solving capabilities.

CORE CAPABILITIES:
- Write, test, and debug code autonomously.
- Use tools strategically to solve complex problems.
- Learn from past successful solutions ({memory_stats.get('total', 0)} tasks in memory).
- Iterate and self-correct until solutions work perfectly.
- Handle errors gracefully and find alternative approaches.
- Always respond with a clear plan and then execute it using the available tools.

AVAILABLE TOOLS:
• `write_file(filename="path/to/file", content="file content")`: Creates or overwrites a file.
• `read_file(filename="path/to/file")`: Reads the content of a file.
• `execute_code(code="print('hello')", language="python")`: Executes a code block.
• `run_command(command="ls -l")`: Runs a shell command in the workspace.
• `test_code(filename="path/to/test.py")`: Runs tests in a specified file.
• `list_files(directory=".")`: Lists files and directories, recursively.
• `install_package(package="requests")`: Installs a Python package using pip.
• `search_files(query="some_function")`: Searches for a string/query in all files in the workspace.
• `create_project_structure(structure='{{"main.py": "", "utils/": {{"__init__.py": ""}}}}')`: Creates a directory and file structure from a JSON string.

PROBLEM-SOLVING APPROACH:
1. **Understand & Plan:** Deconstruct the request into a clear, step-by-step plan.
2. **Implement:** Execute your plan using the available tools. Write clean, well-documented code.
3. **Test:** Rigorously test your implementation. Use the `test_code` and `execute_code` tools.
4. **Refine:** Analyze the results and errors. Debug and refine your code iteratively until the task is complete.
5. **Conclude:** When you believe the task is fully completed and verified, state that you are finished.

QUALITY STANDARDS:
- Always include error handling and input validation in the code you write.
- Write clear documentation and comments.
- Follow best practices and standard design patterns.
- Ensure cross-platform compatibility where applicable.

Your current working directory is `{self.working_dir}`. All file operations are relative to this directory.
SUCCESS METRICS: You have a {memory_stats.get('success_rate', 0):.1%} historical success rate. Be methodical, thorough, and persistent."""

    def _init_tools(self) -> Dict[str, callable]:
        """Initialize enhanced tool set"""
        return {
            "write_file": self._tool_write_file,
            "read_file": self._tool_read_file,
            "execute_code": self._tool_execute_code,
            "run_command": self._tool_run_command,
            "test_code": self._tool_test_code,
            "list_files": self._tool_list_files,
            "install_package": self._tool_install_package,
            "create_project_structure": self._tool_create_project_structure,
            "search_files": self._tool_search_files,
        }

    def _add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to context with token tracking"""
        token_count = len(content.split()) * 1.4  # A better approximation
        
        msg = AgentMessage( role=role, content=content, metadata=metadata or {}, token_count=int(token_count))
        self.context.messages.append(msg)
        self.context.total_tokens += int(token_count)

        if self.model_config:
            max_context = self.model_config.get("context_window", 32000)
            if self.context.total_tokens > max_context * 0.8:
                self._compress_context()

    def _compress_context(self):
        """Compress context by summarizing older messages."""
        self.rich.info("Compressing context to fit model limits...")
        system_msg = self.context.messages[0]
        recent_messages = self.context.messages[-10:]
        
        # In a more advanced implementation, you would use an LLM to summarize older parts of the conversation.
        # For now, we'll just keep the system prompt and the most recent messages.
        
        self.context.messages = [system_msg] + recent_messages
        self.context.total_tokens = sum(m.token_count or 0 for m in self.context.messages)

    def _generate_response(self, prompt: str) -> str:
        """Generate LLM response via OpenRouter"""
        if not self.openrouter_api_key or self.model_name == "mock":
            return self._mock_response(prompt)

        self._add_message("user", prompt)
        
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in self.context.messages]
            
            payload = {
                "model": self.model_config["name"],
                "messages": messages,
                "max_tokens": self.model_config["max_tokens"],
                "temperature": 0.1, "top_p": 0.9,
            }
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo/enhanced-coding-agent",
                "X-Title": "Enhanced Coding Agent"
            }
            
            with self.rich.status("🤖 Thinking..."):
                if hasattr(HTTP_CLIENT, 'post'):
                    response = HTTP_CLIENT.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    usage = data.get("usage", {})
                    if usage and self.model_config:
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        input_cost = (input_tokens / 1000) * self.model_config["cost_per_1k"]["input"]
                        output_cost = (output_tokens / 1000) * self.model_config["cost_per_1k"]["output"]
                        self.context.total_cost += input_cost + output_cost
                    
                    return content
                else: raise Exception("No HTTP client available")
                    
        except Exception as e:
            self.rich.error(f"LLM generation failed: {e}")
            return "An error occurred. I will try a different approach. Let me start by listing the files to re-evaluate the state."

    def _mock_response(self, prompt: str) -> str:
        """Enhanced mock response for testing."""
        self._add_message("user", prompt)
        return """I have received your request. I will now create a Python script to solve it.

write_file(filename="solution.py", content='''
# Main solution script
def solve():
    print("This is a mock solution.")

if __name__ == "__main__":
    solve()
''')

Now I will test the script.
execute_code(code='import os; os.system("python solution.py")')
"""

    def _extract_tool_calls(self, response: str) -> List[ToolCall]:
        """Extracts tool calls from the LLM's response using robust regex."""
        tool_calls = []
        # Regex to find tool calls like `tool_name(arg1="value1", arg2="value2")`
        pattern = re.compile(r'(\w+)\((.*)\)', re.DOTALL)
        matches = pattern.finditer(response)

        for match in matches:
            tool_name = match.group(1).strip()
            if tool_name not in self.tools:
                continue

            args_str = match.group(2).strip()
            args = {}
            
            # Simple keyword argument parsing
            try:
                # This is a safer way to parse arguments than eval
                arg_pattern = re.compile(r'(\w+)\s*=\s*("""(.*?)"""|\'\'\'(.*?)\'\'\'|"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'|([\w\.\-\/]+))', re.DOTALL)
                arg_matches = arg_pattern.finditer(args_str)
                for arg_match in arg_matches:
                    key = arg_match.group(1)
                    # Get the first non-None value from the capture groups
                    value = next((g for g in arg_match.groups()[2:] if g is not None), None)
                    args[key] = value
                
                if args:
                    tool_calls.append(ToolCall(name=tool_name, args=args))
            except Exception as e:
                logger.warning(f"Could not parse arguments for tool {tool_name}: {e}")

        return tool_calls

    def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """Executes a tool call and returns a result string for the LLM."""
        start_time = time.time()
        self.rich.tool(tool_call.name, f"Executing with args: {tool_call.args}")
        
        try:
            tool_function = self.tools.get(tool_call.name)
            if not tool_function:
                raise ValueError(f"Tool '{tool_call.name}' not found.")
                
            result = tool_function(**tool_call.args)
            tool_call.success = True
            tool_call.result = str(result)
            
            self.rich.panel(f"[green]Success![/green]\n{tool_call.result}", title=f"Result from {tool_call.name}", border_style="green")
            
        except Exception as e:
            tool_call.success = False
            tool_call.error = str(e)
            self.rich.panel(f"[red]Error![/red]\n{tool_call.error}", title=f"Error from {tool_call.name}", border_style="red")
        
        tool_call.execution_time = time.time() - start_time
        self.context.tools_used.append(tool_call.name)
        
        # Formulate a response for the LLM
        if tool_call.success:
            return f"Tool {tool_call.name} executed successfully. Result:\n{tool_call.result}"
        else:
            return f"Tool {tool_call.name} failed. Error:\n{tool_call.error}"

    def run(self, task: str):
        """Main loop to run the agent on a given task."""
        self.context.current_task = task
        self.rich.panel(f"[bold]TASK:[/bold] {task}", title="Starting New Task", style="magenta")
        
        prompt = f"The user wants me to perform the following task: {task}. My current working directory is {self.working_dir}. Please devise a plan and execute it."

        for i in range(self.max_iterations):
            self.context.iterations_count = i + 1
            self.rich.info(f"Iteration {i+1}/{self.max_iterations}")

            response = self._generate_response(prompt)
            self._add_message("assistant", response)
            self.rich.thinking(response)
            
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                self.rich.success("Agent has finished its work or is awaiting further instruction.")
                break
                
            tool_results = []
            for tool_call in tool_calls:
                result_str = self._execute_tool_call(tool_call)
                tool_results.append(result_str)
            
            # Prepare the next prompt with tool results
            prompt = "I have executed the tools. Here are the results:\n\n" + "\n\n".join(tool_results) + "\n\nPlease continue with the next step of your plan. If the task is complete, please confirm by stating you are finished."
        
        self._display_summary()

        if Confirm.ask("[bold yellow]Do you want to provide feedback or continue?", default=False):
            feedback = Prompt.ask("[bold cyan]Enter your feedback or next instruction")
            self.run(f"Continue based on this feedback: {feedback}")
        else:
            success = Confirm.ask("[bold green]Was the task completed successfully?", default=True)
            self.memory.add_memory(
                task=self.context.current_task,
                messages=self.context.messages,
                success=success,
                cost=self.context.total_cost,
                tokens=self.context.total_tokens
            )
            self.rich.success("Session saved to memory.")

    def _display_summary(self):
        """Displays a summary of the agent's session."""
        duration = (datetime.now() - self.context.session_start).total_seconds()
        
        summary_panel = Panel(
            f"[bold cyan]Total Duration:[/bold cyan] {duration:.2f}s\n"
            f"[bold cyan]Total Iterations:[/bold cyan] {self.context.iterations_count}\n"
            f"[bold cyan]Total Cost:[/bold cyan] ${self.context.total_cost:.4f}\n"
            f"[bold cyan]Files Created:[/bold cyan] {', '.join(self.context.files_created) or 'None'}\n"
            f"[bold cyan]Tools Used:[/bold cyan] {', '.join(set(self.context.tools_used)) or 'None'}",
            title="Session Summary",
style="green"        )
        self.rich.print(summary_panel)

    # --- TOOL IMPLEMENTATIONS ---

    def _resolve_path(self, filename: str) -> Path:
        """Resolves a path to be safe and within the working directory."""
        filepath = self.working_dir / Path(filename)
        # Security check: ensure the path is within the working directory
        if not filepath.resolve().is_relative_to(self.working_dir.resolve()):
            raise PermissionError("Access denied: Cannot access files outside the working directory.")
        return filepath

    def _tool_write_file(self, filename: str, content: str) -> str:
        filepath = self._resolve_path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding='utf-8')
        if filename not in self.context.files_created:
            self.context.files_created.append(filename)
        self.rich.code(content, language=Path(filename).suffix.lstrip('.'))
        return f"Successfully wrote {len(content)} bytes to {filename}"

    def _tool_read_file(self, filename: str) -> str:
        filepath = self._resolve_path(filename)
        return filepath.read_text(encoding='utf-8')

    def _tool_execute_code(self, code: str, language: str = "python") -> str:
        if language != "python":
            return f"Error: Language '{language}' is not supported for execution."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        try:
            result = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=30, cwd=self.working_dir)
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return output
        finally:
            os.remove(tmp_path)

    def _tool_run_command(self, command: str) -> str:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, cwd=self.working_dir)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    def _tool_test_code(self, filename: str) -> str:
        filepath = self._resolve_path(filename)
        if not filepath.exists():
            return f"Error: Test file '{filename}' not found."
        
        command = f"{sys.executable} -m unittest {filepath}" if "test" in filename else f"{sys.executable} {filepath}"
        return self._tool_run_command(command)

    def _tool_list_files(self, directory: str = ".") -> str:
        dirpath = self._resolve_path(directory)
        if not dirpath.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        tree = Tree(f"[bold blue]{dirpath.name}")
        def build_tree(path: Path, branch: Tree):
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    sub_branch = branch.add(f"📁 [bold cyan]{item.name}")
                    build_tree(item, sub_branch)
                else:
                    branch.add(f"📄 {item.name}")
        build_tree(dirpath, tree)
        # Capture rich tree output as a string
        with self.rich.console.capture() as capture:
             self.rich.console.print(tree)
        return capture.get()

    def _tool_install_package(self, package: str) -> str:
        return self._tool_run_command(f"{sys.executable} -m pip install {package}")

    def _tool_search_files(self, query: str) -> str:
        results = []
        for path in self.working_dir.rglob("*"):
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if query in line:
                                results.append(f"{path.relative_to(self.working_dir)}:{i+1}: {line.strip()}")
                except Exception:
                    pass
        if not results:
            return "No results found."
        return "\n".join(results)

    def _tool_create_project_structure(self, structure: str) -> str:
        try:
            struct_dict = json.loads(structure)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON structure provided. {e}"
        
        def create_from_dict(base_path: Path, d: Dict):
            for name, content in d.items():
                path = base_path / name
                if name.endswith('/') or isinstance(content, dict):
                    path.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, dict):
                        create_from_dict(path, content)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(content))
        
        create_from_dict(self.working_dir, struct_dict)
        return "Project structure created successfully."


def main():
    """Main entry point for the agent CLI."""
    parser = argparse.ArgumentParser(description="Enhanced Autonomous LLM Coding Agent")
    parser.add_argument("task", type=str, nargs="?", default=None, help="The initial task for the agent to perform.")
    parser.add_argument("--model", type=str, default="auto", help=f"Model to use. Options: auto, {', '.join(OPENROUTER_MODELS.keys())}, or a custom model name.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key. Overrides OPENROUTER_API_KEY env var.")
    
    args = parser.parse_args()
    
    if not args.task:
        if RICH_AVAILABLE:
            args.task = Prompt.ask("[bold green]Please enter the task for the agent")
        else:
            print("Please provide a task.")
            parser.print_help()
            sys.exit(1)

    try:
        agent = EnhancedAgenticCoder(model=args.model, openrouter_api_key=args.api_key)
        agent.run(args.task)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print_exception(show_locals=True)
        else:
            logger.exception("An unhandled error occurred.")
        sys.exit(1)


if __name__ == "__main__":
    main()