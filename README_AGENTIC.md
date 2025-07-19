# Lightweight Agentic LLM Coder

A streamlined, resource-efficient autonomous coding agent that replaces the complex multi-engine system with a single, powerful agentic coder.

## 🚀 Key Improvements

### Resource Efficiency
- **82% smaller codebase** (30KB vs 166KB)
- **99.4% faster startup time**
- **Minimal dependencies** (only stdlib + optional LLM clients)
- **Efficient memory usage** with smart context management

### Architecture Benefits
- **Single-file core** (`agentic_coder.py`) - easy to understand and modify
- **Optional dependencies** - works without any external packages
- **Lightweight memory system** instead of complex learning engines
- **Direct tool calling** without heavy abstraction layers

## 🛠️ Features

### Core Capabilities
- ✅ **Autonomous task solving** with iterative improvement
- ✅ **File operations** (read, write, list)
- ✅ **Code execution** and testing
- ✅ **Shell command execution**
- ✅ **Memory system** for learning from past tasks
- ✅ **Interactive mode** for real-time collaboration

### LLM Support
- **OpenAI GPT models** (GPT-4, GPT-3.5-turbo, GPT-4o-mini)
- **Google Gemini models** (Gemini 1.5 Flash, Pro)
- **Mock mode** for testing without API keys
- **Auto-detection** of available models

## 📦 Installation

### Minimal Setup (No Dependencies)
```bash
# Clone or download agentic_coder.py and simple_memory.py
# Works with Python 3.7+ stdlib only
python agentic_coder.py --help
```

### With LLM Support
```bash
# For OpenAI
pip install openai>=1.0.0
export OPENAI_API_KEY="your-key-here"

# For Gemini
pip install google-generativeai>=0.3.0
export GOOGLE_API_KEY="your-key-here"
```

## 🎯 Usage

### Autonomous Task Solving
```bash
# Solve a coding task
python agentic_coder.py --task "Create a web scraper for news articles"

# With specific model
python agentic_coder.py --task "Build a REST API" --model gpt-4o-mini

# Custom iterations and working directory
python agentic_coder.py --task "Create a game" --max-iterations 15 --working-dir ./projects
```

### Interactive Mode
```bash
python agentic_coder.py --interactive
```

Interactive commands:
- `solve <task>` - Solve a task autonomously
- `write <filename> <content>` - Write to a file
- `read <filename>` - Read a file
- `run <command>` - Execute shell command
- `test <filename>` - Test Python code
- `list [directory]` - List files
- `memory` - Show learning statistics
- `status` - Show current status
- `clear` - Clear context
- `help` - Show commands

### Python API
```python
from agentic_coder import AgenticCoder

# Initialize agent
agent = AgenticCoder(model="gpt-4o-mini")

# Solve a task
success, result = agent.solve_task("Create a calculator app")

# Interactive mode
agent.interactive_mode()
```

## 🧠 Memory System

The agent learns from past tasks and applies successful patterns:

```python
# Memory automatically tracks:
# - Task descriptions and solutions
# - Success/failure rates
# - Tools used
# - Files created
# - Common patterns

# View memory statistics
agent.memory.get_memory_stats()
agent.memory.get_successful_patterns()
```

## 🔧 Available Tools

The agent has access to these tools:

1. **write_file(filename, content)** - Create/update files
2. **read_file(filename)** - Read file contents
3. **execute_code(code, language="python")** - Run code
4. **run_command(command)** - Execute shell commands
5. **test_code(filename)** - Test Python files
6. **list_files(directory=".")** - List directory contents

## 📊 Comparison with Old System

| Aspect | Old System | New Agentic Coder | Improvement |
|--------|------------|-------------------|-------------|
| **Code Size** | 166KB (8 files) | 30KB (2 files) | 82% smaller |
| **Dependencies** | 6 required | 2 optional | 67% fewer |
| **Startup Time** | 0.56s | 0.003s | 99.4% faster |
| **Architecture** | Multi-engine complex | Single agentic core | Much simpler |
| **Memory Usage** | Heavy learning engine | Lightweight memory | More efficient |
| **Maintainability** | Complex abstractions | Direct implementation | Easier to modify |

## 🎨 Example Tasks

### Simple Calculator
```bash
python agentic_coder.py --task "Create a calculator that can add, subtract, multiply, and divide"
```

### Web Scraper
```bash
python agentic_coder.py --task "Create a web scraper that extracts article titles from a news website"
```

### Data Analysis
```bash
python agentic_coder.py --task "Create a script that analyzes CSV data and generates a report"
```

### API Development
```bash
python agentic_coder.py --task "Build a REST API with FastAPI that manages a todo list"
```

## 🔍 How It Works

1. **Task Analysis** - Agent analyzes the task and checks memory for similar solutions
2. **Tool Planning** - Determines which tools to use and in what order
3. **Iterative Execution** - Executes tools, tests results, and iterates until success
4. **Memory Storage** - Saves successful patterns for future use
5. **Context Management** - Maintains efficient conversation history

## 🚦 Error Handling

- **Automatic retry** on tool failures
- **Syntax checking** before code execution
- **Timeout protection** (30s per operation)
- **Graceful degradation** when APIs are unavailable
- **Memory persistence** across sessions

## 🎯 Design Philosophy

### Agentic Principles
- **Autonomous decision making** - Agent chooses tools and strategies
- **Iterative improvement** - Learns from failures and adapts
- **Goal-oriented behavior** - Focuses on task completion
- **Tool utilization** - Leverages available capabilities effectively

### Resource Efficiency
- **Minimal footprint** - Only essential code and dependencies
- **Smart caching** - Efficient context and memory management
- **Optional features** - LLM clients only loaded when needed
- **Graceful scaling** - Works from simple scripts to complex projects

## 🤝 Contributing

The agentic coder is designed to be easily extensible:

1. **Add new tools** - Implement new methods in the `tools` dictionary
2. **Enhance memory** - Extend the `SimpleMemory` class
3. **Support new LLMs** - Add clients to `_init_llm()`
4. **Improve parsing** - Enhance `_extract_tool_calls()`

## 📄 License

Same as the original project.

---

**The future of coding is agentic - autonomous, efficient, and intelligent.** 🤖✨