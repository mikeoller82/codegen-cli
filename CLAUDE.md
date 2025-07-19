# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run the CLI**: `python cli.py --help`
- **Start the interactive REPL**: `python cli.py repl`
- **Run tests**: `pytest`

## Architecture

This is a Python-based command-line interface (CLI) tool called "codegen-cli" that uses AI for code generation, reasoning, and auto-fixing.

- **`cli.py`**: The main entry point for the CLI, built using the `click` library. It defines commands like `repl`, `think`, `generate-auto`, and `autonomous`.
- **`core/`**: This directory contains the core logic of the application.
  - **`core/ai.py`**: Manages interactions with different AI models (e.g., OpenAI, Google Gemini).
  - **`core/repl.py`**: Implements the interactive REPL functionality.
  - **`core/reasoning.py`**: Contains the logic for the AI's step-by-step reasoning process.
  - **`core/auto_fix.py`**: Handles the automatic testing and fixing of generated code.
  - **`core/task_manager.py`**: Manages the lifecycle of tasks.
  - **`core/learning_engine.py`**: A system that learns from successful fixes to improve future code generation.
  - **`core/autonomous_agent.py`**: Implements the autonomous mode where the agent can work on tasks iteratively.
- **`tests/`**: Contains unit tests for the application, which can be run using `pytest`.
