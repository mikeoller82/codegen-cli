## 🔄 Interactive REPL Mode

CodeGen CLI now includes a powerful REPL (Read-Eval-Print Loop) mode for continuous operations:

### Starting REPL Mode

\`\`\`bash
# Start REPL from CLI
python cli.py repl

# Or use standalone launcher
python repl.py
\`\`\`

### REPL Features

- **Command History**: Navigate through previous commands with ↑/↓ arrows
- **Auto-completion**: Tab completion for commands and file paths
- **Session Variables**: Store and reuse generated code and results
- **Persistent Context**: Maintain AI conversation context across commands
- **Directory Navigation**: Built-in file system commands (ls, cd, pwd)
- **Configuration Management**: Adjust settings on the fly
- **Session Save/Load**: Persist your work across sessions

### REPL Commands

\`\`\`bash
# AI Operations
generate "create a Flask server"    # Generate code with AI
edit app.py "add error handling"    # Edit files with AI
models                              # Show available AI models

# File Operations  
read app.py                         # Display file with syntax highlighting
write test.py "print('hello')"      # Write content to file
append app.py "# comment"           # Append to file
run script.py                       # Execute Python file safely

# Web Operations
fetch https://api.github.com/users/octocat  # Fetch web content

# Navigation & File System
ls                                  # List files
cd /path/to/directory              # Change directory
pwd                                # Show current directory

# Session Management
memory                             # Show session memory
vars                               # Show session variables
context                            # Show session context
history                            # Show command history

# Configuration
config                             # Show all settings
config verbose true                # Set configuration
save session.json                  # Save session to file
load session.json                  # Load session from file

# Utilities
clear                              # Clear screen
help [command]                     # Show help
quit/exit                          # Exit REPL
\`\`\`

### REPL Example Session

\`\`\`bash
codegen:~ [gpt-3.5-turbo] $ generate "create a simple calculator class"
✓ Generated Calculator class with basic operations

codegen:~ [gpt-3.5-turbo] $ write calculator.py <generated_code>
✓ Content written to calculator.py

codegen:~ [gpt-3.5-turbo] $ run calculator.py
🚀 Execution Results: calculator.py
✓ Success - Calculator works correctly

codegen:~ [gpt-3.5-turbo] $ edit calculator.py "add advanced operations like power and sqrt"
✓ File calculator.py edited successfully

codegen:~ [gpt-3.5-turbo] $ memory
🧠 Session Memory shows 2 generations, 2 file operations

codegen:~ [gpt-3.5-turbo] $ save my_session.json
✓ Session saved to my_session.json
\`\`\`

### REPL Configuration

The REPL supports various configuration options:

\`\`\`bash
config verbose true        # Enable verbose output
config debug true         # Enable debug mode  
config timeout 60         # Set execution timeout
config default_model gpt-4 # Change AI model
\`\`\`

## 🧠 Reasoning and Thinking

CodeGen CLI now includes advanced reasoning capabilities that show the AI's thought process and provide step-by-step problem solving:

### Reasoning Features

- **Multiple Strategies**: Choose from different reasoning approaches
- **Step-by-Step Thinking**: See the AI's thought process broken down
- **Confidence Tracking**: Each reasoning step includes confidence levels
- **Reasoning History**: Track and review past reasoning sessions
- **Integration with Generation**: Use reasoning to improve code quality

### Reasoning Strategies

\`\`\`bash
# Available reasoning strategies:
step_by_step           # Systematic step-by-step analysis
problem_decomposition  # Break complex problems into parts  
first_principles      # Reason from fundamental concepts
analogical           # Use analogies and similar problems
critical_thinking    # Question assumptions and evaluate evidence
\`\`\`

### CLI Reasoning Commands

\`\`\`bash
# Think through a problem
python cli.py think "optimize database performance"

# Use specific reasoning strategy
python cli.py think "design microservices architecture" --strategy problem_decomposition

# Generate code with reasoning
python cli.py generate-with-reasoning "create authentication system" --strategy first_principles

# Hide thinking process (show only results)
python cli.py think "implement caching" --no-thinking
\`\`\`

### REPL Reasoning Commands

\`\`\`bash
# In REPL mode:
think create a REST API                    # Think through problem step-by-step
reason critical_thinking debug this code   # Use specific strategy
analyze app.py                            # Analyze code file
strategies                                # Show available strategies
reasoning                                 # Show reasoning history
generate "create API" --reasoning         # Generate with reasoning
\`\`\`

### Reasoning Example

\`\`\`bash
codegen:~ [gpt-3.5-turbo] $ think "create a scalable web scraper"

🤔 Reasoning through: create a scalable web scraper
Strategy: step_by_step

🔍 Step 1: Problem Analysis
Understanding the problem: "create a scalable web scraper"
- Main objective: Extract data from websites efficiently
- Constraints: Handle rate limiting, avoid blocking
- Technologies: Python requests, async/await, proxies
- Challenges: Anti-bot measures, dynamic content
- Success criteria: Fast, reliable, respectful scraping

🎯 Step 2: Solution Planning  
Based on analysis, here's my solution plan:
1. Choose async architecture for concurrency
2. Implement rate limiting and retry logic
3. Add proxy rotation and user agent rotation
4. Plan for error handling and logging
5. Consider robots.txt compliance

⚙️ Step 3: Implementation Design
Implementation strategy:
- Use aiohttp for async HTTP requests
- Implement exponential backoff for retries
- Add configurable delays between requests
- Include comprehensive error handling
- Make it extensible with plugins

✅ Step 4: Approach Validation
Validating the approach:
✓ Solves scalability through async processing
✓ Addresses rate limiting and blocking
✓ Includes proper error handling
✓ Follows web scraping best practices

✨ Reasoning Complete!
Confidence: 85% | Time: 2.3s | Steps: 4
\`\`\`

### Integration with Code Generation

When using reasoning with code generation, the AI:

1. **Analyzes** the problem systematically
2. **Plans** the solution approach  
3. **Designs** the implementation strategy
4. **Validates** the approach
5. **Generates** code based on the reasoning

This results in higher quality, more thoughtful code that addresses edge cases and follows best practices.
