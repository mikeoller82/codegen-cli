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

## 🔧 Automatic Testing and Fixing

CodeGen CLI now includes powerful auto-fix capabilities that automatically test generated code and fix common issues:

### Auto-Fix Features

- **Automatic Testing**: Every generated code is automatically tested for syntax and runtime errors
- **Intelligent Fixing**: Multiple fix strategies are applied to resolve issues
- **Progress Tracking**: Visual progress indicators show testing and fixing steps
- **Fix Summaries**: Detailed reports of what was fixed and how

### Fix Strategies

The auto-fix engine uses multiple strategies in order:

1. **Syntax Error Fixes**: Correct indentation, missing colons, brackets
2. **Import Error Fixes**: Replace unavailable modules with alternatives
3. **Runtime Error Fixes**: Add missing variable definitions
4. **Error Handling**: Wrap code in try-catch blocks
5. **Structure Fixes**: Add proper function/class structure

### CLI Auto-Fix Commands

\`\`\`bash
# Generate code with automatic testing and fixing
python cli.py generate-auto "create a web scraper"

# Skip reasoning for faster generation
python cli.py generate-auto "create calculator" --no-reasoning

# Skip auto-testing
python cli.py generate-auto "create function" --no-autotest

# Save directly to file
python cli.py generate-auto "create API" -o api.py
\`\`\`

### REPL Auto-Fix Commands

\`\`\`bash
# In REPL mode:
generate-auto "create a Flask server"    # Generate with auto-testing
gen-auto "implement sorting algorithm"   # Short alias
tasks                                    # Show current tasks
tasks stats                             # Show session statistics
tasks 1                                 # Show details for task #1
autotest on/off                         # Toggle auto-testing
next                                    # Ask for next task
\`\`\`

## 📋 Task Management

CodeGen CLI includes a built-in task management system for continuous development:

### Task Features

- **Automatic Task Creation**: Each generation request becomes a tracked task
- **Status Tracking**: Tasks are marked as pending, in-progress, completed, or failed
- **Fix History**: All fix attempts are recorded for each task
- **Session Statistics**: Track success rates and productivity metrics
- **Continuous Workflow**: Automatic prompting for next tasks

### Task Workflow Example

\`\`\`bash
codegen:~ [gpt-3.5-turbo] $ generate-auto "create a calculator class"

🚀 Starting Task #1
create a calculator class

🤔 Reasoning through: create a calculator class
🧪 Testing Generated Code
🔧 Fix Attempt 1/3: Attempting fix...
✅ Fix successful with strategy: _fix_syntax_errors
✅ Code fixed successfully after 1 attempts!

📋 Fix Summary:
✅ Successfully fixed code after 1 attempts
  ✅ Attempt 1: _fix_syntax_errors
     → Fixed syntax_error

📋 Task Recap: #1
Task:     create a calculator class
Status:   completed
Duration: 0:00:15
Fixes Applied: 1

🎯 Ready for next task!
Session stats: 1 completed, 0 pending, 100.0% success rate

What would you like me to work on next?
Examples:
  • Create a web scraper for news articles
  • Add authentication to the existing API
  • Optimize the database queries
  • Write unit tests for the calculator

Next task: add unit tests for the calculator
\`\`\`

### Session Statistics

Track your development productivity:

- **Total Tasks**: Number of tasks attempted
- **Success Rate**: Percentage of successfully completed tasks
- **Fix Attempts**: Average number of fixes needed per task
- **Session Duration**: Total time spent in development session

### Continuous Development Loop

The enhanced workflow creates a continuous development experience:

1. **Generate**: AI creates code based on your request
2. **Test**: Code is automatically tested for issues
3. **Fix**: Any problems are automatically resolved
4. **Recap**: Summary of what was accomplished
5. **Next**: Prompt for the next development task
6. **Repeat**: Seamless continuation of development work

This creates an efficient, automated development assistant that handles the tedious parts of coding while keeping you focused on the creative aspects.

## 🧠 AI Learning System

CodeGen CLI now includes an advanced AI learning system that learns from successful fixes and continuously improves code generation quality:

### Learning Features

- **Pattern Recognition**: Automatically identifies and learns from successful fix patterns
- **Prompt Improvement**: Enhances prompts based on learned patterns and common issues
- **Code Templates**: Builds reusable code templates from successful generations
- **Confidence Tracking**: Tracks success rates and confidence levels for learned patterns
- **Persistent Learning**: Stores learned patterns in a database for future sessions

### How Learning Works

1. **Fix Pattern Learning**: When code is successfully fixed, the system analyzes:
   - What type of error occurred
   - Which fix strategy worked
   - The pattern of the fix applied
   - Success/failure rates over time

2. **Generation Improvement**: The system learns from entire generation sessions:
   - Common issues for different types of prompts
   - Effective approaches for specific problem categories
   - Code templates that work well

3. **Continuous Improvement**: Each session makes the AI better:
   - Prompts are automatically enhanced with learned insights
   - Code templates provide better starting points
   - Fix strategies are prioritized by success rate

### Learning Commands

#### CLI Commands
\`\`\`bash
# Show learning system status and insights
python cli.py learning

# Generate code with learning (default behavior)
python cli.py generate-auto "create a web scraper"
\`\`\`

#### REPL Commands
\`\`\`bash
# In REPL mode:
learning                    # Show learning system status
patterns                    # Show learned fix patterns  
insights                    # Show detailed learning insights and recommendations
learning clear              # Clear all learned patterns (with confirmation)
learning export data.json   # Export learning data to file
\`\`\`

### Learning Insights Example

\`\`\`bash
codegen:~ [gpt-3.5-turbo] $ learning

🧠 AI Learning Status
┌─────────────────────┬─────────────┐
│ Metric              │ Value       │
├─────────────────────┼─────────────┤
│ Status              │ 🟢 Active   │
│ Total Patterns      │ 23          │
│ Active Patterns     │ 18          │
│ Average Confidence  │ 78.5%       │
│ Learning Rate       │ 2.3/hour    │
│ Recent Improvements │ 5           │
└─────────────────────┴─────────────┘

📊 Patterns by Error Type:
┌─────────────────┬──────────┬────────────┐
│ Error Type      │ Patterns │ Percentage │
├─────────────────┼──────────┼────────────┤
│ import_error    │ 8        │ 34.8%      │
│ syntax_error    │ 6        │ 26.1%      │
│ runtime_error   │ 5        │ 21.7%      │
│ logic_error     │ 4        │ 17.4%      │
└─────────────────┴──────────┴────────────┘

🎯 Top Learned Patterns:
┌─────────────────┬─────────────────────┬─────────────────┬──────┐
│ Error Type      │ Fix Strategy        │ Confidence      │ Uses │
├─────────────────┼─────────────────────┼─────────────────┼──────┤
│ import_error    │ Fix Import Errors   │ ██████████ 95%  │ 12   │
│ syntax_error    │ Fix Syntax Errors   │ █████████░ 89%  │ 8    │
│ runtime_error   │ Fix Runtime Errors  │ ████████░░ 82%  │ 6    │
└─────────────────┴─────────────────────┴─────────────────┴──────┘

💡 Recommendations:
• Continue using the system to improve pattern confidence
• The AI learns from every successful fix and improves over time!
\`\`\`

### Learning Demo

Run the learning demo to see the system in action:

\`\`\`bash
python examples/learning_demo.py
\`\`\`

This demo will:
- Show the initial learning state
- Run several coding tasks that trigger learning
- Display how patterns are learned from successful fixes
- Show the improved learning state after the session
- Demonstrate prompt improvements based on learned patterns

### Learning Benefits

- **Faster Generation**: Learned templates provide better starting points
- **Fewer Errors**: Common mistakes are avoided based on learned patterns
- **Better Prompts**: Automatic prompt enhancement based on successful patterns
- **Continuous Improvement**: The system gets better with every use
- **Personalized**: Learns from your specific coding patterns and preferences

The learning system makes CodeGen CLI progressively more intelligent and helpful over time, adapting to your coding style and common problem patterns.
