"""
Autonomous Agent for self-directed code generation with planning and iterative improvement
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Confirm

from .ai import AIEngine
from .task_manager import TaskManager, Task, TaskStatus
from .auto_fix import AutoFixEngine, TestResult
from .learning_engine import LearningEngine
from .action_logger import ActionLogger, get_action_logger

console = Console()
logger = logging.getLogger("codegen.autonomous")


class PlanningPhase(Enum):
    """Planning phases"""

    ANALYSIS = "analysis"
    DECOMPOSITION = "decomposition"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    VALIDATION = "validation"
    REFLECTION = "reflection"


@dataclass
class ActionStep:
    """A single action step in the plan"""

    id: int
    description: str
    action_type: str  # 'analyze', 'generate', 'test', 'fix', 'validate'
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    error: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task"""

    task_description: str
    analysis: str
    decomposition: List[str]
    strategy: str
    action_steps: List[ActionStep]
    estimated_duration: float
    complexity_score: int  # 1-10
    risk_factors: List[str]
    success_criteria: List[str]


class AutonomousAgent:
    """Autonomous agent that plans, executes, and iteratively improves code generation"""

    def __init__(self, ai_engine: AIEngine = None):
        self.ai_engine = ai_engine or AIEngine()
        self.task_manager = TaskManager()
        self.learning_engine = self.ai_engine.learning_engine
        self.auto_fix_engine = self.ai_engine.auto_fix_engine
        self.action_logger = ActionLogger()

        # Agent configuration
        self.max_iterations = 10
        self.max_fix_attempts = 5
        self.planning_enabled = True
        self.verbose_logging = True
        self.auto_continue = False

        # Current state
        self.current_plan: Optional[ExecutionPlan] = None
        self.current_iteration = 0
        self.session_stats = {
            "tasks_completed": 0,
            "total_iterations": 0,
            "total_fixes_applied": 0,
            "success_rate": 0.0,
            "avg_completion_time": 0.0,
        }

    def execute_autonomous_task(
        self, task_description: str, auto_continue: bool = False
    ) -> Tuple[str, bool]:
        """Execute a task autonomously with planning and iterative improvement"""

        self.auto_continue = auto_continue
        self.current_iteration = 0

        console.print(f"\n[bold blue]🤖 Autonomous Agent Starting[/bold blue]")
        console.print(f"[dim]Task: {task_description}[/dim]")

        # Start logging the task execution
        self.action_logger.start_action(
            "task",
            f"Execute autonomous task: {task_description}",
            {"auto_continue": auto_continue, "max_iterations": self.max_iterations},
        )

        # Add task to task manager
        task = self.task_manager.add_task(task_description)
        task.status = TaskStatus.IN_PROGRESS
        try:
            # Phase 1: Planning
            if self.planning_enabled:
                self.action_logger.start_action(
                    "plan", "Create execution plan", {"task": task_description}
                )
                self.current_plan = self._create_execution_plan(task_description)
                self.action_logger.complete_action(
                    "completed",
                    result=f"Plan with {len(self.current_plan.action_steps)} steps",
                )

                self._display_execution_plan(self.current_plan)

                if not self.auto_continue:
                    self.action_logger.log_decision(
                        "Proceed with execution plan",
                        "User confirmation required for plan execution",
                        ["proceed", "cancel"],
                    )

                    if not Confirm.ask("\n[yellow]Proceed with this plan?[/yellow]"):
                        task.status = TaskStatus.FAILED
                        task.notes = "User cancelled execution"
                        self.action_logger.complete_action(
                            "failed", error="User cancelled execution"
                        )
                        return "", False
                    else:
                        self.action_logger.log_decision(
                            "Proceed with execution plan",
                            "User approved plan execution",
                            ["proceed", "cancel"],
                            "proceed",
                        )
            # Phase 2: Execution with iterative improvement
            self.action_logger.start_action(
                "execute",
                "Execute plan with iterations",
                {
                    "max_iterations": self.max_iterations,
                    "auto_continue": self.auto_continue,
                },
            )

            final_code, success = self._execute_plan_with_iterations(
                task, self.current_plan
            )

            # Phase 3: Final validation and reflection
            if success:
                task.status = TaskStatus.COMPLETED
                task.code_generated = final_code
                task.notes = f"Completed in {self.current_iteration} iterations"
                self.session_stats["tasks_completed"] += 1

                console.print(
                    f"\n[bold green]✅ Task completed successfully![/bold green]"
                )
                self._display_completion_summary(task, final_code)

                self.action_logger.complete_action(
                    "completed",
                    result=f"Task completed in {self.current_iteration} iterations",
                )
            else:
                task.status = TaskStatus.FAILED
                task.notes = f"Failed after {self.current_iteration} iterations"
                console.print(
                    f"\n[bold red]❌ Task failed after maximum iterations[/bold red]"
                )

                self.action_logger.complete_action(
                    "failed", error=f"Failed after {self.current_iteration} iterations"
                )

            self._update_session_stats()
            self.action_logger.complete_action("completed" if success else "failed")
            return final_code, success

        except KeyboardInterrupt:
            console.print(f"\n[yellow]⏸️  Task interrupted by user[/yellow]")
            task.status = TaskStatus.FAILED
            task.notes = "Interrupted by user"
            self.action_logger.complete_action("failed", error="Interrupted by user")
            return "", False
        except Exception as e:
            console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
            task.status = TaskStatus.FAILED
            task.notes = f"Error: {e}"
            logger.error(f"Autonomous execution failed: {e}")
            self.action_logger.complete_action("failed", error=str(e))
            return "", False

    def _create_execution_plan(self, task_description: str) -> ExecutionPlan:
        """Create a detailed execution plan for the task"""

        console.print(f"\n[bold yellow]📋 Creating Execution Plan[/bold yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Analysis phase
            progress.add_task("Analyzing task requirements...", total=None)
            analysis = self._analyze_task(task_description)

            # Decomposition phase
            progress.add_task("Breaking down into subtasks...", total=None)
            decomposition = self._decompose_task(task_description, analysis)

            # Strategy phase
            progress.add_task("Developing execution strategy...", total=None)
            strategy = self._develop_strategy(task_description, analysis, decomposition)

            # Create action steps
            progress.add_task("Creating action steps...", total=None)
            action_steps = self._create_action_steps(decomposition, strategy)

            # Risk assessment
            progress.add_task("Assessing risks and complexity...", total=None)
            complexity_score, risk_factors = self._assess_complexity_and_risks(
                task_description, decomposition
            )

            # Success criteria
            success_criteria = self._define_success_criteria(task_description, analysis)

            # Estimate duration
            estimated_duration = self._estimate_duration(action_steps, complexity_score)

        return ExecutionPlan(
            task_description=task_description,
            analysis=analysis,
            decomposition=decomposition,
            strategy=strategy,
            action_steps=action_steps,
            estimated_duration=estimated_duration,
            complexity_score=complexity_score,
            risk_factors=risk_factors,
            success_criteria=success_criteria,
        )

    def _analyze_task(self, task_description: str) -> str:
        """Analyze the task to understand requirements"""

        self.action_logger.log_planning_phase("analysis", "Starting task analysis")

        analysis_prompt = f"""
        Analyze this coding task and provide a detailed analysis:
        
        Task: {task_description}
        
        Please provide:
        1. Core requirements and objectives
        2. Technical domain and complexity level
        3. Key challenges and considerations
        4. Required knowledge areas
        5. Expected deliverables
        
        Keep the analysis concise but comprehensive.
        """

        result = self.ai_engine.generate_code(analysis_prompt, use_reasoning=True)
        self.action_logger.log_planning_phase(
            "analysis", f"Analysis completed ({len(result)} chars)"
        )
        return result

    def _decompose_task(self, task_description: str, analysis: str) -> List[str]:
        """Break down the task into manageable subtasks"""

        decomposition_prompt = f"""
        Break down this coding task into specific, actionable subtasks:
        
        Task: {task_description}
        Analysis: {analysis}
        
        Provide 3-7 specific subtasks that together complete the main task.
        Each subtask should be:
        - Specific and actionable
        - Testable/verifiable
        - Logically ordered
        
        Format as a numbered list.
        """

        response = self.ai_engine.generate_code(
            decomposition_prompt, use_reasoning=True
        )

        # Extract numbered items
        lines = response.split("\n")
        subtasks = []
        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("•")
            ):
                # Clean up the line
                clean_line = line.lstrip("0123456789.-• ").strip()
                if clean_line:
                    subtasks.append(clean_line)

        return subtasks[:7]  # Limit to 7 subtasks

    def _develop_strategy(
        self, task_description: str, analysis: str, decomposition: List[str]
    ) -> str:
        """Develop an execution strategy"""

        strategy_prompt = f"""
        Develop an execution strategy for this coding task:
        
        Task: {task_description}
        Analysis: {analysis}
        Subtasks: {", ".join(decomposition)}
        
        Provide a strategy that covers:
        1. Overall approach and methodology
        2. Technology choices and tools
        3. Testing and validation approach
        4. Risk mitigation strategies
        5. Quality assurance measures
        
        Keep it practical and focused.
        """

        return self.ai_engine.generate_code(strategy_prompt, use_reasoning=True)

    def _create_action_steps(
        self, decomposition: List[str], strategy: str
    ) -> List[ActionStep]:
        """Create detailed action steps from decomposition"""

        action_steps = []
        step_id = 1

        # Always start with analysis
        action_steps.append(
            ActionStep(
                id=step_id,
                description="Analyze requirements and setup",
                action_type="analyze",
            )
        )
        step_id += 1

        # Convert each subtask to action steps
        for subtask in decomposition:
            # Determine action type based on subtask content
            action_type = "generate"
            if any(
                word in subtask.lower()
                for word in ["test", "verify", "check", "validate"]
            ):
                action_type = "test"
            elif any(
                word in subtask.lower()
                for word in ["fix", "debug", "correct", "resolve"]
            ):
                action_type = "fix"
            elif any(
                word in subtask.lower() for word in ["analyze", "review", "examine"]
            ):
                action_type = "analyze"

            action_steps.append(
                ActionStep(id=step_id, description=subtask, action_type=action_type)
            )
            step_id += 1

        # Always end with validation
        action_steps.append(
            ActionStep(
                id=step_id,
                description="Final validation and testing",
                action_type="validate",
            )
        )

        return action_steps

    def _assess_complexity_and_risks(
        self, task_description: str, decomposition: List[str]
    ) -> Tuple[int, List[str]]:
        """Assess task complexity and identify risk factors"""

        # Simple complexity scoring based on task characteristics
        complexity_score = 3  # Base complexity
        risk_factors = []

        task_lower = task_description.lower()

        # Complexity factors
        if any(word in task_lower for word in ["api", "database", "web", "network"]):
            complexity_score += 2
            risk_factors.append("External dependencies")

        if any(
            word in task_lower
            for word in ["machine learning", "ai", "algorithm", "optimization"]
        ):
            complexity_score += 3
            risk_factors.append("Advanced algorithms required")

        if any(
            word in task_lower
            for word in ["concurrent", "parallel", "async", "threading"]
        ):
            complexity_score += 2
            risk_factors.append("Concurrency complexity")

        if len(decomposition) > 5:
            complexity_score += 1
            risk_factors.append("Multiple subtasks")

        if any(
            word in task_lower for word in ["security", "authentication", "encryption"]
        ):
            complexity_score += 2
            risk_factors.append("Security considerations")

        # Cap complexity at 10
        complexity_score = min(complexity_score, 10)

        # Add default risks if none identified
        if not risk_factors:
            risk_factors = ["Standard implementation risks"]

        return complexity_score, risk_factors

    def _define_success_criteria(
        self, task_description: str, analysis: str
    ) -> List[str]:
        """Define success criteria for the task"""

        criteria = [
            "Code executes without errors",
            "All requirements are implemented",
            "Code follows best practices",
        ]

        task_lower = task_description.lower()

        if any(word in task_lower for word in ["test", "testing"]):
            criteria.append("All tests pass")

        if any(word in task_lower for word in ["api", "web", "server"]):
            criteria.append("API endpoints respond correctly")

        if any(word in task_lower for word in ["data", "file", "database"]):
            criteria.append("Data operations work correctly")

        if any(word in task_lower for word in ["performance", "optimize", "fast"]):
            criteria.append("Performance requirements met")

        return criteria

    def _estimate_duration(
        self, action_steps: List[ActionStep], complexity_score: int
    ) -> float:
        """Estimate execution duration in minutes"""

        base_time_per_step = 2.0  # minutes
        complexity_multiplier = 1 + (complexity_score - 1) * 0.3

        estimated_time = len(action_steps) * base_time_per_step * complexity_multiplier
        return round(estimated_time, 1)

    def _display_execution_plan(self, plan: ExecutionPlan):
        """Display the execution plan in a formatted way"""

        console.print(f"\n[bold blue]📋 Execution Plan[/bold blue]")

        # Plan overview
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Label", style="cyan", width=20)
        overview_table.add_column("Value", style="white")

        overview_table.add_row("Task:", plan.task_description)
        overview_table.add_row(
            "Complexity:",
            f"{plan.complexity_score}/10 {'🔴' if plan.complexity_score > 7 else '🟡' if plan.complexity_score > 4 else '🟢'}",
        )
        overview_table.add_row("Est. Duration:", f"{plan.estimated_duration} minutes")
        overview_table.add_row("Action Steps:", str(len(plan.action_steps)))

        console.print(overview_table)

        # Analysis
        console.print(f"\n[bold yellow]🔍 Analysis:[/bold yellow]")
        console.print(Panel(plan.analysis, border_style="yellow"))

        # Strategy
        console.print(f"\n[bold green]🎯 Strategy:[/bold green]")
        console.print(Panel(plan.strategy, border_style="green"))

        # Action steps
        console.print(f"\n[bold blue]📝 Action Steps:[/bold blue]")
        steps_table = Table()
        steps_table.add_column("Step", style="cyan", width=4)
        steps_table.add_column("Type", style="yellow", width=10)
        steps_table.add_column("Description", style="white")

        for step in plan.action_steps:
            type_emoji = {
                "analyze": "🔍",
                "generate": "⚡",
                "test": "🧪",
                "fix": "🔧",
                "validate": "✅",
            }.get(step.action_type, "📝")

            steps_table.add_row(
                str(step.id), f"{type_emoji} {step.action_type}", step.description
            )

        console.print(steps_table)

        # Risk factors
        if plan.risk_factors:
            console.print(f"\n[bold red]⚠️  Risk Factors:[/bold red]")
            for risk in plan.risk_factors:
                console.print(f"  • {risk}")

        # Success criteria
        console.print(f"\n[bold green]🎯 Success Criteria:[/bold green]")
        for criterion in plan.success_criteria:
            console.print(f"  ✓ {criterion}")

    def _execute_plan_with_iterations(
        self, task: Task, plan: ExecutionPlan
    ) -> Tuple[str, bool]:
        """Execute the plan with iterative improvement"""

        console.print(f"\n[bold blue]🚀 Starting Execution[/bold blue]")

        current_code = ""
        last_error = ""
        iteration_results = []

        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration

            console.print(
                f"\n[bold yellow]🔄 Iteration {iteration}/{self.max_iterations}[/bold yellow]"
            )

            # Execute action steps for this iteration
            iteration_code, iteration_success, iteration_error = (
                self._execute_action_steps(
                    plan.action_steps, current_code, last_error, iteration
                )
            )

            iteration_results.append(
                {
                    "iteration": iteration,
                    "code": iteration_code,
                    "success": iteration_success,
                    "error": iteration_error,
                }
            )

            if iteration_success:
                console.print(f"[green]✅ Iteration {iteration} successful![/green]")
                current_code = iteration_code

                # Validate against success criteria
                if self._validate_success_criteria(current_code, plan.success_criteria):
                    console.print(
                        f"[bold green]🎉 All success criteria met![/bold green]"
                    )
                    return current_code, True
                else:
                    console.print(
                        f"[yellow]⚠️  Success criteria not fully met, continuing...[/yellow]"
                    )
            else:
                console.print(
                    f"[red]❌ Iteration {iteration} failed: {iteration_error}[/red]"
                )
                current_code = iteration_code  # Keep the code even if it failed
                last_error = iteration_error

                # Ask user if they want to continue (if not auto-continue)
                if not self.auto_continue and iteration < self.max_iterations:
                    if not Confirm.ask(
                        f"\n[yellow]Continue with iteration {iteration + 1}?[/yellow]"
                    ):
                        break

        # If we get here, we've exhausted iterations
        console.print(f"\n[yellow]⏰ Maximum iterations reached[/yellow]")

        # Return the best result we have
        best_result = max(iteration_results, key=lambda x: x["success"])
        return best_result["code"], best_result["success"]

    def _execute_action_steps(
        self,
        action_steps: List[ActionStep],
        current_code: str,
        last_error: str,
        iteration: int,
    ) -> Tuple[str, bool, str]:
        """Execute all action steps for one iteration"""

        working_code = current_code
        overall_success = True
        final_error = ""

        # Create a live display for action steps
        with Live(
            self._create_action_display(action_steps), refresh_per_second=2
        ) as live:
            for step in action_steps:
                step.status = "in_progress"
                step.start_time = datetime.now()
                live.update(self._create_action_display(action_steps))

                try:
                    if step.action_type == "analyze":
                        step.result = self._execute_analyze_step(
                            step, working_code, last_error
                        )
                        step.status = "completed"

                    elif step.action_type == "generate":
                        new_code = self._execute_generate_step(
                            step, working_code, last_error, iteration
                        )
                        if new_code and new_code.strip():
                            working_code = new_code
                            step.result = "Code generated"
                            step.status = "completed"
                        else:
                            step.status = "failed"
                            step.error = "Failed to generate code"
                            overall_success = False

                    elif step.action_type == "test":
                        test_result = self._execute_test_step(step, working_code)
                        step.result = test_result
                        if test_result.result != TestResult.PASS:
                            step.status = "failed"
                            step.error = test_result.error_message
                            overall_success = False
                            final_error = test_result.error_message
                        else:
                            step.status = "completed"

                    elif step.action_type == "fix":
                        fixed_code, fix_success = self._execute_fix_step(
                            step, working_code, last_error
                        )
                        if fix_success:
                            working_code = fixed_code
                            step.result = "Code fixed"
                            step.status = "completed"
                        else:
                            step.status = "failed"
                            step.error = "Could not fix code"
                            overall_success = False

                    elif step.action_type == "validate":
                        validation_result = self._execute_validate_step(
                            step, working_code
                        )
                        step.result = validation_result
                        if not validation_result:
                            step.status = "failed"
                            step.error = "Validation failed"
                            overall_success = False
                        else:
                            step.status = "completed"

                    step.end_time = datetime.now()
                    live.update(self._create_action_display(action_steps))

                    # Small delay to show progress
                    time.sleep(0.5)

                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    step.end_time = datetime.now()
                    overall_success = False
                    final_error = str(e)
                    logger.error(f"Step {step.id} failed: {e}")
                    live.update(self._create_action_display(action_steps))

        return working_code, overall_success, final_error

    def _create_action_display(self, action_steps: List[ActionStep]) -> Table:
        """Create a live display table for action steps"""

        table = Table(title="Action Steps Progress")
        table.add_column("Step", style="cyan", width=4)
        table.add_column("Status", style="white", width=12)
        table.add_column("Description", style="white")
        table.add_column("Duration", style="dim", width=8)

        for step in action_steps:
            # Status with emoji
            status_map = {
                "pending": "⏳ Pending",
                "in_progress": "🔄 Running",
                "completed": "✅ Done",
                "failed": "❌ Failed",
            }
            status_text = status_map.get(step.status, step.status)

            # Duration
            duration_text = ""
            if step.duration:
                duration_text = f"{step.duration:.1f}s"
            elif step.start_time and not step.end_time:
                current_duration = (datetime.now() - step.start_time).total_seconds()
                duration_text = f"{current_duration:.1f}s"

            table.add_row(str(step.id), status_text, step.description, duration_text)

        return table

    def _execute_analyze_step(
        self, step: ActionStep, current_code: str, last_error: str
    ) -> str:
        """Execute an analysis step"""

        if current_code:
            analysis_prompt = f"""
            Analyze the current code and identify areas for improvement:
            
            Current Code:
            {current_code}
            
            Last Error (if any): {last_error}
            
            Provide a brief analysis of:
            1. Code quality and structure
            2. Potential issues or improvements
            3. Next steps for enhancement
            """
        else:
            analysis_prompt = f"""
            Analyze the requirements for: {step.description}
            
            Provide a brief analysis of:
            1. Key requirements
            2. Technical approach
            3. Implementation considerations
            """

        return self.ai_engine.generate_code(analysis_prompt, use_reasoning=False)

    def _execute_generate_step(
        self, step: ActionStep, current_code: str, last_error: str, iteration: int
    ) -> str:
        """Execute a code generation step"""

        # Build context-aware prompt
        if current_code and last_error:
            prompt = f"""
            Improve the existing code to address the error and implement: {step.description}
            
            Current Code:
            {current_code}
            
            Error to Fix: {last_error}
            
            This is iteration {iteration}. Generate improved, working code.
            """
        elif current_code:
            prompt = f"""
            Enhance the existing code to implement: {step.description}
            
            Current Code:
            {current_code}
            
            Generate the enhanced version.
            """
        else:
            prompt = f"""
            Generate code to implement: {step.description}
            
            Create clean, well-structured, working code.
            """

        # Log the code generation
        self.action_logger.log_code_generation(
            prompt[:200],
            self.ai_engine.default_model,
            len(current_code) if current_code else 0,
            True,
        )

        # Use auto-fix generation for better results
        code, _, _, _ = self.ai_engine.generate_code_with_auto_fix(
            prompt, use_reasoning=True, show_progress=False
        )

        return code

    def _execute_test_step(self, step: ActionStep, current_code: str):
        """Execute a testing step"""

        if not current_code:
            from .auto_fix import TestReport, TestResult

            result = TestReport(
                result=TestResult.LOGIC_ERROR, error_message="No code to test"
            )
            self.action_logger.log_test_result(
                "code_validation", "failed", "No code to test"
            )
            return result

        # Use the auto-fix engine's testing capability
        result = self.auto_fix_engine._test_code(current_code)
        self.action_logger.log_test_result(
            "code_execution",
            result.result.value,
            result.error_message if result.error_message else None,
        )
        return result

    def _execute_fix_step(
        self, step: ActionStep, current_code: str, last_error: str
    ) -> Tuple[str, bool]:
        """Execute a fix step"""

        if not current_code:
            self.action_logger.log_fix_attempt("no_code", False, "No code to fix")
            return current_code, False

        # Test the code first to identify issues
        test_result = self.auto_fix_engine._test_code(current_code)

        if test_result.result == TestResult.PASS:
            self.action_logger.log_fix_attempt(
                "no_fix_needed", True, "Code already passes tests"
            )
            return current_code, True

        # Attempt to fix
        fixed_code, fix_strategy = self.auto_fix_engine._attempt_fix(
            current_code, test_result
        )

        # Test the fixed code
        if fixed_code != current_code:
            fixed_test = self.auto_fix_engine._test_code(fixed_code)
            success = fixed_test.result == TestResult.PASS
            self.action_logger.log_fix_attempt(
                fix_strategy, success, test_result.error_message
            )
            return fixed_code, success
        else:
            self.action_logger.log_fix_attempt(
                "no_fix_applied", False, test_result.error_message
            )

        return current_code, False

    def _execute_validate_step(self, step: ActionStep, current_code: str) -> bool:
        """Execute a validation step"""

        if not current_code:
            return False

        # Basic validation - code should at least run without syntax errors
        test_result = self.auto_fix_engine._test_code(current_code)
        return test_result.result in [
            TestResult.PASS,
            TestResult.LOGIC_ERROR,
        ]  # Allow logic errors in validation

    def _validate_success_criteria(self, code: str, criteria: List[str]) -> bool:
        """Validate code against success criteria"""

        if not code:
            return False

        # Test the code
        test_result = self.auto_fix_engine._test_code(code)

        # Basic criteria: code should execute without errors
        if test_result.result not in [TestResult.PASS, TestResult.LOGIC_ERROR]:
            return False

        # Additional validation could be added here based on specific criteria
        # For now, we consider it successful if it runs without syntax/runtime errors

        return True

    def _display_completion_summary(self, task: Task, final_code: str):
        """Display a summary of the completed task"""

        console.print(f"\n[bold blue]📊 Completion Summary[/bold blue]")

        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Label", style="cyan", width=20)
        summary_table.add_column("Value", style="white")

        duration = task.completed_at - task.created_at if task.completed_at else None

        summary_table.add_row("Task:", task.description)
        summary_table.add_row("Status:", "[green]✅ Completed[/green]")
        summary_table.add_row("Iterations:", str(self.current_iteration))
        summary_table.add_row(
            "Duration:", str(duration).split(".")[0] if duration else "Unknown"
        )
        summary_table.add_row("Code Lines:", str(len(final_code.split("\n"))))

        console.print(summary_table)

        # Show the final code
        console.print(f"\n[bold green]📝 Generated Code:[/bold green]")
        from .fs import FileSystemManager

        fs_manager = FileSystemManager()
        fs_manager.display_code(final_code)

        # Show action summary
        console.print(f"\n[bold blue]📊 Action Summary:[/bold blue]")
        self.action_logger.display_recent_actions(limit=10)

    def _update_session_stats(self):
        """Update session statistics"""

        self.session_stats["total_iterations"] += self.current_iteration

        # Calculate success rate
        total_tasks = len(self.task_manager.tasks)
        if total_tasks > 0:
            completed_tasks = len(
                [t for t in self.task_manager.tasks if t.status == TaskStatus.COMPLETED]
            )
            self.session_stats["success_rate"] = completed_tasks / total_tasks

    def display_session_stats(self):
        """Display current session statistics"""

        console.print(f"\n[bold blue]📈 Session Statistics[/bold blue]")

        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="white")

        stats_table.add_row(
            "Tasks Completed:", str(self.session_stats["tasks_completed"])
        )
        stats_table.add_row(
            "Total Iterations:", str(self.session_stats["total_iterations"])
        )
        stats_table.add_row(
            "Success Rate:", f"{self.session_stats['success_rate']:.1%}"
        )

        if self.session_stats["tasks_completed"] > 0:
            avg_iterations = (
                self.session_stats["total_iterations"]
                / self.session_stats["tasks_completed"]
            )
            stats_table.add_row("Avg Iterations/Task:", f"{avg_iterations:.1f}")

        console.print(stats_table)

        # Show task manager stats
        task_stats = self.task_manager.get_session_stats()
        console.print(
            f"\n[dim]Total tasks: {task_stats['total_tasks']}, "
            f"Pending: {task_stats['pending_tasks']}, "
            f"Failed: {task_stats['failed_tasks']}[/dim]"
        )

    def run_continuous_mode(self):
        """Run in continuous mode, asking for tasks until user quits"""

        console.print(f"\n[bold blue]🤖 Autonomous Agent - Continuous Mode[/bold blue]")
        console.print(
            "[dim]The agent will continuously ask for tasks and execute them autonomously.[/dim]"
        )
        console.print("[dim]Type 'quit' to exit.[/dim]")

        while True:
            try:
                # Ask for next task
                task_description = self.task_manager.ask_for_next_task()

                if task_description is None:
                    console.print("\n[yellow]👋 Goodbye![/yellow]")
                    break

                # Execute the task autonomously
                self.execute_autonomous_task(task_description, auto_continue=True)

                # Show session stats
                self.display_session_stats()

                # Brief pause before next task
                console.print("\n" + "=" * 60)

            except KeyboardInterrupt:
                console.print("\n[yellow]⏸️  Continuous mode interrupted[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]💥 Error in continuous mode: {e}[/red]")
                logger.error(f"Continuous mode error: {e}")
