"""
Reasoning Engine for step-by-step problem solving and thinking
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown

console = Console()
logger = logging.getLogger("codegen.reasoning")

class ThinkingStep(Enum):
    """Types of thinking steps"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    REFLECTION = "reflection"

@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    step_type: ThinkingStep
    title: str
    content: str
    confidence: float = 0.8
    dependencies: List[str] = None
    outputs: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.outputs is None:
            self.outputs = []

@dataclass
class ReasoningChain:
    """A complete reasoning chain for a problem"""
    problem: str
    steps: List[ReasoningStep]
    final_solution: str
    confidence: float
    reasoning_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ReasoningEngine:
    """Engine for structured reasoning and thinking"""
    
    def __init__(self, ai_engine=None):
        self.ai_engine = ai_engine
        self.reasoning_history: List[ReasoningChain] = []
        self.thinking_templates = self._load_thinking_templates()
        self.reasoning_strategies = {
            'step_by_step': self._step_by_step_reasoning,
            'problem_decomposition': self._problem_decomposition,
            'first_principles': self._first_principles_reasoning,
            'analogical': self._analogical_reasoning,
            'critical_thinking': self._critical_thinking
        }
    
    def _load_thinking_templates(self) -> Dict[str, str]:
        """Load templates for different types of thinking"""
        return {
            'code_generation': """
Let me think through this code generation request step by step:

1. ANALYSIS: What exactly is being requested?
   - Understanding the requirements
   - Identifying key components needed
   - Considering constraints and limitations

2. PLANNING: How should I approach this?
   - Breaking down into smaller tasks
   - Choosing appropriate technologies/patterns
   - Considering edge cases and error handling

3. IMPLEMENTATION: What's the best way to implement this?
   - Writing clean, maintainable code
   - Following best practices
   - Adding proper documentation

4. VALIDATION: Does this solution work?
   - Checking for logical errors
   - Ensuring requirements are met
   - Considering potential improvements
""",
            'debugging': """
Let me analyze this debugging problem systematically:

1. PROBLEM IDENTIFICATION: What's going wrong?
   - Understanding the symptoms
   - Reproducing the issue
   - Gathering relevant information

2. HYPOTHESIS FORMATION: What could be causing this?
   - Listing possible causes
   - Prioritizing by likelihood
   - Considering recent changes

3. INVESTIGATION: How can I test these hypotheses?
   - Designing tests to isolate the problem
   - Checking logs and error messages
   - Using debugging tools

4. SOLUTION: How do I fix this?
   - Implementing the fix
   - Testing the solution
   - Preventing similar issues
""",
            'optimization': """
Let me think about how to optimize this:

1. MEASUREMENT: What's the current performance?
   - Identifying bottlenecks
   - Measuring baseline performance
   - Understanding resource usage

2. ANALYSIS: Where are the inefficiencies?
   - Profiling the code
   - Identifying hot spots
   - Understanding algorithmic complexity

3. STRATEGY: What optimization approaches can I use?
   - Algorithmic improvements
   - Data structure optimizations
   - Caching strategies

4. IMPLEMENTATION: How do I apply these optimizations?
   - Making targeted improvements
   - Measuring impact
   - Ensuring correctness is maintained
"""
        }
    
    def reason_through_problem(self, problem: str, strategy: str = 'step_by_step', 
                             show_thinking: bool = True) -> ReasoningChain:
        """Reason through a problem using the specified strategy"""
        import time
        start_time = time.time()
        
        if show_thinking:
            console.print(f"\n[bold blue]🤔 Reasoning through: {problem}[/bold blue]")
            console.print(f"[dim]Strategy: {strategy}[/dim]\n")
        
        # Select reasoning strategy
        reasoning_func = self.reasoning_strategies.get(strategy, self._step_by_step_reasoning)
        
        # Execute reasoning
        steps = reasoning_func(problem, show_thinking)
        
        # Generate final solution
        final_solution = self._synthesize_solution(problem, steps)
        
        # Calculate overall confidence
        avg_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.5
        
        reasoning_time = time.time() - start_time
        
        # Create reasoning chain
        chain = ReasoningChain(
            problem=problem,
            steps=steps,
            final_solution=final_solution,
            confidence=avg_confidence,
            reasoning_time=reasoning_time,
            metadata={'strategy': strategy, 'step_count': len(steps)}
        )
        
        # Store in history
        self.reasoning_history.append(chain)
        
        if show_thinking:
            self._display_reasoning_chain(chain)
        
        return chain
    
    def _step_by_step_reasoning(self, problem: str, show_thinking: bool) -> List[ReasoningStep]:
        """Basic step-by-step reasoning approach"""
        steps = []
        
        # Step 1: Analysis
        analysis_step = self._analyze_problem(problem)
        steps.append(analysis_step)
        if show_thinking:
            self._display_thinking_step(analysis_step, 1)
        
        # Step 2: Planning
        planning_step = self._plan_solution(problem, analysis_step)
        steps.append(planning_step)
        if show_thinking:
            self._display_thinking_step(planning_step, 2)
        
        # Step 3: Implementation Strategy
        impl_step = self._design_implementation(problem, analysis_step, planning_step)
        steps.append(impl_step)
        if show_thinking:
            self._display_thinking_step(impl_step, 3)
        
        # Step 4: Validation
        validation_step = self._validate_approach(problem, steps)
        steps.append(validation_step)
        if show_thinking:
            self._display_thinking_step(validation_step, 4)
        
        return steps
    
    def _problem_decomposition(self, problem: str, show_thinking: bool) -> List[ReasoningStep]:
        """Break down complex problems into smaller parts"""
        steps = []
        
        # Identify main components
        decomp_step = ReasoningStep(
            step_type=ThinkingStep.ANALYSIS,
            title="Problem Decomposition",
            content=self._decompose_problem(problem),
            confidence=0.85
        )
        steps.append(decomp_step)
        if show_thinking:
            self._display_thinking_step(decomp_step, 1)
        
        # Solve each component
        components = self._extract_components(problem)
        for i, component in enumerate(components, 2):
            comp_step = ReasoningStep(
                step_type=ThinkingStep.IMPLEMENTATION,
                title=f"Solve Component: {component}",
                content=self._solve_component(component),
                confidence=0.8
            )
            steps.append(comp_step)
            if show_thinking:
                self._display_thinking_step(comp_step, i)
        
        # Integration step
        integration_step = ReasoningStep(
            step_type=ThinkingStep.PLANNING,
            title="Integration Strategy",
            content=self._plan_integration(components),
            confidence=0.75
        )
        steps.append(integration_step)
        if show_thinking:
            self._display_thinking_step(integration_step, len(steps))
        
        return steps
    
    def _first_principles_reasoning(self, problem: str, show_thinking: bool) -> List[ReasoningStep]:
        """Reason from fundamental principles"""
        steps = []
        
        # Identify fundamental concepts
        principles_step = ReasoningStep(
            step_type=ThinkingStep.ANALYSIS,
            title="Fundamental Principles",
            content=self._identify_principles(problem),
            confidence=0.9
        )
        steps.append(principles_step)
        if show_thinking:
            self._display_thinking_step(principles_step, 1)
        
        # Build up from basics
        building_step = ReasoningStep(
            step_type=ThinkingStep.IMPLEMENTATION,
            title="Build from Fundamentals",
            content=self._build_from_principles(problem),
            confidence=0.85
        )
        steps.append(building_step)
        if show_thinking:
            self._display_thinking_step(building_step, 2)
        
        return steps
    
    def _analogical_reasoning(self, problem: str, show_thinking: bool) -> List[ReasoningStep]:
        """Use analogies and similar problems"""
        steps = []
        
        # Find analogies
        analogy_step = ReasoningStep(
            step_type=ThinkingStep.ANALYSIS,
            title="Find Analogies",
            content=self._find_analogies(problem),
            confidence=0.7
        )
        steps.append(analogy_step)
        if show_thinking:
            self._display_thinking_step(analogy_step, 1)
        
        # Apply analogy
        application_step = ReasoningStep(
            step_type=ThinkingStep.IMPLEMENTATION,
            title="Apply Analogical Solution",
            content=self._apply_analogy(problem),
            confidence=0.75
        )
        steps.append(application_step)
        if show_thinking:
            self._display_thinking_step(application_step, 2)
        
        return steps
    
    def _critical_thinking(self, problem: str, show_thinking: bool) -> List[ReasoningStep]:
        """Apply critical thinking methodology"""
        steps = []
        
        # Question assumptions
        assumptions_step = ReasoningStep(
            step_type=ThinkingStep.ANALYSIS,
            title="Question Assumptions",
            content=self._question_assumptions(problem),
            confidence=0.8
        )
        steps.append(assumptions_step)
        if show_thinking:
            self._display_thinking_step(assumptions_step, 1)
        
        # Evaluate evidence
        evidence_step = ReasoningStep(
            step_type=ThinkingStep.VALIDATION,
            title="Evaluate Evidence",
            content=self._evaluate_evidence(problem),
            confidence=0.85
        )
        steps.append(evidence_step)
        if show_thinking:
            self._display_thinking_step(evidence_step, 2)
        
        # Consider alternatives
        alternatives_step = ReasoningStep(
            step_type=ThinkingStep.PLANNING,
            title="Consider Alternatives",
            content=self._consider_alternatives(problem),
            confidence=0.8
        )
        steps.append(alternatives_step)
        if show_thinking:
            self._display_thinking_step(alternatives_step, 3)
        
        return steps
    
    def _analyze_problem(self, problem: str) -> ReasoningStep:
        """Analyze the problem to understand requirements"""
        analysis_content = f"""
Understanding the problem: "{problem}"

Key aspects to consider:
- What is the main objective?
- What are the constraints and requirements?
- What technologies or approaches are most suitable?
- What are potential challenges or edge cases?
- What would success look like?

Initial assessment: This appears to be a {self._categorize_problem(problem)} problem.
"""
        
        return ReasoningStep(
            step_type=ThinkingStep.ANALYSIS,
            title="Problem Analysis",
            content=analysis_content.strip(),
            confidence=0.85
        )
    
    def _plan_solution(self, problem: str, analysis: ReasoningStep) -> ReasoningStep:
        """Plan the solution approach"""
        planning_content = f"""
Based on the analysis, here's my solution plan:

1. Choose the right approach/architecture
2. Break down into manageable components
3. Consider error handling and edge cases
4. Plan for testing and validation
5. Think about maintainability and extensibility

For this specific problem, I'll focus on:
- {self._get_key_focus_areas(problem)}
"""
        
        return ReasoningStep(
            step_type=ThinkingStep.PLANNING,
            title="Solution Planning",
            content=planning_content.strip(),
            confidence=0.8,
            dependencies=[analysis.title]
        )
    
    def _design_implementation(self, problem: str, analysis: ReasoningStep, 
                             planning: ReasoningStep) -> ReasoningStep:
        """Design the implementation strategy"""
        impl_content = f"""
Implementation strategy:

1. Start with core functionality
2. Add error handling and validation
3. Include proper documentation
4. Follow best practices and patterns
5. Make it extensible and maintainable

Specific implementation considerations:
- {self._get_implementation_details(problem)}
"""
        
        return ReasoningStep(
            step_type=ThinkingStep.IMPLEMENTATION,
            title="Implementation Design",
            content=impl_content.strip(),
            confidence=0.8,
            dependencies=[analysis.title, planning.title]
        )
    
    def _validate_approach(self, problem: str, previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Validate the reasoning and approach"""
        validation_content = f"""
Validating the approach:

✓ Does this solve the original problem?
✓ Are all requirements addressed?
✓ Is the solution robust and maintainable?
✓ Are there any obvious issues or improvements?

Confidence assessment: The approach looks solid with {len(previous_steps)} well-reasoned steps.
"""
        
        return ReasoningStep(
            step_type=ThinkingStep.VALIDATION,
            title="Approach Validation",
            content=validation_content.strip(),
            confidence=0.85,
            dependencies=[step.title for step in previous_steps]
        )
    
    def _synthesize_solution(self, problem: str, steps: List[ReasoningStep]) -> str:
        """Synthesize the final solution from reasoning steps"""
        if self.ai_engine:
            # Use AI to synthesize based on reasoning
            synthesis_prompt = f"""
Based on this reasoning process for the problem "{problem}":

{self._format_steps_for_ai(steps)}

Please provide a comprehensive solution that incorporates all the insights from the reasoning steps.
"""
            try:
                return self.ai_engine.generate_code(synthesis_prompt)
            except Exception as e:
                logger.warning(f"AI synthesis failed: {e}")
        
        # Fallback synthesis
        return self._fallback_synthesis(problem, steps)
    
    def _display_thinking_step(self, step: ReasoningStep, step_number: int):
        """Display a thinking step with rich formatting"""
        step_icon = {
            ThinkingStep.ANALYSIS: "🔍",
            ThinkingStep.PLANNING: "📋",
            ThinkingStep.IMPLEMENTATION: "⚙️",
            ThinkingStep.VALIDATION: "✅",
            ThinkingStep.OPTIMIZATION: "⚡",
            ThinkingStep.REFLECTION: "🤔"
        }.get(step.step_type, "💭")
        
        title = f"{step_icon} Step {step_number}: {step.title}"
        confidence_bar = "█" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
        
        panel = Panel(
            step.content,
            title=title,
            subtitle=f"Confidence: {confidence_bar} {step.confidence:.1%}",
            border_style="blue" if step.confidence > 0.8 else "yellow" if step.confidence > 0.6 else "red"
        )
        console.print(panel)
        console.print()
    
    def _display_reasoning_chain(self, chain: ReasoningChain):
        """Display the complete reasoning chain"""
        console.print(f"\n[bold green]✨ Reasoning Complete![/bold green]")
        
        # Summary table
        summary_table = Table(title="Reasoning Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Problem", chain.problem)
        summary_table.add_row("Strategy", chain.metadata.get('strategy', 'unknown'))
        summary_table.add_row("Steps", str(len(chain.steps)))
        summary_table.add_row("Confidence", f"{chain.confidence:.1%}")
        summary_table.add_row("Time", f"{chain.reasoning_time:.2f}s")
        
        console.print(summary_table)
        console.print()
        
        # Final solution
        if chain.final_solution:
            console.print("[bold blue]🎯 Final Solution:[/bold blue]")
            solution_panel = Panel(
                chain.final_solution,
                title="Generated Solution",
                border_style="green"
            )
            console.print(solution_panel)
    
    def get_reasoning_history(self) -> List[ReasoningChain]:
        """Get the history of reasoning chains"""
        return self.reasoning_history
    
    def export_reasoning(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Export reasoning chain to dictionary"""
        return {
            'problem': chain.problem,
            'steps': [
                {
                    'type': step.step_type.value,
                    'title': step.title,
                    'content': step.content,
                    'confidence': step.confidence,
                    'dependencies': step.dependencies,
                    'outputs': step.outputs
                }
                for step in chain.steps
            ],
            'final_solution': chain.final_solution,
            'confidence': chain.confidence,
            'reasoning_time': chain.reasoning_time,
            'metadata': chain.metadata
        }
    
    # Helper methods for problem analysis
    def _categorize_problem(self, problem: str) -> str:
        """Categorize the type of problem"""
        problem_lower = problem.lower()
        if any(word in problem_lower for word in ['api', 'server', 'web', 'http']):
            return "web development"
        elif any(word in problem_lower for word in ['algorithm', 'sort', 'search', 'optimize']):
            return "algorithmic"
        elif any(word in problem_lower for word in ['data', 'database', 'sql', 'query']):
            return "data management"
        elif any(word in problem_lower for word in ['class', 'object', 'inherit', 'design']):
            return "object-oriented design"
        else:
            return "general programming"
    
    def _get_key_focus_areas(self, problem: str) -> str:
        """Get key focus areas for the problem"""
        return "Clean code structure, proper error handling, and maintainable design"
    
    def _get_implementation_details(self, problem: str) -> str:
        """Get specific implementation details"""
        return "Following best practices, adding comprehensive documentation, and ensuring testability"
    
    def _decompose_problem(self, problem: str) -> str:
        """Decompose problem into components"""
        return f"Breaking down '{problem}' into smaller, manageable components that can be solved independently."
    
    def _extract_components(self, problem: str) -> List[str]:
        """Extract components from problem"""
        # Simple heuristic - in real implementation, this could be more sophisticated
        return ["Core Logic", "Input Validation", "Error Handling", "Output Formatting"]
    
    def _solve_component(self, component: str) -> str:
        """Solve individual component"""
        return f"Implementing {component} with focus on reliability and maintainability."
    
    def _plan_integration(self, components: List[str]) -> str:
        """Plan integration of components"""
        return f"Integrating {len(components)} components with proper interfaces and error propagation."
    
    def _identify_principles(self, problem: str) -> str:
        """Identify fundamental principles"""
        return "Starting from basic programming principles: modularity, separation of concerns, and clear interfaces."
    
    def _build_from_principles(self, problem: str) -> str:
        """Build solution from principles"""
        return "Building up the solution step by step from fundamental concepts."
    
    def _find_analogies(self, problem: str) -> str:
        """Find analogies for the problem"""
        return "Looking for similar problems and patterns that have been solved before."
    
    def _apply_analogy(self, problem: str) -> str:
        """Apply analogical solution"""
        return "Adapting the analogical solution to fit the specific requirements of this problem."
    
    def _question_assumptions(self, problem: str) -> str:
        """Question assumptions in the problem"""
        return "Examining the assumptions and constraints to ensure we're solving the right problem."
    
    def _evaluate_evidence(self, problem: str) -> str:
        """Evaluate evidence and requirements"""
        return "Critically evaluating the requirements and evidence to ensure a robust solution."
    
    def _consider_alternatives(self, problem: str) -> str:
        """Consider alternative approaches"""
        return "Exploring alternative approaches and weighing their trade-offs."
    
    def _format_steps_for_ai(self, steps: List[ReasoningStep]) -> str:
        """Format reasoning steps for AI consumption"""
        formatted = ""
        for i, step in enumerate(steps, 1):
            formatted += f"\nStep {i} ({step.step_type.value}): {step.title}\n"
            formatted += f"{step.content}\n"
            formatted += f"Confidence: {step.confidence:.1%}\n"
        return formatted
    
    def _fallback_synthesis(self, problem: str, steps: List[ReasoningStep]) -> str:
        """Fallback synthesis when AI is not available"""
        return f"""
# Solution for: {problem}

# Based on reasoning through {len(steps)} steps:
# {', '.join(step.title for step in steps)}

def solve_problem():
    \"\"\"
    Generated solution based on systematic reasoning.
    This is a template - implement specific logic based on the problem.
    \"\"\"
    # TODO: Implement based on reasoning steps
    pass

if __name__ == "__main__":
    solve_problem()
"""
