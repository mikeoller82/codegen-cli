import logging
import os
from typing import Tuple, Any, List

import google.generativeai as genai
import openai

from . import config

logger = logging.getLogger(__name__)

# Import the reasoning engine
from .reasoning import ReasoningEngine, ReasoningChain
# Add this import at the top
from .auto_fix import AutoFixEngine
from .learning_engine import LearningEngine

from rich.console import Console
console = Console()


class AIEngine:
    def __init__(self):
        self.default_model = config.get_default_model()
        self.openai_client = None
        self.gemini_model = None

        # Initialize OpenAI client
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables.")

        # Initialize Gemini model
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
                logger.info("Gemini model initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.gemini_model = None
        else:
            logger.warning("GOOGLE_API_KEY not found in environment variables.")

        self.prompt_template = """You are an expert software engineer.
Generate clean, well-documented code to solve the following problem:
{prompt}
```"""

        self.prompt_template_few_shot = """You are an expert software engineer. You are working on a complex project.
You are given a series of examples of problems and solutions.
Use these examples to guide your code generation.

Here are some examples:
{examples}

Now generate clean, well-documented code to solve the following problem:
{prompt}
```"""

        # Initialize learning engine first
        self.learning_engine = LearningEngine(self)
        
        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(self)
        self.use_reasoning = True  # Flag to enable/disable reasoning

        # Initialize auto-fix engine with learning
        self.auto_fix_engine = AutoFixEngine(self, self.learning_engine)
        self.auto_test = True  # Flag to enable/disable auto-testing

    def _generate_with_openai(self, prompt: str, model: str) -> str:
        """Generate code using OpenAI's models."""
        try:
            if self.openai_client:
                chat_completion = self.openai_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                )
                return chat_completion.choices[0].message.content
            else:
                logger.error("OpenAI client not initialized.")
                return "Failed to generate code: OpenAI client not initialized."
        except Exception as e:
            logger.error(f"Error generating code with OpenAI: {e}")
            return f"Failed to generate code: {e}"

    def _generate_with_gemini(self, prompt: str) -> str:
        """Generate code using Gemini model."""
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            else:
                logger.error("Gemini model not initialized.")
                return "Failed to generate code: Gemini model not initialized."
        except Exception as e:
            logger.error(f"Error generating code with Gemini: {e}")
            return f"Failed to generate code: {e}"

    def generate_code(self, prompt: str, model: str = None, use_reasoning: bool = None) -> str:
        """Generate code using the specified AI model with learning improvements"""
        if use_reasoning is None:
            use_reasoning = self.use_reasoning
        
        # Apply learned improvements to the prompt
        improved_prompt, improvement_note = self.learning_engine.improve_prompt(prompt)
        if improvement_note:
            console.print(f"[dim]💡 Applied learned improvement: {improvement_note}[/dim]")
        
        if use_reasoning:
            code, _ = self.generate_code_with_reasoning(improved_prompt, model, show_thinking=False)
            return code
        else:
            model = model or self.default_model
            
            # Check for learned code templates
            template = self.learning_engine.get_code_template(improved_prompt)
            if template:
                console.print("[dim]📋 Using learned code template as starting point[/dim]")
                enhanced_prompt = f"{improved_prompt}\n\nUse this template as a starting point:\n{template}"
            else:
                enhanced_prompt = self.prompt_template.format(prompt=improved_prompt)

            try:
                if model.startswith('gpt') and self.openai_client:
                    return self._generate_with_openai(enhanced_prompt, model)
                elif model.startswith('gemini') and self.gemini_model:
                    return self._generate_with_gemini(enhanced_prompt)
                else:
                    return "Model not supported."
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                return "Failed to generate code."

    def generate_code_with_examples(self, prompt: str, examples: str, model: str = None) -> str:
        """Generate code using the specified AI model with few shot examples"""
        model = model or self.default_model
        
        # Apply learned improvements
        improved_prompt, improvement_note = self.learning_engine.improve_prompt(prompt)
        if improvement_note:
            console.print(f"[dim]💡 Applied learned improvement: {improvement_note}[/dim]")
        
        enhanced_prompt = self.prompt_template_few_shot.format(prompt=improved_prompt, examples=examples)

        try:
            if model.startswith('gpt') and self.openai_client:
                return self._generate_with_openai(enhanced_prompt, model)
            elif model.startswith('gemini') and self.gemini_model:
                return self._generate_with_gemini(enhanced_prompt)
            else:
                return "Model not supported."
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return "Failed to generate code."

    def generate_code_with_reasoning(self, prompt: str, model: str = None, 
                                   reasoning_strategy: str = 'step_by_step',
                                   show_thinking: bool = True) -> Tuple[str, ReasoningChain]:
        """Generate code with explicit reasoning process and learning"""
        model = model or self.default_model
        
        # Apply learned improvements to the prompt
        improved_prompt, improvement_note = self.learning_engine.improve_prompt(prompt)
        if improvement_note and show_thinking:
            console.print(f"[dim]💡 Applied learned improvement: {improvement_note}[/dim]")
        
        # First, reason through the problem
        reasoning_chain = self.reasoning_engine.reason_through_problem(
            improved_prompt, reasoning_strategy, show_thinking
        )
        
        # Check for learned code templates
        template = self.learning_engine.get_code_template(improved_prompt)
        template_context = f"\n\nUse this learned template as guidance:\n{template}" if template else ""
        
        # Use the reasoning to enhance the prompt
        enhanced_prompt = f"""
Based on this systematic reasoning process:

Problem: {improved_prompt}

Reasoning Steps:
{self.reasoning_engine._format_steps_for_ai(reasoning_chain.steps)}

Final Solution Strategy:
{reasoning_chain.final_solution}
{template_context}

Now generate clean, well-documented code that implements this solution:
"""
        
        # Generate the actual code
        try:
            if model.startswith('gpt') and self.openai_client:
                code = self._generate_with_openai(enhanced_prompt, model)
            elif model.startswith('gemini') and self.gemini_model:
                code = self._generate_with_gemini(enhanced_prompt)
            else:
                code = reasoning_chain.final_solution
            
            return code, reasoning_chain
            
        except Exception as e:
            logger.error(f"Code generation with reasoning failed: {e}")
            return reasoning_chain.final_solution, reasoning_chain

    def generate_code_with_auto_fix(self, prompt: str, model: str = None, 
                                  use_reasoning: bool = True,
                                  reasoning_strategy: str = 'step_by_step',
                                  show_progress: bool = True) -> Tuple[str, Any, List, Any]:
        """Generate code with automatic testing, fixing, and learning"""
        
        if show_progress:
            console.print(f"\n[bold blue]🚀 Enhanced AI Generation with Learning[/bold blue]")
            console.print(f"[dim]Prompt: {prompt}[/dim]")
        
        # First generate the code (with or without reasoning)
        if use_reasoning:
            initial_code, reasoning_chain = self.generate_code_with_reasoning(
                prompt, model, reasoning_strategy, show_thinking=show_progress
            )
        else:
            initial_code = self.generate_code(prompt, model, use_reasoning=False)
            reasoning_chain = None
        
        # Test and fix the code with learning
        final_code, fix_attempts, final_test = self.auto_fix_engine.test_and_fix_code(
            initial_code, prompt, show_progress=show_progress
        )
        
        # Generate summary with learning insights
        fix_summary = self.auto_fix_engine.generate_fix_summary(
            initial_code, final_code, fix_attempts, final_test
        )
        
        if show_progress:
            console.print(f"\n[bold blue]📋 Enhanced Fix Summary:[/bold blue]")
            console.print(fix_summary)
            
            # Show learning metrics if available
            if hasattr(self.learning_engine, 'get_learning_metrics'):
                metrics = self.learning_engine.get_learning_metrics()
                if metrics.total_patterns > 0:
                    console.print(f"\n[dim]🧠 Learning Status: {metrics.total_patterns} patterns learned, "
                                f"{metrics.avg_confidence:.1%} avg confidence[/dim]")
        
        return final_code, reasoning_chain, fix_attempts, final_test

    def get_available_models(self) -> List[str]:
        """Get list of available AI models"""
        models = []
        
        if self.openai_client:
            models.extend(['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'])
        
        if self.gemini_model:
            models.extend(['gemini-1.5-pro-latest'])
        
        if not models:
            models = ['mock']  # Fallback for testing
        
        return models

    def get_learning_status(self) -> dict:
        """Get current learning system status"""
        if not self.learning_engine:
            return {"status": "disabled"}
        
        metrics = self.learning_engine.get_learning_metrics()
        return {
            "status": "active",
            "total_patterns": metrics.total_patterns,
            "active_patterns": metrics.active_patterns,
            "avg_confidence": metrics.avg_confidence,
            "learning_rate": metrics.learning_rate,
            "recent_improvements": metrics.recent_improvements
        }

    def display_learning_insights(self):
        """Display learning insights and recommendations"""
        if not self.learning_engine:
            console.print("[yellow]Learning system not available[/yellow]")
            return
        
        console.print("\n[bold blue]🧠 AI Learning Insights[/bold blue]")
        self.learning_engine.display_learning_status()
        
        # Show recent learning activity
        metrics = self.learning_engine.get_learning_metrics()
        if metrics.recent_improvements > 0:
            console.print(f"\n[green]📈 Recent Activity:[/green] {metrics.recent_improvements} new patterns learned this week")
        
        # Learning recommendations
        console.print(f"\n[bold yellow]💡 Recommendations:[/bold yellow]")
        if metrics.avg_confidence < 0.7:
            console.print("• Continue using the system to improve pattern confidence")
        if metrics.learning_rate < 1.0:
            console.print("• Try more diverse coding tasks to expand learning")
        if metrics.active_patterns < 10:
            console.print("• Generate more code to build up the learning database")
        
        console.print("\n[dim]The AI learns from every successful fix and improves over time![/dim]")
