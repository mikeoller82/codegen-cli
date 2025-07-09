import logging
import os
from typing import Tuple

import google.generativeai as genai
import openai

from . import config

logger = logging.getLogger(__name__)

# Import the reasoning engine
from .reasoning import ReasoningEngine, ReasoningChain


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
\`\`\`"""

        self.prompt_template_few_shot = """You are an expert software engineer. You are working on a complex project.
You are given a series of examples of problems and solutions.
Use these examples to guide your code generation.

Here are some examples:
{examples}

Now generate clean, well-documented code to solve the following problem:
{prompt}
\`\`\`"""

        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(self)
        self.use_reasoning = True  # Flag to enable/disable reasoning

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
        """Generate code using the specified AI model"""
        if use_reasoning is None:
            use_reasoning = self.use_reasoning
        
        if use_reasoning:
            code, _ = self.generate_code_with_reasoning(prompt, model, show_thinking=False)
            return code
        else:
            model = model or self.default_model
            prompt = self.prompt_template.format(prompt=prompt)

            try:
                if model.startswith('gpt') and self.openai_client:
                    return self._generate_with_openai(prompt, model)
                elif model.startswith('gemini') and self.gemini_model:
                    return self._generate_with_gemini(prompt)
                else:
                    return "Model not supported."
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                return "Failed to generate code."

    def generate_code_with_examples(self, prompt: str, examples: str, model: str = None) -> str:
        """Generate code using the specified AI model with few shot examples"""
        model = model or self.default_model
        prompt = self.prompt_template_few_shot.format(prompt=prompt, examples=examples)

        try:
            if model.startswith('gpt') and self.openai_client:
                return self._generate_with_openai(prompt, model)
            elif model.startswith('gemini') and self.gemini_model:
                return self._generate_with_gemini(prompt)
            else:
                return "Model not supported."
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return "Failed to generate code."

    def generate_code_with_reasoning(self, prompt: str, model: str = None, 
                                   reasoning_strategy: str = 'step_by_step',
                                   show_thinking: bool = True) -> Tuple[str, ReasoningChain]:
        """Generate code with explicit reasoning process"""
        model = model or self.default_model
        
        # First, reason through the problem
        reasoning_chain = self.reasoning_engine.reason_through_problem(
            prompt, reasoning_strategy, show_thinking
        )
        
        # Use the reasoning to enhance the prompt
        enhanced_prompt = f"""
Based on this systematic reasoning process:

Problem: {prompt}

Reasoning Steps:
{self.reasoning_engine._format_steps_for_ai(reasoning_chain.steps)}

Final Solution Strategy:
{reasoning_chain.final_solution}

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
