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

        # Initialize learning engine
        self.learning_engine = LearningEngine(self)
        
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
                logger.error("Gemini model not initialize
