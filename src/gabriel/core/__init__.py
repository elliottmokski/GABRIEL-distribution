"""Core plumbing components for GABRIEL."""

from .llm_client import LLMClient
from .prompt_template import PromptTemplate
from .pipeline import Pipeline

__all__ = ["LLMClient", "PromptTemplate", "Pipeline"]
