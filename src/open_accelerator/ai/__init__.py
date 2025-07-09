"""
Compound AI System for Open Accelerator

This module provides a comprehensive compound AI system with DSPy integration,
supporting multiple OpenAI models for different AI tasks and reasoning patterns.
"""

from .agents import (
    AgentOrchestrator,
    AnalysisAgent,
    MedicalComplianceAgent,
    OptimizationAgent,
)
from .compound_ai import CompoundAISystem
from .model_registry import ModelConfig, ModelRegistry
from .multimodal_processor import MultimodalProcessor
from .reasoning_chains import ChainOfThought, ReasoningChain, ReflectiveReasoning

__all__ = [
    "CompoundAISystem",
    "ModelRegistry",
    "ModelConfig",
    "ReasoningChain",
    "ChainOfThought",
    "ReflectiveReasoning",
    "MultimodalProcessor",
    "OptimizationAgent",
    "AnalysisAgent",
    "MedicalComplianceAgent",
    "AgentOrchestrator",
]

__version__ = "1.0.0"
