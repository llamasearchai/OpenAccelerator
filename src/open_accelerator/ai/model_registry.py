"""
Model Registry for OpenAI API Models

Manages configuration and selection of optimal OpenAI models for different tasks.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    import openai
    from openai import OpenAI as OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ModelFamily(Enum):
    """OpenAI model families."""

    GPT4X = "gpt-4.x"
    O_SERIES = "o-series"
    IMAGE = "image"
    AUDIO = "audio"


class TaskType(Enum):
    """Types of AI tasks."""

    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    OPTIMIZATION = "optimization"
    MULTIMODAL = "multimodal"
    TRANSCRIPTION = "transcription"
    TEXT_TO_SPEECH = "text_to_speech"
    IMAGE_GENERATION = "image_generation"
    GENERAL_PURPOSE = "general_purpose"
    CLASSIFICATION = "classification"
    COMPLETION = "completion"


@dataclass
class ModelCapabilities:
    """Defines model capabilities and constraints."""

    max_tokens: int
    supports_multimodal: bool = False
    supports_audio: bool = False
    supports_image: bool = False
    reasoning_optimized: bool = False
    cost_tier: str = "standard"  # "budget", "standard", "premium"
    latency_tier: str = "standard"  # "fast", "standard", "slow"
    quality_tier: str = "standard"  # "standard", "high", "premium"


@dataclass
class ModelConfig:
    """Configuration for an OpenAI model."""

    name: str
    family: ModelFamily
    description: str
    capabilities: ModelCapabilities
    optimal_tasks: list[TaskType] = field(default_factory=list)
    api_name: str = ""

    def __post_init__(self):
        if not self.api_name:
            self.api_name = self.name


class ModelRegistry:
    """Registry and manager for OpenAI models."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the model registry."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        # Initialize OpenAI client
        openai.api_key = self.api_key

        # Model registry
        self.models: dict[str, ModelConfig] = {}
        self._init_models()

        # Active model instances
        self._model_instances: dict[str, OpenAIClient] = {}

    def _init_models(self):
        """Initialize the registry with all supported models."""

        # GPT-4.x Family Models
        self.models["gpt-4.1"] = ModelConfig(
            name="gpt-4.1",
            family=ModelFamily.GPT4X,
            description="Flagship GPT-4.1 model with 1M token context, strong general-purpose reasoning",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                supports_multimodal=True,
                supports_image=True,
                cost_tier="premium",
                quality_tier="premium",
            ),
            optimal_tasks=[
                TaskType.REASONING,
                TaskType.ANALYSIS,
                TaskType.CODE_GENERATION,
                TaskType.MULTIMODAL,
                TaskType.GENERAL_PURPOSE,
            ],
        )

        self.models["gpt-4.1-mini"] = ModelConfig(
            name="gpt-4.1-mini",
            family=ModelFamily.GPT4X,
            description="Smaller, faster variant of GPT-4.1 with improved latency",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                supports_multimodal=True,
                supports_image=True,
                cost_tier="standard",
                latency_tier="fast",
                quality_tier="high",
            ),
            optimal_tasks=[
                TaskType.ANALYSIS,
                TaskType.CODE_GENERATION,
                TaskType.GENERAL_PURPOSE,
                TaskType.MULTIMODAL,
            ],
        )

        self.models["gpt-4.1-nano"] = ModelConfig(
            name="gpt-4.1-nano",
            family=ModelFamily.GPT4X,
            description="Lightweight version for classification and autocompletion",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                cost_tier="budget",
                latency_tier="fast",
                quality_tier="standard",
            ),
            optimal_tasks=[
                TaskType.CLASSIFICATION,
                TaskType.COMPLETION,
                TaskType.GENERAL_PURPOSE,
            ],
        )

        self.models["gpt-4o"] = ModelConfig(
            name="gpt-4o",
            family=ModelFamily.GPT4X,
            description="Multimodal 'omni' model with text, image, and audio support",
            capabilities=ModelCapabilities(
                max_tokens=128_000,
                supports_multimodal=True,
                supports_image=True,
                supports_audio=True,
                cost_tier="standard",
                latency_tier="fast",
                quality_tier="high",
            ),
            optimal_tasks=[
                TaskType.MULTIMODAL,
                TaskType.ANALYSIS,
                TaskType.GENERAL_PURPOSE,
            ],
        )

        self.models["gpt-4o-mini"] = ModelConfig(
            name="gpt-4o-mini",
            family=ModelFamily.GPT4X,
            description="Affordable, lighter version of gpt-4o with strong multimodal performance",
            capabilities=ModelCapabilities(
                max_tokens=128_000,
                supports_multimodal=True,
                supports_image=True,
                supports_audio=True,
                cost_tier="budget",
                latency_tier="fast",
                quality_tier="standard",
            ),
            optimal_tasks=[
                TaskType.MULTIMODAL,
                TaskType.CLASSIFICATION,
                TaskType.COMPLETION,
            ],
        )

        # O-Series Models (Reasoning-Optimized)
        self.models["o3-pro"] = ModelConfig(
            name="o3-pro",
            family=ModelFamily.O_SERIES,
            description="State-of-the-art reasoning model, highest accuracy for complex logic",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                reasoning_optimized=True,
                cost_tier="premium",
                latency_tier="slow",
                quality_tier="premium",
            ),
            optimal_tasks=[
                TaskType.REASONING,
                TaskType.OPTIMIZATION,
                TaskType.ANALYSIS,
            ],
        )

        self.models["o3"] = ModelConfig(
            name="o3",
            family=ModelFamily.O_SERIES,
            description="Advanced reasoning model for complex logic and scientific reasoning",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                reasoning_optimized=True,
                cost_tier="standard",
                latency_tier="standard",
                quality_tier="high",
            ),
            optimal_tasks=[
                TaskType.REASONING,
                TaskType.ANALYSIS,
                TaskType.OPTIMIZATION,
            ],
        )

        self.models["o3-mini"] = ModelConfig(
            name="o3-mini",
            family=ModelFamily.O_SERIES,
            description="Optimized for lightweight chain-of-thought reasoning",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                reasoning_optimized=True,
                cost_tier="budget",
                latency_tier="fast",
                quality_tier="standard",
            ),
            optimal_tasks=[TaskType.REASONING, TaskType.ANALYSIS],
        )

        self.models["o4-mini"] = ModelConfig(
            name="o4-mini",
            family=ModelFamily.O_SERIES,
            description="Multimodal reasoning model blending GPT-4 strengths with o-series accuracy",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                supports_multimodal=True,
                supports_image=True,
                reasoning_optimized=True,
                cost_tier="standard",
                latency_tier="standard",
                quality_tier="high",
            ),
            optimal_tasks=[TaskType.REASONING, TaskType.MULTIMODAL, TaskType.ANALYSIS],
        )

        self.models["o1-pro"] = ModelConfig(
            name="o1-pro",
            family=ModelFamily.O_SERIES,
            description="Pro-grade reasoning model for higher accuracy and stable long-form output",
            capabilities=ModelCapabilities(
                max_tokens=1_000_000,
                reasoning_optimized=True,
                cost_tier="premium",
                latency_tier="standard",
                quality_tier="premium",
            ),
            optimal_tasks=[TaskType.REASONING, TaskType.ANALYSIS],
        )

        # Image Models
        self.models["gpt-image-1"] = ModelConfig(
            name="gpt-image-1",
            family=ModelFamily.IMAGE,
            description="High-quality image generation with captioning and style control",
            capabilities=ModelCapabilities(
                max_tokens=4096,
                supports_image=True,
                cost_tier="standard",
                quality_tier="high",
            ),
            optimal_tasks=[TaskType.IMAGE_GENERATION, TaskType.MULTIMODAL],
        )

        # Audio Models
        self.models["gpt-4o-transcribe"] = ModelConfig(
            name="gpt-4o-transcribe",
            family=ModelFamily.AUDIO,
            description="Multilingual ASR model for high-quality transcription",
            capabilities=ModelCapabilities(
                max_tokens=32768,
                supports_audio=True,
                cost_tier="standard",
                quality_tier="high",
            ),
            optimal_tasks=[TaskType.TRANSCRIPTION],
        )

        self.models["gpt-4o-mini-transcribe"] = ModelConfig(
            name="gpt-4o-mini-transcribe",
            family=ModelFamily.AUDIO,
            description="Budget transcription model optimized for mobile and edge use",
            capabilities=ModelCapabilities(
                max_tokens=32768,
                supports_audio=True,
                cost_tier="budget",
                latency_tier="fast",
                quality_tier="standard",
            ),
            optimal_tasks=[TaskType.TRANSCRIPTION],
        )

        self.models["gpt-4o-mini-tts"] = ModelConfig(
            name="gpt-4o-mini-tts",
            family=ModelFamily.AUDIO,
            description="Fast, responsive text-to-speech with emotional expression",
            capabilities=ModelCapabilities(
                max_tokens=4096,
                supports_audio=True,
                cost_tier="budget",
                latency_tier="fast",
                quality_tier="standard",
            ),
            optimal_tasks=[TaskType.TEXT_TO_SPEECH],
        )

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.models.get(name)

    def list_models(self, family: Optional[ModelFamily] = None) -> list[ModelConfig]:
        """List all models, optionally filtered by family."""
        models = list(self.models.values())
        if family:
            models = [m for m in models if m.family == family]
        return models

    def get_optimal_model(
        self,
        task: TaskType,
        cost_preference: str = "standard",
        quality_preference: str = "standard",
        latency_preference: str = "standard",
    ) -> Optional[ModelConfig]:
        """Get the optimal model for a specific task and preferences."""

        # Filter models that can handle the task
        candidate_models = [
            model for model in self.models.values() if task in model.optimal_tasks
        ]

        if not candidate_models:
            return None

        # Score models based on preferences
        def score_model(model: ModelConfig) -> float:
            score = 0.0

            # Cost preference scoring
            cost_scores = {"budget": 3, "standard": 2, "premium": 1}
            if cost_preference == "budget" and model.capabilities.cost_tier == "budget":
                score += 3
            elif cost_preference == "standard" and model.capabilities.cost_tier in [
                "budget",
                "standard",
            ]:
                score += 2
            elif cost_preference == "premium":
                score += cost_scores.get(model.capabilities.cost_tier, 0)

            # Quality preference scoring
            quality_scores = {"standard": 1, "high": 2, "premium": 3}
            if (
                quality_preference == "premium"
                and model.capabilities.quality_tier == "premium"
            ):
                score += 3
            elif quality_preference == "high" and model.capabilities.quality_tier in [
                "high",
                "premium",
            ]:
                score += 2
            elif quality_preference == "standard":
                score += quality_scores.get(model.capabilities.quality_tier, 0)

            # Latency preference scoring
            if (
                latency_preference == "fast"
                and model.capabilities.latency_tier == "fast"
            ):
                score += 2
            elif latency_preference == "standard":
                score += 1

            # Reasoning optimization bonus for reasoning tasks
            if task == TaskType.REASONING and model.capabilities.reasoning_optimized:
                score += 2

            return score

        # Return the highest scoring model
        best_model = max(candidate_models, key=score_model)
        return best_model

    def get_openai_client(self, model_name: str, **kwargs) -> OpenAIClient:
        """Get an OpenAI client instance."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
            
        if model_name not in self._model_instances:
            model_config = self.get_model(model_name)
            if not model_config:
                raise ValueError(f"Model {model_name} not found in registry")

            # Create OpenAI client instance
            self._model_instances[model_name] = OpenAIClient(
                api_key=self.api_key,
                **kwargs
            )

        return self._model_instances[model_name]

    def get_model_recommendations(self, task: TaskType) -> list[dict[str, Any]]:
        """Get ranked model recommendations for a task."""
        recommendations = []

        for model in self.models.values():
            if task in model.optimal_tasks:
                recommendations.append(
                    {
                        "name": model.name,
                        "family": model.family.value,
                        "description": model.description,
                        "cost_tier": model.capabilities.cost_tier,
                        "quality_tier": model.capabilities.quality_tier,
                        "latency_tier": model.capabilities.latency_tier,
                        "max_tokens": model.capabilities.max_tokens,
                        "reasoning_optimized": model.capabilities.reasoning_optimized,
                        "multimodal": model.capabilities.supports_multimodal,
                    }
                )

        # Sort by quality tier, then by cost efficiency
        quality_order = {"premium": 3, "high": 2, "standard": 1}
        cost_order = {"budget": 3, "standard": 2, "premium": 1}

        recommendations.sort(
            key=lambda x: (
                quality_order.get(x["quality_tier"], 0),
                cost_order.get(x["cost_tier"], 0),
            ),
            reverse=True,
        )

        return recommendations

    def estimate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost for a model call based on current OpenAI pricing."""
        model = self.get_model(model_name)
        if not model:
            return 0.0

        # OpenAI pricing as of 2024 (subject to change)
        base_rates = {
            "budget": {"input": 0.0001, "output": 0.0002},
            "standard": {"input": 0.001, "output": 0.002},
            "premium": {"input": 0.01, "output": 0.03},
        }

        rates = base_rates.get(model.capabilities.cost_tier, base_rates["standard"])
        return input_tokens * rates["input"] + output_tokens * rates["output"]
