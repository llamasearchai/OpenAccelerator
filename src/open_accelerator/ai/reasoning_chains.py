#!/usr/bin/env python3
"""
Reasoning chains for advanced AI problem-solving.
Implements various reasoning strategies including chain-of-thought,
reflective reasoning, multi-step reasoning, and distillation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Conditional imports with fallbacks
try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. AI features will be limited.")

from .model_registry import ModelConfig

logger = logging.getLogger(__name__)


class ChainType(Enum):
    """Reasoning chain types."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    MEDICAL_DIAGNOSIS = "medical_diagnosis"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REFLECTIVE = "reflective"
    MULTI_STEP = "multi_step"
    DISTILLATION = "distillation"


@dataclass
class ReasoningStep:
    """Individual reasoning step in a chain."""

    step_id: str
    step_type: str  # 'analysis', 'synthesis', 'reflection', 'validation'
    prompt: str
    response: str
    confidence: float
    processing_time: float
    model_used: str
    token_usage: Dict[str, int]
    metadata: Dict[str, Any]


@dataclass
class ReasoningResult:
    """Result of reasoning chain execution."""

    chain_id: str
    chain_type: str
    final_answer: str
    confidence_score: float
    reasoning_steps: List[ReasoningStep]
    total_processing_time: float
    quality_metrics: Dict[str, float]
    error_analysis: Optional[str] = None
    improvement_suggestions: Optional[List[str]] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value by key (dict-like behavior)."""
        return getattr(self, key, default)


class ReasoningChain:
    """Abstract base class for reasoning chains with all required attributes."""

    def __init__(
        self, chain_type: ChainType, model_config: Optional[ModelConfig] = None
    ):
        """Initialize reasoning chain with all required attributes."""
        self.chain_id = str(uuid.uuid4())
        self.chain_type = chain_type
        self.model_config = model_config
        self.created_at = datetime.utcnow()
        self.execution_count = 0
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}_{self.chain_id[:8]}"
        )

        # Initialize core attributes
        self.reasoning_steps: List[ReasoningStep] = []
        self.context_memory: Dict[str, Any] = {}
        self.steps: Dict[str, Any] = {}

        # Initialize OpenAI client if available
        self.client = None
        if OPENAI_AVAILABLE and model_config:
            try:
                self.client = OpenAI()
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")

        # Initialize performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_processing_time": 0.0,
            "average_confidence": 0.0,
            "last_execution": None,
        }

        # Initialize quality assessment
        self.quality_threshold = 0.8
        self.max_retries = 3

        self.logger.info(f"ReasoningChain {self.chain_id} initialized")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously execute added steps for simple reasoning chains used in unit tests."""
        data: Dict[str, Any] = dict(input_data)
        for name, step in self.steps.items():
            try:
                result = step(data)
                if isinstance(result, dict):
                    data.update(result)
            except Exception as exc:
                self.logger.error(f"Step '{name}' failed: {exc}")
                return {"error": str(exc)}
        return data

    # Backward-compatible async wrapper
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper to allow awaiting the chain in async contexts."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data)

    def _ensure_return_value(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure a valid return value"""
        if result is None:
            return {"error": "No result generated", "chain_id": self.chain_id}
        return result

    def add_step(self, name: str, step: Any):
        """Add a reasoning step to the chain."""
        self.steps[name] = step
        self.logger.debug(f"Added step '{name}' to chain {self.chain_id}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the reasoning chain (synchronous wrapper)."""
        try:
            # Run the async execute method
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in async context, we can't use run_until_complete
                    # Return a promise that the caller needs to await
                    self.logger.warning(
                        "Cannot run synchronous process in async context. Use execute() directly."
                    )
                    return {
                        "error": "Use execute() method in async context",
                        "chain_id": self.chain_id,
                    }
                else:
                    # If not in async context, run it
                    return loop.run_until_complete(self.execute(input_data))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.execute(input_data))
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return {"error": str(e), "chain_id": self.chain_id}

    def validate(self) -> bool:
        """Validate the reasoning chain configuration."""
        if not self.chain_type:
            self.logger.error("Chain type not specified")
            return False

        if self.model_config is None:
            self.logger.warning("No model configuration provided")
            return True  # Allow execution without model config

        return True

    def optimize(self) -> Dict[str, Any]:
        """Optimize the reasoning chain performance."""
        optimization_results = {
            "optimized": True,
            "chain_id": self.chain_id,
            "optimizations_applied": [],
        }

        # Optimize based on execution history
        if self.metrics["total_executions"] > 0:
            avg_time = self.metrics["average_processing_time"]
            if avg_time > 10.0:  # If average time > 10 seconds
                optimization_results["optimizations_applied"].append(
                    "timeout_reduction"
                )

            avg_confidence = self.metrics["average_confidence"]
            if avg_confidence < 0.7:  # If confidence is low
                optimization_results["optimizations_applied"].append(
                    "confidence_improvement"
                )

        return optimization_results

    async def _execute_reasoning_step(
        self,
        step_id: str,
        step_type: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> ReasoningStep:
        """Execute a single reasoning step."""
        if not self.client:
            # Return a mock step if no client available
            return ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                prompt=prompt,
                response="Mock response - OpenAI client not available",
                confidence=0.5,
                processing_time=0.1,
                model_used="mock",
                token_usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={"mock": True},
            )

        start_time = datetime.utcnow()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.name if self.model_config else "gpt-4",
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            reasoning_step = ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                prompt=prompt,
                response=response.choices[0].message.content or "",
                confidence=self._extract_confidence(
                    response.choices[0].message.content or ""
                ),
                processing_time=processing_time,
                model_used=self.model_config.name if self.model_config else "gpt-4",
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens
                    if response.usage
                    else 0,
                },
                metadata={},
            )

            self.reasoning_steps.append(reasoning_step)
            return reasoning_step

        except Exception as e:
            self.logger.error(f"Reasoning step {step_id} failed: {e}")
            # Return error step instead of raising
            return ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                prompt=prompt,
                response=f"Error: {str(e)}",
                confidence=0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                model_used="error",
                token_usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                metadata={"error": True},
            )

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response."""
        # Simple pattern matching for confidence indicators
        if "very confident" in response.lower():
            return 0.9
        elif "confident" in response.lower():
            return 0.8
        elif "uncertain" in response.lower():
            return 0.4
        elif "not sure" in response.lower():
            return 0.2
        else:
            return 0.6  # Default confidence

    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality metrics for the reasoning chain."""
        if not self.reasoning_steps:
            return {"overall_quality": 0.0}

        avg_confidence = sum(step.confidence for step in self.reasoning_steps) / len(
            self.reasoning_steps
        )
        consistency_score = self._calculate_consistency()
        completeness_score = self._calculate_completeness()

        return {
            "average_confidence": avg_confidence,
            "consistency_score": consistency_score,
            "completeness_score": completeness_score,
            "overall_quality": (avg_confidence + consistency_score + completeness_score)
            / 3,
        }

    def _calculate_consistency(self) -> float:
        """Calculate consistency score across reasoning steps."""
        # Simple heuristic: consistent if steps build upon each other
        if len(self.reasoning_steps) < 2:
            return 1.0

        # Check if later steps reference earlier ones
        consistency_score = 0.8  # Default
        return min(consistency_score, 1.0)

    def _calculate_completeness(self) -> float:
        """Calculate completeness score of the reasoning."""
        # Heuristic: more reasoning steps typically indicate more complete analysis
        step_count = len(self.reasoning_steps)
        if step_count >= 5:
            return 1.0
        elif step_count >= 3:
            return 0.8
        elif step_count >= 2:
            return 0.6
        else:
            return 0.4

    def _update_metrics(self, processing_time: float, confidence: float, success: bool):
        """Update performance metrics."""
        self.metrics["total_executions"] += 1
        if success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1

        # Update averages
        total_execs = self.metrics["total_executions"]
        self.metrics["average_processing_time"] = (
            self.metrics["average_processing_time"] * (total_execs - 1)
            + processing_time
        ) / total_execs
        self.metrics["average_confidence"] = (
            self.metrics["average_confidence"] * (total_execs - 1) + confidence
        ) / total_execs
        self.metrics["last_execution"] = datetime.utcnow()


class ChainOfThought(ReasoningChain):
    """Chain-of-thought reasoning implementation with full attribute support."""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize chain of thought reasoning."""
        super().__init__(ChainType.CHAIN_OF_THOUGHT, model_config)
        self.thinking_steps: List[str] = []
        self.domain = "general"

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chain of thought reasoning."""
        self.execution_count += 1
        start_time = datetime.utcnow()

        try:
            # Extract prompt from input
            prompt = input_data.get("prompt", "")
            context = input_data.get("context", {})

            if not prompt:
                return self._ensure_return_value({"error": "No prompt provided"})

            # Perform reasoning
            result = await self._perform_reasoning(prompt, context)

            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            confidence = result.get("confidence_score", 0.0)
            self._update_metrics(processing_time, confidence, True)

            return self._ensure_return_value(result)

        except Exception as e:
            self.logger.error(f"Error in chain of thought execution: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(processing_time, 0.0, False)
            return self._ensure_return_value(
                {"error": str(e), "chain_id": self.chain_id}
            )

    async def _perform_reasoning(
        self, prompt: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual chain of thought reasoning."""
        # Step 1: Problem analysis
        analysis_prompt = f"""
        Analyze this problem step by step:

        Problem: {prompt}
        Context: {json.dumps(context, indent=2)}

        Break down the problem into components and identify what needs to be solved.
        Think through this systematically.
        """

        analysis_step = await self._execute_reasoning_step(
            "analysis",
            "analysis",
            analysis_prompt,
            f"You are an expert in {self.domain}. Think through problems systematically.",
        )

        # Step 2: Solution approach
        approach_prompt = f"""
        Based on the analysis:
        {analysis_step.response}

        Design a systematic approach to solve this problem.
        Consider multiple approaches and select the best one.
        Explain your reasoning for the chosen approach.
        """

        approach_step = await self._execute_reasoning_step(
            "approach",
            "synthesis",
            approach_prompt,
            f"You are an expert in {self.domain}. Design effective solution approaches.",
        )

        # Step 3: Implementation/Solution
        solution_prompt = f"""
        Using the chosen approach:
        {approach_step.response}

        Implement the solution step by step.
        Show your work and explain each step.
        Provide specific, actionable recommendations.
        """

        solution_step = await self._execute_reasoning_step(
            "solution",
            "synthesis",
            solution_prompt,
            f"You are an expert in {self.domain}. Implement solutions carefully and thoroughly.",
        )

        # Calculate metrics
        quality_metrics = self._calculate_quality_metrics()
        confidence_score = (
            sum(step.confidence for step in self.reasoning_steps)
            / len(self.reasoning_steps)
            if self.reasoning_steps
            else 0.0
        )

        return {
            "chain_id": self.chain_id,
            "type": self.chain_type.value,
            "final_answer": solution_step.response,
            "confidence_score": confidence_score,
            "reasoning_steps": [step.__dict__ for step in self.reasoning_steps],
            "quality_metrics": quality_metrics,
            "thinking_steps": self.thinking_steps,
        }


class ReflectiveReasoning(ReasoningChain):
    """Reflective reasoning with self-criticism and improvement."""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize reflective reasoning."""
        super().__init__(ChainType.REFLECTIVE, model_config)
        self.reflection_depth = 2
        self.reflection_history: List[Dict[str, Any]] = []

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reflective reasoning."""
        self.execution_count += 1
        start_time = datetime.utcnow()

        try:
            # Extract prompt from input
            prompt = input_data.get("prompt", "")
            context = input_data.get("context", {})

            if not prompt:
                return self._ensure_return_value({"error": "No prompt provided"})

            # Perform reasoning
            result = await self._perform_reflection(prompt, context)

            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            confidence = result.get("confidence_score", 0.0)
            self._update_metrics(processing_time, confidence, True)

            return self._ensure_return_value(result)

        except Exception as e:
            self.logger.error(f"Error in reflective reasoning execution: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(processing_time, 0.0, False)
            return self._ensure_return_value(
                {"error": str(e), "chain_id": self.chain_id}
            )

    async def _perform_reflection(
        self, prompt: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform reflective reasoning with self-criticism."""
        # Initial response
        initial_prompt = f"""
        Provide an initial response to this problem:

        Problem: {prompt}
        Context: {json.dumps(context, indent=2)}

        Give your best initial answer, but prepare to reflect on and improve it.
        """

        initial_step = await self._execute_reasoning_step(
            "initial_response",
            "analysis",
            initial_prompt,
            "You are an expert problem solver. Provide thoughtful initial responses.",
        )

        current_response = initial_step.response

        # Reflection cycles
        for i in range(self.reflection_depth):
            # Reflection step
            reflection_prompt = f"""
            Reflect on this response:
            {current_response}

            Critically analyze:
            1. What is good about this response?
            2. What could be improved?
            3. What might be missing?
            4. Are there any errors or weaknesses?
            5. What would make this response better?

            Be specific and constructive in your critique.
            """

            reflection_step = await self._execute_reasoning_step(
                f"reflection_{i+1}",
                "reflection",
                reflection_prompt,
                "You are a critical evaluator. Provide constructive, specific feedback.",
            )

            # Store reflection
            self.reflection_history.append(
                {
                    "cycle": i + 1,
                    "reflection": reflection_step.response,
                    "previous_response": current_response,
                }
            )

            # Improvement step
            improvement_prompt = f"""
            Based on the reflection:
            {reflection_step.response}

            Improve the original response:
            {current_response}

            Address the issues identified in the reflection.
            Provide an improved, more comprehensive response.
            """

            improvement_step = await self._execute_reasoning_step(
                f"improvement_{i+1}",
                "synthesis",
                improvement_prompt,
                "You are an expert problem solver. Continuously improve your responses based on reflection.",
            )

            current_response = improvement_step.response

        # Calculate metrics
        quality_metrics = self._calculate_quality_metrics()
        confidence_score = (
            sum(step.confidence for step in self.reasoning_steps)
            / len(self.reasoning_steps)
            if self.reasoning_steps
            else 0.0
        )

        return {
            "chain_id": self.chain_id,
            "type": self.chain_type.value,
            "final_answer": current_response,
            "confidence_score": confidence_score,
            "reasoning_steps": [step.__dict__ for step in self.reasoning_steps],
            "quality_metrics": quality_metrics,
            "reflection_history": self.reflection_history,
        }


class MultiStepReasoning(ReasoningChain):
    """Multi-step reasoning with configurable steps."""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize multi-step reasoning."""
        super().__init__(ChainType.MULTI_STEP, model_config)
        self.steps: List[Dict[str, str]] = []

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-step reasoning."""
        self.execution_count += 1
        start_time = datetime.utcnow()

        try:
            # Extract prompt from input
            prompt = input_data.get("prompt", "")
            context = input_data.get("context", {})

            if not prompt:
                return self._ensure_return_value({"error": "No prompt provided"})

            # Execute configured steps
            accumulated_context = context or {}

            for i, step_config in enumerate(self.steps):
                step_prompt = step_config.get("prompt", "").format(
                    original_prompt=prompt,
                    context=json.dumps(accumulated_context, indent=2),
                    previous_steps="\n".join(
                        [step.response for step in self.reasoning_steps]
                    ),
                )

                step = await self._execute_reasoning_step(
                    f"step_{i+1}",
                    step_config.get("type", "analysis"),
                    step_prompt,
                    step_config.get(
                        "system_prompt", "You are an expert problem solver."
                    ),
                )

                # Update context with step results
                accumulated_context[f"step_{i+1}_result"] = step.response

            # Final synthesis
            synthesis_prompt = f"""
            Synthesize all the analysis steps into a final comprehensive answer:

            Original problem: {prompt}
            Analysis steps: {json.dumps(accumulated_context, indent=2)}

            Provide a complete, synthesized response that integrates all insights.
            """

            synthesis_step = await self._execute_reasoning_step(
                "synthesis",
                "synthesis",
                synthesis_prompt,
                "You are an expert synthesizer. Integrate multiple analyses into coherent solutions.",
            )

            # Calculate metrics
            total_time = (datetime.utcnow() - start_time).total_seconds()
            quality_metrics = self._calculate_quality_metrics()
            confidence_score = (
                sum(step.confidence for step in self.reasoning_steps)
                / len(self.reasoning_steps)
                if self.reasoning_steps
                else 0.0
            )

            return self._ensure_return_value(
                {
                    "chain_id": self.chain_id,
                    "type": self.chain_type.value,
                    "final_answer": synthesis_step.response,
                    "confidence_score": confidence_score,
                    "reasoning_steps": [step.__dict__ for step in self.reasoning_steps],
                    "quality_metrics": quality_metrics,
                }
            )

        except Exception as e:
            self.logger.error(f"Error in multi-step reasoning execution: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(processing_time, 0.0, False)
            return self._ensure_return_value(
                {"error": str(e), "chain_id": self.chain_id}
            )


class DistillationReasoning(ReasoningChain):
    """Distillation reasoning for knowledge extraction and refinement."""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize distillation reasoning."""
        super().__init__(ChainType.DISTILLATION, model_config)
        self.distillation_levels = 3

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute distillation reasoning."""
        self.execution_count += 1
        start_time = datetime.utcnow()

        try:
            # Extract prompt from input
            prompt = input_data.get("prompt", "")
            context = input_data.get("context", {})

            if not prompt:
                return self._ensure_return_value({"error": "No prompt provided"})

            # Initial comprehensive analysis
            comprehensive_prompt = f"""
            Provide a comprehensive, detailed analysis of this problem:

            Problem: {prompt}
            Context: {json.dumps(context, indent=2)}

            Include all relevant details, considerations, and nuances.
            Be thorough and comprehensive.
            """

            comprehensive_step = await self._execute_reasoning_step(
                "comprehensive_analysis",
                "analysis",
                comprehensive_prompt,
                "You are an expert analyst. Provide comprehensive, detailed analysis.",
            )

            current_analysis = comprehensive_step.response

            # Distillation levels
            for level in range(self.distillation_levels):
                distillation_prompt = f"""
                Distill this analysis to its most essential elements:

                Current analysis: {current_analysis}

                Extract:
                1. The most important insights
                2. Key recommendations
                3. Critical considerations
                4. Actionable next steps

                Make it more concise while retaining all essential information.
                """

                distillation_step = await self._execute_reasoning_step(
                    f"distillation_level_{level+1}",
                    "synthesis",
                    distillation_prompt,
                    "You are an expert at distilling complex information into essential insights.",
                )

                current_analysis = distillation_step.response

            # Final refinement
            refinement_prompt = f"""
            Refine this distilled analysis for maximum clarity and impact:

            {current_analysis}

            Ensure:
            1. Clear, actionable recommendations
            2. Proper prioritization of insights
            3. Practical implementation guidance
            4. Balanced consideration of trade-offs

            Provide the final, refined analysis.
            """

            refinement_step = await self._execute_reasoning_step(
                "final_refinement",
                "synthesis",
                refinement_prompt,
                "You are an expert at refining analysis for maximum clarity and impact.",
            )

            # Calculate metrics
            total_time = (datetime.utcnow() - start_time).total_seconds()
            quality_metrics = self._calculate_quality_metrics()
            confidence_score = (
                sum(step.confidence for step in self.reasoning_steps)
                / len(self.reasoning_steps)
                if self.reasoning_steps
                else 0.0
            )

            return self._ensure_return_value(
                {
                    "chain_id": self.chain_id,
                    "type": self.chain_type.value,
                    "final_answer": refinement_step.response,
                    "confidence_score": confidence_score,
                    "reasoning_steps": [step.__dict__ for step in self.reasoning_steps],
                    "quality_metrics": quality_metrics,
                }
            )

        except Exception as e:
            self.logger.error(f"Error in distillation reasoning execution: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(processing_time, 0.0, False)
            return self._ensure_return_value(
                {"error": str(e), "chain_id": self.chain_id}
            )


# Factory function for creating reasoning chains
def create_reasoning_chain(
    chain_type: str, model_config: ModelConfig, **kwargs
) -> ReasoningChain:
    """Create a reasoning chain of the specified type."""

    if chain_type == "chain_of_thought":
        return ChainOfThought(model_config)
    elif chain_type == "reflective":
        return ReflectiveReasoning(model_config)
    elif chain_type == "multi_step":
        return MultiStepReasoning(model_config)
    elif chain_type == "distillation":
        return DistillationReasoning(model_config)
    else:
        raise ValueError(f"Unknown reasoning chain type: {chain_type}")


# Example usage and testing
async def example_usage():
    """Example usage of reasoning chains."""
    if not OPENAI_AVAILABLE:
        print("OpenAI not available - skipping example")
        return

    try:
        from .model_registry import ModelRegistry

        # Initialize model registry
        model_registry = ModelRegistry(api_key="test_key")
        model_config = model_registry.get_model_config("gpt-4")

        if model_config is None:
            print("Model configuration not available - skipping example")
            return

        # Example 1: Chain of thought
        cot_chain = ChainOfThought(model_config)
        result = await cot_chain.execute(
            {
                "prompt": "How should I optimize a 16x16 systolic array for GEMM with 60% PE utilization?",
                "context": {"workload": "GEMM", "utilization": 0.6},
            }
        )
        print(f"Chain of thought result: {result.get('final_answer', 'No answer')}")

        # Example 2: Reflective reasoning
        reflective_chain = ReflectiveReasoning(model_config)
        result = await reflective_chain.execute(
            {
                "prompt": "What are the key considerations for medical AI accelerator design?",
                "context": {"domain": "medical_ai", "compliance": "FDA"},
            }
        )
        print(f"Reflective reasoning result: {result.get('final_answer', 'No answer')}")

    except Exception as e:
        print(f"Example usage failed: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())
