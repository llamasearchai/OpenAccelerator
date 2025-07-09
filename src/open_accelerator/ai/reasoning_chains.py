"""
Advanced Reasoning Chains for Accelerator Optimization.

Implements sophisticated reasoning patterns including chain-of-thought and reflective reasoning
for complex accelerator optimization and analysis tasks.
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .model_registry import ModelConfig

logger = logging.getLogger(__name__)


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


class ReasoningChain(ABC):
    """Abstract base class for reasoning chains."""

    def __init__(self, model_config: ModelConfig, chain_id: str = None):
        """Initialize reasoning chain."""
        self.model_config = model_config
        self.chain_id = chain_id or f"chain_{datetime.now().isoformat()}"
        self.client = None

        if OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.reasoning_steps: List[ReasoningStep] = []
        self.context_memory: Dict[str, Any] = {}

    @abstractmethod
    async def process(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """Process a prompt through the reasoning chain."""
        pass

    async def _execute_reasoning_step(
        self, step_id: str, step_type: str, prompt: str, system_prompt: str = None
    ) -> ReasoningStep:
        """Execute a single reasoning step."""
        if not self.client:
            raise RuntimeError("OpenAI client not available")

        start_time = datetime.now()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.name,
                messages=messages,
                temperature=0.1,
                max_tokens=2000,
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            reasoning_step = ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                prompt=prompt,
                response=response.choices[0].message.content,
                confidence=0.8,  # Default confidence
                processing_time=processing_time,
                model_used=self.model_config.name,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={},
            )

            self.reasoning_steps.append(reasoning_step)
            return reasoning_step

        except Exception as e:
            logger.error(f"Reasoning step {step_id} failed: {e}")
            raise

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


class ChainOfThought(ReasoningChain):
    """Chain-of-thought reasoning for complex problem solving."""

    def __init__(self, model_config: ModelConfig, domain: str = "general"):
        """Initialize chain-of-thought reasoning."""
        super().__init__(model_config, f"cot_{domain}")
        self.domain = domain
        self.thinking_steps = []

    async def process(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """Process with chain-of-thought reasoning."""
        start_time = datetime.now()

        # Step 1: Problem analysis
        analysis_prompt = f"""
        Analyze this problem step by step:

        Problem: {prompt}
        Context: {json.dumps(context or {}, indent=2)}

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

        # Step 4: Verification
        verification_prompt = f"""
        Review the solution:
        {solution_step.response}

        Verify the solution by:
        1. Checking if it addresses the original problem
        2. Identifying potential issues or edge cases
        3. Evaluating the quality of the solution
        4. Suggesting improvements if needed
        """

        verification_step = await self._execute_reasoning_step(
            "verification",
            "validation",
            verification_prompt,
            f"You are an expert in {self.domain}. Critically evaluate solutions for correctness and completeness.",
        )

        # Extract final answer
        final_answer = solution_step.response

        # Calculate metrics
        total_time = (datetime.now() - start_time).total_seconds()
        quality_metrics = self._calculate_quality_metrics()
        confidence_score = sum(step.confidence for step in self.reasoning_steps) / len(
            self.reasoning_steps
        )

        return ReasoningResult(
            chain_id=self.chain_id,
            chain_type="chain_of_thought",
            final_answer=final_answer,
            confidence_score=confidence_score,
            reasoning_steps=self.reasoning_steps,
            total_processing_time=total_time,
            quality_metrics=quality_metrics,
        )


class ReflectiveReasoning(ReasoningChain):
    """Reflective reasoning with self-criticism and improvement."""

    def __init__(self, model_config: ModelConfig, reflection_depth: int = 2):
        """Initialize reflective reasoning."""
        super().__init__(model_config, "reflective")
        self.reflection_depth = reflection_depth

    async def process(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """Process with reflective reasoning."""
        start_time = datetime.now()

        # Initial response
        initial_prompt = f"""
        Provide an initial response to this problem:

        Problem: {prompt}
        Context: {json.dumps(context or {}, 'indent=2')}

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

        # Final validation
        validation_prompt = f"""
        Validate this final response:
        {current_response}

        Evaluate:
        1. Is this response accurate and complete?
        2. Does it address the original problem fully?
        3. What is the confidence level?
        4. Are there any remaining issues?

        Provide a final assessment.
        """

        validation_step = await self._execute_reasoning_step(
            "final_validation",
            "validation",
            validation_prompt,
            "You are a quality assessor. Provide final validation of solutions.",
        )

        # Extract improvement suggestions
        improvement_suggestions = []
        for step in self.reasoning_steps:
            if step.step_type == "reflection":
                # Extract suggestions from reflection
                if (
                    "improve" in step.response.lower()
                    or "better" in step.response.lower()
                ):
                    improvement_suggestions.append(step.response)

        # Calculate metrics
        total_time = (datetime.now() - start_time).total_seconds()
        quality_metrics = self._calculate_quality_metrics()
        confidence_score = self._extract_confidence(validation_step.response)

        return ReasoningResult(
            chain_id=self.chain_id,
            chain_type="reflective_reasoning",
            final_answer=current_response,
            confidence_score=confidence_score,
            reasoning_steps=self.reasoning_steps,
            total_processing_time=total_time,
            quality_metrics=quality_metrics,
            improvement_suggestions=improvement_suggestions,
        )


class MultiStepReasoning(ReasoningChain):
    """Multi-step reasoning for complex analysis tasks."""

    def __init__(self, model_config: ModelConfig, steps: List[Dict[str, str]]):
        """Initialize multi-step reasoning with custom steps."""
        super().__init__(model_config, "multi_step")
        self.custom_steps = steps

    async def process(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """Process with custom multi-step reasoning."""
        start_time = datetime.now()

        accumulated_context = context or {}

        for i, step_config in enumerate(self.custom_steps):
            step_prompt = step_config["prompt"].format(
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
                step_config.get("system_prompt", "You are an expert problem solver."),
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
        total_time = (datetime.now() - start_time).total_seconds()
        quality_metrics = self._calculate_quality_metrics()
        confidence_score = sum(step.confidence for step in self.reasoning_steps) / len(
            self.reasoning_steps
        )

        return ReasoningResult(
            chain_id=self.chain_id,
            chain_type="multi_step_reasoning",
            final_answer=synthesis_step.response,
            confidence_score=confidence_score,
            reasoning_steps=self.reasoning_steps,
            total_processing_time=total_time,
            quality_metrics=quality_metrics,
        )


class DistillationReasoning(ReasoningChain):
    """Distillation reasoning for knowledge extraction and refinement."""

    def __init__(self, model_config: ModelConfig, distillation_levels: int = 3):
        """Initialize distillation reasoning."""
        super().__init__(model_config, "distillation")
        self.distillation_levels = distillation_levels

    async def process(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> ReasoningResult:
        """Process with distillation reasoning."""
        start_time = datetime.now()

        # Initial comprehensive analysis
        comprehensive_prompt = f"""
        Provide a comprehensive, detailed analysis of this problem:

        Problem: {prompt}
        Context: {json.dumps(context or {}, indent=2)}

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
        total_time = (datetime.now() - start_time).total_seconds()
        quality_metrics = self._calculate_quality_metrics()
        confidence_score = sum(step.confidence for step in self.reasoning_steps) / len(
            self.reasoning_steps
        )

        return ReasoningResult(
            chain_id=self.chain_id,
            chain_type="distillation_reasoning",
            final_answer=refinement_step.response,
            confidence_score=confidence_score,
            reasoning_steps=self.reasoning_steps,
            total_processing_time=total_time,
            quality_metrics=quality_metrics,
        )


# Factory function for creating reasoning chains
def create_reasoning_chain(
    chain_type: str, model_config: ModelConfig, **kwargs
) -> ReasoningChain:
    """Create a reasoning chain of the specified type."""

    if chain_type == "chain_of_thought":
        return ChainOfThought(model_config, kwargs.get("domain", "general"))
    elif chain_type == "reflective":
        return ReflectiveReasoning(model_config, kwargs.get("reflection_depth", 2))
    elif chain_type == "multi_step":
        return MultiStepReasoning(model_config, kwargs.get("steps", []))
    elif chain_type == "distillation":
        return DistillationReasoning(model_config, kwargs.get("distillation_levels", 3))
    else:
        raise ValueError(f"Unknown reasoning chain type: {chain_type}")


# Example usage and testing
async def example_usage():
    """Example usage of reasoning chains."""
    if not OPENAI_AVAILABLE:
        print("OpenAI not available - skipping example")
        return

    from .model_registry import ModelRegistry

    # Initialize model registry
    model_registry = ModelRegistry()
    model_config = model_registry.get_model_config("gpt-4")

    # Example 1: Chain of thought
    cot_chain = ChainOfThought(model_config, "accelerator_optimization")
    result = await cot_chain.process(
        "How should I optimize a 16x16 systolic array for GEMM with 60% PE utilization?",
        {"workload": "GEMM", "utilization": 0.6},
    )
    print(f"Chain of thought result: {result.final_answer}")

    # Example 2: Reflective reasoning
    reflective_chain = ReflectiveReasoning(model_config, reflection_depth=2)
    result = await reflective_chain.process(
        "What are the key considerations for medical AI accelerator design?",
        {"domain": "medical_ai", "compliance": "FDA"},
    )
    print(f"Reflective reasoning result: {result.final_answer}")


if __name__ == "__main__":
    asyncio.run(example_usage())
