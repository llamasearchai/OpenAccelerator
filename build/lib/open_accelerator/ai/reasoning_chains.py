"""
Reasoning Chains with DSPy Integration

Implements structured reasoning patterns using DSPy for compound AI workflows.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import dspy
from dspy import ChainOfThought, InputField, OutputField, Predict, Signature

from .model_registry import ModelRegistry, TaskType


class ReasoningSignature(Signature):
    """Base signature for reasoning tasks."""

    context = InputField(desc="Context information for reasoning")
    question = InputField(desc="Question or problem to solve")
    reasoning = OutputField(desc="Step-by-step reasoning process")
    answer = OutputField(desc="Final answer or solution")


class AnalysisSignature(Signature):
    """Signature for analysis tasks."""

    data = InputField(desc="Data to analyze")
    analysis_type = InputField(desc="Type of analysis to perform")
    insights = OutputField(desc="Key insights and findings")
    recommendations = OutputField(desc="Actionable recommendations")


class OptimizationSignature(Signature):
    """Signature for optimization tasks."""

    problem_description = InputField(desc="Description of optimization problem")
    constraints = InputField(desc="Constraints and limitations")
    objectives = InputField(desc="Optimization objectives")
    solution = OutputField(desc="Proposed optimization solution")
    trade_offs = OutputField(desc="Analysis of trade-offs")


class CodeGenerationSignature(Signature):
    """Signature for code generation tasks."""

    requirements = InputField(desc="Code requirements and specifications")
    context = InputField(desc="Additional context and constraints")
    language = InputField(desc="Programming language")
    code = OutputField(desc="Generated code")
    explanation = OutputField(desc="Code explanation and documentation")
    tests = OutputField(desc="Test cases for the code")


class ReasoningChain(ABC):
    """Abstract base class for reasoning chains."""

    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the reasoning chain."""
        pass

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the reasoning history."""
        return self.history

    def clear_history(self):
        """Clear the reasoning history."""
        self.history.clear()


class ChainOfThoughtReasoning(ReasoningChain):
    """Chain-of-thought reasoning implementation."""

    def __init__(self, model_registry: ModelRegistry, model_name: Optional[str] = None):
        super().__init__(model_registry)

        # Select optimal model for reasoning
        if model_name:
            self.model_config = model_registry.get_model(model_name)
            if not self.model_config:
                raise ValueError(f"Model {model_name} not found")
        else:
            self.model_config = model_registry.get_optimal_model(
                TaskType.REASONING, quality_preference="high"
            )

        # Initialize DSPy components
        self.lm = model_registry.get_dspy_model(self.model_config.name)
        dspy.configure(lm=self.lm)

        # Define reasoning modules
        self.cot_reasoner = ChainOfThought(ReasoningSignature)
        self.analyzer = Predict(AnalysisSignature)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through chain-of-thought reasoning."""

        context = input_data.get("context", "")
        question = input_data.get("question", "")

        # Perform chain-of-thought reasoning
        reasoning_result = self.cot_reasoner(context=context, question=question)

        # Analyze the reasoning process
        analysis_result = self.analyzer(
            data=reasoning_result.reasoning, analysis_type="reasoning_quality"
        )

        result = {
            "reasoning": reasoning_result.reasoning,
            "answer": reasoning_result.answer,
            "insights": analysis_result.insights,
            "recommendations": analysis_result.recommendations,
            "model_used": self.model_config.name,
            "confidence": self._estimate_confidence(reasoning_result),
        }

        # Store in history
        self.history.append(
            {
                "input": input_data,
                "output": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def _estimate_confidence(self, reasoning_result) -> float:
        """Estimate confidence in the reasoning result."""
        # Simple heuristic - could be improved with more sophisticated methods
        reasoning_length = len(reasoning_result.reasoning.split())
        answer_length = len(reasoning_result.answer.split())

        # Longer reasoning with concise answers typically indicate higher confidence
        if reasoning_length > 50 and answer_length < 100:
            return 0.85
        elif reasoning_length > 20:
            return 0.75
        else:
            return 0.65


class ReflectiveReasoning(ReasoningChain):
    """Reflective reasoning with self-correction."""

    def __init__(self, model_registry: ModelRegistry, model_name: Optional[str] = None):
        super().__init__(model_registry)

        # Use high-quality reasoning model
        if model_name:
            self.model_config = model_registry.get_model(model_name)
        else:
            self.model_config = model_registry.get_optimal_model(
                TaskType.REASONING, quality_preference="premium"
            )

        self.lm = model_registry.get_dspy_model(self.model_config.name)
        dspy.configure(lm=self.lm)

        # Define reasoning and reflection modules
        self.initial_reasoner = ChainOfThought(ReasoningSignature)
        self.reflector = Predict(self._create_reflection_signature())
        self.final_reasoner = ChainOfThought(ReasoningSignature)

    def _create_reflection_signature(self):
        """Create signature for reflection step."""

        class ReflectionSignature(Signature):
            initial_reasoning = InputField(desc="Initial reasoning process")
            initial_answer = InputField(desc="Initial answer")
            question = InputField(desc="Original question")
            critique = OutputField(desc="Critique of initial reasoning")
            improvements = OutputField(desc="Suggested improvements")
            confidence = OutputField(desc="Confidence assessment")

        return ReflectionSignature

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through reflective reasoning."""

        context = input_data.get("context", "")
        question = input_data.get("question", "")

        # Initial reasoning
        initial_result = self.initial_reasoner(context=context, question=question)

        # Reflection step
        reflection = self.reflector(
            initial_reasoning=initial_result.reasoning,
            initial_answer=initial_result.answer,
            question=question,
        )

        # Enhanced context with reflection
        enhanced_context = f"{context}\n\nInitial reasoning: {initial_result.reasoning}\n\nReflection: {reflection.critique}\n\nImprovements to consider: {reflection.improvements}"

        # Final reasoning with reflection
        final_result = self.final_reasoner(context=enhanced_context, question=question)

        result = {
            "initial_reasoning": initial_result.reasoning,
            "initial_answer": initial_result.answer,
            "reflection_critique": reflection.critique,
            "improvements": reflection.improvements,
            "final_reasoning": final_result.reasoning,
            "final_answer": final_result.answer,
            "model_used": self.model_config.name,
            "confidence": float(reflection.confidence)
            if reflection.confidence.replace(".", "").isdigit()
            else 0.8,
        }

        # Store in history
        self.history.append(
            {
                "input": input_data,
                "output": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result


class MultiModelReasoning(ReasoningChain):
    """Multi-model reasoning combining different model strengths."""

    def __init__(self, model_registry: ModelRegistry):
        super().__init__(model_registry)

        # Get different models for different reasoning aspects
        self.creative_model = model_registry.get_optimal_model(
            TaskType.GENERAL_PURPOSE, quality_preference="high"
        )
        self.analytical_model = model_registry.get_optimal_model(
            TaskType.REASONING, quality_preference="premium"
        )
        self.synthesis_model = model_registry.get_optimal_model(
            TaskType.ANALYSIS, quality_preference="high"
        )

        # Initialize DSPy components for each model
        self.creative_lm = model_registry.get_dspy_model(self.creative_model.name)
        self.analytical_lm = model_registry.get_dspy_model(self.analytical_model.name)
        self.synthesis_lm = model_registry.get_dspy_model(self.synthesis_model.name)

        # Define reasoning modules
        dspy.configure(lm=self.creative_lm)
        self.creative_reasoner = Predict(ReasoningSignature)

        dspy.configure(lm=self.analytical_lm)
        self.analytical_reasoner = ChainOfThought(ReasoningSignature)

        dspy.configure(lm=self.synthesis_lm)
        self.synthesizer = Predict(self._create_synthesis_signature())

    def _create_synthesis_signature(self):
        """Create signature for synthesis step."""

        class SynthesisSignature(Signature):
            creative_reasoning = InputField(desc="Creative reasoning perspective")
            analytical_reasoning = InputField(desc="Analytical reasoning perspective")
            question = InputField(desc="Original question")
            synthesized_reasoning = OutputField(
                desc="Synthesized reasoning combining both perspectives"
            )
            final_answer = OutputField(desc="Final synthesized answer")
            confidence = OutputField(desc="Confidence in synthesized result")

        return SynthesisSignature

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through multi-model reasoning."""

        context = input_data.get("context", "")
        question = input_data.get("question", "")

        # Creative reasoning
        dspy.configure(lm=self.creative_lm)
        creative_result = self.creative_reasoner(context=context, question=question)

        # Analytical reasoning
        dspy.configure(lm=self.analytical_lm)
        analytical_result = self.analytical_reasoner(context=context, question=question)

        # Synthesis
        dspy.configure(lm=self.synthesis_lm)
        synthesis_result = self.synthesizer(
            creative_reasoning=f"{creative_result.reasoning}\nAnswer: {creative_result.answer}",
            analytical_reasoning=f"{analytical_result.reasoning}\nAnswer: {analytical_result.answer}",
            question=question,
        )

        result = {
            "creative_perspective": {
                "reasoning": creative_result.reasoning,
                "answer": creative_result.answer,
                "model": self.creative_model.name,
            },
            "analytical_perspective": {
                "reasoning": analytical_result.reasoning,
                "answer": analytical_result.answer,
                "model": self.analytical_model.name,
            },
            "synthesized_result": {
                "reasoning": synthesis_result.synthesized_reasoning,
                "answer": synthesis_result.final_answer,
                "model": self.synthesis_model.name,
                "confidence": float(synthesis_result.confidence)
                if synthesis_result.confidence.replace(".", "").isdigit()
                else 0.8,
            },
            "models_used": [
                self.creative_model.name,
                self.analytical_model.name,
                self.synthesis_model.name,
            ],
        }

        # Store in history
        self.history.append(
            {
                "input": input_data,
                "output": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result


class OptimizationReasoning(ReasoningChain):
    """Specialized reasoning for optimization problems."""

    def __init__(self, model_registry: ModelRegistry, model_name: Optional[str] = None):
        super().__init__(model_registry)

        # Use optimization-capable model
        if model_name:
            self.model_config = model_registry.get_model(model_name)
        else:
            self.model_config = model_registry.get_optimal_model(
                TaskType.OPTIMIZATION, quality_preference="high"
            )

        self.lm = model_registry.get_dspy_model(self.model_config.name)
        dspy.configure(lm=self.lm)

        # Define optimization modules
        self.optimizer = ChainOfThought(OptimizationSignature)
        self.analyzer = Predict(AnalysisSignature)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process optimization problem through specialized reasoning."""

        problem_description = input_data.get("problem_description", "")
        constraints = input_data.get("constraints", "")
        objectives = input_data.get("objectives", "")

        # Optimization reasoning
        optimization_result = self.optimizer(
            problem_description=problem_description,
            constraints=constraints,
            objectives=objectives,
        )

        # Analyze the solution
        analysis_result = self.analyzer(
            data=optimization_result.solution, analysis_type="optimization_analysis"
        )

        result = {
            "solution": optimization_result.solution,
            "trade_offs": optimization_result.trade_offs,
            "analysis": {
                "insights": analysis_result.insights,
                "recommendations": analysis_result.recommendations,
            },
            "model_used": self.model_config.name,
            "optimization_quality": self._assess_optimization_quality(
                optimization_result
            ),
        }

        # Store in history
        self.history.append(
            {
                "input": input_data,
                "output": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def _assess_optimization_quality(self, optimization_result) -> Dict[str, Any]:
        """Assess the quality of optimization result."""
        # Simple heuristic assessment
        solution_completeness = len(optimization_result.solution.split()) > 100
        trade_off_analysis = "trade-off" in optimization_result.trade_offs.lower()

        return {
            "completeness": "high" if solution_completeness else "medium",
            "trade_off_analysis": "present" if trade_off_analysis else "limited",
            "overall_quality": "high"
            if solution_completeness and trade_off_analysis
            else "medium",
        }


class CodeGenerationReasoning(ReasoningChain):
    """Specialized reasoning for code generation tasks."""

    def __init__(self, model_registry: ModelRegistry, model_name: Optional[str] = None):
        super().__init__(model_registry)

        # Use code generation optimized model
        if model_name:
            self.model_config = model_registry.get_model(model_name)
        else:
            self.model_config = model_registry.get_optimal_model(
                TaskType.CODE_GENERATION, quality_preference="high"
            )

        self.lm = model_registry.get_dspy_model(self.model_config.name)
        dspy.configure(lm=self.lm)

        # Define code generation modules
        self.code_generator = ChainOfThought(CodeGenerationSignature)
        self.code_reviewer = Predict(self._create_review_signature())

    def _create_review_signature(self):
        """Create signature for code review."""

        class CodeReviewSignature(Signature):
            code = InputField(desc="Code to review")
            requirements = InputField(desc="Original requirements")
            review = OutputField(desc="Code review feedback")
            improvements = OutputField(desc="Suggested improvements")
            quality_score = OutputField(desc="Quality score (1-10)")

        return CodeReviewSignature

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process code generation request through specialized reasoning."""

        requirements = input_data.get("requirements", "")
        context = input_data.get("context", "")
        language = input_data.get("language", "python")

        # Code generation
        generation_result = self.code_generator(
            requirements=requirements, context=context, language=language
        )

        # Code review
        review_result = self.code_reviewer(
            code=generation_result.code, requirements=requirements
        )

        result = {
            "code": generation_result.code,
            "explanation": generation_result.explanation,
            "tests": generation_result.tests,
            "review": {
                "feedback": review_result.review,
                "improvements": review_result.improvements,
                "quality_score": review_result.quality_score,
            },
            "language": language,
            "model_used": self.model_config.name,
        }

        # Store in history
        self.history.append(
            {
                "input": input_data,
                "output": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result
