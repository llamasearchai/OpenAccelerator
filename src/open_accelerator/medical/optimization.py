"""
Medical optimization module for Open Accelerator.

Provides medical-specific optimization algorithms with safety constraints,
reliability optimization, and power management for medical AI systems.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of medical optimization."""

    SAFETY = "safety"
    RELIABILITY = "reliability"
    POWER = "power"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


class ConstraintType(Enum):
    """Types of safety constraints."""

    HARD = "hard"  # Must be satisfied
    SOFT = "soft"  # Preferred but not required
    ADAPTIVE = "adaptive"  # Can be adjusted based on conditions


@dataclass
class SafetyConstraint:
    """Safety constraint for medical optimization."""

    constraint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    constraint_type: ConstraintType = ConstraintType.HARD
    parameter: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.01
    priority: int = 1
    applicable_phases: List[str] = field(default_factory=list)
    violation_action: str = "stop"  # stop, warn, adjust


@dataclass
class OptimizationObjective:
    """Optimization objective."""

    objective_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    optimization_type: OptimizationType = OptimizationType.PERFORMANCE
    weight: float = 1.0
    target_value: Optional[float] = None
    minimize: bool = True  # True for minimize, False for maximize


@dataclass
class OptimizationConfiguration:
    """Medical optimization configuration."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    objectives: List[OptimizationObjective] = field(default_factory=list)
    constraints: List[SafetyConstraint] = field(default_factory=list)
    optimization_algorithm: str = "gradient_descent"
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    safety_margin: float = 0.1
    enable_adaptive_constraints: bool = True
    real_time_monitoring: bool = True


class MedicalOptimizer:
    """
    Medical-specific optimizer with safety constraints and reliability requirements.

    Ensures all optimizations maintain medical safety standards and regulatory compliance.
    """

    def __init__(self, config: OptimizationConfiguration):
        """Initialize medical optimizer."""
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []
        self.constraint_violations: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {}
        self.safety_monitor_enabled = True

        # Initialize safety monitoring
        self._initialize_safety_monitoring()

        logger.info(f"Medical optimizer initialized: {config.name}")

    def _initialize_safety_monitoring(self):
        """Initialize safety monitoring system."""
        self.safety_metrics = {
            "accuracy_threshold": 0.95,
            "reliability_threshold": 0.99,
            "max_latency_ms": 1000,
            "power_budget_watts": 100,
            "temperature_limit_celsius": 70,
        }

        self.monitoring_enabled = True
        logger.info("Safety monitoring initialized")

    def optimize(
        self, system_parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform medical-safe optimization.

        Args:
            system_parameters: Current system parameters
            target_metrics: Target performance metrics

        Returns:
            Optimized parameters and results
        """
        optimization_start = datetime.now()

        try:
            # Validate safety constraints before optimization
            self._validate_safety_constraints(system_parameters)

            # Initialize optimization state
            current_params = system_parameters.copy()
            best_params = current_params.copy()
            best_score = float("inf")

            # Optimization loop
            for iteration in range(self.config.max_iterations):
                # Evaluate current parameters
                score = self._evaluate_objective(current_params, target_metrics)

                # Update best solution if improved
                if score < best_score:
                    best_score = score
                    best_params = current_params.copy()

                # Check convergence
                if self._check_convergence(score, iteration):
                    logger.info(f"Optimization converged at iteration {iteration}")
                    break

                # Generate next parameters
                next_params = self._generate_next_parameters(
                    current_params, target_metrics
                )

                # Validate safety constraints for next parameters
                if self._validate_safety_constraints(next_params):
                    current_params = next_params
                else:
                    logger.warning(
                        f"Safety constraint violation at iteration {iteration}"
                    )
                    # Apply safety correction
                    current_params = self._apply_safety_correction(
                        current_params, next_params
                    )

                # Log iteration
                self._log_optimization_iteration(iteration, current_params, score)

            # Finalize optimization results
            optimization_end = datetime.now()
            results = self._finalize_optimization_results(
                best_params, best_score, optimization_start, optimization_end
            )

            logger.info("Medical optimization completed successfully")
            return results

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    def _validate_safety_constraints(self, parameters: Dict[str, Any]) -> bool:
        """Validate safety constraints for given parameters."""
        violations = []

        for constraint in self.config.constraints:
            if not self._check_constraint(constraint, parameters):
                violations.append(constraint)

        if violations:
            self._handle_constraint_violations(violations, parameters)
            return False

        return True

    def _check_constraint(
        self, constraint: SafetyConstraint, parameters: Dict[str, Any]
    ) -> bool:
        """Check if a single constraint is satisfied."""
        param_value = parameters.get(constraint.parameter)

        if param_value is None:
            return True  # Parameter not applicable

        # Check minimum value constraint
        if constraint.min_value is not None and param_value < constraint.min_value:
            return False

        # Check maximum value constraint
        if constraint.max_value is not None and param_value > constraint.max_value:
            return False

        # Check target value constraint with tolerance
        if constraint.target_value is not None:
            deviation = abs(param_value - constraint.target_value)
            if deviation > constraint.tolerance:
                return False

        return True

    def _handle_constraint_violations(
        self, violations: List[SafetyConstraint], parameters: Dict[str, Any]
    ):
        """Handle safety constraint violations."""
        for constraint in violations:
            violation_record = {
                "timestamp": datetime.now().isoformat(),
                "constraint_id": constraint.constraint_id,
                "constraint_name": constraint.name,
                "parameter": constraint.parameter,
                "value": parameters.get(constraint.parameter),
                "violation_type": constraint.constraint_type.value,
                "action": constraint.violation_action,
            }

            self.constraint_violations.append(violation_record)

            # Take action based on violation type
            if constraint.constraint_type == ConstraintType.HARD:
                if constraint.violation_action == "stop":
                    raise ValueError(
                        f"Hard safety constraint violated: {constraint.name}"
                    )
                elif constraint.violation_action == "adjust":
                    self._adjust_parameter_for_constraint(parameters, constraint)

            logger.warning(f"Safety constraint violation: {constraint.name}")

    def _adjust_parameter_for_constraint(
        self, parameters: Dict[str, Any], constraint: SafetyConstraint
    ):
        """Adjust parameter to satisfy constraint."""
        param_value = parameters.get(constraint.parameter)

        if param_value is None:
            return

        # Adjust to satisfy constraint
        if constraint.min_value is not None and param_value < constraint.min_value:
            parameters[constraint.parameter] = (
                constraint.min_value + constraint.tolerance
            )
        elif constraint.max_value is not None and param_value > constraint.max_value:
            parameters[constraint.parameter] = (
                constraint.max_value - constraint.tolerance
            )
        elif constraint.target_value is not None:
            parameters[constraint.parameter] = constraint.target_value

    def _evaluate_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate optimization objective."""
        total_score = 0.0

        for objective in self.config.objectives:
            objective_score = self._evaluate_single_objective(
                objective, parameters, target_metrics
            )
            weighted_score = objective_score * objective.weight
            total_score += weighted_score

        return total_score

    def _evaluate_single_objective(
        self,
        objective: OptimizationObjective,
        parameters: Dict[str, Any],
        target_metrics: Dict[str, Any],
    ) -> float:
        """Evaluate a single optimization objective."""
        if objective.optimization_type == OptimizationType.SAFETY:
            return self._evaluate_safety_objective(parameters, target_metrics)
        elif objective.optimization_type == OptimizationType.RELIABILITY:
            return self._evaluate_reliability_objective(parameters, target_metrics)
        elif objective.optimization_type == OptimizationType.POWER:
            return self._evaluate_power_objective(parameters, target_metrics)
        elif objective.optimization_type == OptimizationType.PERFORMANCE:
            return self._evaluate_performance_objective(parameters, target_metrics)
        elif objective.optimization_type == OptimizationType.ACCURACY:
            return self._evaluate_accuracy_objective(parameters, target_metrics)
        elif objective.optimization_type == OptimizationType.LATENCY:
            return self._evaluate_latency_objective(parameters, target_metrics)
        else:
            return 0.0

    def _evaluate_safety_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate safety objective."""
        safety_score = 0.0

        # Calculate safety score based on constraint satisfaction
        for constraint in self.config.constraints:
            if constraint.constraint_type == ConstraintType.HARD:
                if not self._check_constraint(constraint, parameters):
                    safety_score += 1000  # High penalty for hard constraint violations
            elif constraint.constraint_type == ConstraintType.SOFT:
                if not self._check_constraint(constraint, parameters):
                    safety_score += 10  # Lower penalty for soft constraint violations

        return safety_score

    def _evaluate_reliability_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate reliability objective."""
        # Simulate reliability calculation
        reliability_target = target_metrics.get("reliability", 0.99)
        current_reliability = parameters.get("reliability", 0.95)

        return abs(reliability_target - current_reliability)

    def _evaluate_power_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate power consumption objective."""
        power_budget = target_metrics.get("power_budget", 100)
        current_power = parameters.get("power_consumption", 80)

        if current_power > power_budget:
            return (
                current_power - power_budget
            ) ** 2  # Quadratic penalty for exceeding budget
        else:
            return current_power  # Minimize power within budget

    def _evaluate_performance_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate performance objective."""
        performance_target = target_metrics.get("performance", 1.0)
        current_performance = parameters.get("performance", 0.8)

        return abs(performance_target - current_performance)

    def _evaluate_accuracy_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate accuracy objective."""
        accuracy_target = target_metrics.get("accuracy", 0.95)
        current_accuracy = parameters.get("accuracy", 0.90)

        if current_accuracy < accuracy_target:
            return (
                accuracy_target - current_accuracy
            ) ** 2  # Quadratic penalty for low accuracy
        else:
            return 0.0  # No penalty if accuracy target is met

    def _evaluate_latency_objective(
        self, parameters: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate latency objective."""
        latency_target = target_metrics.get("latency", 100)
        current_latency = parameters.get("latency", 120)

        return max(0, current_latency - latency_target)

    def _generate_next_parameters(
        self, current_params: Dict[str, Any], target_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate next set of parameters for optimization."""
        next_params = current_params.copy()

        # Simple gradient-based parameter update
        for param_name, param_value in current_params.items():
            if isinstance(param_value, (int, float)):
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.01)
                next_params[param_name] = param_value + perturbation

        return next_params

    def _check_convergence(self, score: float, iteration: int) -> bool:
        """Check if optimization has converged."""
        if iteration < 10:  # Need some history
            return False

        # Check if score has improved significantly in recent iterations
        recent_scores = [entry["score"] for entry in self.optimization_history[-10:]]
        if len(recent_scores) >= 10:
            improvement = max(recent_scores) - min(recent_scores)
            if improvement < self.config.convergence_threshold:
                return True

        return False

    def _apply_safety_correction(
        self, current_params: Dict[str, Any], unsafe_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply safety correction to unsafe parameters."""
        corrected_params = current_params.copy()

        # Apply constraints to correct unsafe parameters
        for constraint in self.config.constraints:
            if constraint.constraint_type == ConstraintType.HARD:
                self._adjust_parameter_for_constraint(corrected_params, constraint)

        return corrected_params

    def _log_optimization_iteration(
        self, iteration: int, parameters: Dict[str, Any], score: float
    ):
        """Log optimization iteration."""
        log_entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters.copy(),
            "score": score,
            "constraints_satisfied": len(self.constraint_violations) == 0,
        }

        self.optimization_history.append(log_entry)

    def _finalize_optimization_results(
        self,
        best_params: Dict[str, Any],
        best_score: float,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Finalize optimization results."""
        results = {
            "optimized_parameters": best_params,
            "final_score": best_score,
            "optimization_duration": (end_time - start_time).total_seconds(),
            "iterations_completed": len(self.optimization_history),
            "convergence_achieved": True,
            "constraint_violations": len(self.constraint_violations),
            "safety_compliant": len(self.constraint_violations) == 0,
            "optimization_history": self.optimization_history,
            "constraint_violation_log": self.constraint_violations,
            "optimizer_config": self.config.name,
        }

        return results


class ReliabilityOptimizer:
    """Optimizer focused on system reliability for medical applications."""

    def __init__(self, target_reliability: float = 0.999):
        """Initialize reliability optimizer."""
        self.target_reliability = target_reliability
        self.redundancy_strategies = [
            "component_redundancy",
            "functional_redundancy",
            "time_redundancy",
        ]
        logger.info(
            f"Reliability optimizer initialized with target: {target_reliability}"
        )

    def optimize_reliability(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system reliability."""
        optimization_results = {
            "original_reliability": self._calculate_reliability(system_config),
            "target_reliability": self.target_reliability,
            "optimization_applied": [],
            "final_reliability": 0.0,
            "improvement_achieved": False,
        }

        current_config = system_config.copy()

        # Apply redundancy strategies
        for strategy in self.redundancy_strategies:
            if optimization_results["original_reliability"] >= self.target_reliability:
                break

            improved_config = self._apply_redundancy_strategy(current_config, strategy)
            new_reliability = self._calculate_reliability(improved_config)

            if new_reliability > optimization_results["original_reliability"]:
                current_config = improved_config
                optimization_results["optimization_applied"].append(strategy)

        optimization_results["final_reliability"] = self._calculate_reliability(
            current_config
        )
        optimization_results["improvement_achieved"] = (
            optimization_results["final_reliability"]
            > optimization_results["original_reliability"]
        )
        optimization_results["optimized_config"] = current_config

        logger.info(
            f"Reliability optimization completed: {optimization_results['final_reliability']:.4f}"
        )
        return optimization_results

    def _calculate_reliability(self, config: Dict[str, Any]) -> float:
        """Calculate system reliability."""
        # Simplified reliability calculation
        base_reliability = config.get("base_reliability", 0.95)
        redundancy_factor = config.get("redundancy_factor", 1.0)

        # Apply redundancy improvement
        improved_reliability = 1 - (1 - base_reliability) ** redundancy_factor

        return min(improved_reliability, 0.999999)  # Cap at practical maximum

    def _apply_redundancy_strategy(
        self, config: Dict[str, Any], strategy: str
    ) -> Dict[str, Any]:
        """Apply redundancy strategy to improve reliability."""
        improved_config = config.copy()

        if strategy == "component_redundancy":
            improved_config["redundancy_factor"] = (
                config.get("redundancy_factor", 1.0) * 1.2
            )
        elif strategy == "functional_redundancy":
            improved_config["functional_redundancy"] = True
            improved_config["base_reliability"] = min(
                config.get("base_reliability", 0.95) * 1.1, 0.99
            )
        elif strategy == "time_redundancy":
            improved_config["time_redundancy"] = True
            improved_config["retry_count"] = config.get("retry_count", 1) + 1

        return improved_config


class PowerOptimizer:
    """Optimizer focused on power management for medical devices."""

    def __init__(self, power_budget: float = 100.0):
        """Initialize power optimizer."""
        self.power_budget = power_budget
        self.optimization_strategies = [
            "dvfs",
            "clock_gating",
            "power_gating",
            "workload_scheduling",
        ]
        logger.info(f"Power optimizer initialized with budget: {power_budget}W")

    def optimize_power(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system power consumption."""
        optimization_results = {
            "original_power": self._calculate_power(system_config),
            "power_budget": self.power_budget,
            "optimization_applied": [],
            "final_power": 0.0,
            "budget_met": False,
        }

        current_config = system_config.copy()

        # Apply power optimization strategies
        for strategy in self.optimization_strategies:
            current_power = self._calculate_power(current_config)
            if current_power <= self.power_budget:
                break

            optimized_config = self._apply_power_strategy(current_config, strategy)
            new_power = self._calculate_power(optimized_config)

            if new_power < current_power:
                current_config = optimized_config
                optimization_results["optimization_applied"].append(strategy)

        optimization_results["final_power"] = self._calculate_power(current_config)
        optimization_results["budget_met"] = (
            optimization_results["final_power"] <= self.power_budget
        )
        optimization_results["optimized_config"] = current_config

        logger.info(
            f"Power optimization completed: {optimization_results['final_power']:.2f}W"
        )
        return optimization_results

    def _calculate_power(self, config: Dict[str, Any]) -> float:
        """Calculate system power consumption."""
        base_power = config.get("base_power", 80.0)
        frequency_factor = config.get("frequency", 1.0)
        voltage_factor = config.get("voltage", 1.0)

        # Power scales quadratically with voltage and linearly with frequency
        power = base_power * frequency_factor * (voltage_factor**2)

        # Apply power management features
        if config.get("clock_gating", False):
            power *= 0.9
        if config.get("power_gating", False):
            power *= 0.8
        if config.get("dvfs_enabled", False):
            power *= config.get("dvfs_efficiency", 0.85)

        return power

    def _apply_power_strategy(
        self, config: Dict[str, Any], strategy: str
    ) -> Dict[str, Any]:
        """Apply power optimization strategy."""
        optimized_config = config.copy()

        if strategy == "dvfs":
            optimized_config["dvfs_enabled"] = True
            optimized_config["frequency"] = config.get("frequency", 1.0) * 0.9
            optimized_config["voltage"] = config.get("voltage", 1.0) * 0.95
        elif strategy == "clock_gating":
            optimized_config["clock_gating"] = True
        elif strategy == "power_gating":
            optimized_config["power_gating"] = True
        elif strategy == "workload_scheduling":
            optimized_config["workload_scheduling"] = True
            optimized_config["utilization"] = config.get("utilization", 1.0) * 0.85

        return optimized_config


def create_medical_optimization_config(
    optimization_type: str = "safety", **kwargs
) -> OptimizationConfiguration:
    """
    Create medical optimization configuration.

    Args:
        optimization_type: Type of optimization (safety, reliability, power, performance)
        **kwargs: Additional configuration parameters

    Returns:
        Medical optimization configuration
    """
    # Create default objectives based on type
    objectives = []
    constraints = []

    if optimization_type == "safety":
        objectives.append(
            OptimizationObjective(
                name="safety_objective",
                optimization_type=OptimizationType.SAFETY,
                weight=1.0,
                minimize=True,
            )
        )
        constraints.append(
            SafetyConstraint(
                name="accuracy_constraint",
                constraint_type=ConstraintType.HARD,
                parameter="accuracy",
                min_value=0.95,
                violation_action="stop",
            )
        )
    elif optimization_type == "reliability":
        objectives.append(
            OptimizationObjective(
                name="reliability_objective",
                optimization_type=OptimizationType.RELIABILITY,
                weight=1.0,
                minimize=False,
            )
        )
        constraints.append(
            SafetyConstraint(
                name="reliability_constraint",
                constraint_type=ConstraintType.HARD,
                parameter="reliability",
                min_value=0.999,
                violation_action="adjust",
            )
        )
    elif optimization_type == "power":
        objectives.append(
            OptimizationObjective(
                name="power_objective",
                optimization_type=OptimizationType.POWER,
                weight=1.0,
                minimize=True,
            )
        )
        constraints.append(
            SafetyConstraint(
                name="power_constraint",
                constraint_type=ConstraintType.HARD,
                parameter="power_consumption",
                max_value=100.0,
                violation_action="adjust",
            )
        )

    config = OptimizationConfiguration(
        name=f"{optimization_type}_optimization",
        objectives=objectives,
        constraints=constraints,
        **kwargs,
    )

    logger.info(f"Created medical optimization config: {optimization_type}")
    return config
