"""
High-level simulation orchestration and control.

Provides comprehensive simulation capabilities with workload management,
performance analysis, and result generation.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..analysis.metrics import PerformanceAnalyzer
from ..core.accelerator import (
    AcceleratorController,
    create_datacenter_accelerator,
    create_edge_accelerator,
    create_medical_accelerator,
)
from ..utils.config import AcceleratorConfig
from ..workloads.base import BaseWorkload

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation execution."""

    max_simulation_time: float = 3600.0  # 1 hour max
    progress_reporting_interval: int = 1000  # cycles
    enable_real_time_monitoring: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10000  # cycles
    output_directory: str = "simulation_results"
    enable_parallel_execution: bool = False
    max_parallel_workers: int = 4

    # Analysis configuration
    enable_performance_analysis: bool = True
    enable_power_analysis: bool = True
    enable_reliability_analysis: bool = True
    enable_security_analysis: bool = True

    # Output configuration
    save_detailed_trace: bool = False
    save_performance_history: bool = True
    save_subsystem_metrics: bool = True
    generate_visualizations: bool = True


@dataclass
class SimulationResult:
    """Complete simulation result with all metrics and analysis."""

    simulation_id: str
    workload_name: str
    accelerator_config: str
    execution_time_seconds: float

    # Core metrics
    total_cycles: int
    total_operations: int
    throughput_ops_per_second: float

    # Performance metrics
    pe_utilization: float
    memory_utilization: float
    energy_efficiency_tops_per_watt: float

    # Subsystem results
    power_metrics: Dict[str, Any]
    reliability_metrics: Dict[str, Any]
    security_metrics: Dict[str, Any]

    # Analysis results
    performance_analysis: Optional[Dict[str, Any]] = None
    bottleneck_analysis: Optional[Dict[str, Any]] = None
    optimization_recommendations: List[str] = field(default_factory=list)

    # Raw data (optional)
    detailed_trace: Optional[List[Dict[str, Any]]] = None
    performance_history: Optional[List[Dict[str, Any]]] = None


class SimulationOrchestrator:
    """
    High-level simulation orchestrator.

    Manages multiple simulations, workload execution, and result analysis.
    """

    def __init__(self, config: SimulationConfig = None):
        """
        Initialize simulation orchestrator.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.simulation_id = f"sim_{int(time.time())}"

        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Simulation state
        self.is_running = False
        self.current_simulation = None
        self.simulation_history: List[SimulationResult] = []

        # Performance monitoring
        self.monitor_thread = None
        self.monitor_stop_event = threading.Event()

        # Thread pool for parallel simulations
        if self.config.enable_parallel_execution:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_parallel_workers
            )
        else:
            self.thread_pool = None

        logger.info(
            f"Simulation orchestrator initialized with ID: {self.simulation_id}"
        )

    def run_single_simulation(
        self,
        accelerator_config: AcceleratorConfig,
        workload: BaseWorkload,
        simulation_name: Optional[str] = None,
    ) -> SimulationResult:
        """
        Run a single simulation with specified configuration and workload.

        Args:
            accelerator_config: Accelerator configuration
            workload: Workload to execute
            simulation_name: Optional simulation name

        Returns:
            Complete simulation results
        """
        sim_name = simulation_name or f"{workload.get_name()}_{int(time.time())}"
        logger.info(f"Starting simulation: {sim_name}")

        start_time = time.time()

        try:
            # Create accelerator
            accelerator = AcceleratorController(accelerator_config)

            # Start monitoring if enabled
            if self.config.enable_real_time_monitoring:
                self._start_monitoring(accelerator, sim_name)

            # Execute workload
            execution_results = accelerator.execute_workload(workload)

            # Stop monitoring
            if self.config.enable_real_time_monitoring:
                self._stop_monitoring()

            # Generate comprehensive results
            simulation_result = self._generate_simulation_result(
                sim_name, accelerator_config, workload, execution_results, start_time
            )

            # Perform analysis
            if self.config.enable_performance_analysis:
                simulation_result = self._perform_analysis(
                    simulation_result, execution_results
                )

            # Save results
            self._save_simulation_result(simulation_result)

            # Add to history
            self.simulation_history.append(simulation_result)

            logger.info(f"Simulation {sim_name} completed successfully")
            return simulation_result

        except Exception as e:
            logger.error(f"Simulation {sim_name} failed: {e}")
            raise

    def run_batch_simulations(
        self, simulation_specs: List[Dict[str, Any]]
    ) -> List[SimulationResult]:
        """
        Run multiple simulations in batch.

        Args:
            simulation_specs: List of simulation specifications

        Returns:
            List of simulation results
        """
        logger.info(
            f"Starting batch simulation with {len(simulation_specs)} simulations"
        )

        results = []

        if self.config.enable_parallel_execution and self.thread_pool:
            # Parallel execution
            future_to_spec = {}

            for spec in simulation_specs:
                future = self.thread_pool.submit(self._run_simulation_from_spec, spec)
                future_to_spec[future] = spec

            # Collect results as they complete
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed simulation: {spec.get('name', 'unnamed')}")
                except Exception as e:
                    logger.error(
                        f"Simulation failed: {spec.get('name', 'unnamed')}: {e}"
                    )

        else:
            # Sequential execution
            for i, spec in enumerate(simulation_specs):
                try:
                    logger.info(f"Running simulation {i+1}/{len(simulation_specs)}")
                    result = self._run_simulation_from_spec(spec)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Simulation {i+1} failed: {e}")

        # Generate batch analysis
        if len(results) > 1:
            self._generate_batch_analysis(results)

        logger.info(
            f"Batch simulation completed. {len(results)}/{len(simulation_specs)} successful"
        )
        return results

    def _run_simulation_from_spec(self, spec: Dict[str, Any]) -> SimulationResult:
        """Run simulation from specification dictionary."""
        # Extract configuration
        accel_config = spec["accelerator_config"]
        workload = spec["workload"]
        name = spec.get("name", f"sim_{int(time.time())}")

        return self.run_single_simulation(accel_config, workload, name)

    def run_design_space_exploration(
        self,
        base_config: AcceleratorConfig,
        workload: BaseWorkload,
        parameter_sweeps: Dict[str, List[Any]],
    ) -> List[SimulationResult]:
        """
        Run design space exploration across parameter ranges.

        Args:
            base_config: Base accelerator configuration
            workload: Workload to execute
            parameter_sweeps: Dictionary of parameters to sweep

        Returns:
            List of simulation results for all parameter combinations
        """
        logger.info("Starting design space exploration")

        # Generate all parameter combinations
        import itertools

        param_names = list(parameter_sweeps.keys())
        param_values = list(parameter_sweeps.values())

        simulation_specs = []

        for combination in itertools.product(*param_values):
            # Create modified configuration
            config = self._modify_config(base_config, param_names, combination)

            # Create simulation spec
            spec = {
                "name": f"dse_{self._params_to_string(param_names, combination)}",
                "accelerator_config": config,
                "workload": workload,
                "parameters": dict(zip(param_names, combination)),
            }
            simulation_specs.append(spec)

        logger.info(f"Generated {len(simulation_specs)} parameter combinations")

        # Run batch simulations
        results = self.run_batch_simulations(simulation_specs)

        # Generate DSE analysis
        self._generate_dse_analysis(results, parameter_sweeps)

        return results

    def _modify_config(
        self,
        base_config: AcceleratorConfig,
        param_names: List[str],
        param_values: Tuple[Any, ...],
    ) -> AcceleratorConfig:
        """Modify configuration with new parameter values."""
        import copy

        config = copy.deepcopy(base_config)

        for param_name, param_value in zip(param_names, param_values):
            if param_name == "array_rows":
                config.array.rows = param_value
            elif param_name == "array_cols":
                config.array.cols = param_value
            elif param_name == "frequency":
                config.array.frequency = param_value
            elif param_name == "voltage":
                config.array.voltage = param_value
            elif param_name == "data_type":
                config.data_type = param_value
            elif param_name == "input_buffer_size":
                config.input_buffer.buffer_size = param_value
            elif param_name == "weight_buffer_size":
                config.weight_buffer.buffer_size = param_value
            elif param_name == "output_buffer_size":
                config.output_buffer.buffer_size = param_value
            # Add more parameter mappings as needed

        return config

    def _params_to_string(
        self, param_names: List[str], param_values: Tuple[Any, ...]
    ) -> str:
        """Convert parameter combination to string."""
        param_pairs = [
            f"{name}_{value}" for name, value in zip(param_names, param_values)
        ]
        return "_".join(param_pairs)

    def _generate_simulation_result(
        self,
        sim_name: str,
        accelerator_config: AcceleratorConfig,
        workload: BaseWorkload,
        execution_results: Dict[str, Any],
        start_time: float,
    ) -> SimulationResult:
        """Generate comprehensive simulation result."""
        execution_time = time.time() - start_time

        # Extract core metrics
        exec_summary = execution_results.get("execution_summary", {})
        perf_metrics = execution_results.get("performance_metrics", {})
        subsystem_metrics = execution_results.get("subsystem_metrics", {})

        # Create simulation result
        result = SimulationResult(
            simulation_id=f"{self.simulation_id}_{sim_name}",
            workload_name=workload.get_name(),
            accelerator_config=accelerator_config.name,
            execution_time_seconds=execution_time,
            # Core metrics
            total_cycles=exec_summary.get("total_cycles", 0),
            total_operations=exec_summary.get("total_operations", 0),
            throughput_ops_per_second=exec_summary.get(
                "throughput_ops_per_second", 0.0
            ),
            # Performance metrics
            pe_utilization=perf_metrics.get("pe_utilization", 0.0),
            memory_utilization=perf_metrics.get("memory_utilization", 0.0),
            energy_efficiency_tops_per_watt=perf_metrics.get(
                "energy_efficiency_tops_per_watt", 0.0
            ),
            # Subsystem results
            power_metrics=subsystem_metrics.get("power_management", {}),
            reliability_metrics=subsystem_metrics.get("reliability", {}),
            security_metrics=subsystem_metrics.get("security", {}),
        )

        # Add optional detailed data
        if self.config.save_detailed_trace:
            result.detailed_trace = execution_results.get("raw_results", {}).get(
                "execution_trace", []
            )

        if self.config.save_performance_history:
            result.performance_history = execution_results.get(
                "performance_history", []
            )

        return result

    def _perform_analysis(
        self, simulation_result: SimulationResult, execution_results: Dict[str, Any]
    ) -> SimulationResult:
        """Perform comprehensive analysis on simulation results."""

        # Performance analysis
        if simulation_result.performance_history:
            analyzer = PerformanceAnalyzer()

            # Add performance samples
            for sample in simulation_result.performance_history:
                analyzer.add_sample(
                    cycle=sample.get("cycle", 0),
                    utilization=sample.get("utilization", 0.0),
                    power=sample.get("power_watts", 0.0),
                    throughput=sample.get("throughput", 0.0),
                )

            simulation_result.performance_analysis = analyzer.get_analysis_report()

        # Bottleneck analysis
        simulation_result.bottleneck_analysis = self._analyze_bottlenecks(
            execution_results
        )

        # Generate optimization recommendations
        simulation_result.optimization_recommendations = (
            self._generate_optimization_recommendations(
                simulation_result, execution_results
            )
        )

        return simulation_result

    def _analyze_bottlenecks(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        bottlenecks = {
            "compute_bound": False,
            "memory_bound": False,
            "power_bound": False,
            "thermal_bound": False,
            "primary_bottleneck": "unknown",
            "bottleneck_severity": "low",
        }

        # Analyze subsystem metrics
        subsystem_metrics = execution_results.get("subsystem_metrics", {})

        # Check memory bottleneck
        memory_metrics = subsystem_metrics.get("memory_hierarchy", {})
        if memory_metrics:
            hierarchy_metrics = memory_metrics.get("hierarchy_metrics", {})
            l1_hit_rate = hierarchy_metrics.get("l1_hit_rate", 1.0)
            l2_hit_rate = hierarchy_metrics.get("l2_hit_rate", 1.0)

            if l1_hit_rate < 0.8 or l2_hit_rate < 0.9:
                bottlenecks["memory_bound"] = True
                bottlenecks["primary_bottleneck"] = "memory"
                bottlenecks["bottleneck_severity"] = (
                    "high" if l1_hit_rate < 0.6 else "medium"
                )

        # Check power bottleneck
        power_metrics = subsystem_metrics.get("power_management", {})
        if power_metrics:
            budget_status = power_metrics.get("budget_status", {})
            budget_util = budget_status.get("budget_utilization_percent", 0.0)

            if budget_util > 90:
                bottlenecks["power_bound"] = True
                if bottlenecks["primary_bottleneck"] == "unknown":
                    bottlenecks["primary_bottleneck"] = "power"
                    bottlenecks["bottleneck_severity"] = "high"

        # Check thermal bottleneck
        thermal_status = power_metrics.get("thermal_status", {})
        if thermal_status:
            thermal_util = thermal_status.get("thermal_utilization_percent", 0.0)

            if thermal_util > 85:
                bottlenecks["thermal_bound"] = True
                if bottlenecks["primary_bottleneck"] == "unknown":
                    bottlenecks["primary_bottleneck"] = "thermal"
                    bottlenecks["bottleneck_severity"] = "high"

        # Check compute bottleneck (default if no other bottlenecks)
        if bottlenecks["primary_bottleneck"] == "unknown":
            bottlenecks["compute_bound"] = True
            bottlenecks["primary_bottleneck"] = "compute"
            bottlenecks["bottleneck_severity"] = "low"

        return bottlenecks

    def _generate_optimization_recommendations(
        self, simulation_result: SimulationResult, execution_results: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # PE utilization recommendations
        if simulation_result.pe_utilization < 0.5:
            recommendations.append(
                "Low PE utilization detected. Consider optimizing workload mapping or "
                "reducing array size for better efficiency."
            )
        elif simulation_result.pe_utilization > 0.95:
            recommendations.append(
                "Very high PE utilization. Consider increasing array size for higher throughput."
            )

        # Memory utilization recommendations
        if simulation_result.memory_utilization < 0.3:
            recommendations.append(
                "Low memory utilization. Consider reducing memory buffer sizes to save area and power."
            )
        elif simulation_result.memory_utilization > 0.9:
            recommendations.append(
                "High memory utilization. Consider increasing buffer sizes or improving memory hierarchy."
            )

        # Energy efficiency recommendations
        if simulation_result.energy_efficiency_tops_per_watt < 10:
            recommendations.append(
                "Low energy efficiency. Consider enabling power management features or "
                "optimizing for lower power operation."
            )

        # Bottleneck-specific recommendations
        if hasattr(simulation_result, "bottleneck_analysis"):
            bottleneck = simulation_result.bottleneck_analysis.get(
                "primary_bottleneck", "unknown"
            )

            if bottleneck == "memory":
                recommendations.append(
                    "Memory bottleneck detected. Consider increasing cache sizes, "
                    "improving memory bandwidth, or optimizing data access patterns."
                )
            elif bottleneck == "power":
                recommendations.append(
                    "Power bottleneck detected. Consider enabling DVFS, power gating, "
                    "or reducing operating frequency/voltage."
                )
            elif bottleneck == "thermal":
                recommendations.append(
                    "Thermal bottleneck detected. Consider improving cooling, "
                    "reducing power consumption, or thermal-aware scheduling."
                )

        return recommendations

    def _start_monitoring(self, accelerator: AcceleratorController, sim_name: str):
        """Start real-time monitoring of simulation."""

        def monitor_loop():
            while not self.monitor_stop_event.is_set():
                try:
                    status = accelerator.get_real_time_status()

                    # Log status periodically
                    if (
                        status["current_cycle"]
                        % self.config.progress_reporting_interval
                        == 0
                    ):
                        logger.info(
                            f"Simulation {sim_name} - Cycle: {status['current_cycle']}, "
                            f"Power: {status['current_power']:.1f}W, "
                            f"Utilization: {status['performance_snapshot']['utilization']:.1%}"
                        )

                    time.sleep(0.1)  # 100ms monitoring interval

                except Exception as e:
                    logger.warning(f"Monitoring error: {e}")
                    break

        self.monitor_stop_event.clear()
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitoring(self):
        """Stop real-time monitoring."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_stop_event.set()
            self.monitor_thread.join(timeout=1.0)

    def _save_simulation_result(self, result: SimulationResult):
        """Save simulation result to file."""
        try:
            result_file = self.output_dir / f"{result.simulation_id}_result.json"

            # Convert result to dictionary for JSON serialization
            result_dict = {
                "simulation_id": result.simulation_id,
                "workload_name": result.workload_name,
                "accelerator_config": result.accelerator_config,
                "execution_time_seconds": result.execution_time_seconds,
                "total_cycles": result.total_cycles,
                "total_operations": result.total_operations,
                "throughput_ops_per_second": result.throughput_ops_per_second,
                "pe_utilization": result.pe_utilization,
                "memory_utilization": result.memory_utilization,
                "energy_efficiency_tops_per_watt": result.energy_efficiency_tops_per_watt,
                "power_metrics": result.power_metrics,
                "reliability_metrics": result.reliability_metrics,
                "security_metrics": result.security_metrics,
                "performance_analysis": result.performance_analysis,
                "bottleneck_analysis": result.bottleneck_analysis,
                "optimization_recommendations": result.optimization_recommendations,
            }

            with open(result_file, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)

            logger.info(f"Simulation result saved to {result_file}")

        except Exception as e:
            logger.error(f"Failed to save simulation result: {e}")

    def _generate_batch_analysis(self, results: List[SimulationResult]):
        """Generate analysis across multiple simulation results."""
        if not results:
            return

        batch_analysis = {
            "total_simulations": len(results),
            "summary_statistics": {},
            "performance_comparison": {},
            "recommendations": [],
        }

        # Calculate summary statistics
        throughputs = [r.throughput_ops_per_second for r in results]
        utilizations = [r.pe_utilization for r in results]
        efficiencies = [r.energy_efficiency_tops_per_watt for r in results]

        batch_analysis["summary_statistics"] = {
            "throughput": {
                "mean": np.mean(throughputs),
                "std": np.std(throughputs),
                "min": np.min(throughputs),
                "max": np.max(throughputs),
            },
            "utilization": {
                "mean": np.mean(utilizations),
                "std": np.std(utilizations),
                "min": np.min(utilizations),
                "max": np.max(utilizations),
            },
            "efficiency": {
                "mean": np.mean(efficiencies),
                "std": np.std(efficiencies),
                "min": np.min(efficiencies),
                "max": np.max(efficiencies),
            },
        }

        # Find best and worst performers
        best_throughput = max(results, key=lambda r: r.throughput_ops_per_second)
        best_efficiency = max(results, key=lambda r: r.energy_efficiency_tops_per_watt)

        batch_analysis["performance_comparison"] = {
            "best_throughput": {
                "simulation_id": best_throughput.simulation_id,
                "value": best_throughput.throughput_ops_per_second,
            },
            "best_efficiency": {
                "simulation_id": best_efficiency.simulation_id,
                "value": best_efficiency.energy_efficiency_tops_per_watt,
            },
        }

        # Save batch analysis
        try:
            analysis_file = self.output_dir / f"batch_analysis_{int(time.time())}.json"
            with open(analysis_file, "w") as f:
                json.dump(batch_analysis, f, indent=2, default=str)

            logger.info(f"Batch analysis saved to {analysis_file}")

        except Exception as e:
            logger.error(f"Failed to save batch analysis: {e}")

    def _generate_dse_analysis(
        self, results: List[SimulationResult], parameter_sweeps: Dict[str, List[Any]]
    ):
        """Generate design space exploration analysis."""
        if not results:
            return

        dse_analysis = {
            "parameter_sweeps": parameter_sweeps,
            "total_configurations": len(results),
            "pareto_frontier": [],
            "parameter_sensitivity": {},
            "optimal_configurations": {},
        }

        # Find Pareto frontier (throughput vs efficiency)
        pareto_points = []
        for result in results:
            is_pareto = True
            for other in results:
                if (
                    other.throughput_ops_per_second >= result.throughput_ops_per_second
                    and other.energy_efficiency_tops_per_watt
                    >= result.energy_efficiency_tops_per_watt
                    and (
                        other.throughput_ops_per_second
                        > result.throughput_ops_per_second
                        or other.energy_efficiency_tops_per_watt
                        > result.energy_efficiency_tops_per_watt
                    )
                ):
                    is_pareto = False
                    break

            if is_pareto:
                pareto_points.append(
                    {
                        "simulation_id": result.simulation_id,
                        "throughput": result.throughput_ops_per_second,
                        "efficiency": result.energy_efficiency_tops_per_watt,
                    }
                )

        dse_analysis["pareto_frontier"] = pareto_points

        # Save DSE analysis
        try:
            dse_file = self.output_dir / f"dse_analysis_{int(time.time())}.json"
            with open(dse_file, "w") as f:
                json.dump(dse_analysis, f, indent=2, default=str)

            logger.info(f"DSE analysis saved to {dse_file}")

        except Exception as e:
            logger.error(f"Failed to save DSE analysis: {e}")

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of all simulations run by this orchestrator."""
        return {
            "orchestrator_id": self.simulation_id,
            "total_simulations": len(self.simulation_history),
            "output_directory": str(self.output_dir),
            "configuration": self.config.__dict__,
            "simulation_history": [
                {
                    "simulation_id": r.simulation_id,
                    "workload_name": r.workload_name,
                    "execution_time_seconds": r.execution_time_seconds,
                    "throughput_ops_per_second": r.throughput_ops_per_second,
                    "energy_efficiency_tops_per_watt": r.energy_efficiency_tops_per_watt,
                }
                for r in self.simulation_history
            ],
        }

    def cleanup(self):
        """Clean up resources."""
        self._stop_monitoring()

        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        logger.info("Simulation orchestrator cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


# Convenience functions for common simulation scenarios


def run_quick_simulation(
    workload: BaseWorkload,
    array_size: Tuple[int, int] = (16, 16),
    accelerator_type: str = "balanced",
) -> SimulationResult:
    """
    Run a quick simulation with default settings.

    Args:
        workload: Workload to execute
        array_size: Systolic array dimensions
        accelerator_type: Type of accelerator ("medical", "edge", "datacenter", "balanced")

    Returns:
        Simulation result
    """
    # Create accelerator based on type
    if accelerator_type == "medical":
        accelerator = create_medical_accelerator(array_size)
    elif accelerator_type == "edge":
        accelerator = create_edge_accelerator(array_size)
    elif accelerator_type == "datacenter":
        accelerator = create_datacenter_accelerator(array_size)
    else:  # balanced
        from ..utils.config import AcceleratorConfig, ArrayConfig, DataType

        config = AcceleratorConfig(
            name="Balanced_Accelerator",
            array=ArrayConfig(rows=array_size[0], cols=array_size[1]),
            data_type=DataType.FLOAT32,
        )
        accelerator = AcceleratorController(config)

    # Run simulation
    with SimulationOrchestrator() as orchestrator:
        return orchestrator.run_single_simulation(
            accelerator.config, workload, f"quick_sim_{workload.get_name()}"
        )


def run_comparison_study(
    workloads: List[BaseWorkload], accelerator_configs: List[AcceleratorConfig]
) -> List[SimulationResult]:
    """
    Run a comparison study across multiple workloads and configurations.

    Args:
        workloads: List of workloads to test
        accelerator_configs: List of accelerator configurations to test

    Returns:
        List of simulation results
    """
    simulation_specs = []

    for workload in workloads:
        for config in accelerator_configs:
            # Check compatibility
            if hasattr(workload, "get_requirements"):
                requirements = workload.get_requirements()
                if hasattr(requirements, "required_array_size"):
                    req_rows, req_cols = requirements.required_array_size
                    if req_rows != config.array.rows or req_cols != config.array.cols:
                        continue  # Skip incompatible combinations

            spec = {
                "name": f"{workload.get_name()}_{config.name}",
                "accelerator_config": config,
                "workload": workload,
            }
            simulation_specs.append(spec)

    # Run batch simulation
    with SimulationOrchestrator() as orchestrator:
        return orchestrator.run_batch_simulations(simulation_specs)
