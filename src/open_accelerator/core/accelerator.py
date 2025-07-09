"""
Advanced accelerator orchestration and control.

Integrates all accelerator components including systolic array, memory hierarchy,
power management, reliability, and security for comprehensive simulation.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..utils.config import AcceleratorConfig
from ..workloads.base import BaseWorkload
from .memory import MemoryHierarchy
from .power_management import PowerManager, create_automotive_power_config
from .reliability import ReliabilityManager, create_medical_reliability_config
from .security import SecurityManager, create_medical_security_config
from .systolic_array import SystolicArray

logger = logging.getLogger(__name__)


@dataclass
class AcceleratorMetrics:
    """Comprehensive accelerator performance metrics."""

    # Performance metrics
    total_cycles: int = 0
    total_operations: int = 0
    throughput_ops_per_second: float = 0.0
    latency_cycles: int = 0

    # Utilization metrics
    pe_utilization: float = 0.0
    memory_utilization: float = 0.0
    interconnect_utilization: float = 0.0

    # Power metrics
    average_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    total_energy_joules: float = 0.0
    energy_efficiency_tops_per_watt: float = 0.0

    # Reliability metrics
    errors_detected: int = 0
    errors_corrected: int = 0
    system_availability: float = 1.0

    # Security metrics
    security_violations: int = 0
    encryption_operations: int = 0

    # Medical compliance metrics
    precision_violations: int = 0
    safety_events: int = 0
    regulatory_compliance_score: float = 1.0


class AcceleratorController:
    """
    Main accelerator controller that orchestrates all subsystems.

    Provides unified interface for simulation, monitoring, and control
    of the complete accelerator system.
    """

    def __init__(self, config: AcceleratorConfig):
        """
        Initialize accelerator controller.

        Args:
            config: Complete accelerator configuration
        """
        self.config = config
        self.current_cycle = 0
        self.simulation_start_time = 0.0
        self.is_running = False

        # Initialize core components
        self._initialize_components()

        logger.info("Accelerator controller initialization complete")

    def _initialize_components(self):
        """Initialize all accelerator subsystems."""
        logger.info("Initializing accelerator subsystems...")

        # Systolic array
        self.systolic_array = SystolicArray(self.config)
        logger.info(
            f"Initialized {self.config.array.rows}x{self.config.array.cols} systolic array"
        )

        # Memory hierarchy
        self.memory_hierarchy = MemoryHierarchy(self.config)
        logger.info("Initialized memory hierarchy")

        # Power management
        if self.config.medical_mode:
            power_config = create_automotive_power_config()  # Conservative for medical
        else:
            power_config = create_automotive_power_config()
        self.power_manager = PowerManager(power_config)
        logger.info("Initialized power management")

        # Reliability (enhanced for medical mode)
        if self.config.medical_mode:
            reliability_config = create_medical_reliability_config()
        else:
            reliability_config = create_medical_reliability_config()
            reliability_config.safety_critical = False
        self.reliability_manager = ReliabilityManager(reliability_config)
        logger.info("Initialized reliability management")

        # Security (enhanced for medical mode)
        if self.config.medical_mode:
            security_config = create_medical_security_config()
        else:
            security_config = create_medical_security_config()
            security_config.hipaa_compliant = False
            security_config.fda_compliant = False
        self.security_manager = SecurityManager(security_config)
        logger.info("Initialized security management")

        # Metrics and monitoring
        self.metrics = AcceleratorMetrics()
        self.performance_history: list[dict[str, Any]] = []

        # Workload management
        self.current_workload: Optional[BaseWorkload] = None
        self.workload_queue: list[BaseWorkload] = []

        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def load_workload(self, workload: BaseWorkload) -> bool:
        """
        Load a workload for execution.

        Args:
            workload: Workload to execute

        Returns:
            True if workload loaded successfully
        """
        try:
            # Validate workload compatibility
            if not self._validate_workload_compatibility(workload):
                logger.error("Workload incompatible with current configuration")
                return False

            # Security check for workload
            if self.config.medical_mode:
                workload_data = workload.get_input_data()
                if not self.security_manager.verify_data_integrity(workload_data):
                    logger.error("Workload failed security verification")
                    return False

            self.current_workload = workload
            logger.info(f"Loaded workload: {workload.get_name()}")
            return True

        except Exception as e:
            logger.error(f"Failed to load workload: {e}")
            return False

    def _validate_workload_compatibility(self, workload: BaseWorkload) -> bool:
        """Validate that workload is compatible with accelerator configuration."""
        workload_requirements = workload.get_requirements()

        # Check array size compatibility
        # First check if there's a specific required array size
        if hasattr(workload_requirements, "required_array_size"):
            required_rows, required_cols = workload_requirements.required_array_size
            if (
                required_rows > self.config.array.rows
                or required_cols > self.config.array.cols
            ):
                logger.error(f"Required array size {required_rows}x{required_cols} exceeds available {self.config.array.rows}x{self.config.array.cols}")
                return False
        
        # Check if workload fits within max array size constraints
        if hasattr(workload_requirements, "max_array_size"):
            max_rows, max_cols = workload_requirements.max_array_size
            # Workload should be able to run if accelerator array is within max size
            if (
                self.config.array.rows > max_rows
                or self.config.array.cols > max_cols
            ):
                logger.warning(f"Accelerator array {self.config.array.rows}x{self.config.array.cols} exceeds workload max {max_rows}x{max_cols}, but proceeding")
        
        # Check minimum array size requirements
        if hasattr(workload_requirements, "min_array_size"):
            min_rows, min_cols = workload_requirements.min_array_size
            if (
                self.config.array.rows < min_rows
                or self.config.array.cols < min_cols
            ):
                logger.error(f"Accelerator array {self.config.array.rows}x{self.config.array.cols} is smaller than minimum required {min_rows}x{min_cols}")
                return False

        # Check memory requirements
        if hasattr(workload_requirements, "memory_requirements"):
            total_memory_needed = sum(
                workload_requirements.memory_requirements.values()
            )
            available_memory = (
                self.config.input_buffer.buffer_size
                + self.config.weight_buffer.buffer_size
                + self.config.output_buffer.buffer_size
            )
            if total_memory_needed > available_memory:
                logger.error(f"Memory requirements {total_memory_needed} MB exceed available {available_memory} MB")
                return False

        # Check data type compatibility
        if hasattr(workload_requirements, "supported_data_types"):
            if self.config.data_type.value not in workload_requirements.supported_data_types:
                logger.error(f"Data type {self.config.data_type.value} not supported by workload")
                return False

        return True

    def execute_workload(
        self, workload: Optional[BaseWorkload] = None
    ) -> dict[str, Any]:
        """
        Execute a workload on the accelerator.

        Args:
            workload: Optional workload to execute (uses current if None)

        Returns:
            Execution results and metrics
        """
        if workload:
            if not self.load_workload(workload):
                raise ValueError("Failed to load workload")

        if not self.current_workload:
            raise ValueError("No workload loaded")

        logger.info(f"Starting execution of {self.current_workload.get_name()}")

        # Reset system state
        self._reset_system_state()

        # Start simulation
        self.is_running = True
        self.simulation_start_time = time.time()

        try:
            # Execute workload with full system simulation
            results = self._execute_with_full_simulation()

            # Generate comprehensive metrics
            final_metrics = self._generate_final_metrics(results)

            logger.info("Workload execution completed successfully")
            return final_metrics

        except Exception as e:
            logger.error(f"Workload execution failed: {e}")
            self.is_running = False
            raise
        finally:
            self.is_running = False

    def _reset_system_state(self):
        """Reset all subsystems to initial state."""
        self.current_cycle = 0
        self.systolic_array.reset()
        self.memory_hierarchy.reset()
        self.power_manager.reset()
        self.reliability_manager.reset_metrics()
        self.metrics = AcceleratorMetrics()
        self.performance_history.clear()

    def _execute_with_full_simulation(self) -> dict[str, Any]:
        """Execute workload with full system simulation including all subsystems."""
        results: dict[str, Any] = {
            "execution_trace": [],
            "power_trace": [],
            "reliability_events": [],
            "security_events": [],
            "performance_samples": [],
        }

        # Get workload operations
        if self.current_workload is None:
            raise ValueError("No workload loaded")
        operations = self.current_workload.get_operations()
        total_operations = len(operations)

        logger.info(f"Executing {total_operations} operations")

        for op_idx, operation in enumerate(operations):
            cycle_start_time = time.time()

            # Execute single cycle
            cycle_results = self._execute_single_cycle(operation, op_idx)

            # Update subsystems
            self._update_subsystems(cycle_results)

            # Collect metrics
            self._collect_cycle_metrics(cycle_results, cycle_start_time)

            # Store results
            results["execution_trace"].append(cycle_results)

            # Progress logging
            if (op_idx + 1) % max(1, total_operations // 10) == 0:
                progress = ((op_idx + 1) / total_operations) * 100
                logger.info(f"Execution progress: {progress:.1f}%")

            self.current_cycle += 1

        # Finalize results
        array_results = self.systolic_array.get_results()
        if hasattr(array_results, 'tolist'):
            results["final_output"] = array_results.tolist()
        else:
            results["final_output"] = list(array_results)
        results["total_cycles"] = int(self.current_cycle)

        return results

    def _execute_single_cycle(
        self, operation: dict[str, Any], op_idx: int
    ) -> dict[str, Any]:
        """Execute a single operation cycle with full system modeling."""
        cycle_results = {
            "cycle": self.current_cycle,
            "operation_index": op_idx,
            "operation_type": operation.get("type", "unknown"),
            "systolic_results": {},
            "memory_results": {},
            "power_results": {},
            "reliability_results": {},
            "security_results": {},
        }

        try:
            # Security processing (if enabled)
            if hasattr(self.config, 'medical_mode') and self.config.medical_mode:
                operation_data = operation.get("data", np.array([]))
                if operation_data.size > 0:
                    encrypted_data = self.security_manager.encrypt_data(operation_data)
                    decrypted_data = self.security_manager.decrypt_data(
                        encrypted_data
                    )
                    if isinstance(decrypted_data, tuple):
                        operation["data"] = decrypted_data[0]
                    else:
                        operation["data"] = decrypted_data
                    cycle_results["security_results"]["encryption_performed"] = True

            # Reliability processing
            input_data = operation.get("data", np.array([]))
            if input_data.size > 0:
                (
                    processed_data,
                    reliability_errors,
                ) = self.reliability_manager.process_data_with_reliability(
                    input_data, component="systolic_array", cycle=self.current_cycle
                )
                operation["data"] = processed_data
                cycle_results["reliability_results"] = {
                    "errors_detected": len(reliability_errors),
                    "errors_corrected": len(
                        [e for e in reliability_errors if e.corrected]
                    ),
                }

            # Memory operations
            if "memory_ops" in operation:
                for mem_op in operation["memory_ops"]:
                    if mem_op["type"] == "read":
                        data, latency = self.memory_hierarchy.read_request(
                            mem_op["address"], mem_op["size"]
                        )
                        cycle_results["memory_results"][f"read_{mem_op['address']}"] = {
                            "data_size": len(data),
                            "latency": latency,
                        }
                    elif mem_op["type"] == "write":
                        success, latency = self.memory_hierarchy.write_request(
                            mem_op["address"], mem_op["data"]
                        )
                        cycle_results["memory_results"][
                            f"write_{mem_op['address']}"
                        ] = {"success": success, "latency": latency}

            # Systolic array execution
            if "systolic_inputs" in operation:
                systolic_output = self.systolic_array.cycle(
                    operation["systolic_inputs"], operation.get("control_signals", {})
                )
                cycle_results["systolic_results"] = systolic_output

            # Power management update
            component_utilizations = {
                "systolic_array": cycle_results["systolic_results"].get(
                    "utilization", 0.0
                ),
                "memory": cycle_results["memory_results"].get("utilization", 0.0),
                "control": 0.3,  # Baseline control utilization
                "io": 0.1,  # Baseline I/O utilization
            }

            for component, utilization in component_utilizations.items():
                self.power_manager.update_component_utilization(component, utilization)

            self.power_manager.cycle_update()
            cycle_results["power_results"] = {
                "current_power": self.power_manager.get_power_metrics().current_power_watts,
                "temperature": self.power_manager.get_power_metrics().current_temperature_c,
            }

        except Exception as e:
            logger.error(f"Error in cycle {self.current_cycle}: {e}")
            cycle_results["error"] = str(e)

        return cycle_results

    def _update_subsystems(self, cycle_results: dict[str, Any]):
        """Update all subsystems based on cycle results."""
        # Update memory hierarchy
        self.memory_hierarchy.cycle()

        # Update power management (already done in _execute_single_cycle)

        # Update reliability metrics
        if "reliability_results" in cycle_results:
            rel_results = cycle_results["reliability_results"]
            self.metrics.errors_detected += rel_results.get("errors_detected", 0)
            self.metrics.errors_corrected += rel_results.get("errors_corrected", 0)

        # Update security metrics
        if "security_results" in cycle_results:
            sec_results = cycle_results["security_results"]
            if sec_results.get("encryption_performed", False):
                self.metrics.encryption_operations += 1

    def _collect_cycle_metrics(
        self, cycle_results: dict[str, Any], cycle_start_time: float
    ):
        """Collect and update metrics for the current cycle."""
        cycle_time = time.time() - cycle_start_time
        self.metrics.total_cycles = self.current_cycle + 1  # +1 because we're 0-indexed

        # Update operation count
        if "systolic_results" in cycle_results:
            self.metrics.total_operations += cycle_results["systolic_results"].get(
                "operations", 0
            )

        # Update power metrics
        if "power_results" in cycle_results:
            power_data = cycle_results["power_results"]
            current_power = power_data.get("current_power", 0.0)
            
            # Avoid division by zero
            if self.current_cycle > 0:
                self.metrics.average_power_watts = (
                    self.metrics.average_power_watts * (self.current_cycle - 1)
                    + current_power
                ) / self.current_cycle
            else:
                self.metrics.average_power_watts = current_power
                
            self.metrics.peak_power_watts = max(
                self.metrics.peak_power_watts, current_power
            )
            self.metrics.total_energy_joules += current_power * cycle_time

        # Store performance sample
        performance_sample = {
            "cycle": self.current_cycle,
            "timestamp": time.time(),
            "cycle_time_seconds": cycle_time,
            "power_watts": cycle_results.get("power_results", {}).get(
                "current_power", 0.0
            ),
            "utilization": cycle_results.get("systolic_results", {}).get(
                "utilization", 0.0
            ),
        }
        self.performance_history.append(performance_sample)

    def _generate_final_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive final metrics and analysis."""
        execution_time = time.time() - self.simulation_start_time

        # Calculate derived metrics
        if execution_time > 0:
            self.metrics.throughput_ops_per_second = (
                self.metrics.total_operations / execution_time
            )

        if self.metrics.average_power_watts > 0:
            # Assume peak performance of 100 TOPS for efficiency calculation
            estimated_tops = self.metrics.throughput_ops_per_second / 1e12 * 100
            self.metrics.energy_efficiency_tops_per_watt = (
                estimated_tops / self.metrics.average_power_watts
            )

        # Get subsystem metrics
        systolic_metrics = self.systolic_array.get_metrics()
        memory_metrics = self.memory_hierarchy.get_hierarchy_metrics()
        power_status = self.power_manager.get_power_status()
        reliability_report = self.reliability_manager.get_reliability_report()
        security_status = self.security_manager.get_security_status()

        # Compile comprehensive results
        final_results = {
            "execution_summary": {
                "workload_name": self.current_workload.get_name() if self.current_workload else "unknown",
                "total_cycles": self.metrics.total_cycles,
                "total_operations": self.metrics.total_operations,
                "execution_time_seconds": execution_time,
                "throughput_ops_per_second": self.metrics.throughput_ops_per_second,
            },
            "performance_metrics": self.metrics.__dict__,
            "subsystem_metrics": {
                "systolic_array": systolic_metrics,
                "memory_hierarchy": memory_metrics,
                "power_management": power_status,
                "reliability": reliability_report,
                "security": security_status,
            },
            "raw_results": results,
            "performance_history": self.performance_history,
        }

        # Medical compliance analysis (if applicable)
        if self.config.medical_mode:
            final_results["medical_compliance"] = self._analyze_medical_compliance()

        return final_results

    def _analyze_medical_compliance(self) -> dict[str, Any]:
        """Analyze medical compliance metrics."""
        compliance_report = {
            "regulatory_compliance_score": self.metrics.regulatory_compliance_score,
            "safety_events": self.metrics.safety_events,
            "precision_violations": self.metrics.precision_violations,
            "reliability_metrics": {
                "availability": self.reliability_manager.get_reliability_report()[
                    "reliability_metrics"
                ]["availability"],
                "fault_coverage": self.reliability_manager.get_reliability_report()[
                    "reliability_metrics"
                ]["fault_coverage"],
            },
            "security_compliance": {
                "encryption_operations": self.metrics.encryption_operations,
                "security_violations": self.metrics.security_violations,
            },
            "recommendations": [],
        }

        # Generate compliance recommendations
        if compliance_report["safety_events"] > 0:
            compliance_report["recommendations"].append(
                "Safety events detected. Review safety protocols and error handling."
            )

        if compliance_report["precision_violations"] > 0:
            compliance_report["recommendations"].append(
                "Precision violations detected. Consider higher precision data types or enhanced error correction."
            )

        if compliance_report["reliability_metrics"]["availability"] < 0.9999:
            compliance_report["recommendations"].append(
                "Availability below medical requirements. Enhance fault tolerance mechanisms."
            )

        return compliance_report

    def get_real_time_status(self) -> dict[str, Any]:
        """Get real-time accelerator status."""
        return {
            "is_running": self.is_running,
            "current_cycle": self.current_cycle,
            "current_workload": self.current_workload.get_name()
            if self.current_workload is not None
            else None,
            "current_power": self.power_manager.get_power_metrics().current_power_watts,
            "current_temperature": self.power_manager.get_power_metrics().current_temperature_c,
            "system_health": {
                "systolic_array": "operational",
                "memory": "operational",
                "power": "normal"
                if not self.power_manager.get_throttle_factor() < 1.0
                else "throttled",
                "reliability": "operational"
                if self.reliability_manager.system_operational
                else "degraded",
                "security": "secure",
            },
            "performance_snapshot": {
                "utilization": self.performance_history[-1]["utilization"]
                if self.performance_history
                else 0.0,
                "throughput": self.metrics.throughput_ops_per_second,
                "efficiency": self.metrics.energy_efficiency_tops_per_watt,
            },
        }

    def emergency_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """Perform emergency shutdown of the accelerator."""
        logger.warning(f"Emergency shutdown initiated: {reason}")

        try:
            # Stop current execution
            self.is_running = False

            # Security shutdown
            self.security_manager.emergency_shutdown(reason)

            # Power management shutdown
            self.power_manager.reset()

            # Save current state for recovery
            self._save_emergency_state()

            logger.info("Emergency shutdown completed successfully")
            return True

        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            return False

    def _save_emergency_state(self):
        """Save current state for emergency recovery."""
        emergency_state = {
            "timestamp": time.time(),
            "current_cycle": self.current_cycle,
            "metrics": self.metrics.__dict__,
            "performance_history": self.performance_history[-100:],  # Last 100 samples
            "subsystem_status": {
                "power": self.power_manager.get_power_status(),
                "reliability": self.reliability_manager.get_reliability_report(),
                "security": self.security_manager.get_security_status(),
            },
        }

        try:
            emergency_file = Path("emergency_state.json")
            import json

            with open(emergency_file, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump(emergency_state, f, indent=2, default=str)
            logger.info(f"Emergency state saved to {emergency_file}")
        except Exception as e:
            logger.error(f"Failed to save emergency state: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.is_running:
            self.emergency_shutdown("Context manager exit")

        # Cleanup thread pool
        self.thread_pool.shutdown(wait=True)

        if exc_type:
            logger.error(f"Exception during accelerator operation: {exc_val}")

        return False  # Don't suppress exceptions


# Factory functions for common accelerator configurations


def create_medical_accelerator(
    array_size: tuple[int, int] = (16, 16),
) -> AcceleratorController:
    """Create accelerator optimized for medical AI applications."""
    from ..utils.config import AcceleratorConfig, ArrayConfig, DataType

    config = AcceleratorConfig(
        name="MedicalAI_Accelerator",
        array=ArrayConfig(
            rows=array_size[0],
            cols=array_size[1],
            frequency=1.2e9,  # Conservative frequency for reliability
            voltage=1.0,  # Nominal voltage for stability
        ),
        data_type=DataType.FLOAT32,  # High precision for medical
        enable_power_modeling=True,
        enable_thermal_modeling=True,
    )

    return AcceleratorController(config)


def create_edge_accelerator(
    array_size: tuple[int, int] = (8, 8),
) -> AcceleratorController:
    """Create accelerator optimized for edge AI applications."""
    from ..utils.config import AcceleratorConfig, ArrayConfig, DataType

    config = AcceleratorConfig(
        name="EdgeAI_Accelerator",
        array=ArrayConfig(
            rows=array_size[0],
            cols=array_size[1],
            frequency=800e6,  # Lower frequency for power efficiency
            voltage=0.9,  # Reduced voltage for power savings
        ),
        data_type=DataType.FLOAT16,  # Reduced precision for efficiency
        enable_power_modeling=True,
        enable_thermal_modeling=False,
    )

    return AcceleratorController(config)


def create_datacenter_accelerator(
    array_size: tuple[int, int] = (32, 32),
) -> AcceleratorController:
    """Create accelerator optimized for datacenter AI applications."""
    from ..utils.config import AcceleratorConfig, ArrayConfig, DataType

    config = AcceleratorConfig(
        name="DatacenterAI_Accelerator",
        array=ArrayConfig(
            rows=array_size[0],
            cols=array_size[1],
            frequency=2.0e9,  # High frequency for performance
            voltage=1.2,  # Higher voltage for performance
        ),
        data_type=DataType.FLOAT32,
        enable_power_modeling=True,
        enable_thermal_modeling=True,
    )

    return AcceleratorController(config)
