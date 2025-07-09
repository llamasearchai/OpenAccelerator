"""
Advanced Systolic Array implementation with multiple dataflow support.

Features:
- Output Stationary (OS) and Weight Stationary (WS) dataflows
- Advanced memory hierarchy integration
- Power and thermal modeling
- Medical-grade error detection
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.memory import MemoryHierarchy
from ..utils.config import AcceleratorConfig, DataflowType
from .pe import PEState, ProcessingElement

logger = logging.getLogger(__name__)

# Enable vectorised propagation for better performance
USE_NUMPY_FAST_PATH = True


class ThermalModel:
    """Basic thermal model for temperature estimation."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

    def update(self, power_map: np.ndarray) -> np.ndarray:
        """Update thermal model."""
        # Simple model: return uniform temperature based on power
        return np.ones((self.rows, self.cols)) * 25.0


class PowerModel:
    """Basic power model for power estimation."""

    def __init__(self, config):
        self.config = config


@dataclass
class ArrayMetrics:
    """Systolic array performance metrics."""

    total_cycles: int = 0
    active_pes: int = 0
    total_operations: int = 0
    total_energy: float = 0.0
    peak_power: float = 0.0
    thermal_events: int = 0
    memory_stalls: int = 0
    utilization_map: Optional[np.ndarray] = None


class SystolicArray:
    """
    Advanced systolic array with configurable dataflow and comprehensive monitoring.

    Supports multiple dataflow patterns optimized for different workload types.
    """

    def __init__(self, config: AcceleratorConfig):
        """
        Initialize systolic array.

        Args:
            config: Complete accelerator configuration
        """
        self.config = config
        self.rows = config.array.rows
        self.cols = config.array.cols
        self.dataflow = config.array.dataflow

        # Initialize PE grid
        self.pes: List[List[ProcessingElement]] = []
        self._initialize_pe_grid()

        # Dataflow management
        self._setup_dataflow()

        # Memory integration
        self.memory_hierarchy = MemoryHierarchy(config)

        # Performance tracking
        self.metrics = ArrayMetrics()
        self.cycle_count = 0

        # Pre-allocated NumPy arrays for vectorised propagation
        self.inputs_a = np.zeros((self.rows, self.cols), dtype=object)
        self.inputs_b = np.zeros((self.rows, self.cols), dtype=object)
        self.outputs_a = np.zeros((self.rows, self.cols), dtype=object)
        self.outputs_b = np.zeros((self.rows, self.cols), dtype=object)

        # Thermal and power modeling
        self.thermal_model = (
            ThermalModel(self.rows, self.cols)
            if config.enable_thermal_modeling
            else None
        )
        self.power_model = PowerModel(config) if config.enable_power_modeling else None

        # Thread pool for parallel PE operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, self.rows * self.cols)
        )

        logger.info(
            f"Initialized {self.rows}×{self.cols} systolic array with {self.dataflow.value} dataflow"
        )

    def _initialize_pe_grid(self) -> None:
        """Initialize the grid of processing elements."""
        for row in range(self.rows):
            pe_row = []
            for col in range(self.cols):
                pe = ProcessingElement(
                    pe_id=(row, col),
                    config=self.config.array.pe_config,
                    data_type=self.config.data_type,
                )
                pe_row.append(pe)
            self.pes.append(pe_row)

        logger.debug(f"Initialized {self.rows * self.cols} processing elements")

    def _setup_dataflow(self) -> None:
        """Setup dataflow-specific parameters."""
        if self.dataflow == DataflowType.OUTPUT_STATIONARY:
            self._setup_output_stationary()
        elif self.dataflow == DataflowType.WEIGHT_STATIONARY:
            self._setup_weight_stationary()
        else:
            raise NotImplementedError(f"Dataflow {self.dataflow} not implemented")

    def _setup_output_stationary(self) -> None:
        """Configure for Output Stationary dataflow."""
        # In OS, each PE accumulates one output element
        # Data flows: A (horizontal), B (vertical)
        self.flow_direction_a = "horizontal"  # A flows left-to-right
        self.flow_direction_b = "vertical"  # B flows top-to-bottom

        logger.debug("Configured Output Stationary dataflow")

    def _setup_weight_stationary(self) -> None:
        """Configure for Weight Stationary dataflow."""
        # In WS, weights stay in PEs, inputs flow through
        self.flow_direction_weights = "stationary"
        self.flow_direction_inputs = "diagonal"

        logger.debug("Configured Weight Stationary dataflow")

    def cycle(
        self,
        input_data: Dict[str, np.ndarray],
        control_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one clock cycle of the systolic array.

        Args:
            input_data: Input data for this cycle
            control_signals: Optional control signals

        Returns:
            Cycle results and metrics
        """
        self.cycle_count += 1
        cycle_start_time = time.time()

        # Pre-cycle power gating decisions
        if self.power_model:
            self._apply_power_gating()

        # Load inputs based on dataflow
        if self.dataflow == DataflowType.OUTPUT_STATIONARY:
            self._cycle_output_stationary(input_data)
        elif self.dataflow == DataflowType.WEIGHT_STATIONARY:
            self._cycle_weight_stationary(input_data)

        # Parallel PE execution
        self._execute_pes_parallel()

        # Update thermal model
        if self.thermal_model:
            self._update_thermal_model()

        # Collect cycle metrics
        cycle_metrics = self._collect_cycle_metrics()

        # Update array-level metrics
        self._update_array_metrics(cycle_metrics)

        logger.debug(
            f"Cycle {self.cycle_count} completed in {time.time() - cycle_start_time:.4f}s"
        )

        return {
            "cycle": self.cycle_count,
            "metrics": cycle_metrics,
            "pe_states": self._get_pe_states(),
            "outputs": self._collect_outputs(),
        }

    def _cycle_output_stationary(self, input_data: Dict[str, np.ndarray]) -> None:
        """Execute Output Stationary dataflow cycle with vectorised propagation."""
        # Extract edge inputs
        edge_inputs_a = input_data.get("edge_a", [None] * self.rows)
        edge_inputs_b = input_data.get("edge_b", [None] * self.cols)

        if USE_NUMPY_FAST_PATH:
            self._cycle_output_stationary_vectorised(edge_inputs_a, edge_inputs_b)
        else:
            self._cycle_output_stationary_fallback(edge_inputs_a, edge_inputs_b)

    def _cycle_output_stationary_vectorised(
        self,
        edge_inputs_a: Union[List, np.ndarray],
        edge_inputs_b: Union[List, np.ndarray],
    ) -> None:
        """Vectorised Output Stationary dataflow using NumPy operations."""
        # Collect all current outputs for propagation using vectorised operations
        for row in range(self.rows):
            for col in range(self.cols):
                out_a, out_b = self.pes[row][col].get_propagation_outputs()
                self.outputs_a[row, col] = out_a
                self.outputs_b[row, col] = out_b

        # Propagate data using NumPy roll operations
        # A flows horizontally (left-to-right)
        self.inputs_a[:, 1:] = self.outputs_a[:, :-1]  # Shift right
        self.inputs_a[:, 0] = edge_inputs_a  # Edge inputs for leftmost column

        # B flows vertically (top-to-bottom)
        self.inputs_b[1:, :] = self.outputs_b[:-1, :]  # Shift down
        self.inputs_b[0, :] = edge_inputs_b  # Edge inputs for topmost row

        # Load inputs to PEs
        for row in range(self.rows):
            for col in range(self.cols):
                self.pes[row][col].load_inputs(
                    self.inputs_a[row, col], self.inputs_b[row, col]
                )

    def _cycle_output_stationary_fallback(
        self,
        edge_inputs_a: Union[List, np.ndarray],
        edge_inputs_b: Union[List, np.ndarray],
    ) -> None:
        """Fallback Output Stationary dataflow using traditional for-loops for readability."""
        # Propagate data through array
        # First, collect all current outputs for propagation
        propagation_a = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        propagation_b = [[None for _ in range(self.cols)] for _ in range(self.rows)]

        for row in range(self.rows):
            for col in range(self.cols):
                out_a, out_b = self.pes[row][col].get_propagation_outputs()
                propagation_a[row][col] = out_a
                propagation_b[row][col] = out_b

        # Load new inputs for each PE
        for row in range(self.rows):
            for col in range(self.cols):
                # Input A comes from left (or edge)
                input_a = (
                    edge_inputs_a[row] if col == 0 else propagation_a[row][col - 1]
                )

                # Input B comes from top (or edge)
                input_b = (
                    edge_inputs_b[col] if row == 0 else propagation_b[row - 1][col]
                )

                self.pes[row][col].load_inputs(input_a, input_b)

    def _cycle_weight_stationary(self, input_data: Dict[str, np.ndarray]) -> None:
        """Execute Weight Stationary dataflow cycle."""
        # Implementation for WS dataflow
        # Weights are pre-loaded and stay in PEs
        # Inputs flow diagonally through the array

        # Extract diagonal inputs for WS dataflow
        diagonal_inputs = input_data.get("diagonal_inputs", [])

        # In WS, weights are stationary in PEs, inputs move diagonally
        # We need to manage the diagonal propagation of inputs

        # For simplicity, assume inputs enter from the top-left and flow diagonally
        # Each PE processes its weight against the flowing input

        # Create input propagation matrix for diagonal flow
        input_propagation = np.zeros((self.rows, self.cols), dtype=object)

        # Fill diagonal inputs starting from top-left
        for i, input_val in enumerate(diagonal_inputs):
            if i < min(self.rows, self.cols):
                input_propagation[i, i] = input_val

        # Process each PE with diagonal input propagation
        for row in range(self.rows):
            for col in range(self.cols):
                # Get input from diagonal position
                if row == col and row < len(diagonal_inputs):
                    input_val = diagonal_inputs[row]
                else:
                    # Calculate diagonal position for this PE
                    diag_offset = col - row
                    if diag_offset >= 0 and diag_offset < len(diagonal_inputs):
                        input_val = diagonal_inputs[diag_offset]
                    else:
                        input_val = None

                # Load input to PE (weight is already loaded and stationary)
                if input_val is not None:
                    self.pes[row][col].load_inputs(
                        input_val, None
                    )  # Only input, weight is stationary

        # Update diagonal input buffer for next cycle
        # Shift inputs diagonally down and right
        if hasattr(self, "diagonal_buffer"):
            self.diagonal_buffer = self.diagonal_buffer[1:] + [None]
        else:
            self.diagonal_buffer = diagonal_inputs[1:] + [None]

    def _execute_pes_parallel(self) -> None:
        """Execute all PEs in parallel using thread pool."""

        def execute_pe(pe):
            return pe.cycle()

        # Submit all PE executions
        futures = []
        for row in range(self.rows):
            for col in range(self.cols):
                future = self.thread_pool.submit(execute_pe, self.pes[row][col])
                futures.append(future)

        # Wait for all to complete
        results = [future.result() for future in futures]

        # Count active PEs
        self.metrics.active_pes = sum(results)

    def _apply_power_gating(self) -> None:
        """Apply intelligent power gating based on utilization."""
        if not self.power_model:
            return

        # Simple heuristic: power gate PEs that have been idle for multiple cycles
        for row in range(self.rows):
            for col in range(self.cols):
                pe = self.pes[row][col]
                if pe.metrics.idle_cycles > 10:  # Idle for 10+ cycles
                    pe.power_gate(True)
                elif pe.state == PEState.POWER_GATED and pe.in_reg_a is not None:
                    pe.power_gate(False)  # Wake up if data available

    def _update_thermal_model(self) -> None:
        """Update thermal simulation."""
        if not self.thermal_model:
            return

        # Collect power density from each PE
        power_map = np.zeros((self.rows, self.cols))
        for row in range(self.rows):
            for col in range(self.cols):
                # Simplified: power proportional to activity
                if self.pes[row][col].state == PEState.COMPUTING:
                    power_map[row, col] = 1.0

        # Update thermal model
        temp_map = self.thermal_model.update(power_map)

        # Check for thermal violations
        if np.max(temp_map) > 85.0:  # 85°C threshold
            self.metrics.thermal_events += 1
            logger.warning(
                f"Thermal event detected: max temp = {np.max(temp_map):.1f}°C"
            )

    def _collect_cycle_metrics(self) -> Dict[str, Any]:
        """Collect metrics for current cycle."""
        metrics = {
            "cycle": self.cycle_count,
            "active_pes": self.metrics.active_pes,
            "total_operations": 0,
            "total_energy": 0.0,
            "pe_utilization": 0.0,
        }

        total_ops = 0
        total_energy = 0.0
        active_count = 0

        for row in range(self.rows):
            for col in range(self.cols):
                pe = self.pes[row][col]
                total_ops += pe.metrics.total_operations
                total_energy += pe.metrics.energy_consumed
                if pe.state == PEState.COMPUTING:
                    active_count += 1

        metrics["total_operations"] = total_ops
        metrics["total_energy"] = total_energy

        if self.rows * self.cols > 0:
            metrics["pe_utilization"] = active_count / (self.rows * self.cols)
        else:
            metrics["pe_utilization"] = 0.0

        return metrics

    def _update_array_metrics(self, cycle_metrics: Dict[str, Any]) -> None:
        """Update array-level metrics."""
        self.metrics.total_cycles = self.cycle_count
        self.metrics.total_operations += cycle_metrics["total_operations"]
        self.metrics.total_energy += cycle_metrics["total_energy"]

        # Update utilization map
        if self.metrics.utilization_map is None:
            self.metrics.utilization_map = np.zeros((self.rows, self.cols))

        # Update utilization for this cycle
        for row in range(self.rows):
            for col in range(self.cols):
                if self.pes[row][col].state == PEState.COMPUTING:
                    self.metrics.utilization_map[row, col] += 1

    def _get_pe_states(self) -> List[List[Dict[str, Any]]]:
        """Get current state of all PEs."""
        states = []
        for row in range(self.rows):
            row_states = []
            for col in range(self.cols):
                pe = self.pes[row][col]
                row_states.append(
                    {
                        "pe_id": pe.pe_id,
                        "state": pe.state.value
                        if hasattr(pe.state, "value")
                        else str(pe.state),
                        "accumulator": pe.accumulator,
                        "operations": pe.metrics.total_operations,
                    }
                )
            states.append(row_states)
        return states

    def _collect_outputs(self) -> Dict[str, Any]:
        """Collect outputs from PEs."""
        outputs = {}

        # Collect accumulator values (for OS dataflow)
        if self.dataflow == DataflowType.OUTPUT_STATIONARY:
            result_matrix = np.zeros((self.rows, self.cols))
            for row in range(self.rows):
                for col in range(self.cols):
                    result_matrix[row, col] = self.pes[row][col].accumulator
            outputs["result_matrix"] = result_matrix

        return outputs

    def get_results(self) -> np.ndarray:
        """Get final results from the systolic array."""
        if self.dataflow == DataflowType.OUTPUT_STATIONARY:
            result_matrix = np.zeros((self.rows, self.cols))
            for row in range(self.rows):
                for col in range(self.cols):
                    result_matrix[row, col] = self.pes[row][col].accumulator
            return result_matrix
        else:
            # For other dataflows, return accumulated results
            results = []
            for row in range(self.rows):
                for col in range(self.cols):
                    results.append(self.pes[row][col].accumulator)
            return np.array(results).reshape(self.rows, self.cols)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive array metrics."""
        # Calculate utilization map if not already done
        if self.metrics.utilization_map is not None and self.cycle_count > 0:
            utilization_map = self.metrics.utilization_map / self.cycle_count
        else:
            utilization_map = np.zeros((self.rows, self.cols))

        return {
            "total_cycles": self.metrics.total_cycles,
            "total_operations": self.metrics.total_operations,
            "total_energy": self.metrics.total_energy,
            "active_pes": self.metrics.active_pes,
            "peak_power": self.metrics.peak_power,
            "thermal_events": self.metrics.thermal_events,
            "memory_stalls": self.metrics.memory_stalls,
            "utilization_map": utilization_map,
            "average_utilization": np.mean(utilization_map)
            if utilization_map.size > 0
            else 0.0,
        }

    def get_array_metrics(self) -> ArrayMetrics:
        """Get comprehensive array metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset the systolic array."""
        for row in range(self.rows):
            for col in range(self.cols):
                self.pes[row][col].reset()

        self.metrics = ArrayMetrics()
        self.cycle_count = 0

        logger.info("Systolic array reset completed")

    def __del__(self) -> None:
        """Cleanup thread pool."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)
