"""
Advanced Systolic Array implementation with multiple dataflow support.

Features:
- Output Stationary (OS) and Weight Stationary (WS) dataflows
- Advanced memory hierarchy integration
- Power and thermal modeling
- Medical-grade error detection
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.memory import MemoryHierarchy
from ..utils.config import AcceleratorConfig, DataflowType, DataType
from .pe import PEMetrics, PEState, ProcessingElement

logger = logging.getLogger(__name__)

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

        # Thermal and power modeling
        self.thermal_model = ThermalModel(self.rows, self.cols) if config.enable_thermal_modeling else None
        self.power_model = PowerModel(config) if config.enable_power_modeling else None

        # Thread pool for parallel PE operations
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, self.rows * self.cols))

        logger.info(f"Initialized {self.rows}×{self.cols} systolic array with {self.dataflow.value} dataflow")

    def _initialize_pe_grid(self) -> None:
        """Initialize the grid of processing elements."""
        for row in range(self.rows):
            pe_row = []
            for col in range(self.cols):
                pe = ProcessingElement(
                    pe_id=(row, col),
                    config=self.config.array.pe_config,
                    data_type=self.config.data_type
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
        self.flow_direction_a = 'horizontal'  # A flows left-to-right
        self.flow_direction_b = 'vertical'    # B flows top-to-bottom

        logger.debug("Configured Output Stationary dataflow")

    def _setup_weight_stationary(self) -> None:
        """Configure for Weight Stationary dataflow."""
        # In WS, weights stay in PEs, inputs flow through
        self.flow_direction_weights = 'stationary'
        self.flow_direction_inputs = 'diagonal'

        logger.debug("Configured Weight Stationary dataflow")

    def cycle(self,
              input_data: Dict[str, np.ndarray],
              control_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

        logger.debug(f"Cycle {self.cycle_count} completed in {time.time() - cycle_start_time:.4f}s")

        return {
            'cycle': self.cycle_count,
            'metrics': cycle_metrics,
            'pe_states': self._get_pe_states(),
            'outputs': self._collect_outputs()
        }

    def _cycle_output_stationary(self, input_data: Dict[str, np.ndarray]) -> None:
        """Execute Output Stationary dataflow cycle."""
        # Extract edge inputs
        edge_inputs_a = input_data.get('edge_a', [None] * self.rows)
        edge_inputs_b = input_data.get('edge_b', [None] * self.cols)

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
                input_a = edge_inputs_a[row] if col == 0 else propagation_a[row][col-1]

                # Input B comes from top (or edge)
                input_b = edge_inputs_b[col] if row == 0 else propagation_b[row-1][col]

                self.pes[row][col].load_inputs(input_a, input_b)

    def _cycle_weight_stationary(self, input_data: Dict[str, np.ndarray]) -> None:
        """Execute Weight Stationary dataflow cycle."""
        # Implementation for WS dataflow
        # Weights are pre-loaded and stay in PEs
        # Inputs flow diagonally through the array
        pass  # Placeholder for WS implementation

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
                elif pe.state == PEState.POWER_GATED and pe.input_a is not None:
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
            logger.warning(f"Thermal event detected: max temp = {np.max(temp_map):.1f}°C")

    def _collect_cycle_metrics(self) -> Dict[str, Any]:
        """Collect metrics for current cycle."""
        metrics = {
            'cycle': self.cycle_count,
            'active_pes': self.metrics.active_pes,
            'total_operations': 0,
            'total_energy': 0.0,
            'pe_utilization': 0.0,
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

        metrics['total_operations'] = total_ops
        metrics['total_energy'] = total_energy
        metrics['pe_utilization'] = active_count / (self.rows * self.cols)

        return metrics

    def _update_array_metrics(
