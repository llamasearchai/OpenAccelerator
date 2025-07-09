"""
Advanced Processing Element (PE) implementation.

Features:
- Multiple data types support
- Sparsity handling
- Power modeling
- Medical-grade precision
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np

from ..utils.config import DataType, PEConfig

logger = logging.getLogger(__name__)


class PEState(Enum):
    """Processing Element operational states."""

    IDLE = "idle"
    COMPUTING = "computing"
    POWER_GATED = "power_gated"
    ERROR = "error"


@dataclass
class PEMetrics:
    """Performance and power metrics for a PE."""

    total_operations: int = 0
    active_cycles: int = 0
    idle_cycles: int = 0
    power_gated_cycles: int = 0
    energy_consumed: float = 0.0  # in pJ
    sparsity_detected: int = 0
    errors_detected: int = 0


class ProcessingElement:
    """
    Advanced Processing Element with medical-grade precision and power modeling.

    Supports multiple data types, sparsity detection, and comprehensive metrics.
    """

    def __init__(
        self,
        pe_id: tuple[int, int],
        config: PEConfig,
        data_type: DataType = DataType.FLOAT32,
    ):
        """
        Initialize Processing Element.

        Args:
            pe_id: (row, col) position in systolic array
            config: PE configuration parameters
            data_type: Numerical data type for computations
        """
        self.pe_id = pe_id
        self.config = config
        self.data_type = data_type

        # Initialize numerical type
        self._setup_data_type()

        # PE state
        self.state = PEState.IDLE
        self.accumulator = self.zero_value

        # Input/output registers
        self.input_a: Optional[Any] = None
        self.input_b: Optional[Any] = None
        self.output_a: Optional[Any] = None
        self.output_b: Optional[Any] = None

        # Metrics tracking
        self.metrics = PEMetrics()

        # Medical mode features
        self.medical_mode = (
            config.enable_sparsity if hasattr(config, "enable_sparsity") else False
        )
        self.precision_guard = True  # Always verify precision in medical mode

        logger.debug(f"Initialized PE {pe_id} with {data_type.value} precision")

    def _setup_data_type(self):
        """Configure numerical data type and associated parameters."""
        type_map = {
            DataType.INT8: (np.int8, 0, 1e-6),
            DataType.INT16: (np.int16, 0, 1e-6),
            DataType.INT32: (np.int32, 0, 1e-6),
            DataType.FLOAT16: (np.float16, 0.0, 1e-4),
            DataType.FLOAT32: (np.float32, 0.0, 1e-6),
            DataType.BFLOAT16: (np.float32, 0.0, 1e-3),  # Approximate as float32
        }

        self.numpy_type, self.zero_value, self.precision_threshold = type_map[
            self.data_type
        ]

        # Configure accumulator precision (always higher than input)
        if self.data_type in [DataType.INT8, DataType.INT16]:
            self.accumulator_type = np.int32
        elif self.data_type == DataType.INT32:
            self.accumulator_type = np.int64
        else:  # Float types
            self.accumulator_type = np.float64 if self.medical_mode else np.float32

    def load_inputs(self, input_a: Optional[Any], input_b: Optional[Any]) -> None:
        """
        Load input values for computation.

        Args:
            input_a: First operand (from north or west neighbor)
            input_b: Second operand (from west or north neighbor)
        """
        self.input_a = self._convert_input(input_a)
        self.input_b = self._convert_input(input_b)

        # Prepare outputs for propagation (same as inputs in systolic flow)
        self.output_a = self.input_a
        self.output_b = self.input_b

    def _convert_input(self, value: Optional[Any]) -> Optional[Any]:
        """Convert input to appropriate data type with error checking."""
        if value is None:
            return None

        try:
            # Convert to target type
            converted = self.numpy_type(value)

            # Medical mode: verify no precision loss
            if self.medical_mode and self.precision_guard:
                if abs(float(converted) - float(value)) > self.precision_threshold:
                    logger.warning(
                        f"PE {self.pe_id}: Precision loss detected in conversion"
                    )
                    self.metrics.errors_detected += 1

            return converted

        except (ValueError, OverflowError) as e:
            logger.error(f"PE {self.pe_id}: Input conversion error: {e}")
            self.metrics.errors_detected += 1
            self.state = PEState.ERROR
            return None

    def cycle(self) -> bool:
        """
        Execute one computation cycle.

        Returns:
            True if computation was performed, False if idle
        """
        if self.state == PEState.POWER_GATED:
            self.metrics.power_gated_cycles += 1
            return False

        if self.state == PEState.ERROR:
            return False

        # Check if we have valid inputs for computation
        if self.input_a is not None and self.input_b is not None:
            return self._perform_mac()
        else:
            self.state = PEState.IDLE
            self.metrics.idle_cycles += 1
            return False

    def _perform_mac(self) -> bool:
        """
        Perform Multiply-Accumulate operation.

        Returns:
            True if MAC was performed successfully
        """
        try:
            # Check for sparsity
            if self.config.enable_sparsity:
                if (
                    abs(self.input_a) < self.config.sparsity_threshold
                    or abs(self.input_b) < self.config.sparsity_threshold
                ):
                    self.metrics.sparsity_detected += 1
                    self.state = PEState.IDLE
                    self.metrics.idle_cycles += 1
                    return False

            # Perform MAC with higher precision accumulator
            product = self.accumulator_type(self.input_a) * self.accumulator_type(
                self.input_b
            )
            self.accumulator = self.accumulator_type(self.accumulator) + product

            # Update metrics
            self.metrics.total_operations += 1
            self.metrics.active_cycles += 1
            self.state = PEState.COMPUTING

            # Power consumption (simplified model)
            self.metrics.energy_consumed += self._compute_energy()

            logger.debug(
                f"PE {self.pe_id}: MAC completed, accumulator = {self.accumulator}"
            )
            return True

        except Exception as e:
            logger.error(f"PE {self.pe_id}: MAC operation failed: {e}")
            self.state = PEState.ERROR
            self.metrics.errors_detected += 1
            return False

    def _compute_energy(self) -> float:
        """
        Compute energy consumption for current operation.

        Returns:
            Energy in picojoules
        """
        # Simplified power model based on data type and operation
        base_energy = {
            DataType.INT8: 0.1,
            DataType.INT16: 0.2,
            DataType.INT32: 0.5,
            DataType.FLOAT16: 0.8,
            DataType.FLOAT32: 1.5,
            DataType.BFLOAT16: 1.2,
        }

        energy = base_energy.get(self.data_type, 1.0)

        # Scale by voltage squared (P ∝ V²f)
        voltage_scale = (1.2 / 1.0) ** 2  # Assuming 1.2V vs 1.0V nominal
        energy *= voltage_scale

        return energy

    def get_output(self) -> Any:
        """
        Get final accumulated result.

        Returns:
            Final accumulator value converted to output data type
        """
        try:
            # Convert back to original data type for output
            if self.data_type in [
                DataType.FLOAT16,
                DataType.FLOAT32,
                DataType.BFLOAT16,
            ]:
                return self.numpy_type(self.accumulator)
            else:
                # For integer types, ensure no overflow
                info = np.iinfo(self.numpy_type)
                clamped = np.clip(self.accumulator, info.min, info.max)
                if clamped != self.accumulator:
                    logger.warning(f"PE {self.pe_id}: Output clamped due to overflow")
                    self.metrics.errors_detected += 1
                return self.numpy_type(clamped)
        except Exception as e:
            logger.error(f"PE {self.pe_id}: Output conversion failed: {e}")
            return self.zero_value

    def get_propagation_outputs(self) -> tuple[Optional[Any], Optional[Any]]:
        """
        Get values to propagate to neighboring PEs.

        Returns:
            Tuple of (output_a, output_b) for systolic flow
        """
        return self.output_a, self.output_b

    def power_gate(self, enable: bool = True) -> None:
        """
        Enable/disable power gating for energy efficiency.

        Args:
            enable: True to power gate, False to wake up
        """
        if enable and self.config.power_gating:
            self.state = PEState.POWER_GATED
            logger.debug(f"PE {self.pe_id}: Power gated")
        else:
            self.state = PEState.IDLE
            logger.debug(f"PE {self.pe_id}: Woke from power gating")

    def reset(self) -> None:
        """Reset PE state for new computation."""
        self.state = PEState.IDLE
        self.accumulator = self.zero_value
        self.input_a = None
        self.input_b = None
        self.output_a = None
        self.output_b = None

        # Reset metrics but preserve cumulative counters
        previous_total = self.metrics.total_operations
        previous_energy = self.metrics.energy_consumed
        self.metrics = PEMetrics()
        self.metrics.total_operations = previous_total
        self.metrics.energy_consumed = previous_energy

    def get_utilization(self) -> float:
        """
        Calculate PE utilization percentage.

        Returns:
            Utilization as fraction (0.0 to 1.0)
        """
        total_cycles = (
            self.metrics.active_cycles
            + self.metrics.idle_cycles
            + self.metrics.power_gated_cycles
        )

        if total_cycles == 0:
            return 0.0

        return self.metrics.active_cycles / total_cycles

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary of performance and power metrics
        """
        return {
            "pe_id": self.pe_id,
            "state": self.state.value,
            "utilization": self.get_utilization(),
            "total_operations": self.metrics.total_operations,
            "energy_consumed": self.metrics.energy_consumed,
            "errors_detected": self.metrics.errors_detected,
            "sparsity_detected": self.metrics.sparsity_detected,
            "data_type": self.data_type.value,
        }

    def __repr__(self) -> str:
        return (
            f"PE{self.pe_id}[{self.state.value}, ops={self.metrics.total_operations}]"
        )
