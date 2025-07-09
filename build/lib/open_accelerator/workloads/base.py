"""
Base workload classes and interfaces.

Provides abstract base classes and common functionality for all workload types
including GEMM, convolution, and medical AI workloads.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class WorkloadType(Enum):
    """Types of supported workloads."""

    GEMM = "gemm"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    MEDICAL_IMAGING = "medical_imaging"
    NEURAL_NETWORK = "neural_network"
    CUSTOM = "custom"


class DataLayout(Enum):
    """Data layout formats."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    NHWC = "nhwc"  # Batch, Height, Width, Channels
    NCHW = "nchw"  # Batch, Channels, Height, Width
    BLOCKED = "blocked"


@dataclass
class WorkloadRequirements:
    """Requirements and constraints for workload execution."""

    required_array_size: Optional[Tuple[int, int]] = None
    memory_requirements: Dict[str, int] = field(default_factory=dict)
    data_type_requirements: Optional[str] = None
    precision_requirements: Optional[str] = None
    latency_constraints: Optional[float] = None
    throughput_requirements: Optional[float] = None
    power_constraints: Optional[float] = None
    medical_compliance: bool = False


@dataclass
class WorkloadConfig:
    """Base configuration for all workloads."""

    name: str = "base_workload"
    workload_type: WorkloadType = WorkloadType.CUSTOM
    data_layout: DataLayout = DataLayout.ROW_MAJOR
    batch_size: int = 1
    precision: str = "float32"
    seed: int = 42

    # Performance hints
    enable_sparsity: bool = False
    sparsity_ratio: float = 0.0
    enable_quantization: bool = False
    quantization_bits: int = 8

    # Medical/safety requirements
    medical_grade: bool = False
    safety_critical: bool = False
    regulatory_compliance: bool = False


class BaseWorkload(ABC):
    """
    Abstract base class for all workloads.

    Defines the interface that all workloads must implement for execution
    on the accelerator simulator.
    """

    def __init__(self, config: WorkloadConfig):
        """
        Initialize base workload.

        Args:
            config: Workload configuration
        """
        self.config = config
        self.name = config.name
        self.workload_type = config.workload_type

        # Data storage
        self.input_data: Dict[str, np.ndarray] = {}
        self.output_data: Dict[str, np.ndarray] = {}
        self.intermediate_data: Dict[str, np.ndarray] = {}

        # Execution state
        self.is_generated = False
        self.is_validated = False
        self.execution_operations: List[Dict[str, Any]] = []

        # Metrics
        self.generation_time: float = 0.0
        self.validation_time: float = 0.0

        logger.debug(f"Initialized {self.workload_type.value} workload: {self.name}")

    @abstractmethod
    def generate_data(self, seed: Optional[int] = None) -> None:
        """
        Generate input data for the workload.

        Args:
            seed: Random seed for reproducible data generation
        """
        pass

    @abstractmethod
    def get_operations(self) -> List[Dict[str, Any]]:
        """
        Get the sequence of operations to execute on the accelerator.

        Returns:
            List of operation dictionaries with type, data, and parameters
        """
        pass

    @abstractmethod
    def validate_output(self, computed_output: np.ndarray) -> bool:
        """
        Validate computed output against expected results.

        Args:
            computed_output: Output computed by the accelerator

        Returns:
            True if output is valid
        """
        pass

    def get_name(self) -> str:
        """Get workload name."""
        return self.name

    def get_type(self) -> WorkloadType:
        """Get workload type."""
        return self.workload_type

    def get_requirements(self) -> WorkloadRequirements:
        """
        Get workload requirements and constraints.

        Returns:
            WorkloadRequirements object
        """
        # Default implementation - subclasses should override
        return WorkloadRequirements()

    def get_input_data(self) -> Dict[str, np.ndarray]:
        """Get input data for the workload."""
        if not self.is_generated:
            raise RuntimeError(
                "Workload data not generated. Call generate_data() first."
            )
        return self.input_data.copy()

    def get_expected_output(self) -> Dict[str, np.ndarray]:
        """Get expected output data for validation."""
        if not self.is_generated:
            raise RuntimeError(
                "Workload data not generated. Call generate_data() first."
            )
        return self.output_data.copy()

    def get_workload_info(self) -> Dict[str, Any]:
        """Get comprehensive workload information."""
        return {
            "name": self.name,
            "type": self.workload_type.value,
            "config": self.config.__dict__,
            "requirements": self.get_requirements().__dict__,
            "data_shapes": {name: data.shape for name, data in self.input_data.items()},
            "is_generated": self.is_generated,
            "is_validated": self.is_validated,
            "operation_count": len(self.execution_operations),
        }

    def estimate_complexity(self) -> Dict[str, Any]:
        """
        Estimate computational complexity of the workload.

        Returns:
            Dictionary with complexity metrics
        """
        total_ops = 0
        memory_footprint = 0

        # Calculate from input data
        for data in self.input_data.values():
            memory_footprint += data.nbytes

        # Default complexity estimation - subclasses should override
        complexity = {
            "total_operations": total_ops,
            "memory_footprint_bytes": memory_footprint,
            "computational_intensity": 0.0,  # ops per byte
            "parallelism_potential": 1.0,  # 0.0 to 1.0
            "memory_access_pattern": "sequential",  # sequential, random, structured
        }

        if memory_footprint > 0:
            complexity["computational_intensity"] = total_ops / memory_footprint

        return complexity

    def apply_transformations(self, transformations: List[str]) -> None:
        """
        Apply data transformations to the workload.

        Args:
            transformations: List of transformation names to apply
        """
        for transform in transformations:
            if transform == "normalize":
                self._normalize_data()
            elif transform == "quantize":
                self._quantize_data()
            elif transform == "add_noise":
                self._add_noise()
            elif transform == "sparsify":
                self._sparsify_data()
            else:
                logger.warning(f"Unknown transformation: {transform}")

    def _normalize_data(self):
        """Normalize input data to [0, 1] range."""
        for name, data in self.input_data.items():
            if data.dtype in [np.float32, np.float64]:
                data_min = np.min(data)
                data_max = np.max(data)
                if data_max > data_min:
                    self.input_data[name] = (data - data_min) / (data_max - data_min)

    def _quantize_data(self):
        """Quantize data based on configuration."""
        if not self.config.enable_quantization:
            return

        bits = self.config.quantization_bits
        max_val = 2 ** (bits - 1) - 1
        min_val = -(2 ** (bits - 1))

        for name, data in self.input_data.items():
            if data.dtype in [np.float32, np.float64]:
                # Scale to quantization range
                data_min = np.min(data)
                data_max = np.max(data)
                if data_max > data_min:
                    scaled = (data - data_min) / (data_max - data_min)
                    quantized = np.round(scaled * (max_val - min_val) + min_val)
                    self.input_data[name] = np.clip(quantized, min_val, max_val).astype(
                        np.int32
                    )

    def _add_noise(self, noise_level: float = 0.01):
        """Add Gaussian noise to input data."""
        np.random.seed(self.config.seed)
        for name, data in self.input_data.items():
            if data.dtype in [np.float32, np.float64]:
                noise = np.random.normal(0, noise_level * np.std(data), data.shape)
                self.input_data[name] = data + noise.astype(data.dtype)

    def _sparsify_data(self):
        """Apply sparsity to input data."""
        if not self.config.enable_sparsity or self.config.sparsity_ratio <= 0:
            return

        np.random.seed(self.config.seed)
        for name, data in self.input_data.items():
            # Create sparsity mask
            mask = np.random.random(data.shape) > self.config.sparsity_ratio
            self.input_data[name] = data * mask

    def save_data(self, filepath: str) -> None:
        """
        Save workload data to file.

        Args:
            filepath: Path to save data
        """
        try:
            save_data = {
                "config": self.config.__dict__,
                "input_data": {name: data for name, data in self.input_data.items()},
                "output_data": {name: data for name, data in self.output_data.items()},
                "workload_info": self.get_workload_info(),
            }
            np.savez_compressed(filepath, **save_data)
            logger.info(f"Workload data saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save workload data: {e}")
            raise

    def load_data(self, filepath: str) -> None:
        """
        Load workload data from file.

        Args:
            filepath: Path to load data from
        """
        try:
            loaded_data = np.load(filepath, allow_pickle=True)

            # Restore input and output data
            if "input_data" in loaded_data:
                self.input_data = loaded_data["input_data"].item()
            if "output_data" in loaded_data:
                self.output_data = loaded_data["output_data"].item()

            self.is_generated = True
            logger.info(f"Workload data loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load workload data: {e}")
            raise

    def __str__(self) -> str:
        """String representation of workload."""
        return f"{self.workload_type.value}_{self.name}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type={self.workload_type.value}, "
            f"generated={self.is_generated})"
        )


class MatrixWorkload(BaseWorkload):
    """
    Base class for matrix-based workloads (GEMM, convolution, etc.).

    Provides common functionality for workloads that operate on matrices.
    """

    def __init__(self, config: WorkloadConfig):
        """Initialize matrix workload."""
        super().__init__(config)

        # Matrix dimensions (to be set by subclasses)
        self.matrix_dimensions: Dict[str, Tuple[int, ...]] = {}

        # Matrix data
        self.matrices: Dict[str, np.ndarray] = {}

        # Operation metadata
        self.operation_type = "matrix_operation"
        self.flop_count = 0

    def get_matrix_info(self) -> Dict[str, Any]:
        """Get information about matrices in the workload."""
        return {
            "dimensions": self.matrix_dimensions,
            "shapes": {name: matrix.shape for name, matrix in self.matrices.items()},
            "dtypes": {
                name: str(matrix.dtype) for name, matrix in self.matrices.items()
            },
            "memory_usage": {
                name: matrix.nbytes for name, matrix in self.matrices.items()
            },
            "total_memory": sum(matrix.nbytes for matrix in self.matrices.values()),
            "flop_count": self.flop_count,
        }

    def estimate_complexity(self) -> Dict[str, Any]:
        """Estimate complexity for matrix workloads."""
        complexity = super().estimate_complexity()

        # Update with matrix-specific metrics
        complexity.update(
            {
                "total_operations": self.flop_count,
                "matrix_count": len(self.matrices),
                "largest_matrix_size": max(
                    matrix.size for matrix in self.matrices.values()
                )
                if self.matrices
                else 0,
                "computational_intensity": self.flop_count
                / complexity["memory_footprint_bytes"]
                if complexity["memory_footprint_bytes"] > 0
                else 0.0,
            }
        )

        return complexity

    def validate_matrix_dimensions(self) -> bool:
        """Validate that matrix dimensions are compatible for the operation."""
        # Default implementation - subclasses should override
        return True

    def convert_to_systolic_operations(self) -> List[Dict[str, Any]]:
        """
        Convert matrix operations to systolic array operations.

        Returns:
            List of systolic array operation dictionaries
        """
        # Default implementation - subclasses should override
        operations = []

        if "A" in self.matrices and "B" in self.matrices:
            # Basic matrix multiplication operation
            operation = {
                "type": "matrix_multiply",
                "matrix_a": self.matrices["A"],
                "matrix_b": self.matrices["B"],
                "systolic_inputs": self._prepare_systolic_inputs(),
                "control_signals": self._generate_control_signals(),
            }
            operations.append(operation)

        return operations

    def _prepare_systolic_inputs(self) -> Dict[str, np.ndarray]:
        """Prepare inputs for systolic array execution."""
        # Default implementation for matrix multiplication
        inputs = {}

        if "A" in self.matrices and "B" in self.matrices:
            inputs["input_a"] = self.matrices["A"]
            inputs["input_b"] = self.matrices["B"]

        return inputs

    def _generate_control_signals(self) -> Dict[str, Any]:
        """Generate control signals for systolic array."""
        return {
            "operation_type": self.operation_type,
            "data_type": str(self.matrices["A"].dtype)
            if "A" in self.matrices
            else "float32",
            "enable_accumulation": True,
            "enable_sparsity": self.config.enable_sparsity,
        }


class SequentialWorkload(BaseWorkload):
    """
    Base class for sequential workloads with multiple operations.

    Useful for neural networks, attention mechanisms, and complex pipelines.
    """

    def __init__(self, config: WorkloadConfig):
        """Initialize sequential workload."""
        super().__init__(config)

        # Operation sequence
        self.operation_sequence: List[Dict[str, Any]] = []
        self.current_operation_index = 0

        # Layer/stage information
        self.layers: List[Dict[str, Any]] = []
        self.layer_outputs: Dict[int, np.ndarray] = {}

    def add_operation(self, operation: Dict[str, Any]) -> None:
        """
        Add an operation to the sequence.

        Args:
            operation: Operation dictionary with type, parameters, and data
        """
        operation["index"] = len(self.operation_sequence)
        self.operation_sequence.append(operation)

    def add_layer(self, layer_type: str, parameters: Dict[str, Any]) -> None:
        """
        Add a layer to the workload.

        Args:
            layer_type: Type of layer (e.g., 'dense', 'conv', 'attention')
            parameters: Layer parameters
        """
        layer = {
            "type": layer_type,
            "index": len(self.layers),
            "parameters": parameters,
        }
        self.layers.append(layer)

    def get_operations(self) -> List[Dict[str, Any]]:
        """Get all operations in sequence."""
        if not self.operation_sequence:
            # Generate operations from layers if not already done
            self._generate_operations_from_layers()

        return self.operation_sequence

    def _generate_operations_from_layers(self) -> None:
        """Generate operation sequence from layer definitions."""
        # Default implementation - subclasses should override
        for layer in self.layers:
            operation = {
                "type": layer["type"],
                "parameters": layer["parameters"],
                "layer_index": layer["index"],
            }
            self.add_operation(operation)

    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about layers in the workload."""
        return {
            "layer_count": len(self.layers),
            "operation_count": len(self.operation_sequence),
            "layer_types": [layer["type"] for layer in self.layers],
            "total_parameters": sum(
                layer["parameters"].get("parameter_count", 0) for layer in self.layers
            ),
        }


# Factory functions for creating common workload types


def create_simple_gemm_workload(
    M: int, K: int, P: int, name: str = "simple_gemm"
) -> "GEMMWorkload":
    """
    Create a simple GEMM workload.

    Args:
        M, K, P: Matrix dimensions for C[M,P] = A[M,K] * B[K,P]
        name: Workload name

    Returns:
        GEMMWorkload instance
    """
    from .gemm import GEMMWorkload, GEMMWorkloadConfig

    config = GEMMWorkloadConfig(
        name=name, M=M, K=K, P=P, workload_type=WorkloadType.GEMM
    )

    workload = GEMMWorkload(config)
    workload.generate_data()

    return workload


def create_conv_workload(
    input_shape: Tuple[int, int, int, int],
    kernel_shape: Tuple[int, int, int, int],
    name: str = "conv_workload",
) -> "ConvolutionWorkload":
    """
    Create a convolution workload.

    Args:
        input_shape: Input tensor shape (N, H, W, C)
        kernel_shape: Kernel shape (H, W, C_in, C_out)
        name: Workload name

    Returns:
        ConvolutionWorkload instance
    """
    from .conv import ConvolutionWorkload, ConvWorkloadConfig

    config = ConvWorkloadConfig(
        name=name,
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        workload_type=WorkloadType.CONVOLUTION,
    )

    workload = ConvolutionWorkload(config)
    workload.generate_data()

    return workload
