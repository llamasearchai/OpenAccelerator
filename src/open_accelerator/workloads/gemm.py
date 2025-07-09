"""
GEMM (General Matrix Multiplication) workload implementation.

Provides a complete implementation of matrix multiplication workloads
for the OpenAccelerator simulation framework.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseWorkload, WorkloadRequirements

logger = logging.getLogger(__name__)


@dataclass
class GEMMWorkloadConfig:
    """Configuration for GEMM workload."""

    M: int = 16  # Rows of A and C
    K: int = 16  # Cols of A, Rows of B
    P: int = 16  # Cols of B and C

    # Data generation parameters
    data_range: tuple[float, float] = (-1.0, 1.0)
    seed: int = 42
    use_integers: bool = False
    sparsity: float = 0.0  # 0.0 = dense, 0.9 = 90% sparse

    def __post_init__(self):
        if self.M <= 0 or self.K <= 0 or self.P <= 0:
            raise ValueError("Matrix dimensions must be positive")
        if not (0.0 <= self.sparsity < 1.0):
            raise ValueError("Sparsity must be in range [0.0, 1.0)")


class GEMMWorkload(BaseWorkload):
    """
    General Matrix Multiplication workload.

    Implements C = A @ B where:
    - A is M x K matrix
    - B is K x P matrix
    - C is M x P matrix (result)
    """

    def __init__(self, config: GEMMWorkloadConfig, name: str = "GEMM"):
        """Initialize GEMM workload."""
        super().__init__(name)
        self.config = config

        # Set requirements based on config
        self.requirements = WorkloadRequirements(
            min_array_size=(1, 1),
            max_array_size=(config.M, config.P),
            required_array_size=(config.M, config.P),  # Set to actual matrix dimensions
            required_memory_mb=self._estimate_memory_mb(),
            required_bandwidth_gbps=self._estimate_bandwidth_gbps(),
            supported_data_types=["float32", "int32"]
            if config.use_integers
            else ["float32"],
        )

        # Data storage
        self.matrix_A: Optional[np.ndarray] = None
        self.matrix_B: Optional[np.ndarray] = None
        self.expected_C: Optional[np.ndarray] = None

        logger.info(
            f"Initialized GEMM workload: {config.M}x{config.K} @ {config.K}x{config.P}"
        )

    # Compatibility properties for legacy API
    @property
    def M(self) -> int:
        """Matrix A rows / Matrix C rows."""
        return self.config.M

    @property
    def K(self) -> int:
        """Matrix A cols / Matrix B rows."""
        return self.config.K

    @property
    def P(self) -> int:
        """Matrix B cols / Matrix C cols."""
        return self.config.P

    @property
    def expected_result(self) -> Optional[np.ndarray]:
        """Expected result matrix (backward compatibility)."""
        return self.expected_C

    def get_complexity(self) -> Dict[str, Any]:
        """Get workload complexity metrics (backward compatibility)."""
        return {
            "total_operations": 2 * self.config.M * self.config.K * self.config.P,
            "memory_accesses": self.config.M * self.config.K
            + self.config.K * self.config.P
            + self.config.M * self.config.P,
            "arithmetic_intensity": (2 * self.config.M * self.config.K * self.config.P)
            / (
                self.config.M * self.config.K
                + self.config.K * self.config.P
                + self.config.M * self.config.P
            ),
            "matrix_dimensions": {
                "A": (self.config.M, self.config.K),
                "B": (self.config.K, self.config.P),
                "C": (self.config.M, self.config.P),
            },
        }

    def prepare(self, **kwargs) -> None:
        """Prepare GEMM workload by generating matrices."""
        # Override config parameters if provided
        seed = kwargs.get("seed", self.config.seed)
        data_range = kwargs.get("data_range", self.config.data_range)

        np.random.seed(seed)

        # Generate matrix A (M x K)
        if self.config.use_integers:
            self.matrix_A = np.random.randint(
                int(data_range[0]),
                int(data_range[1]) + 1,
                size=(self.config.M, self.config.K),
                dtype=np.int32,
            )
        else:
            self.matrix_A = np.random.uniform(
                data_range[0], data_range[1], size=(self.config.M, self.config.K)
            ).astype(np.float32)

        # Generate matrix B (K x P)
        if self.config.use_integers:
            self.matrix_B = np.random.randint(
                int(data_range[0]),
                int(data_range[1]) + 1,
                size=(self.config.K, self.config.P),
                dtype=np.int32,
            )
        else:
            self.matrix_B = np.random.uniform(
                data_range[0], data_range[1], size=(self.config.K, self.config.P)
            ).astype(np.float32)

        # Apply sparsity if requested
        if self.config.sparsity > 0:
            self._apply_sparsity()

        # Compute expected result
        assert self.matrix_A is not None and self.matrix_B is not None
        self.expected_C = np.dot(self.matrix_A, self.matrix_B)

        # Update metrics
        self.metrics.total_operations = (
            2 * self.config.M * self.config.K * self.config.P
        )  # MACs

        self.is_prepared = True
        logger.info(
            f"GEMM workload prepared with {self.metrics.total_operations} operations"
        )

    def generate_data(self, **kwargs) -> None:
        """Backward-compatibility helper.

        The legacy public API exposed a *generate_data()* method for GEMM workloads.
        The new implementation uses :pymeth:`prepare`, so we simply forward any
        parameters to that method.  All keyword arguments accepted by
        :pymeth:`prepare` (``seed``, ``data_range`` and future ones) are supported.
        """
        # Default behaviour stays identical to *prepare*.
        self.prepare(**kwargs)

    def get_input_data(self) -> Dict[str, np.ndarray]:
        """Get input matrices for GEMM."""
        if not self.is_prepared:
            raise RuntimeError("Workload not prepared. Call prepare() first.")

        assert self.matrix_A is not None and self.matrix_B is not None
        return {
            "matrix_A": self.matrix_A,
            "matrix_B": self.matrix_B,
        }

    def get_expected_output(self) -> Dict[str, np.ndarray]:
        """Get expected output matrix."""
        if not self.is_prepared:
            raise RuntimeError("Workload not prepared. Call prepare() first.")

        assert self.expected_C is not None
        return {
            "matrix_C": self.expected_C,
        }

    def validate_output(self, actual_output: Dict[str, np.ndarray]) -> bool:
        """Validate computed matrix against expected result."""
        if "matrix_C" not in actual_output:
            logger.error("Missing matrix_C in output")
            return False

        actual_C = actual_output["matrix_C"]
        assert self.expected_C is not None
        expected_C = self.expected_C

        # Check shape
        if actual_C.shape != expected_C.shape:
            logger.error(
                f"Shape mismatch: expected {expected_C.shape}, got {actual_C.shape}"
            )
            return False

        # Check values with appropriate tolerance
        if self.config.use_integers:
            # For integers, require exact match
            is_valid = np.array_equal(actual_C, expected_C)
        else:
            # For floats, use relative tolerance
            is_valid = np.allclose(actual_C, expected_C, rtol=1e-5, atol=1e-6)

        if is_valid:
            logger.info("GEMM output validation passed")
            self.execution_results["validation_passed"] = True
        else:
            logger.error("GEMM output validation failed")
            self.execution_results["validation_passed"] = False

            # Log some debugging info
            if not self.config.use_integers:
                max_error = np.max(np.abs(actual_C - expected_C))
                logger.error(f"Maximum absolute error: {max_error}")

        return is_valid

    def get_operations(self) -> List[Dict[str, Any]]:
        """Get list of operations for simulation execution."""
        if not self.is_prepared:
            raise RuntimeError("Workload not prepared. Call prepare() first.")

        operations = []

        # Generate operations for GEMM computation
        # For simplicity, we'll create operations that represent the matrix multiplication
        assert (
            self.matrix_A is not None
            and self.matrix_B is not None
            and self.expected_C is not None
        )

        # For each output element C[i,j], we need to compute sum(A[i,k] * B[k,j]) for k in range(K)
        for i in range(self.config.M):
            for j in range(self.config.P):
                for k in range(self.config.K):
                    operation = {
                        "type": "multiply_accumulate",
                        "output_position": (i, j),
                        "input_a": self.matrix_A[i, k],
                        "input_b": self.matrix_B[k, j],
                        "data": np.array([self.matrix_A[i, k], self.matrix_B[k, j]]),
                        "memory_ops": [
                            {
                                "type": "read",
                                "address": f"A[{i},{k}]",
                                "size": 4,  # 4 bytes for float32
                                "data": self.matrix_A[i, k],
                            },
                            {
                                "type": "read",
                                "address": f"B[{k},{j}]",
                                "size": 4,
                                "data": self.matrix_B[k, j],
                            },
                        ],
                    }
                    operations.append(operation)

        # Add final write operations for each output element
        for i in range(self.config.M):
            for j in range(self.config.P):
                operation = {
                    "type": "write_result",
                    "output_position": (i, j),
                    "data": np.array([self.expected_C[i, j]]),
                    "memory_ops": [
                        {
                            "type": "write",
                            "address": f"C[{i},{j}]",
                            "size": 4,
                            "data": self.expected_C[i, j],
                        }
                    ],
                }
                operations.append(operation)

        return operations

    def _apply_sparsity(self) -> None:
        """Apply sparsity to input matrices."""
        assert self.matrix_A is not None and self.matrix_B is not None
        # Create sparsity masks
        mask_A = np.random.random(self.matrix_A.shape) > self.config.sparsity
        mask_B = np.random.random(self.matrix_B.shape) > self.config.sparsity

        # Apply masks
        self.matrix_A = self.matrix_A * mask_A
        self.matrix_B = self.matrix_B * mask_B

        logger.info(f"Applied {self.config.sparsity:.1%} sparsity to matrices")

    def _estimate_memory_mb(self) -> float:
        """Estimate memory requirements in MB."""
        # Assume float32 (4 bytes per element)
        bytes_per_element = 4
        total_elements = (
            self.config.M * self.config.K  # Matrix A
            + self.config.K * self.config.P  # Matrix B
            + self.config.M * self.config.P  # Matrix C
        )
        return (total_elements * bytes_per_element) / (1024 * 1024)

    def _estimate_bandwidth_gbps(self) -> float:
        """Estimate bandwidth requirements in GB/s."""
        # Simple heuristic: assume we need to read A and B once, write C once
        # and achieve reasonable utilization
        memory_mb = self._estimate_memory_mb()
        # Assume 1 second execution time as baseline
        return memory_mb / 1024  # Convert to GB

    def get_description(self) -> str:
        """Get description of GEMM workload."""
        sparsity_str = (
            f", {self.config.sparsity:.1%} sparse" if self.config.sparsity > 0 else ""
        )
        dtype_str = "int32" if self.config.use_integers else "float32"

        return (
            f"GEMM {self.config.M}x{self.config.K} @ {self.config.K}x{self.config.P} "
            f"({dtype_str}{sparsity_str})"
        )

    def get_workload_details(self) -> Dict[str, Any]:
        """Get detailed information about the workload."""
        return {
            "type": "GEMM",
            "dimensions": {
                "M": self.config.M,
                "K": self.config.K,
                "P": self.config.P,
            },
            "config": {
                "data_range": self.config.data_range,
                "seed": self.config.seed,
                "use_integers": self.config.use_integers,
                "sparsity": self.config.sparsity,
            },
            "complexity": {
                "total_operations": self.metrics.total_operations,
                "memory_mb": self._estimate_memory_mb(),
                "bandwidth_gbps": self._estimate_bandwidth_gbps(),
            },
            "data_info": {
                "matrix_A_shape": (self.config.M, self.config.K),
                "matrix_B_shape": (self.config.K, self.config.P),
                "matrix_C_shape": (self.config.M, self.config.P),
                "dtype": "int32" if self.config.use_integers else "float32",
            }
            if self.is_prepared
            else None,
        }


# Convenience functions for creating GEMM workloads
def create_small_gemm(name: str = "SmallGEMM") -> GEMMWorkload:
    """Create a small GEMM workload for testing."""
    config = GEMMWorkloadConfig(M=4, K=4, P=4, use_integers=True)
    workload = GEMMWorkload(config, name)
    workload.prepare()
    return workload


def create_medium_gemm(name: str = "MediumGEMM") -> GEMMWorkload:
    """Create a medium GEMM workload."""
    config = GEMMWorkloadConfig(M=16, K=16, P=16)
    workload = GEMMWorkload(config, name)
    workload.prepare()
    return workload


def create_large_gemm(name: str = "LargeGEMM") -> GEMMWorkload:
    """Create a large GEMM workload."""
    config = GEMMWorkloadConfig(M=64, K=64, P=64)
    workload = GEMMWorkload(config, name)
    workload.prepare()
    return workload


def create_rectangular_gemm(
    M: int, K: int, P: int, name: str = "RectangularGEMM"
) -> GEMMWorkload:
    """Create a GEMM workload with custom dimensions."""
    config = GEMMWorkloadConfig(M=M, K=K, P=P)
    workload = GEMMWorkload(config, name)
    workload.prepare()
    return workload


def create_sparse_gemm(sparsity: float = 0.5, name: str = "SparseGEMM") -> GEMMWorkload:
    """Create a sparse GEMM workload."""
    config = GEMMWorkloadConfig(M=16, K=16, P=16, sparsity=sparsity)
    workload = GEMMWorkload(config, name)
    workload.prepare()
    return workload
