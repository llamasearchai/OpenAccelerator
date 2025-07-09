"""
GEMM (General Matrix Multiplication) workload implementation.

Provides comprehensive GEMM workload with support for various matrix sizes,
data types, and optimization features like sparsity and quantization.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import MatrixWorkload, WorkloadConfig, WorkloadRequirements, WorkloadType

logger = logging.getLogger(__name__)


@dataclass
class GEMMWorkloadConfig(WorkloadConfig):
    """Configuration for GEMM workload."""

    # Matrix dimensions: C[M,P] = A[M,K] * B[K,P]
    M: int = 16
    K: int = 16
    P: int = 16

    # Data generation parameters
    data_range: Tuple[float, float] = (-10.0, 10.0)
    integer_data: bool = False

    # GEMM variants
    transposed_a: bool = False
    transposed_b: bool = False
    alpha: float = 1.0  # Scaling factor for A*B
    beta: float = 0.0  # Scaling factor for C (if adding to existing C)

    # Optimization features
    enable_blocking: bool = False
    block_size: Tuple[int, int] = (8, 8)

    def __post_init__(self):
        """Post-initialization validation."""
        if self.M <= 0 or self.K <= 0 or self.P <= 0:
            raise ValueError("Matrix dimensions must be positive")

        if self.workload_type != WorkloadType.GEMM:
            self.workload_type = WorkloadType.GEMM

        if not self.name.startswith("gemm"):
            self.name = f"gemm_{self.name}"


class GEMMWorkload(MatrixWorkload):
    """
    GEMM workload implementation.

    Implements general matrix multiplication C = α*A*B + β*C
    with support for various optimizations and data layouts.
    """

    def __init__(self, config: GEMMWorkloadConfig):
        """
        Initialize GEMM workload.

        Args:
            config: GEMM workload configuration
        """
        super().__init__(config)
        self.gemm_config = config
        self.operation_type = "gemm"

        # Store matrix dimensions
        self.M = config.M
        self.K = config.K
        self.P = config.P

        # Update matrix dimensions dict
        self.matrix_dimensions = {
            "A": (self.M, self.K),
            "B": (self.K, self.P),
            "C": (self.M, self.P),
        }

        # Calculate FLOP count
        self.flop_count = 2 * self.M * self.K * self.P  # 2 ops per MAC

        logger.debug(
            f"Initialized GEMM workload: {self.M}x{self.K} * {self.K}x{self.P} = {self.M}x{self.P}"
        )

    def generate_data(self, seed: Optional[int] = None) -> None:
        """
        Generate matrices A, B, and compute expected output C.

        Args:
            seed: Random seed for reproducible generation
        """
        start_time = time.time()

        if seed is None:
            seed = self.config.seed

        np.random.seed(seed)
        logger.info(f"Generating GEMM data with seed {seed}")

        # Determine data type
        if self.config.precision == "float32":
            dtype = np.float32
        elif self.config.precision == "float64":
            dtype = np.float64
        elif self.config.precision == "float16":
            dtype = np.float16
        else:
            dtype = np.float32
            logger.warning(f"Unknown precision {self.config.precision}, using float32")

        # Generate matrices
        data_min, data_max = self.gemm_config.data_range

        if self.gemm_config.integer_data:
            # Generate integer data
            matrix_A = np.random.randint(
                int(data_min), int(data_max) + 1, size=(self.M, self.K)
            ).astype(dtype)
            matrix_B = np.random.randint(
                int(data_min), int(data_max) + 1, size=(self.K, self.P)
            ).astype(dtype)
        else:
            # Generate floating-point data
            matrix_A = np.random.uniform(
                data_min, data_max, size=(self.M, self.K)
            ).astype(dtype)
            matrix_B = np.random.uniform(
                data_min, data_max, size=(self.K, self.P)
            ).astype(dtype)

        # Handle transposition
        if self.gemm_config.transposed_a:
            matrix_A = matrix_A.T
            self.matrix_dimensions["A"] = (self.K, self.M)

        if self.gemm_config.transposed_b:
            matrix_B = matrix_B.T
            self.matrix_dimensions["B"] = (self.P, self.K)

        # Store matrices
        self.matrices["A"] = matrix_A
        self.matrices["B"] = matrix_B

        # Store in input_data for base class compatibility
        self.input_data["matrix_A"] = matrix_A
        self.input_data["matrix_B"] = matrix_B

        # Compute expected output
        expected_C = self._compute_expected_output()
        self.matrices["C"] = expected_C
        self.output_data["matrix_C"] = expected_C

        # Apply transformations if enabled
        if self.config.enable_sparsity or self.config.enable_quantization:
            self.apply_transformations(["sparsify", "quantize"])

        self.is_generated = True
        self.generation_time = time.time() - start_time

        logger.info(f"GEMM data generation completed in {self.generation_time:.3f}s")
        logger.debug(f"Matrix A shape: {matrix_A.shape}, dtype: {matrix_A.dtype}")
        logger.debug(f"Matrix B shape: {matrix_B.shape}, dtype: {matrix_B.dtype}")
        logger.debug(f"Matrix C shape: {expected_C.shape}, dtype: {expected_C.dtype}")

    def _compute_expected_output(self) -> np.ndarray:
        """Compute expected output using NumPy."""
        A = self.matrices["A"]
        B = self.matrices["B"]

        # Perform matrix multiplication
        if self.gemm_config.transposed_a and self.gemm_config.transposed_b:
            C = np.dot(A.T, B.T)
        elif self.gemm_config.transposed_a:
            C = np.dot(A.T, B)
        elif self.gemm_config.transposed_b:
            C = np.dot(A, B.T)
        else:
            C = np.dot(A, B)

        # Apply scaling factors
        C = self.gemm_config.alpha * C

        # Add beta*C if beta != 0 (requires existing C matrix)
        if self.gemm_config.beta != 0.0:
            # For simplicity, assume initial C is zeros
            # In practice, this would be a provided matrix
            pass

        return C

    def get_operations(self) -> List[Dict[str, Any]]:
        """
        Get systolic array operations for GEMM execution.

        Returns:
            List of operations for systolic array execution
        """
        if not self.is_generated:
            raise RuntimeError("Data not generated. Call generate_data() first.")

        if self.execution_operations:
            return self.execution_operations

        # Generate operations based on blocking strategy
        if self.gemm_config.enable_blocking:
            self.execution_operations = self._generate_blocked_operations()
        else:
            self.execution_operations = self._generate_simple_operations()

        return self.execution_operations

    def _generate_simple_operations(self) -> List[Dict[str, Any]]:
        """Generate simple (non-blocked) GEMM operations."""
        operations = []

        # Single GEMM operation
        operation = {
            "type": "gemm",
            "operation_index": 0,
            "matrix_a": self.matrices["A"],
            "matrix_b": self.matrices["B"],
            "dimensions": {"M": self.M, "K": self.K, "P": self.P},
            "parameters": {
                "alpha": self.gemm_config.alpha,
                "beta": self.gemm_config.beta,
                "transposed_a": self.gemm_config.transposed_a,
                "transposed_b": self.gemm_config.transposed_b,
            },
            "systolic_inputs": self._prepare_systolic_inputs(),
            "control_signals": self._generate_control_signals(),
            "memory_ops": self._generate_memory_operations(),
        }

        operations.append(operation)
        return operations

    def _generate_blocked_operations(self) -> List[Dict[str, Any]]:
        """Generate blocked GEMM operations for better cache locality."""
        operations = []
        block_m, block_k = self.gemm_config.block_size
        block_p = block_k  # Use same block size for consistency

        operation_index = 0

        # Iterate through blocks
        for i in range(0, self.M, block_m):
            for j in range(0, self.P, block_p):
                for k in range(0, self.K, block_k):
                    # Calculate block boundaries
                    i_end = min(i + block_m, self.M)
                    j_end = min(j + block_p, self.P)
                    k_end = min(k + block_k, self.K)

                    # Extract blocks
                    block_A = self.matrices["A"][i:i_end, k:k_end]
                    block_B = self.matrices["B"][k:k_end, j:j_end]

                    # Create block operation
                    operation = {
                        "type": "gemm_block",
                        "operation_index": operation_index,
                        "block_indices": (i, j, k),
                        "block_size": (i_end - i, j_end - j, k_end - k),
                        "matrix_a": block_A,
                        "matrix_b": block_B,
                        "output_position": (i, j),
                        "accumulate": k > 0,  # Accumulate if not first k-block
                        "systolic_inputs": self._prepare_block_systolic_inputs(
                            block_A, block_B
                        ),
                        "control_signals": self._generate_control_signals(),
                        "memory_ops": self._generate_block_memory_operations(
                            i, j, k, i_end, j_end, k_end
                        ),
                    }

                    operations.append(operation)
                    operation_index += 1

        logger.info(f"Generated {len(operations)} blocked GEMM operations")
        return operations

    def _prepare_block_systolic_inputs(
        self, block_A: np.ndarray, block_B: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare systolic inputs for a block operation."""
        return {
            "input_a": block_A,
            "input_b": block_B,
            "block_operation": True,
            "input_dimensions": {"A_shape": block_A.shape, "B_shape": block_B.shape},
        }

    def _generate_memory_operations(self) -> List[Dict[str, Any]]:
        """Generate memory operations for the GEMM."""
        memory_ops = []

        # Load matrix A
        memory_ops.append(
            {
                "type": "read",
                "address": "matrix_A_base",
                "size": self.matrices["A"].nbytes,
                "data_type": str(self.matrices["A"].dtype),
                "access_pattern": "sequential",
            }
        )

        # Load matrix B
        memory_ops.append(
            {
                "type": "read",
                "address": "matrix_B_base",
                "size": self.matrices["B"].nbytes,
                "data_type": str(self.matrices["B"].dtype),
                "access_pattern": "sequential",
            }
        )

        # Store result C
        memory_ops.append(
            {
                "type": "write",
                "address": "matrix_C_base",
                "size": self.matrices["C"].nbytes,
                "data_type": str(self.matrices["C"].dtype),
                "access_pattern": "sequential",
            }
        )

        return memory_ops

    def _generate_block_memory_operations(
        self, i: int, j: int, k: int, i_end: int, j_end: int, k_end: int
    ) -> List[Dict[str, Any]]:
        """Generate memory operations for a block."""
        memory_ops = []

        # Block sizes
        block_m = i_end - i
        block_k = k_end - k
        block_p = j_end - j

        # Load block A
        memory_ops.append(
            {
                "type": "read",
                "address": f"matrix_A_base + {i * self.K + k}",
                "size": block_m * block_k * self.matrices["A"].itemsize,
                "data_type": str(self.matrices["A"].dtype),
                "access_pattern": "block",
                "block_info": {
                    "start_row": i,
                    "start_col": k,
                    "rows": block_m,
                    "cols": block_k,
                },
            }
        )

        # Load block B
        memory_ops.append(
            {
                "type": "read",
                "address": f"matrix_B_base + {k * self.P + j}",
                "size": block_k * block_p * self.matrices["B"].itemsize,
                "data_type": str(self.matrices["B"].dtype),
                "access_pattern": "block",
                "block_info": {
                    "start_row": k,
                    "start_col": j,
                    "rows": block_k,
                    "cols": block_p,
                },
            }
        )

        # Store/accumulate block C
        memory_ops.append(
            {
                "type": "write" if k == 0 else "read_modify_write",
                "address": f"matrix_C_base + {i * self.P + j}",
                "size": block_m * block_p * self.matrices["C"].itemsize,
                "data_type": str(self.matrices["C"].dtype),
                "access_pattern": "block",
                "block_info": {
                    "start_row": i,
                    "start_col": j,
                    "rows": block_m,
                    "cols": block_p,
                },
            }
        )

        return memory_ops

    def validate_output(self, computed_output: np.ndarray) -> bool:
        """
        Validate computed output against expected result.

        Args:
            computed_output: Output computed by accelerator

        Returns:
            True if output is valid within tolerance
        """
        if not self.is_generated:
            raise RuntimeError(
                "Expected output not available. Call generate_data() first."
            )

        expected_output = self.matrices["C"]

        # Check shape
        if computed_output.shape != expected_output.shape:
            logger.error(
                f"Shape mismatch: expected {expected_output.shape}, got {computed_output.shape}"
            )
            return False

        # Check values with tolerance
        if self.config.precision == "float16":
            rtol, atol = 1e-3, 1e-3
        elif self.config.precision == "float32":
            rtol, atol = 1e-5, 1e-6
        else:  # float64
            rtol, atol = 1e-12, 1e-12

        try:
            is_close = np.allclose(
                computed_output, expected_output, rtol=rtol, atol=atol
            )

            if not is_close:
                # Detailed error analysis
                diff = np.abs(computed_output - expected_output)
                max_error = np.max(diff)
                mean_error = np.mean(diff)

                logger.error(
                    f"Validation failed - Max error: {max_error:.2e}, Mean error: {mean_error:.2e}"
                )
                logger.error(f"Tolerance: rtol={rtol}, atol={atol}")

                # Show some mismatched values
                mismatch_indices = np.where(
                    ~np.isclose(computed_output, expected_output, rtol=rtol, atol=atol)
                )
                if len(mismatch_indices[0]) > 0:
                    logger.error("First few mismatches:")
                    for i in range(min(5, len(mismatch_indices[0]))):
                        idx = (mismatch_indices[0][i], mismatch_indices[1][i])
                        logger.error(
                            f"  [{idx[0]}, {idx[1]}]: expected {expected_output[idx]:.6f}, got {computed_output[idx]:.6f}"
                        )

            self.is_validated = is_close
            return is_close

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def get_requirements(self) -> WorkloadRequirements:
        """Get GEMM workload requirements."""
        # For Output Stationary, array size should match M x P
        required_array_size = (self.M, self.P)

        # Calculate memory requirements
        memory_requirements = {
            "input_buffer": self.matrices["A"].nbytes + self.matrices["B"].nbytes,
            "weight_buffer": max(self.matrices["A"].nbytes, self.matrices["B"].nbytes),
            "output_buffer": self.matrices["C"].nbytes,
            "total": self.matrices["A"].nbytes
            + self.matrices["B"].nbytes
            + self.matrices["C"].nbytes,
        }

        return WorkloadRequirements(
            required_array_size=required_array_size,
            memory_requirements=memory_requirements,
            data_type_requirements=self.config.precision,
            precision_requirements="high"
            if self.config.precision in ["float64", "float32"]
            else "medium",
            medical_compliance=self.config.medical_grade,
        )

    def estimate_complexity(self) -> Dict[str, Any]:
        """Estimate GEMM computational complexity."""
        complexity = super().estimate_complexity()

        # GEMM-specific complexity metrics
        complexity.update(
            {
                "flop_count": self.flop_count,
                "arithmetic_intensity": self.flop_count
                / (
                    self.matrices["A"].nbytes
                    + self.matrices["B"].nbytes
                    + self.matrices["C"].nbytes
                ),
                "parallelism_potential": min(self.M, self.P)
                / max(self.M, self.P),  # How well it can be parallelized
                "memory_access_pattern": "structured",
                "cache_efficiency": "high"
                if self.gemm_config.enable_blocking
                else "medium",
                "computational_density": self.flop_count
                / (self.M * self.P),  # FLOPs per output element
                "reuse_factor": {
                    "A": self.P,  # Each A element is reused P times
                    "B": self.M,  # Each B element is reused M times
                    "C": self.K,  # Each C element is accumulated K times
                },
            }
        )

        return complexity

    def get_performance_characteristics(self) -> Dict[str, Any]:
        """Get performance characteristics for the GEMM workload."""
        return {
            "workload_type": "compute_bound",
            "parallelization_strategy": "output_stationary",
            "data_reuse_pattern": {
                "A": "row_wise_streaming",
                "B": "column_wise_streaming",
                "C": "accumulation",
            },
            "memory_access_intensity": {"reads": 2, "writes": 1},  # A and B  # C
            "computational_intensity": self.flop_count
            / (
                self.matrices["A"].nbytes
                + self.matrices["B"].nbytes
                + self.matrices["C"].nbytes
            ),
            "scalability": {
                "array_size_sensitivity": "high",
                "memory_bandwidth_sensitivity": "medium",
                "frequency_sensitivity": "high",
            },
        }

    def generate_test_cases(self) -> List["GEMMWorkload"]:
        """Generate a set of test cases with different configurations."""
        test_cases = []

        # Different matrix sizes
        sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 16, 8)]

        for M, K, P in sizes:
            config = GEMMWorkloadConfig(
                name=f"test_gemm_{M}x{K}x{P}", M=M, K=K, P=P, seed=self.config.seed
            )

            test_case = GEMMWorkload(config)
            test_case.generate_data()
            test_cases.append(test_case)

        # Different data types
        for precision in ["float16", "float32"]:
            config = GEMMWorkloadConfig(
                name=f"test_gemm_{precision}",
                M=8,
                K=8,
                P=8,
                precision=precision,
                seed=self.config.seed,
            )

            test_case = GEMMWorkload(config)
            test_case.generate_data()
            test_cases.append(test_case)

        # With sparsity
        config = GEMMWorkloadConfig(
            name="test_gemm_sparse",
            M=16,
            K=16,
            P=16,
            enable_sparsity=True,
            sparsity_ratio=0.5,
            seed=self.config.seed,
        )

        test_case = GEMMWorkload(config)
        test_case.generate_data()
        test_cases.append(test_case)

        return test_cases


# Convenience functions for creating common GEMM workloads


def create_square_gemm(size: int, name: str = None) -> GEMMWorkload:
    """Create a square GEMM workload (NxN * NxN = NxN)."""
    config = GEMMWorkloadConfig(
        name=name or f"square_gemm_{size}", M=size, K=size, P=size
    )

    workload = GEMMWorkload(config)
    workload.generate_data()
    return workload


def create_tall_gemm(M: int, K: int, name: str = None) -> GEMMWorkload:
    """Create a tall GEMM workload (MxK * KxK = MxK)."""
    config = GEMMWorkloadConfig(name=name or f"tall_gemm_{M}x{K}", M=M, K=K, P=K)

    workload = GEMMWorkload(config)
    workload.generate_data()
    return workload


def create_wide_gemm(K: int, P: int, name: str = None) -> GEMMWorkload:
    """Create a wide GEMM workload (KxK * KxP = KxP)."""
    config = GEMMWorkloadConfig(name=name or f"wide_gemm_{K}x{P}", M=K, K=K, P=P)

    workload = GEMMWorkload(config)
    workload.generate_data()
    return workload


def create_medical_gemm(size: int, name: str = None) -> GEMMWorkload:
    """Create a medical-grade GEMM workload with high precision."""
    config = GEMMWorkloadConfig(
        name=name or f"medical_gemm_{size}",
        M=size,
        K=size,
        P=size,
        precision="float64",
        medical_grade=True,
        safety_critical=True,
        data_range=(-1.0, 1.0),  # Normalized range for medical data
    )

    workload = GEMMWorkload(config)
    workload.generate_data()
    return workload
