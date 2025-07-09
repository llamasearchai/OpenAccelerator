"""
Base workload classes for OpenAccelerator.

Provides abstract base classes and common functionality for all workload types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorkloadRequirements:
    """Workload requirements for accelerator compatibility."""
    min_array_size: Tuple[int, int] = (1, 1)
    max_array_size: Tuple[int, int] = (1024, 1024)
    required_array_size: Tuple[int, int] = (16, 16)  # Specific required array size
    required_memory_mb: float = 1.0
    required_bandwidth_gbps: float = 1.0
    supported_data_types: List[str] = field(default_factory=lambda: ["float32"])
    memory_requirements: Dict[str, float] = field(default_factory=lambda: {"input": 1.0, "weight": 1.0, "output": 1.0})
    
    # Medical-specific requirements
    requires_medical_mode: bool = False
    requires_phi_compliance: bool = False
    requires_fda_validation: bool = False


@dataclass
class WorkloadMetrics:
    """Performance metrics for workload execution."""
    
    total_operations: int = 0
    total_cycles: int = 0
    throughput_ops_per_second: float = 0.0
    energy_consumed_joules: float = 0.0
    memory_bandwidth_utilized_gbps: float = 0.0
    
    # Accuracy metrics (for ML workloads)
    output_accuracy: Optional[float] = None
    numerical_precision: Optional[float] = None
    
    # Medical-specific metrics
    diagnostic_accuracy: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None


class BaseWorkload(ABC):
    """
    Abstract base class for all workloads.
    
    Defines the interface that all workloads must implement to be
    compatible with the OpenAccelerator simulation framework.
    """
    
    def __init__(self, name: str = "BaseWorkload"):
        """Initialize base workload."""
        self.name = name
        self.requirements = WorkloadRequirements()
        self.metrics = WorkloadMetrics()
        self.is_prepared = False
        self.execution_results: Dict[str, Any] = {}
        
        logger.info(f"Initialized workload: {self.name}")
    
    @abstractmethod
    def prepare(self, **kwargs) -> None:
        """
        Prepare the workload for execution.
        
        This method should generate or load all necessary data,
        validate parameters, and set up any required state.
        """
        pass
    
    @abstractmethod
    def get_input_data(self) -> Dict[str, np.ndarray]:
        """
        Get input data for the workload.
        
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        pass
    
    @abstractmethod
    def get_expected_output(self) -> Dict[str, np.ndarray]:
        """
        Get expected output for validation.
        
        Returns:
            Dictionary mapping output names to expected numpy arrays
        """
        pass
    
    @abstractmethod
    def validate_output(self, actual_output: Dict[str, np.ndarray]) -> bool:
        """
        Validate actual output against expected results.
        
        Args:
            actual_output: Dictionary of actual output arrays
            
        Returns:
            True if output is valid, False otherwise
        """
        pass
    
    def get_name(self) -> str:
        """Get workload name."""
        return self.name
    
    def get_requirements(self) -> WorkloadRequirements:
        """Get workload requirements for accelerator compatibility checking."""
        return self.requirements
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Get list of operations for simulation execution."""
        if not self.is_prepared:
            raise RuntimeError("Workload not prepared. Call prepare() first.")
        
        # Default implementation returns empty list
        # Subclasses should override this method
        return []
    
    def get_metrics(self) -> WorkloadMetrics:
        """Get performance metrics."""
        return self.metrics
    
    def set_metrics(self, metrics: WorkloadMetrics) -> None:
        """Set performance metrics."""
        self.metrics = metrics
    
    def is_ready(self) -> bool:
        """Check if workload is ready for execution."""
        return self.is_prepared
    
    def get_description(self) -> str:
        """Get human-readable description of the workload."""
        return f"{self.name}: {self.__class__.__doc__ or 'No description available'}"
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of last execution."""
        return {
            "workload_name": self.name,
            "total_operations": self.metrics.total_operations,
            "total_cycles": self.metrics.total_cycles,
            "throughput": self.metrics.throughput_ops_per_second,
            "energy_consumed": self.metrics.energy_consumed_joules,
            "is_valid": self.execution_results.get("validation_passed", False),
        }
    
    def reset(self) -> None:
        """Reset workload state for re-execution."""
        self.metrics = WorkloadMetrics()
        self.execution_results = {}
        self.is_prepared = False
        logger.info(f"Reset workload: {self.name}")
    
    def __str__(self) -> str:
        """String representation of workload."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"ready={self.is_prepared}, "
            f"operations={self.metrics.total_operations})"
        )


class ComputeWorkload(BaseWorkload):
    """Base class for compute-intensive workloads."""
    
    def __init__(self, name: str = "ComputeWorkload"):
        super().__init__(name)
        self.requirements.required_memory_mb = 10.0
        self.requirements.required_bandwidth_gbps = 10.0

    # ------------------------------------------------------------------
    # Concrete implementations (no-op) so the class is instantiable.      
    # These can be overridden by derived workloads with real logic.      
    # ------------------------------------------------------------------

    def prepare(self, **kwargs) -> None:  # type: ignore[override]
        """Dummy prepare – marks workload as ready without data."""
        self.is_prepared = True

    def get_input_data(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        return {}

    def get_expected_output(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        return {}

    def validate_output(self, actual_output: Dict[str, np.ndarray]) -> bool:  # type: ignore[override]
        return True


class MLWorkload(BaseWorkload):
    """Base class for machine learning workloads."""
    
    def __init__(self, name: str = "MLWorkload"):
        super().__init__(name)
        self.requirements.required_memory_mb = 100.0
        self.requirements.required_bandwidth_gbps = 50.0
        self.requirements.supported_data_types = ["float32", "float16", "bfloat16"]
        
        # ML-specific attributes
        self.model_parameters = 0
        self.inference_mode = True
        self.batch_size = 1

    # Provide concrete implementations (no-op) – can be overridden later.
    def prepare(self, **kwargs) -> None:  # type: ignore[override]
        self.is_prepared = True

    def get_input_data(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        return {}

    def get_expected_output(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        return {}

    def validate_output(self, actual_output: Dict[str, np.ndarray]) -> bool:  # type: ignore[override]
        return True


class MedicalWorkload(MLWorkload):
    """Base class for medical AI workloads."""
    
    def __init__(self, name: str = "MedicalWorkload"):
        super().__init__(name)
        self.requirements.requires_medical_mode = True
        self.requirements.requires_phi_compliance = True
        self.requirements.supported_data_types = ["float32"]  # Medical requires high precision
        
        # Medical-specific attributes
        self.patient_id: Optional[str] = None
        self.study_date: Optional[str] = None
        self.modality: str = "CT"
        self.anonymized = True

    # Concrete method stubs
    def prepare(self, **kwargs) -> None:  # type: ignore[override]
        self.is_prepared = True

    def get_input_data(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        return {}

    def get_expected_output(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        return {}

    def validate_output(self, actual_output: Dict[str, np.ndarray]) -> bool:  # type: ignore[override]
        return True


# Utility functions for workload management
def create_workload_from_spec(spec: Dict[str, Any]) -> BaseWorkload:
    """Create workload instance from specification dictionary."""
    workload_type = spec.get("type", "base")
    name = spec.get("name", f"{workload_type}_workload")
    
    if workload_type == "compute":
        return ComputeWorkload(name)
    elif workload_type == "ml":
        return MLWorkload(name)
    elif workload_type == "medical":
        return MedicalWorkload(name)
    else:
        # Return a minimal concrete implementation for testing
        return _TestWorkload(name)


class _TestWorkload(BaseWorkload):
    """Test workload implementation for development."""
    
    def __init__(self, name: str = "TestWorkload"):
        super().__init__(name)
        self.input_data = {}
        self.expected_output = {}
    
    def prepare(self, **kwargs) -> None:
        """Prepare test workload."""
        size = kwargs.get("size", (4, 4))
        self.input_data = {
            "input_a": np.random.rand(*size).astype(np.float32),
            "input_b": np.random.rand(*size).astype(np.float32),
        }
        self.expected_output = {
            "output": self.input_data["input_a"] @ self.input_data["input_b"]
        }
        self.is_prepared = True
        self.metrics.total_operations = size[0] * size[1] * size[0]  # Approximate for GEMM
    
    def get_input_data(self) -> Dict[str, np.ndarray]:
        """Get test input data."""
        return self.input_data
    
    def get_expected_output(self) -> Dict[str, np.ndarray]:
        """Get expected test output."""
        return self.expected_output
    
    def validate_output(self, actual_output: Dict[str, np.ndarray]) -> bool:
        """Validate test output."""
        if "output" not in actual_output:
            return False
        
        expected = self.expected_output["output"]
        actual = actual_output["output"]
        
        return np.allclose(expected, actual, rtol=1e-5, atol=1e-6)
