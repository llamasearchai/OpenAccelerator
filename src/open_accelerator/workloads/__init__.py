"""
Workload implementations for OpenAccelerator.

This module provides various workload types for testing and benchmarking
the accelerator simulation framework.
"""

from .base import (
    BaseWorkload,
    ComputeWorkload,
    MedicalWorkload,
    MLWorkload,
    create_workload_from_spec,
)
from .gemm import (
    GEMMWorkload,
    GEMMWorkloadConfig,
    create_large_gemm,
    create_medium_gemm,
    create_small_gemm,
)

# Only import modules that exist
try:
    from .medical import MedicalImagingWorkload

    HAS_MEDICAL = True
except ImportError:
    HAS_MEDICAL = False

__all__ = [
    # Base classes
    "BaseWorkload",
    "ComputeWorkload",
    "MLWorkload",
    "MedicalWorkload",
    "create_workload_from_spec",
    # GEMM workloads
    "GEMMWorkload",
    "GEMMWorkloadConfig",
    "create_small_gemm",
    "create_medium_gemm",
    "create_large_gemm",
]

# Add medical workloads if available
if HAS_MEDICAL:
    __all__.append("MedicalImagingWorkload")

__version__ = "1.0.0"
