"""
OpenAccelerator: Advanced ML Accelerator Simulator for Medical AI Applications

A comprehensive, production-ready simulator for exploring ML accelerator architectures
with specialized focus on medical imaging and healthcare AI workloads.

Copyright 2024 LlamaSearch AI Research
Licensed under the Apache License, Version 2.0
"""

# Version information
__version__ = "1.0.0"
__author__ = "LlamaSearch AI Research"
__email__ = "contact@llamasearch.ai"
__license__ = "Apache-2.0"

# Core imports with error handling
try:
    from .analysis.performance_analysis import PerformanceAnalyzer
    from .core.accelerator import AcceleratorController
    from .core.memory import MemoryHierarchy
    from .core.pe import ProcessingElement
    from .core.systolic_array import SystolicArray
    from .simulation.simulator import Simulator
    from .utils.config import AcceleratorConfig
    from .workloads.base import BaseWorkload
    from .workloads.gemm import GEMMWorkload

except ImportError:
    # Don't raise - allow partial functionality
    pass


# Export main components
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "AcceleratorController",
    "MemoryHierarchy",
    "ProcessingElement",
    "SystolicArray",
    "Simulator",
    "AcceleratorConfig",
    "BaseWorkload",
    "GEMMWorkload",
    "PerformanceAnalyzer",
]
