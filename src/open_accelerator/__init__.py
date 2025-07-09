"""
OpenAccelerator: Advanced ML Accelerator Simulator for Medical AI Applications

A comprehensive, production-ready simulator for exploring ML accelerator architectures
with specialized focus on medical imaging and healthcare AI workloads.

Copyright 2024 LlamaSearch AI Research
Licensed under the Apache License, Version 2.0
"""

# Version information
__version__ = "1.0.1"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
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


def get_config(key=None):
    """Get global configuration or specific key."""
    global _global_config
    if _global_config is None:
        _global_config = {}

    if key is None:
        try:
            from .utils.config import AcceleratorConfig

            config = AcceleratorConfig()
            # Convert to dict format for testing
            return {
                "name": config.name,
                "accelerator_type": config.accelerator_type.value,
                "data_type": config.data_type.value,
                "array_rows": config.array.rows,
                "array_cols": config.array.cols,
                "max_cycles": config.max_cycles,
                "debug_mode": config.debug_mode,
                "enable_logging": config.enable_logging,
                **_global_config,
            }
        except (ImportError, AttributeError):
            return _global_config
    else:
        return _global_config.get(key)


def set_config(key, value):
    """Set global configuration key-value pair."""
    global _global_config
    if _global_config is None:
        _global_config = {}
    _global_config[key] = value
    return True


def reset_config():
    """Reset configuration to defaults."""
    global _global_config
    _global_config = {}
    return True


# Global configuration instance
_global_config = None


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
    "get_config",
    "set_config",
    "reset_config",
]
