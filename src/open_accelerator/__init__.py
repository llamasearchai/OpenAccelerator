"""
OpenAccelerator: Advanced ML Accelerator Simulator for Medical AI Applications

A comprehensive, production-ready simulator for exploring ML accelerator architectures
with specialized focus on medical imaging and healthcare AI workloads.

Copyright 2024 Nik Jois <nikjois@llamasearch.ai>
Licensed under the Apache License, Version 2.0
"""

import logging
import sys
from typing import Any

# Version information
__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "Apache-2.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("openaccelerator.log", mode="a"),
    ],
)

logger = logging.getLogger(__name__)

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

    logger.info(f"OpenAccelerator v{__version__} initialized successfully")

except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    # Don't raise - allow partial functionality
    logger.warning("Some modules may not be available")

# Global configuration
_config: dict[str, Any] = {
    "simulation": {
        "max_cycles": 1_000_000,
        "precision": "float32",
        "debug_mode": False,
    },
    "hardware": {
        "default_array_size": (16, 16),
        "default_frequency": 1e9,  # 1 GHz
        "default_voltage": 1.2,  # 1.2V
    },
    "medical": {
        "dicom_cache_size": "1GB",
        "enable_phi_compliance": True,
        "default_modality": "CT",
    },
    "visualization": {
        "theme": "dark",
        "animation_fps": 30,
        "export_dpi": 300,
    },
}


def get_config(key: str | None = None) -> Any:
    """Get global configuration value."""
    if key is None:
        return _config
    return _config.get(key, {})


def set_config(key: str, value: Any) -> None:
    """Set global configuration value."""
    _config[key] = value


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = {
        "simulation": {
            "max_cycles": 1_000_000,
            "precision": "float32",
            "debug_mode": False,
        },
        "hardware": {
            "default_array_size": (16, 16),
            "default_frequency": 1e9,
            "default_voltage": 1.2,
        },
        "medical": {
            "dicom_cache_size": "1GB",
            "enable_phi_compliance": True,
            "default_modality": "CT",
        },
        "visualization": {
            "theme": "dark",
            "animation_fps": 30,
            "export_dpi": 300,
        },
    }


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

# Ensure proper cleanup on exit
import atexit


def _cleanup():
    """Cleanup function called on exit."""
    logger.info("OpenAccelerator shutdown complete")


atexit.register(_cleanup)
