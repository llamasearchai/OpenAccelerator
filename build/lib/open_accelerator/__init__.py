"""
OpenAccelerator: Advanced ML Accelerator Simulator for Medical AI Applications

A comprehensive, production-ready simulator for exploring ML accelerator architectures
with specialized focus on medical imaging and healthcare AI workloads.

Copyright 2024 OpenAccelerator Team
Licensed under the Apache License, Version 2.0
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Version information
__version__ = "1.0.0"
__author__ = "OpenAccelerator Team"
__email__ = "team@openaccelerator.ai"
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
    from .analysis.metrics import PerformanceAnalyzer, PowerAnalyzer
    from .core.accelerator import Accelerator
    from .core.dataflow import OutputStationaryController, WeightStationaryController
    from .core.memory import Buffer, MemoryHierarchy
    from .core.pe import ProcessingElement
    from .core.systolic_array import SystolicArray
    from .simulation.simulator import SimulationConfig, Simulator
    from .utils.config import AcceleratorConfig, load_config, save_config
    from .utils.profiler import HardwareProfiler
    from .visualization.dashboard import AcceleratorDashboard
    from .workloads.base import BaseWorkload
    from .workloads.conv import ConvolutionWorkload
    from .workloads.gemm import GEMMWorkload
    from .workloads.medical import MedicalImagingWorkload

    logger.info(f"OpenAccelerator v{__version__} initialized successfully")

except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    raise

# Global configuration
_config: Dict[str, Any] = {
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


def get_config(key: str = None) -> Any:
    """Get global configuration value."""
    if key is None:
        return _config
    return _config.get(key, {})


def set_config(key: str, value: Any) -> None:
    """Set global configuration value."""
    _config[key] = value
    logger.debug(f"Configuration updated: {key} = {value}")


# Package metadata
__all__ = [
    # Core classes
    "Accelerator",
    "ProcessingElement",
    "SystolicArray",
    "MemoryHierarchy",
    "Buffer",
    "OutputStationaryController",
    "WeightStationaryController",
    # Simulation
    "Simulator",
    "SimulationConfig",
    # Workloads
    "BaseWorkload",
    "GEMMWorkload",
    "ConvolutionWorkload",
    "MedicalImagingWorkload",
    # Analysis
    "PerformanceAnalyzer",
    "PowerAnalyzer",
    # Visualization
    "AcceleratorDashboard",
    # Configuration
    "AcceleratorConfig",
    "load_config",
    "save_config",
    # Utilities
    "HardwareProfiler",
    "get_config",
    "set_config",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


# Compatibility checks
def _check_dependencies():
    """Check for optional dependencies and warn if missing."""
    optional_deps = {
        "cupy": "GPU acceleration",
        "nibabel": "Medical imaging (NIfTI)",
        "pydicom": "Medical imaging (DICOM)",
        "monai": "Medical AI workflows",
    }

    missing_deps = []
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep} ({description})")

    if missing_deps:
        logger.warning(
            f"Optional dependencies missing: {', '.join(missing_deps)}. "
            "Some features may be unavailable."
        )


_check_dependencies()
