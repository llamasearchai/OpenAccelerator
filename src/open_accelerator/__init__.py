"""
OpenAccelerator: Enterprise-Grade Systolic Array Computing Framework

A comprehensive, production-ready simulator for exploring ML accelerator architectures
with specialized focus on medical imaging and healthcare AI workloads.

Author: Nik Jois <nikjois@llamasearch.ai>
Copyright 2024 Nik Jois
Licensed under the MIT License
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from open_accelerator.utils.config import (
    AcceleratorConfig,
    get_default_configs,
    load_config,
)

__version__ = "1.0.2"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "MIT"

# Global configuration dictionary
_config: Optional[Dict[str, Any]] = None

# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    Get a configuration value.

    If key is None, returns the entire configuration dictionary.
    """
    global _config
    if _config is None:
        try:
            _config = load_config("config.yaml").__dict__
        except (FileNotFoundError, AttributeError):
            _config = get_default_configs().__dict__

    if key is None:
        return _config

    return _config.get(key, default)


def set_config(key: str, value: Any) -> bool:
    """Set a configuration value."""
    global _config
    if _config is None:
        _config = {}
    _config[key] = value
    return True


def reset_config() -> bool:
    """Reset the global configuration."""
    global _config
    _config = None
    return True
