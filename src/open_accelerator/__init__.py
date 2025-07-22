"""
OpenAccelerator: Advanced ML Accelerator Simulator for Medical AI Applications

A comprehensive, production-ready simulator for exploring ML accelerator architectures
with specialized focus on medical imaging and healthcare AI workloads.

Copyright 2024 LlamaSearch AI Research
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

__version__ = "1.0.1"

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
        from open_accelerator.utils.config import (
            DEFAULT_CONFIG,
            load_config_from_file,
        )

        _config = load_config_from_file() or DEFAULT_CONFIG

    if key is None:
        return _config

    return _config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set a configuration value."""
    global _config
    if _config is None:
        _config = {}
    _config[key] = value


def reset_config() -> None:
    """Reset the global configuration."""
    global _config
    _config = None
