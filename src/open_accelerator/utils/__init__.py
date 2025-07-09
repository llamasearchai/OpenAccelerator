"""Utility sub-package public interface.

This *init* module re-exports the high-level configuration dataclasses so that
call-sites can simply write::

    from open_accelerator.utils import AcceleratorConfig, WorkloadConfig

without having to know the internal file structure (``.config``).
"""

from __future__ import annotations

from typing import Optional
from .config import (
    AcceleratorConfig as _RichAcceleratorConfig,
    WorkloadConfig,
    ArrayConfig,
    BufferConfig,
    MemoryConfig,
    MemoryHierarchyConfig,
    PowerConfig,
    MedicalConfig,
    DataType,
    DataflowType,
    WorkloadType,
    AcceleratorType,
)

import numpy as _np


class AcceleratorConfig(_RichAcceleratorConfig):  # type: ignore[misc]
    """Backward-compatibility wrapper.

    Early releases allowed constructing a flat *AcceleratorConfig* like::

        AcceleratorConfig(array_rows=4, array_cols=4, input_buffer_size=512, ...)

    The modern implementation expects nested dataclasses.  This subclass
    converts the legacy keyword arguments into the appropriate nested
    structures before delegating to the rich parent dataclass.
    """

    def __init__(
        self,
        *,
        array_rows: Optional[int] = None,
        array_cols: Optional[int] = None,
        pe_mac_latency: Optional[int] = None,
        input_buffer_size: Optional[int] = None,
        weight_buffer_size: Optional[int] = None,
        output_buffer_size: Optional[int] = None,
        data_type: Optional[type] = None,
        **kwargs,
    ) -> None:

        # Start with defaults from parent class
        array_config = ArrayConfig()
        input_buffer_config = BufferConfig()
        weight_buffer_config = BufferConfig()
        output_buffer_config = BufferConfig()

        # Apply legacy flat parameters to nested configs
        if array_rows is not None:
            array_config.rows = array_rows
        if array_cols is not None:
            array_config.cols = array_cols

        if input_buffer_size is not None:
            input_buffer_config.buffer_size = input_buffer_size
        if weight_buffer_size is not None:
            weight_buffer_config.buffer_size = weight_buffer_size
        if output_buffer_size is not None:
            output_buffer_config.buffer_size = output_buffer_size

        # Set up kwargs for parent constructor
        parent_kwargs = {
            "array": array_config,
            "input_buffer": input_buffer_config,
            "weight_buffer": weight_buffer_config,
            "output_buffer": output_buffer_config,
        }

        if data_type is not None:
            parent_kwargs["data_type"] = data_type

        # Merge with any additional kwargs
        parent_kwargs.update(kwargs)

        super().__init__(**parent_kwargs)


__all__: list[str] = [
    "AcceleratorConfig",  # wrapper class defined above
    "WorkloadConfig",
    "ArrayConfig",
    "BufferConfig",
    "MemoryConfig",
    "MemoryHierarchyConfig",
    "PowerConfig",
    "MedicalConfig",
    "DataType",
    "DataflowType",
    "WorkloadType",
    "AcceleratorType",
] 