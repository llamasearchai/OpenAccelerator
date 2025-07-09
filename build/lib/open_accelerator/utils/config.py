"""
Configuration management for OpenAccelerator.

Provides robust configuration handling with validation, serialization,
and environment-specific settings.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for simulation."""

    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


class DataflowType(Enum):
    """Supported dataflow patterns."""

    OUTPUT_STATIONARY = "output_stationary"
    WEIGHT_STATIONARY = "weight_stationary"
    INPUT_STATIONARY = "input_stationary"


class MemoryType(Enum):
    """Memory hierarchy types."""

    SRAM = "sram"
    DRAM = "dram"
    HBM = "hbm"
    CACHE = "cache"


@dataclass
class PEConfig:
    """Processing Element configuration."""

    mac_latency: int = 1
    add_latency: int = 1
    mult_latency: int = 1
    accumulator_bits: int = 32
    enable_sparsity: bool = False
    sparsity_threshold: float = 1e-6
    power_gating: bool = True

    def __post_init__(self):
        if self.mac_latency < 1:
            raise ValueError("MAC latency must be at least 1 cycle")


@dataclass
class MemoryConfig:
    """Memory subsystem configuration."""

    buffer_size: int = 1024  # in elements
    bandwidth: int = 16  # elements per cycle
    latency: int = 1  # access latency in cycles
    energy_per_access: float = 1.0  # pJ per access
    memory_type: MemoryType = MemoryType.SRAM

    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")


@dataclass
class ArrayConfig:
    """Systolic array configuration."""

    rows: int = 16
    cols: int = 16
    pe_config: PEConfig = field(default_factory=PEConfig)
    dataflow: DataflowType = DataflowType.OUTPUT_STATIONARY
    frequency: float = 1e9  # Hz
    voltage: float = 1.2  # V

    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Array dimensions must be positive")
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")


@dataclass
class AcceleratorConfig:
    """Complete accelerator configuration."""

    name: str = "OpenAccelerator"
    array: ArrayConfig = field(default_factory=ArrayConfig)

    # Memory hierarchy
    input_buffer: MemoryConfig = field(
        default_factory=lambda: MemoryConfig(
            buffer_size=8192, bandwidth=32, memory_type=MemoryType.SRAM
        )
    )
    weight_buffer: MemoryConfig = field(
        default_factory=lambda: MemoryConfig(
            buffer_size=16384, bandwidth=64, memory_type=MemoryType.SRAM
        )
    )
    output_buffer: MemoryConfig = field(
        default_factory=lambda: MemoryConfig(
            buffer_size=4096, bandwidth=16, memory_type=MemoryType.SRAM
        )
    )

    # Data type configuration
    data_type: DataType = DataType.FLOAT32

    # Simulation parameters
    max_cycles: int = 1_000_000
    enable_power_modeling: bool = True
    enable_thermal_modeling: bool = False
    debug_mode: bool = False

    # Medical AI specific settings
    medical_mode: bool = False
    phi_compliance: bool = True
    precision_requirements: str = "medical_grade"  # medical_grade, research, prototype

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AcceleratorConfig":
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if "array" in data and isinstance(data["array"], dict):
            if "pe_config" in data["array"] and isinstance(
                data["array"]["pe_config"], dict
            ):
                data["array"]["pe_config"] = PEConfig(**data["array"]["pe_config"])
            data["array"] = ArrayConfig(**data["array"])

        for buffer_name in ["input_buffer", "weight_buffer", "output_buffer"]:
            if buffer_name in data and isinstance(data[buffer_name], dict):
                data[buffer_name] = MemoryConfig(**data[buffer_name])

        # Handle enums
        if "data_type" in data and isinstance(data["data_type"], str):
            data["data_type"] = DataType(data["data_type"])

        return cls(**data)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check array configuration
        total_pes = self.array.rows * self.array.cols
        if total_pes > 10000:
            issues.append(
                f"Large array size ({total_pes} PEs) may impact simulation performance"
            )

        # Check memory configuration
        total_memory = (
            self.input_buffer.buffer_size
            + self.weight_buffer.buffer_size
            + self.output_buffer.buffer_size
        )
        if total_memory > 1e6:
            issues.append(
                f"Large total memory ({total_memory} elements) may impact performance"
            )

        # Medical mode checks
        if self.medical_mode:
            if self.data_type in [DataType.INT8, DataType.FLOAT16]:
                issues.append(
                    "Low precision data types may not meet medical requirements"
                )
            if not self.phi_compliance:
                issues.append("PHI compliance disabled in medical mode")

        return issues


def load_config(path: Union[str, Path]) -> AcceleratorConfig:
    """Load configuration from file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        config = AcceleratorConfig.from_dict(data)
        logger.info(f"Configuration loaded from {path}")

        # Validate configuration
        issues = config.validate()
        if issues:
            logger.warning(f"Configuration issues: {'; '.join(issues)}")

        return config

    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}")
        raise


def save_config(config: AcceleratorConfig, path: Union[str, Path]) -> None:
    """Save configuration to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = config.to_dict()

        with open(path, "w") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif path.suffix.lower() == ".json":
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info(f"Configuration saved to {path}")

    except Exception as e:
        logger.error(f"Failed to save configuration to {path}: {e}")
        raise


def get_default_configs() -> Dict[str, AcceleratorConfig]:
    """Get pre-defined configuration templates."""
    configs = {}

    # Small research configuration
    configs["small"] = AcceleratorConfig(
        name="Small Research Array",
        array=ArrayConfig(rows=8, cols=8, frequency=500e6),
        data_type=DataType.FLOAT32,
    )

    # Large production configuration
    configs["large"] = AcceleratorConfig(
        name="Large Production Array",
        array=ArrayConfig(rows=64, cols=64, frequency=1.5e9),
        input_buffer=MemoryConfig(buffer_size=32768, bandwidth=128),
        weight_buffer=MemoryConfig(buffer_size=65536, bandwidth=256),
        output_buffer=MemoryConfig(buffer_size=16384, bandwidth=64),
        data_type=DataType.FLOAT32,
    )

    # Medical imaging optimized
    configs["medical"] = AcceleratorConfig(
        name="Medical Imaging Array",
        array=ArrayConfig(rows=32, cols=32, frequency=1.2e9),
        input_buffer=MemoryConfig(
            buffer_size=524288, bandwidth=512
        ),  # Large for images
        weight_buffer=MemoryConfig(buffer_size=131072, bandwidth=128),
        output_buffer=MemoryConfig(buffer_size=65536, bandwidth=64),
        data_type=DataType.FLOAT32,
        medical_mode=True,
        phi_compliance=True,
        precision_requirements="medical_grade",
        enable_power_modeling=True,
    )

    # Edge inference configuration
    configs["edge"] = AcceleratorConfig(
        name="Edge Inference Array",
        array=ArrayConfig(rows=16, cols=16, frequency=800e6, voltage=0.9),
        input_buffer=MemoryConfig(buffer_size=4096, bandwidth=32),
        weight_buffer=MemoryConfig(buffer_size=8192, bandwidth=64),
        output_buffer=MemoryConfig(buffer_size=2048, bandwidth=16),
        data_type=DataType.INT8,
        enable_power_modeling=True,
    )

    return configs


# Environment variable support
def load_config_from_env() -> Optional[AcceleratorConfig]:
    """Load configuration from environment variables."""
    env_config_path = os.getenv("OPENACCEL_CONFIG")
    if env_config_path:
        return load_config(env_config_path)
    return None


def get_config_search_paths() -> List[Path]:
    """Get standard configuration file search paths."""
    paths = []

    # Current directory
    paths.append(Path.cwd() / "openaccel_config.yaml")
    paths.append(Path.cwd() / "config.yaml")

    # User config directory
    if "HOME" in os.environ:
        home = Path(os.environ["HOME"])
        paths.append(home / ".config" / "openaccel" / "config.yaml")
        paths.append(home / ".openaccel.yaml")

    # System config directory
    paths.append(Path("/etc/openaccel/config.yaml"))

    return paths
