"""
Configuration management for OpenAccelerator.

Provides comprehensive configuration classes and utilities for all aspects
of the accelerator simulation system.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for accelerator operations."""
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


# ---------------------------------------------------------------------------
# Array dataflow enumeration – used by advanced core models
# ---------------------------------------------------------------------------

class DataflowType(Enum):
    """Supported dataflow patterns for the systolic array."""

    OUTPUT_STATIONARY = "output_stationary"
    WEIGHT_STATIONARY = "weight_stationary"
    INPUT_STATIONARY = "input_stationary"


class AcceleratorType(Enum):
    """Types of accelerator configurations."""
    MEDICAL = "medical"
    EDGE = "edge"
    DATACENTER = "datacenter"
    BALANCED = "balanced"


class WorkloadType(Enum):
    """Supported workload types."""
    GEMM = "gemm"
    CONVOLUTION = "convolution"
    MEDICAL_IMAGING = "medical_imaging"
    CUSTOM = "custom"


class MemoryType(Enum):
    """Memory types supported by the accelerator."""
    SRAM = "sram"
    DRAM = "dram"
    HBM = "hbm"
    CACHE = "cache"


@dataclass
class ArrayConfig:
    """Configuration for the systolic array."""
    rows: int = 16
    cols: int = 16
    frequency: float = 1e9  # 1 GHz
    voltage: float = 1.2  # 1.2V

    # Advanced parameters
    dataflow: DataflowType = DataflowType.OUTPUT_STATIONARY
    pe_config: 'PEConfig' = field(default_factory=lambda: PEConfig())

    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Array dimensions must be positive")
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if self.voltage <= 0:
            raise ValueError("Voltage must be positive")

    def __setattr__(self, name: str, value):
        """Validate attribute changes."""
        if name == "rows" and value <= 0:
            raise ValueError("Array dimensions must be positive")
        elif name == "cols" and value <= 0:
            raise ValueError("Array dimensions must be positive")
        elif name == "frequency" and value <= 0:
            raise ValueError("Frequency must be positive")
        elif name == "voltage" and value <= 0:
            raise ValueError("Voltage must be positive")
        super().__setattr__(name, value)


@dataclass
class BufferConfig:
    """Configuration for memory buffers."""
    buffer_size: int = 1024  # Elements
    bandwidth: int = 16  # Elements per cycle
    latency: int = 1  # Cycles

    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        if self.latency < 0:
            raise ValueError("Latency must be non-negative")


@dataclass
class PowerConfig:
    """Configuration for power management."""
    enable_power_gating: bool = True
    enable_dvfs: bool = True
    thermal_design_power: float = 100.0  # Watts
    operating_temperature: float = 85.0  # Celsius

    def __post_init__(self):
        if self.thermal_design_power <= 0:
            raise ValueError("TDP must be positive")
        if self.operating_temperature <= 0:
            raise ValueError("Operating temperature must be positive")


@dataclass
class MedicalConfig:
    """Configuration for medical AI features."""
    enable_medical_mode: bool = False
    dicom_support: bool = True
    nifti_support: bool = True
    phi_compliance: bool = True
    fda_validation: bool = False

    # Medical imaging parameters
    default_modality: str = "CT"
    max_image_size: tuple[int, int] = (512, 512)
    supported_modalities: List[str] = field(default_factory=lambda: ["CT", "MRI", "X-Ray", "Ultrasound"])


@dataclass
class MemoryConfig:
    """Configuration for memory buffers."""
    buffer_size: int = 1024
    bandwidth: int = 16
    latency: int = 1
    memory_type: MemoryType = MemoryType.SRAM
    energy_per_access: float = 0.1  # pJ per bit
    
    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive")
        if self.latency < 0:
            raise ValueError("Latency must be non-negative")


@dataclass
class MemoryHierarchyConfig:
    """Configuration for memory hierarchy including caches and main memory."""
    # L1 Cache configuration
    l1_size: int = 8192  # 8KB
    l1_bandwidth: int = 32  # Elements per cycle
    l1_latency: int = 1  # Cycles
    l1_associativity: int = 4
    
    # L2 Cache configuration  
    l2_size: int = 65536  # 64KB
    l2_bandwidth: int = 16  # Elements per cycle
    l2_latency: int = 8  # Cycles
    l2_associativity: int = 8
    
    # Main memory configuration
    main_memory_size: int = 134217728  # 128MB
    main_memory_bandwidth: int = 8  # Elements per cycle
    main_memory_latency: int = 100  # Cycles
    enable_hbm: bool = False  # Use HBM instead of DRAM
    
    # Memory management
    enable_prefetching: bool = True
    prefetch_distance: int = 4
    write_policy: str = "write_through"  # "write_through" or "write_back"
    replacement_policy: str = "LRU"  # "LRU", "FIFO", "RANDOM"
    
    def __post_init__(self):
        if self.l1_size <= 0 or self.l2_size <= 0 or self.main_memory_size <= 0:
            raise ValueError("Memory sizes must be positive")
        if self.l1_bandwidth <= 0 or self.l2_bandwidth <= 0 or self.main_memory_bandwidth <= 0:
            raise ValueError("Memory bandwidths must be positive")
        if self.l1_latency < 0 or self.l2_latency < 0 or self.main_memory_latency < 0:
            raise ValueError("Memory latencies must be non-negative")
        if self.l1_size >= self.l2_size:
            raise ValueError("L1 cache size must be smaller than L2 cache size")
        if self.l2_size >= self.main_memory_size:
            raise ValueError("L2 cache size must be smaller than main memory size")


@dataclass
class PEConfig:
    """Configuration parameters specific to a Processing Element (PE)."""
    # Sparsity support
    enable_sparsity: bool = False
    sparsity_threshold: float = 1e-6  # Values with magnitude below this are treated as zero

    # Power management
    power_gating: bool = False  # Enable power gating when PE is idle

    # Reserved for future extensions (e.g., fault-tolerance, voltage scaling)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.sparsity_threshold < 0:
            raise ValueError("sparsity_threshold must be non-negative")


@dataclass
class WorkloadConfig:
    """Configuration for workloads."""
    workload_type: WorkloadType = WorkloadType.GEMM
    name: str = "default_workload"

    # GEMM parameters
    gemm_M: Optional[int] = None
    gemm_K: Optional[int] = None
    gemm_P: Optional[int] = None

    # Convolution parameters
    conv_input_height: Optional[int] = None
    conv_input_width: Optional[int] = None
    conv_input_channels: Optional[int] = None
    conv_output_channels: Optional[int] = None
    conv_kernel_size: Optional[int] = None
    conv_stride: Optional[int] = None
    conv_padding: Optional[int] = None

    # Medical imaging parameters
    medical_modality: Optional[str] = None
    medical_image_size: Optional[Tuple[int, int]] = None
    medical_slice_thickness: Optional[float] = None
    medical_pixel_spacing: Optional[Tuple[float, float]] = None

    # Data generation parameters
    data_type: DataType = DataType.FLOAT32
    random_seed: int = 42

    # Validation parameters
    validate_output: bool = True
    tolerance: float = 1e-6

    def __post_init__(self):
        """Validate workload configuration."""
        if self.workload_type == WorkloadType.GEMM:
            if not all([self.gemm_M, self.gemm_K, self.gemm_P]):
                raise ValueError("GEMM workload requires M, K, P dimensions")
            if any(dim <= 0 for dim in [self.gemm_M, self.gemm_K, self.gemm_P] if dim is not None):
                raise ValueError("GEMM dimensions must be positive")

        elif self.workload_type == WorkloadType.CONVOLUTION:
            required_params = [
                self.conv_input_height, self.conv_input_width,
                self.conv_input_channels, self.conv_output_channels,
                self.conv_kernel_size
            ]
            if not all(param is not None for param in required_params):
                raise ValueError("Convolution workload requires all input/output dimensions")
            if any(param <= 0 for param in required_params if param is not None):
                raise ValueError("Convolution dimensions must be positive")

        elif self.workload_type == WorkloadType.MEDICAL_IMAGING:
            if not self.medical_modality:
                raise ValueError("Medical imaging workload requires modality")
            if not self.medical_image_size:
                raise ValueError("Medical imaging workload requires image size")


@dataclass
class AcceleratorConfig:
    """Complete accelerator configuration."""
    name: str = "OpenAccelerator"
    accelerator_type: AcceleratorType = AcceleratorType.BALANCED
    data_type: DataType = DataType.FLOAT32

    # Core components
    array: ArrayConfig = field(default_factory=ArrayConfig)
    input_buffer: BufferConfig = field(default_factory=BufferConfig)
    weight_buffer: BufferConfig = field(default_factory=BufferConfig)
    output_buffer: BufferConfig = field(default_factory=BufferConfig)

    # Memory hierarchy
    memory: MemoryHierarchyConfig = field(default_factory=MemoryHierarchyConfig)

    # Advanced features
    power: PowerConfig = field(default_factory=PowerConfig)
    medical: MedicalConfig = field(default_factory=MedicalConfig)

    # Simulation parameters
    max_cycles: int = 1_000_000
    debug_mode: bool = False
    enable_logging: bool = True
    
    # Advanced modeling features
    enable_thermal_modeling: bool = False
    enable_power_modeling: bool = False

    def __post_init__(self):
        # Validate configuration consistency
        if self.accelerator_type == AcceleratorType.MEDICAL:
            self.medical.enable_medical_mode = True

        # Adjust buffer sizes based on array size
        if self.input_buffer.buffer_size < self.array.rows * self.array.cols:
            logger.warning("Input buffer size may be too small for array dimensions")

        # Adjust memory hierarchy for accelerator type
        if self.accelerator_type == AcceleratorType.EDGE:
            # Reduce memory sizes for edge devices
            self.memory.l1_size = min(self.memory.l1_size, 4096)  # 4KB max
            self.memory.l2_size = min(self.memory.l2_size, 32768)  # 32KB max
            self.memory.main_memory_size = min(self.memory.main_memory_size, 16777216)  # 16MB max
        elif self.accelerator_type == AcceleratorType.DATACENTER:
            # Increase memory sizes for datacenter
            self.memory.enable_hbm = True
            self.memory.main_memory_size = max(self.memory.main_memory_size, 1073741824)  # 1GB min

    @property
    def medical_mode(self) -> bool:
        """Convenience property for medical mode status."""
        return self.medical.enable_medical_mode
    
    # Backward compatibility properties
    @property
    def array_rows(self) -> int:
        """Convenience property for array rows."""
        return self.array.rows
    
    @array_rows.setter
    def array_rows(self, value: int):
        """Setter for array rows."""
        if value <= 0:
            raise ValueError("Array rows must be positive")
        self.array.rows = value
    
    @property
    def array_cols(self) -> int:
        """Convenience property for array columns."""
        return self.array.cols
    
    @array_cols.setter
    def array_cols(self, value: int):
        """Setter for array columns."""
        if value <= 0:
            raise ValueError("Array columns must be positive")
        self.array.cols = value
    
    @property
    def frequency_mhz(self) -> float:
        """Convenience property for frequency in MHz."""
        return self.array.frequency / 1e6
    
    @frequency_mhz.setter
    def frequency_mhz(self, value: float):
        """Setter for frequency in MHz."""
        self.array.frequency = value * 1e6
    
    @property
    def memory_hierarchy(self) -> Optional[MemoryHierarchyConfig]:
        """Convenience property for memory hierarchy."""
        return self.memory
    
    @memory_hierarchy.setter  
    def memory_hierarchy(self, value: Optional[MemoryHierarchyConfig]):
        """Setter for memory hierarchy."""
        if value is not None:
            self.memory = value


# Configuration exceptions
class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class ConfigurationFileError(ConfigurationError):
    """Raised when configuration file operations fail."""
    pass


# Default configurations for different accelerator types
def get_default_configs() -> Dict[str, AcceleratorConfig]:
    """Get default configurations for different accelerator types."""

    configs = {}

    # Small/Edge configuration
    configs["small"] = AcceleratorConfig(
        name="EdgeAccelerator",
        accelerator_type=AcceleratorType.EDGE,
        data_type=DataType.INT8,
        array=ArrayConfig(rows=8, cols=8, frequency=500e6, voltage=1.0),
        input_buffer=BufferConfig(buffer_size=512, bandwidth=8),
        weight_buffer=BufferConfig(buffer_size=512, bandwidth=8),
        output_buffer=BufferConfig(buffer_size=256, bandwidth=8),
        power=PowerConfig(thermal_design_power=10.0, enable_dvfs=True),
    )

    # Large/Datacenter configuration
    configs["large"] = AcceleratorConfig(
        name="DatacenterAccelerator",
        accelerator_type=AcceleratorType.DATACENTER,
        data_type=DataType.FLOAT32,
        array=ArrayConfig(rows=32, cols=32, frequency=2e9, voltage=1.5),
        input_buffer=BufferConfig(buffer_size=4096, bandwidth=64),
        weight_buffer=BufferConfig(buffer_size=4096, bandwidth=64),
        output_buffer=BufferConfig(buffer_size=2048, bandwidth=32),
        power=PowerConfig(thermal_design_power=300.0, enable_power_gating=True),
    )

    # Medical configuration
    configs["medical"] = AcceleratorConfig(
        name="MedicalAccelerator",
        accelerator_type=AcceleratorType.MEDICAL,
        data_type=DataType.FLOAT32,
        array=ArrayConfig(rows=16, cols=16, frequency=1e9, voltage=1.2),
        input_buffer=BufferConfig(buffer_size=2048, bandwidth=32),
        weight_buffer=BufferConfig(buffer_size=2048, bandwidth=32),
        output_buffer=BufferConfig(buffer_size=1024, bandwidth=16),
        power=PowerConfig(thermal_design_power=50.0, enable_power_gating=True),
        medical=MedicalConfig(
            enable_medical_mode=True,
            dicom_support=True,
            nifti_support=True,
            phi_compliance=True,
            fda_validation=True,
        ),
    )

    return configs


# Default workload configurations
def get_default_workload_configs() -> Dict[str, WorkloadConfig]:
    """Get default workload configurations."""

    configs = {}

    # Small GEMM
    configs["gemm_small"] = WorkloadConfig(
        workload_type=WorkloadType.GEMM,
        name="Small GEMM",
        gemm_M=8,
        gemm_K=8,
        gemm_P=8,
    )

    # Large GEMM
    configs["gemm_large"] = WorkloadConfig(
        workload_type=WorkloadType.GEMM,
        name="Large GEMM",
        gemm_M=1024,
        gemm_K=1024,
        gemm_P=1024,
    )

    # Convolution
    configs["conv_2d"] = WorkloadConfig(
        workload_type=WorkloadType.CONVOLUTION,
        name="2D Convolution",
        conv_input_height=224,
        conv_input_width=224,
        conv_input_channels=3,
        conv_output_channels=64,
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
    )

    # Medical imaging
    configs["medical_ct"] = WorkloadConfig(
        workload_type=WorkloadType.MEDICAL_IMAGING,
        name="CT Scan Processing",
        medical_modality="CT",
        medical_image_size=(512, 512),
        medical_slice_thickness=1.0,
        medical_pixel_spacing=(0.5, 0.5),
    )

    return configs


def load_config(config_path: Union[str, Path]) -> AcceleratorConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Convert nested dictionaries to dataclass instances
        config = _dict_to_config(config_data)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def save_config(config: AcceleratorConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert dataclass to dictionary
        config_dict = _config_to_dict(config)

        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def _dict_to_config(config_dict: Dict[str, Any]) -> AcceleratorConfig:
    """Convert dictionary to AcceleratorConfig."""

    # Handle enums
    if 'accelerator_type' in config_dict:
        config_dict['accelerator_type'] = AcceleratorType(config_dict['accelerator_type'])

    if 'data_type' in config_dict:
        config_dict['data_type'] = DataType(config_dict['data_type'])

    # Handle nested configurations
    if 'array' in config_dict:
        config_dict['array'] = ArrayConfig(**config_dict['array'])

    if 'input_buffer' in config_dict:
        config_dict['input_buffer'] = BufferConfig(**config_dict['input_buffer'])

    if 'weight_buffer' in config_dict:
        config_dict['weight_buffer'] = BufferConfig(**config_dict['weight_buffer'])

    if 'output_buffer' in config_dict:
        config_dict['output_buffer'] = BufferConfig(**config_dict['output_buffer'])

    if 'power' in config_dict:
        config_dict['power'] = PowerConfig(**config_dict['power'])

    if 'medical' in config_dict:
        config_dict['medical'] = MedicalConfig(**config_dict['medical'])

    return AcceleratorConfig(**config_dict)


def _config_to_dict(config: AcceleratorConfig) -> Dict[str, Any]:
    """Convert AcceleratorConfig to dictionary."""

    def _dataclass_to_dict(obj):
        """Recursively convert dataclass to dictionary."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_value in obj.__dict__.items():
                if isinstance(field_value, Enum):
                    result[field_name] = field_value.value
                elif hasattr(field_value, '__dataclass_fields__'):
                    result[field_name] = _dataclass_to_dict(field_value)
                else:
                    result[field_name] = field_value
            return result
        else:
            return obj

    result = _dataclass_to_dict(config)
    # Ensure we return a proper dictionary
    if isinstance(result, dict):
        return result
    else:
        return {}


def get_config_template() -> str:
    """Get a YAML configuration template."""
    template = """
# OpenAccelerator Configuration Template

name: "MyAccelerator"
accelerator_type: "balanced"  # Options: medical, edge, datacenter, balanced
data_type: "float32"  # Options: int8, int16, int32, float16, float32, bfloat16

# Systolic Array Configuration
array:
  rows: 16
  cols: 16
  frequency: 1000000000  # 1 GHz
  voltage: 1.2  # 1.2V

# Buffer Configurations
input_buffer:
  buffer_size: 1024  # Elements
  bandwidth: 16  # Elements per cycle
  latency: 1  # Cycles

weight_buffer:
  buffer_size: 1024
  bandwidth: 16
  latency: 1

output_buffer:
  buffer_size: 512
  bandwidth: 8
  latency: 1

# Power Management
power:
  enable_power_gating: true
  enable_dvfs: true
  thermal_design_power: 100.0  # Watts
  operating_temperature: 85.0  # Celsius

# Medical AI Features
medical:
  enable_medical_mode: false
  dicom_support: true
  nifti_support: true
  phi_compliance: true
  fda_validation: false
  default_modality: "CT"
  max_image_size: [512, 512]
  supported_modalities: ["CT", "MRI", "X-Ray", "Ultrasound"]

# Simulation Parameters
max_cycles: 1000000
debug_mode: false
enable_logging: true
"""
    return template.strip()


def validate_config(config: AcceleratorConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []

    # Check array dimensions
    if config.array.rows * config.array.cols > 1024:
        warnings.append("Large array size may impact simulation performance")

    # Check buffer sizes
    min_buffer_size = config.array.rows * config.array.cols
    if config.input_buffer.buffer_size < min_buffer_size:
        warnings.append(f"Input buffer size ({config.input_buffer.buffer_size}) may be too small for array size ({min_buffer_size})")

    # Check power settings
    if config.power.thermal_design_power < 1.0:
        warnings.append("Very low TDP may not be realistic")

    # Check medical configuration
    if config.medical.enable_medical_mode and not config.medical.phi_compliance:
        warnings.append("Medical mode enabled but PHI compliance disabled")

    return warnings


# Environment variable overrides
def apply_env_overrides(config: AcceleratorConfig) -> AcceleratorConfig:
    """Apply environment variable overrides to configuration."""

    # Array configuration
    array_rows = os.getenv("OPENACCEL_ARRAY_ROWS")
    if array_rows:
        config.array.rows = int(array_rows)

    array_cols = os.getenv("OPENACCEL_ARRAY_COLS")
    if array_cols:
        config.array.cols = int(array_cols)

    frequency = os.getenv("OPENACCEL_FREQUENCY")
    if frequency:
        config.array.frequency = float(frequency)

    # Data type
    data_type = os.getenv("OPENACCEL_DATA_TYPE")
    if data_type:
        config.data_type = DataType(data_type)

    # Debug mode
    debug_mode = os.getenv("OPENACCEL_DEBUG")
    if debug_mode:
        config.debug_mode = debug_mode.lower() in ("true", "1", "yes")

    # Medical mode
    medical_mode = os.getenv("OPENACCEL_MEDICAL_MODE")
    if medical_mode:
        config.medical.enable_medical_mode = medical_mode.lower() in ("true", "1", "yes")

    logger.info("Applied environment variable overrides")
    return config
