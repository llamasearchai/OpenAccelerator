"""
Comprehensive tests for configuration management.

Tests all configuration classes, validation, serialization, and file operations.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from open_accelerator.utils.config import (
    AcceleratorConfig, ArrayConfig, BufferConfig, PowerConfig, 
    MedicalConfig, PEConfig, WorkloadConfig,
    AcceleratorType, DataType, DataflowType, WorkloadType,
    get_default_configs, get_default_workload_configs,
    load_config, save_config, validate_config, apply_env_overrides,
    ConfigurationError, InvalidConfigurationError
)


class TestDataTypes:
    """Test data type enumerations."""
    
    def test_data_type_enum_values(self):
        """Test DataType enum has correct values."""
        assert DataType.INT8.value == "int8"
        assert DataType.INT16.value == "int16"
        assert DataType.INT32.value == "int32"
        assert DataType.FLOAT16.value == "float16"
        assert DataType.FLOAT32.value == "float32"
        assert DataType.BFLOAT16.value == "bfloat16"
    
    def test_accelerator_type_enum_values(self):
        """Test AcceleratorType enum has correct values."""
        assert AcceleratorType.MEDICAL.value == "medical"
        assert AcceleratorType.EDGE.value == "edge"
        assert AcceleratorType.DATACENTER.value == "datacenter"
        assert AcceleratorType.BALANCED.value == "balanced"
    
    def test_dataflow_type_enum_values(self):
        """Test DataflowType enum has correct values."""
        assert DataflowType.OUTPUT_STATIONARY.value == "output_stationary"
        assert DataflowType.WEIGHT_STATIONARY.value == "weight_stationary"
        assert DataflowType.INPUT_STATIONARY.value == "input_stationary"
    
    def test_workload_type_enum_values(self):
        """Test WorkloadType enum has correct values."""
        assert WorkloadType.GEMM.value == "gemm"
        assert WorkloadType.CONVOLUTION.value == "convolution"
        assert WorkloadType.MEDICAL_IMAGING.value == "medical_imaging"
        assert WorkloadType.CUSTOM.value == "custom"


class TestArrayConfig:
    """Test ArrayConfig class."""
    
    def test_array_config_defaults(self):
        """Test ArrayConfig default values."""
        config = ArrayConfig()
        assert config.rows == 16
        assert config.cols == 16
        assert config.frequency == 1e9
        assert config.voltage == 1.2
        assert config.dataflow == DataflowType.OUTPUT_STATIONARY
        assert config.pe_config is not None
    
    def test_array_config_custom_values(self):
        """Test ArrayConfig with custom values."""
        config = ArrayConfig(
            rows=32, cols=32, frequency=2e9, voltage=1.5
        )
        assert config.rows == 32
        assert config.cols == 32
        assert config.frequency == 2e9
        assert config.voltage == 1.5
    
    def test_array_config_validation(self):
        """Test ArrayConfig validation."""
        with pytest.raises(ValueError, match="Array dimensions must be positive"):
            ArrayConfig(rows=0, cols=16)
        
        with pytest.raises(ValueError, match="Array dimensions must be positive"):
            ArrayConfig(rows=16, cols=-1)
        
        with pytest.raises(ValueError, match="Frequency must be positive"):
            ArrayConfig(frequency=0)
        
        with pytest.raises(ValueError, match="Voltage must be positive"):
            ArrayConfig(voltage=-1.0)


class TestBufferConfig:
    """Test BufferConfig class."""
    
    def test_buffer_config_defaults(self):
        """Test BufferConfig default values."""
        config = BufferConfig()
        assert config.buffer_size == 1024
        assert config.bandwidth == 16
        assert config.latency == 1
    
    def test_buffer_config_custom_values(self):
        """Test BufferConfig with custom values."""
        config = BufferConfig(buffer_size=2048, bandwidth=32, latency=2)
        assert config.buffer_size == 2048
        assert config.bandwidth == 32
        assert config.latency == 2
    
    def test_buffer_config_validation(self):
        """Test BufferConfig validation."""
        with pytest.raises(ValueError, match="Buffer size must be positive"):
            BufferConfig(buffer_size=0)
        
        with pytest.raises(ValueError, match="Bandwidth must be positive"):
            BufferConfig(bandwidth=-1)
        
        with pytest.raises(ValueError, match="Latency must be non-negative"):
            BufferConfig(latency=-1)


class TestPowerConfig:
    """Test PowerConfig class."""
    
    def test_power_config_defaults(self):
        """Test PowerConfig default values."""
        config = PowerConfig()
        assert config.enable_power_gating is True
        assert config.enable_dvfs is True
        assert config.thermal_design_power == 100.0
        assert config.operating_temperature == 85.0
    
    def test_power_config_validation(self):
        """Test PowerConfig validation."""
        with pytest.raises(ValueError, match="TDP must be positive"):
            PowerConfig(thermal_design_power=0)
        
        with pytest.raises(ValueError, match="Operating temperature must be positive"):
            PowerConfig(operating_temperature=0)


class TestMedicalConfig:
    """Test MedicalConfig class."""
    
    def test_medical_config_defaults(self):
        """Test MedicalConfig default values."""
        config = MedicalConfig()
        assert config.enable_medical_mode is False
        assert config.dicom_support is True
        assert config.nifti_support is True
        assert config.phi_compliance is True
        assert config.fda_validation is False
        assert config.default_modality == "CT"
        assert config.max_image_size == (512, 512)
        assert "CT" in config.supported_modalities
        assert "MRI" in config.supported_modalities


class TestPEConfig:
    """Test PEConfig class."""
    
    def test_pe_config_defaults(self):
        """Test PEConfig default values."""
        config = PEConfig()
        assert config.enable_sparsity is False
        assert config.sparsity_threshold == 1e-6
        assert config.power_gating is False
        assert isinstance(config.extra_params, dict)
    
    def test_pe_config_validation(self):
        """Test PEConfig validation."""
        with pytest.raises(ValueError, match="sparsity_threshold must be non-negative"):
            PEConfig(sparsity_threshold=-1.0)


class TestWorkloadConfig:
    """Test WorkloadConfig class."""
    
    def test_workload_config_defaults(self):
        """Test WorkloadConfig default values."""
        config = WorkloadConfig(
            workload_type=WorkloadType.GEMM,
            gemm_M=8, gemm_K=8, gemm_P=8
        )
        assert config.workload_type == WorkloadType.GEMM
        assert config.name == "default_workload"
        assert config.data_type == DataType.FLOAT32
        assert config.random_seed == 42
        assert config.validate_output is True
        assert config.tolerance == 1e-6
    
    def test_gemm_workload_config(self):
        """Test GEMM workload configuration."""
        config = WorkloadConfig(
            workload_type=WorkloadType.GEMM,
            gemm_M=64, gemm_K=32, gemm_P=16
        )
        assert config.gemm_M == 64
        assert config.gemm_K == 32
        assert config.gemm_P == 16
    
    def test_convolution_workload_config(self):
        """Test convolution workload configuration."""
        config = WorkloadConfig(
            workload_type=WorkloadType.CONVOLUTION,
            conv_input_height=224, conv_input_width=224,
            conv_input_channels=3, conv_output_channels=64,
            conv_kernel_size=3, conv_stride=1, conv_padding=1
        )
        assert config.conv_input_height == 224
        assert config.conv_input_channels == 3
        assert config.conv_kernel_size == 3
    
    def test_medical_workload_config(self):
        """Test medical imaging workload configuration."""
        config = WorkloadConfig(
            workload_type=WorkloadType.MEDICAL_IMAGING,
            medical_modality="CT",
            medical_image_size=(512, 512),
            medical_slice_thickness=1.0,
            medical_pixel_spacing=(0.5, 0.5)
        )
        assert config.medical_modality == "CT"
        assert config.medical_image_size == (512, 512)
    
    def test_workload_config_validation(self):
        """Test workload configuration validation."""
        # Test GEMM validation - missing parameters
        with pytest.raises(ValueError, match="GEMM workload requires M, K, P dimensions"):
            WorkloadConfig(workload_type=WorkloadType.GEMM, gemm_M=64)
        
        # Test GEMM validation - zero dimensions  
        with pytest.raises(ValueError, match="GEMM dimensions must be positive"):
            WorkloadConfig(workload_type=WorkloadType.GEMM, gemm_M=-1, gemm_K=32, gemm_P=16)
        
        # Test convolution validation
        with pytest.raises(ValueError, match="Convolution workload requires all input/output dimensions"):
            WorkloadConfig(workload_type=WorkloadType.CONVOLUTION, conv_input_height=224)
        
        # Test medical imaging validation
        with pytest.raises(ValueError, match="Medical imaging workload requires modality"):
            WorkloadConfig(workload_type=WorkloadType.MEDICAL_IMAGING)


class TestAcceleratorConfig:
    """Test AcceleratorConfig class."""
    
    def test_accelerator_config_defaults(self):
        """Test AcceleratorConfig default values."""
        config = AcceleratorConfig()
        assert config.name == "OpenAccelerator"
        assert config.accelerator_type == AcceleratorType.BALANCED
        assert config.data_type == DataType.FLOAT32
        assert config.max_cycles == 1_000_000
        assert config.debug_mode is False
        assert config.enable_logging is True
        assert isinstance(config.array, ArrayConfig)
        assert isinstance(config.power, PowerConfig)
        assert isinstance(config.medical, MedicalConfig)
    
    def test_accelerator_config_medical_mode(self):
        """Test medical mode configuration."""
        config = AcceleratorConfig(accelerator_type=AcceleratorType.MEDICAL)
        assert config.medical.enable_medical_mode is True
        assert config.medical_mode is True
    
    def test_accelerator_config_property(self):
        """Test accelerator config properties."""
        config = AcceleratorConfig()
        config.medical.enable_medical_mode = True
        assert config.medical_mode is True


class TestDefaultConfigs:
    """Test default configuration functions."""
    
    def test_get_default_configs(self):
        """Test get_default_configs function."""
        configs = get_default_configs()
        assert isinstance(configs, dict)
        assert "small" in configs
        assert "large" in configs
        assert "medical" in configs
        
        # Test small config
        small_config = configs["small"]
        assert small_config.accelerator_type == AcceleratorType.EDGE
        assert small_config.array.rows == 8
        assert small_config.array.cols == 8
        
        # Test medical config
        medical_config = configs["medical"]
        assert medical_config.accelerator_type == AcceleratorType.MEDICAL
        assert medical_config.medical.enable_medical_mode is True
    
    def test_get_default_workload_configs(self):
        """Test get_default_workload_configs function."""
        configs = get_default_workload_configs()
        assert isinstance(configs, dict)
        assert "gemm_small" in configs
        assert "gemm_large" in configs
        assert "conv_2d" in configs
        assert "medical_ct" in configs
        
        # Test GEMM configs
        gemm_small = configs["gemm_small"]
        assert gemm_small.workload_type == WorkloadType.GEMM
        assert gemm_small.gemm_M == 8
        
        # Test medical config
        medical_ct = configs["medical_ct"]
        assert medical_ct.workload_type == WorkloadType.MEDICAL_IMAGING
        assert medical_ct.medical_modality == "CT"


class TestConfigFileFunctions:
    """Test configuration file operations."""
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            
            # Create test configuration
            original_config = AcceleratorConfig(
                name="TestAccelerator",
                accelerator_type=AcceleratorType.MEDICAL,
                data_type=DataType.FLOAT16,
                max_cycles=500_000
            )
            
            # Save configuration
            save_config(original_config, config_path)
            assert config_path.exists()
            
            # Load configuration
            loaded_config = load_config(config_path)
            assert loaded_config.name == "TestAccelerator"
            assert loaded_config.accelerator_type == AcceleratorType.MEDICAL
            assert loaded_config.data_type == DataType.FLOAT16
            assert loaded_config.max_cycles == 500_000
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_save_config_directory_creation(self):
        """Test configuration saving creates directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "subdir" / "test_config.yaml"
            config = AcceleratorConfig()
            
            save_config(config, config_path)
            assert config_path.exists()
            assert config_path.parent.exists()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_config_warnings(self):
        """Test configuration validation warnings."""
        # Test large array warning
        config = AcceleratorConfig()
        config.array.rows = 64
        config.array.cols = 64
        warnings = validate_config(config)
        assert any("Large array size" in warning for warning in warnings)
        
        # Test buffer size warning
        config = AcceleratorConfig()
        config.input_buffer.buffer_size = 10
        warnings = validate_config(config)
        assert any("Input buffer size" in warning for warning in warnings)
        
        # Test low TDP warning
        config = AcceleratorConfig()
        config.power.thermal_design_power = 0.5
        warnings = validate_config(config)
        assert any("Very low TDP" in warning for warning in warnings)
        
        # Test medical mode warning
        config = AcceleratorConfig()
        config.medical.enable_medical_mode = True
        config.medical.phi_compliance = False
        warnings = validate_config(config)
        assert any("PHI compliance disabled" in warning for warning in warnings)
    
    def test_validate_config_no_warnings(self):
        """Test configuration validation with no warnings."""
        config = AcceleratorConfig()
        warnings = validate_config(config)
        assert isinstance(warnings, list)


class TestEnvironmentOverrides:
    """Test environment variable overrides."""
    
    def test_apply_env_overrides(self, monkeypatch):
        """Test applying environment variable overrides."""
        config = AcceleratorConfig()
        
        # Set environment variables
        monkeypatch.setenv("OPENACCEL_ARRAY_ROWS", "32")
        monkeypatch.setenv("OPENACCEL_ARRAY_COLS", "64")
        monkeypatch.setenv("OPENACCEL_FREQUENCY", "2000000000")
        monkeypatch.setenv("OPENACCEL_DATA_TYPE", "int8")
        monkeypatch.setenv("OPENACCEL_DEBUG", "true")
        monkeypatch.setenv("OPENACCEL_MEDICAL_MODE", "yes")
        
        # Apply overrides
        config = apply_env_overrides(config)
        
        # Check overrides applied
        assert config.array.rows == 32
        assert config.array.cols == 64
        assert config.array.frequency == 2e9
        assert config.data_type == DataType.INT8
        assert config.debug_mode is True
        assert config.medical.enable_medical_mode is True
    
    def test_apply_env_overrides_no_env_vars(self):
        """Test applying overrides with no environment variables."""
        config = AcceleratorConfig()
        original_rows = config.array.rows
        
        config = apply_env_overrides(config)
        assert config.array.rows == original_rows


class TestConfigurationExceptions:
    """Test configuration exceptions."""
    
    def test_configuration_error_hierarchy(self):
        """Test configuration error class hierarchy."""
        assert issubclass(InvalidConfigurationError, ConfigurationError)
        assert issubclass(ConfigurationError, Exception)
    
    def test_configuration_errors_can_be_raised(self):
        """Test that configuration errors can be raised."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test error")
        
        with pytest.raises(InvalidConfigurationError):
            raise InvalidConfigurationError("Test invalid config") 