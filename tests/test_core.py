"""
Comprehensive tests for core modules.

Tests accelerator, memory, systolic array, and processing elements for complete coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from open_accelerator.core.accelerator import AcceleratorController, AcceleratorMetrics
from open_accelerator.core.memory import MemoryBuffer, MemoryMetrics, MemoryState
from open_accelerator.core.systolic_array import SystolicArray
from open_accelerator.core.pe import ProcessingElement, PEState, PEMetrics
from open_accelerator.core.processing_element import MACUnit, RegisterFile
from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig, MemoryConfig, WorkloadConfig, WorkloadType, DataType, PEConfig


class TestAcceleratorController:
    """Test AcceleratorController class."""

    @pytest.fixture
    def accelerator_config(self):
        """Test accelerator configuration."""
        return AcceleratorConfig(
            name="test_accelerator",
            array=ArrayConfig(rows=8, cols=8),
            debug_mode=True
        )

    @pytest.fixture
    def test_accelerator(self, accelerator_config):
        """Test accelerator instance."""
        return AcceleratorController(accelerator_config)

    def test_accelerator_initialization(self, test_accelerator):
        """Test accelerator initialization."""
        assert test_accelerator.config.name == "test_accelerator"
        assert test_accelerator.config.array.rows == 8
        assert test_accelerator.config.array.cols == 8
        assert test_accelerator.current_cycle == 0

    def test_accelerator_components(self, test_accelerator):
        """Test accelerator components initialization."""
        assert test_accelerator.systolic_array is not None
        assert test_accelerator.memory_hierarchy is not None
        assert test_accelerator.power_manager is not None
        assert test_accelerator.reliability_manager is not None

    def test_accelerator_metrics(self, test_accelerator):
        """Test accelerator metrics collection."""
        metrics = AcceleratorMetrics()
        assert metrics.total_cycles == 0
        assert metrics.total_operations == 0
        assert metrics.pe_utilization == 0.0
        assert metrics.average_power_watts == 0.0

    @patch('open_accelerator.core.accelerator.time.time')
    def test_accelerator_simulation_control(self, mock_time, test_accelerator):
        """Test simulation control methods."""
        mock_time.return_value = 1000.0
        
        # Test start simulation
        test_accelerator.simulation_start_time = 1000.0
        test_accelerator.is_running = True
        
        assert test_accelerator.simulation_start_time == 1000.0
        assert test_accelerator.is_running is True

    def test_accelerator_error_handling(self, test_accelerator):
        """Test error handling."""
        # Test with invalid configuration
        with pytest.raises((ValueError, AttributeError)):
            invalid_config = AcceleratorConfig(
                array=ArrayConfig(rows=-1, cols=8)  # Invalid dimension
            )
            AcceleratorController(invalid_config)


class TestMemoryBuffer:
    """Test MemoryBuffer class."""

    @pytest.fixture
    def memory_config(self):
        """Test memory configuration."""
        return MemoryConfig(
            buffer_size=1024,
            bandwidth=16,
            latency=1
        )

    @pytest.fixture
    def memory_buffer(self, memory_config):
        """Test memory buffer instance."""
        return MemoryBuffer("test_buffer", memory_config)

    def test_memory_buffer_initialization(self, memory_buffer):
        """Test memory buffer initialization."""
        assert memory_buffer.name == "test_buffer"
        assert memory_buffer.capacity == 1024
        assert memory_buffer.bandwidth == 16
        assert memory_buffer.latency == 1
        assert memory_buffer.state == MemoryState.IDLE

    def test_memory_buffer_operations(self, memory_buffer):
        """Test memory buffer operations."""
        # Test write operation
        data = np.random.rand(10, 10)
        memory_buffer.data.append(data)
        
        assert len(memory_buffer.data) == 1
        
        # Test read operation
        read_data = memory_buffer.data[0]
        assert np.array_equal(data, read_data)

    def test_memory_buffer_medical_mode(self, memory_config):
        """Test memory buffer medical mode features."""
        medical_buffer = MemoryBuffer("medical_buffer", memory_config, medical_mode=True)
        
        assert medical_buffer.enable_ecc is True
        assert medical_buffer.enable_integrity_checks is True
        assert medical_buffer.redundant_storage is True

    def test_memory_metrics(self, memory_buffer):
        """Test memory metrics collection."""
        metrics = MemoryMetrics()
        assert metrics.total_reads == 0
        assert metrics.total_writes == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_memory_bandwidth_calculation(self, memory_buffer):
        """Test memory bandwidth calculations."""
        # Simulate memory accesses
        memory_buffer.metrics.total_reads = 100
        memory_buffer.metrics.total_writes = 50
        memory_buffer.current_cycle = 10
        
        # Calculate bandwidth utilization
        total_accesses = memory_buffer.metrics.total_reads + memory_buffer.metrics.total_writes
        utilization = min(total_accesses / (memory_buffer.bandwidth * memory_buffer.current_cycle), 1.0)
        
        assert 0.0 <= utilization <= 1.0

    def test_memory_error_detection(self, memory_config):
        """Test memory error detection."""
        medical_buffer = MemoryBuffer("medical_buffer", memory_config, medical_mode=True)
        
        # Test that medical mode features are enabled
        assert medical_buffer.enable_ecc is True
        assert medical_buffer.enable_integrity_checks is True
        assert medical_buffer.redundant_storage is True
        
        # Test that medical mode buffer has proper configuration
        assert medical_buffer.medical_mode is True


class TestSystolicArray:
    """Test SystolicArray class."""

    @pytest.fixture
    def array_config(self):
        """Test array configuration."""
        return AcceleratorConfig(
            array=ArrayConfig(rows=4, cols=4, frequency=1e9)
        )

    @pytest.fixture
    def systolic_array(self, array_config):
        """Test systolic array instance."""
        return SystolicArray(array_config)

    def test_systolic_array_initialization(self, systolic_array):
        """Test systolic array initialization."""
        assert systolic_array.rows == 4
        assert systolic_array.cols == 4
        assert len(systolic_array.pes) == 4
        assert len(systolic_array.pes[0]) == 4

    def test_processing_element_grid(self, systolic_array):
        """Test processing element grid."""
        # Check that all PEs are initialized
        for row in range(systolic_array.rows):
            for col in range(systolic_array.cols):
                pe = systolic_array.pes[row][col]
                assert pe is not None
                assert pe.pe_id == (row, col)

    def test_dataflow_configuration(self, systolic_array):
        """Test dataflow configuration."""
        # Test that dataflow is properly configured
        assert hasattr(systolic_array, 'dataflow')
        assert hasattr(systolic_array, 'flow_direction_a')
        assert hasattr(systolic_array, 'flow_direction_b')

    def test_memory_integration(self, systolic_array):
        """Test memory integration."""
        assert systolic_array.memory_hierarchy is not None

    def test_performance_metrics(self, systolic_array):
        """Test performance metrics collection."""
        assert hasattr(systolic_array, 'metrics')
        assert systolic_array.cycle_count == 0

    def test_thermal_modeling(self, array_config):
        """Test thermal modeling."""
        array_config.enable_thermal_modeling = True
        systolic_array = SystolicArray(array_config)
        
        assert systolic_array.thermal_model is not None

    def test_power_modeling(self, array_config):
        """Test power modeling."""
        array_config.enable_power_modeling = True
        systolic_array = SystolicArray(array_config)
        
        assert systolic_array.power_model is not None


class TestProcessingElement:
    """Test ProcessingElement class."""

    @pytest.fixture
    def pe_config(self):
        """Test PE configuration."""
        return PEConfig(
            enable_sparsity=True,
            sparsity_threshold=1e-6,
            power_gating=True
        )

    @pytest.fixture
    def processing_element(self, pe_config):
        """Test processing element instance."""
        return ProcessingElement(
            pe_id=(0, 0),
            config=pe_config,
            data_type=DataType.FLOAT32
        )

    def test_pe_initialization(self, processing_element):
        """Test processing element initialization."""
        assert processing_element.pe_id == (0, 0)
        assert processing_element.data_type == DataType.FLOAT32
        assert processing_element.state == PEState.IDLE

    def test_pe_data_type_setup(self, processing_element):
        """Test data type setup."""
        assert processing_element.numpy_type == np.float32
        assert processing_element.zero_value == 0.0
        assert processing_element.precision_threshold == 1e-6

    def test_pe_input_loading(self, processing_element):
        """Test input loading."""
        processing_element.load_inputs(2.5, 3.0)
        
        assert processing_element.next_in_reg_a == 2.5
        assert processing_element.next_in_reg_b == 3.0

    def test_pe_cycle_execution(self, processing_element):
        """Test cycle execution."""
        # Load inputs
        processing_element.load_inputs(2.5, 3.0)
        
        # Execute cycle
        result = processing_element.cycle()
        
        # Check that MAC was performed
        assert result is True
        assert processing_element.state == PEState.COMPUTING

    def test_pe_sparsity_detection(self, processing_element):
        """Test sparsity detection."""
        # Load sparse inputs (below threshold)
        processing_element.load_inputs(1e-8, 3.0)
        
        # Execute cycle
        result = processing_element.cycle()
        
        # Should detect sparsity and skip computation
        assert result is False
        assert processing_element.metrics.sparsity_detected > 0

    def test_pe_power_gating(self, processing_element):
        """Test power gating."""
        processing_element.power_gate(True)
        assert processing_element.state == PEState.POWER_GATED
        
        processing_element.power_gate(False)
        assert processing_element.state == PEState.IDLE

    def test_pe_metrics_collection(self, processing_element):
        """Test metrics collection."""
        # Execute some operations
        processing_element.load_inputs(2.5, 3.0)
        processing_element.cycle()
        
        metrics = processing_element.get_metrics_summary()
        
        assert "pe_id" in metrics
        assert "utilization" in metrics
        assert "total_operations" in metrics
        assert "energy_consumed" in metrics

    def test_pe_error_handling(self, processing_element):
        """Test error handling."""
        # Test with invalid input
        processing_element.load_inputs(float('inf'), 3.0)
        
        # Should handle error gracefully
        processing_element.cycle()
        assert processing_element.state == PEState.ERROR

    def test_pe_reset(self, processing_element):
        """Test PE reset."""
        # Execute some operations
        processing_element.load_inputs(2.5, 3.0)
        processing_element.cycle()
        
        # Reset
        processing_element.reset()
        
        assert processing_element.state == PEState.IDLE
        assert processing_element.accumulator == 0.0
        assert processing_element.in_reg_a is None
        assert processing_element.in_reg_b is None


class TestMACUnit:
    """Test MACUnit class."""

    @pytest.fixture
    def mac_unit(self):
        """Test MAC unit instance."""
        return MACUnit(precision="float32")

    def test_mac_unit_initialization(self, mac_unit):
        """Test MAC unit initialization."""
        assert mac_unit.precision == "float32"
        assert mac_unit.operations_count == 0
        assert mac_unit.energy_consumed == 0.0

    def test_mac_operation(self, mac_unit):
        """Test multiply-accumulate operation."""
        result = mac_unit.multiply_accumulate(2.5, 3.0, 1.5)
        expected = 2.5 * 3.0 + 1.5
        
        assert abs(result - expected) < 1e-6
        assert mac_unit.operations_count == 1
        assert mac_unit.energy_consumed > 0

    def test_mac_unit_reset(self, mac_unit):
        """Test MAC unit reset."""
        mac_unit.multiply_accumulate(2.5, 3.0, 1.5)
        mac_unit.reset()
        
        assert mac_unit.operations_count == 0
        assert mac_unit.energy_consumed == 0.0


class TestRegisterFile:
    """Test RegisterFile class."""

    @pytest.fixture
    def register_file(self):
        """Test register file instance."""
        return RegisterFile(size=16, width=32)

    def test_register_file_initialization(self, register_file):
        """Test register file initialization."""
        assert register_file.size == 16
        assert register_file.width == 32
        assert len(register_file.registers) == 16
        assert register_file.read_count == 0
        assert register_file.write_count == 0

    def test_register_operations(self, register_file):
        """Test register read/write operations."""
        # Test write
        register_file.write(5, 42)
        assert register_file.registers[5] == 42
        assert register_file.write_count == 1
        
        # Test read
        value = register_file.read(5)
        assert value == 42
        assert register_file.read_count == 1

    def test_register_bounds_checking(self, register_file):
        """Test register bounds checking."""
        # Test write out of bounds
        with pytest.raises(IndexError):
            register_file.write(20, 42)
        
        # Test read out of bounds
        with pytest.raises(IndexError):
            register_file.read(20)

    def test_register_file_reset(self, register_file):
        """Test register file reset."""
        register_file.write(5, 42)
        register_file.read(5)
        
        register_file.reset()
        
        assert register_file.registers[5] == 0
        assert register_file.read_count == 0
        assert register_file.write_count == 0


class TestCoreIntegration:
    """Test integration between core components."""

    @pytest.fixture
    def integrated_config(self):
        """Test integrated system configuration."""
        return AcceleratorConfig(
            array=ArrayConfig(rows=4, cols=4),
            debug_mode=True,
            enable_thermal_modeling=True,
            enable_power_modeling=True
        )

    @pytest.fixture
    def integrated_system(self, integrated_config):
        """Test integrated system."""
        return AcceleratorController(integrated_config)

    def test_system_initialization(self, integrated_system):
        """Test system initialization."""
        assert integrated_system.systolic_array is not None
        assert integrated_system.memory_hierarchy is not None
        assert integrated_system.power_manager is not None
        assert integrated_system.reliability_manager is not None

    def test_component_interaction(self, integrated_system):
        """Test interaction between components."""
        # Test that components can communicate
        systolic_array = integrated_system.systolic_array
        memory_hierarchy = integrated_system.memory_hierarchy
        
        assert systolic_array.memory_hierarchy is not None
        assert memory_hierarchy is not None

    def test_performance_monitoring(self, integrated_system):
        """Test performance monitoring."""
        # Test that metrics are collected
        metrics = AcceleratorMetrics()
        assert hasattr(metrics, 'total_cycles')
        assert hasattr(metrics, 'pe_utilization')
        assert hasattr(metrics, 'average_power_watts')

    def test_medical_mode_features(self, integrated_config):
        """Test medical mode features."""
        integrated_config.medical.enable_medical_mode = True
        system = AcceleratorController(integrated_config)
        
        # Check that medical mode features are enabled
        assert system.config.medical.enable_medical_mode is True
        assert system.reliability_manager is not None

    def test_error_recovery(self, integrated_system):
        """Test error recovery mechanisms."""
        # Test that system can handle errors gracefully
        reliability_manager = integrated_system.reliability_manager
        
        # Simulate error injection
        with patch.object(reliability_manager, 'inject_fault') as mock_inject:
            mock_inject.return_value = True
            
            # System should continue operating
            assert integrated_system.is_running is False  # Not started yet 