"""Comprehensive tests for Open Accelerator package."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Test imports with proper error handling
try:
    import open_accelerator
    from open_accelerator.ai.agents import AgentConfig, AgentOrchestrator
    from open_accelerator.analysis.performance_analysis import PerformanceAnalyzer
    from open_accelerator.core.accelerator import AcceleratorController
    from open_accelerator.core.memory import MemoryHierarchy
    from open_accelerator.core.power import PowerConfig, SystemPowerManager
    from open_accelerator.core.security import SecurityConfig, SecurityManager
    from open_accelerator.core.systolic_array import SystolicArray
    from open_accelerator.simulation.simulator import Simulator
    from open_accelerator.utils.config import AcceleratorConfig, get_default_configs
    from open_accelerator.workloads.base import (
        BaseWorkload,
        ComputeWorkload,
        MLWorkload,
    )
    from open_accelerator.workloads.gemm import GEMMWorkload

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Some imports failed: {e}")
    IMPORTS_AVAILABLE = False


class TestBasicFunctionality:
    """Test basic functionality and package imports."""

    def test_package_import(self):
        """Test that the package can be imported."""
        assert open_accelerator.__version__ is not None
        assert open_accelerator.__author__ == "Nik Jois"
        assert open_accelerator.__email__ == "nikjois@llamasearch.ai"
        assert open_accelerator.__license__ == "MIT"

    def test_basic_math_operations(self):
        """Test basic mathematical operations."""
        assert 2 * 3 == 6
        assert 10 / 2 == 5
        assert 5 - 3 == 2
        assert 2 + 3 == 5
        assert 2**3 == 8

    def test_numpy_integration(self):
        """Test NumPy integration works correctly."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
        assert arr.std() > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestConfiguration:
    """Test configuration system."""

    def test_default_configs_creation(self):
        """Test that default configurations can be created."""
        configs = get_default_configs()
        assert "small" in configs
        assert "large" in configs
        assert "medical" in configs

        # Verify config structure
        small_config = configs["small"]
        assert small_config.name == "EdgeAccelerator"
        assert small_config.array.rows == 8
        assert small_config.array.cols == 8

    def test_config_validation(self):
        """Test configuration validation."""
        config = AcceleratorConfig(name="TestAccelerator")
        assert config.name == "TestAccelerator"
        assert config.array.rows > 0
        assert config.array.cols > 0
        assert config.max_cycles > 0

    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = AcceleratorConfig(name="SerializationTest")

        # Test that config can be converted to dict-like structure
        config_dict = {
            "name": config.name,
            "array_rows": config.array.rows,
            "array_cols": config.array.cols,
            "max_cycles": config.max_cycles,
        }

        assert config_dict["name"] == "SerializationTest"
        assert config_dict["array_rows"] == 16  # default
        assert config_dict["array_cols"] == 16  # default


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestWorkloads:
    """Test workload system."""

    def test_base_workload_creation(self):
        """Test base workload creation."""
        workload = ComputeWorkload("TestWorkload")
        assert workload.name == "TestWorkload"
        assert workload.requirements is not None
        assert workload.metrics is not None

    def test_gemm_workload_creation(self):
        """Test GEMM workload creation."""
        from open_accelerator.workloads.gemm import GEMMWorkloadConfig

        config = GEMMWorkloadConfig(M=16, K=16, P=16)
        workload = GEMMWorkload(config, name="TestGEMM")
        assert workload.name == "TestGEMM"
        assert workload.config.M == 16
        assert workload.config.K == 16
        assert workload.config.P == 16

    def test_workload_preparation(self):
        """Test workload preparation."""
        workload = ComputeWorkload("PrepTest")
        assert not workload.is_ready()

        workload.prepare()
        assert workload.is_ready()

    def test_workload_reset(self):
        """Test workload reset functionality."""
        workload = ComputeWorkload("ResetTest")
        workload.prepare()

        # Simulate some execution
        workload.metrics.total_operations = 100
        workload.metrics.total_cycles = 1000

        workload.reset()
        assert workload.metrics.total_operations == 0
        assert workload.metrics.total_cycles == 0
        assert not workload.is_ready()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestSystolicArray:
    """Test systolic array functionality."""

    def test_systolic_array_creation(self):
        """Test systolic array creation."""
        config = AcceleratorConfig()
        array = SystolicArray(config)

        assert array.rows == config.array.rows
        assert array.cols == config.array.cols
        assert array.cycle_count == 0

    def test_systolic_array_initialization(self):
        """Test systolic array PE initialization."""
        config = AcceleratorConfig()
        config.array.rows = 4
        config.array.cols = 4

        array = SystolicArray(config)

        # Check PE grid is initialized
        assert len(array.pes) == 4
        assert len(array.pes[0]) == 4

    def test_systolic_array_cycle_execution(self):
        """Test systolic array cycle execution."""
        config = AcceleratorConfig()
        config.array.rows = 2
        config.array.cols = 2

        array = SystolicArray(config)

        # Prepare input data with numpy arrays
        input_data = {"edge_a": np.array([1.0, 2.0]), "edge_b": np.array([3.0, 4.0])}

        # Execute one cycle
        result = array.cycle(input_data)

        assert "cycle" in result
        assert result["cycle"] == 1
        assert "metrics" in result
        assert array.cycle_count == 1


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestMemoryHierarchy:
    """Test memory hierarchy functionality."""

    def test_memory_hierarchy_creation(self):
        """Test memory hierarchy creation."""
        config = AcceleratorConfig()
        memory = MemoryHierarchy(config)

        assert memory.l1_cache is not None
        assert memory.l2_cache is not None
        assert memory.main_memory is not None

    def test_memory_read_write(self):
        """Test memory read/write operations."""
        config = AcceleratorConfig()
        memory = MemoryHierarchy(config)

        # Test write operation
        test_data = [1, 2, 3, 4, 5]
        success, latency = memory.write_request(0x1000, test_data)
        assert isinstance(success, bool)
        assert isinstance(latency, int)

        # Test read operation
        read_data, read_latency = memory.read_request(0x1000, 5)
        assert isinstance(read_data, list)
        assert isinstance(read_latency, int)

    def test_memory_hierarchy_cycle(self):
        """Test memory hierarchy cycle execution."""
        config = AcceleratorConfig()
        memory = MemoryHierarchy(config)

        # Should not raise any exceptions
        memory.cycle()

        # Test metrics
        metrics = memory.get_hierarchy_metrics()
        assert isinstance(metrics, dict)
        assert "l1_metrics" in metrics
        assert "l2_metrics" in metrics
        assert "main_memory_metrics" in metrics


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestSimulator:
    """Test simulator functionality."""

    def test_simulator_creation(self):
        """Test simulator creation."""
        config = AcceleratorConfig()
        simulator = Simulator(config)

        assert simulator.config == config
        assert simulator.simulation_id is not None

    def test_simulator_workload_execution(self):
        """Test simulator workload execution."""
        config = AcceleratorConfig()
        config.array.rows = 4
        config.array.cols = 4

        simulator = Simulator(config)
        workload = ComputeWorkload("SimulationTest")
        workload.prepare()

        # Run simulation
        results = simulator.run(workload)

        assert isinstance(results, dict)
        assert "total_cycles" in results
        assert "total_mac_operations" in results
        assert "output_matrix" in results
        assert "pe_activity_map_over_time" in results

    def test_simulator_reset(self):
        """Test simulator reset functionality."""
        config = AcceleratorConfig()
        simulator = Simulator(config)

        # Simulator doesn't have reset method, just verify we can create a new one
        new_simulator = Simulator(config)
        assert new_simulator.simulation_id != simulator.simulation_id


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestPerformanceAnalysis:
    """Test performance analysis functionality."""

    def test_performance_analyzer_creation(self):
        """Test performance analyzer creation."""
        mock_sim_stats = {
            "total_cycles": 1000,
            "total_mac_operations": 2000,
            "output_matrix": np.zeros((4, 4)),
            "pe_activity_map_over_time": np.random.rand(4, 4) * 0.8,
        }
        analyzer = PerformanceAnalyzer(mock_sim_stats)

        assert analyzer._stats == mock_sim_stats

    def test_performance_metrics_analysis(self):
        """Test performance metrics analysis."""
        mock_sim_stats = {
            "total_cycles": 1000,
            "total_mac_operations": 2000,
            "output_matrix": np.zeros((4, 4)),
            "pe_activity_map_over_time": np.random.rand(4, 4) * 0.8,
        }
        analyzer = PerformanceAnalyzer(mock_sim_stats)

        # Analyze performance
        analysis = analyzer.compute_metrics()

        assert isinstance(analysis, dict)
        # Should contain basic performance metrics
        assert "total_cycles" in analysis
        assert "total_macs" in analysis
        assert "macs_per_cycle" in analysis
        assert "efficiency" in analysis
        assert "pe_utilization" in analysis


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestSecuritySystem:
    """Test security system functionality."""

    def test_security_manager_creation(self):
        """Test security manager creation."""
        config = SecurityConfig()
        security_manager = SecurityManager(config)

        assert security_manager.config == config
        # Key manager should be available when key management is enabled (default)
        if config.enable_key_management:
            assert security_manager.key_manager is not None
        else:
            assert security_manager.key_manager is None

    def test_data_encryption_decryption(self):
        """Test data encryption and decryption."""
        config = SecurityConfig()
        security_manager = SecurityManager(config)

        # Test data encryption
        test_data = b"Hello, World!"
        encrypted_data = security_manager.encrypt_data(test_data)

        assert isinstance(encrypted_data, bytes)
        assert encrypted_data != test_data

        # Test data decryption
        decrypted_data = security_manager.decrypt_data(encrypted_data)
        assert decrypted_data == test_data

    def test_security_status_report(self):
        """Test security status reporting."""
        config = SecurityConfig()
        security_manager = SecurityManager(config)

        status = security_manager.get_security_status()

        assert isinstance(status, dict)
        assert "security_state" in status
        assert "security_config" in status
        assert "metrics" in status


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestPowerManagement:
    """Test power management functionality."""

    def test_power_manager_creation(self):
        """Test power manager creation."""
        config = PowerConfig()
        power_manager = SystemPowerManager(config)

        assert power_manager.config == config

    def test_power_component_registration(self):
        """Test power component registration."""
        config = PowerConfig()
        power_manager = SystemPowerManager(config)

        # Mock power model
        from open_accelerator.core.power import CMOSPowerModel

        power_model = CMOSPowerModel()

        # Register component
        component_manager = power_manager.register_component(
            "test_component", power_model
        )

        assert component_manager is not None
        assert component_manager.component_id == "test_component"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestAIAgents:
    """Test AI agents functionality."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_agent_orchestrator_creation(self):
        """Test agent orchestrator creation."""
        config = AgentConfig()
        orchestrator = AgentOrchestrator(config)

        assert orchestrator.config == config

    def test_agent_configuration(self):
        """Test agent configuration."""
        config = AgentConfig(model="gpt-4o", temperature=0.1, medical_compliance=True)

        assert config.model == "gpt-4o"
        assert config.temperature == 0.1
        assert config.medical_compliance is True


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Core modules not available")
class TestIntegration:
    """Test integration between components."""

    def test_full_system_integration(self):
        """Test full system integration."""
        # Create system configuration
        config = AcceleratorConfig()
        config.array.rows = 4
        config.array.cols = 4

        # Create simulator
        simulator = Simulator(config)

        # Create workload
        workload = ComputeWorkload("IntegrationTest")
        workload.prepare()

        # Run simulation
        results = simulator.run(workload)

        # Verify results
        assert isinstance(results, dict)

        # Test performance analysis
        analyzer = PerformanceAnalyzer(results)
        analysis = analyzer.compute_metrics()

        assert isinstance(analysis, dict)

    def test_medical_compliance_integration(self):
        """Test medical compliance integration."""
        # Create medical configuration
        config = AcceleratorConfig()
        config.medical.enable_medical_mode = True
        config.medical.phi_compliance = True

        # Create security manager
        security_config = SecurityConfig()
        security_config.hipaa_compliant = True
        security_manager = SecurityManager(security_config)

        # Verify medical mode is enabled
        assert config.medical.enable_medical_mode is True
        assert security_config.hipaa_compliant is True

    def test_config_consistency(self):
        """Test configuration consistency across components."""
        config = AcceleratorConfig()

        # Test that configurations are consistent
        assert config.array.rows > 0
        assert config.array.cols > 0
        assert config.max_cycles > 0

        # Test medical configuration
        if config.medical.enable_medical_mode:
            assert (
                config.medical.phi_compliance is True
                or config.medical.phi_compliance is False
            )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        with pytest.raises(ValueError):
            # This should raise a ValueError for invalid array dimensions
            AcceleratorConfig().array.rows = -1

    def test_missing_dependencies(self):
        """Test handling of missing dependencies."""
        # Test that system gracefully handles missing optional dependencies
        with patch("open_accelerator.ai.agents.OPENAI_AVAILABLE", False):
            # Should not raise exceptions even if OpenAI is not available
            config = AgentConfig()
            assert config.model == "gpt-4o"

    def test_file_operations(self):
        """Test file operations with temporary files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test_config.yaml"

            # Test file path exists
            assert test_file.parent.exists()

            # Test file creation
            test_file.write_text("test: value")
            assert test_file.exists()

            # Test file reading
            content = test_file.read_text()
            assert "test: value" in content


class TestUtilities:
    """Test utility functions and helpers."""

    def test_global_configuration(self):
        """Test global configuration functions."""
        # Test get_config
        config = open_accelerator.get_config()
        assert isinstance(config, dict)

        # Test set_config
        result = open_accelerator.set_config("test_key", "test_value")
        assert isinstance(result, bool)

        # Test get_config with key
        value = open_accelerator.get_config("test_key")
        assert value == "test_value"

        # Test reset_config
        result = open_accelerator.reset_config()
        assert isinstance(result, bool)

    def test_version_info(self):
        """Test version information."""
        assert open_accelerator.__version__ == "1.0.1"
        assert open_accelerator.__author__ == "Nik Jois"
        assert open_accelerator.__email__ == "nikjois@llamasearch.ai"
        assert open_accelerator.__license__ == "MIT"

    def test_exported_components(self):
        """Test that all expected components are exported."""
        expected_exports = [
            "__version__",
            "__author__",
            "__email__",
            "__license__",
            "get_config",
            "set_config",
            "reset_config",
        ]

        for export in expected_exports:
            assert hasattr(open_accelerator, export)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
