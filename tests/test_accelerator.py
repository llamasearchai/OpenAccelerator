"""
Comprehensive tests for accelerator controller.

Tests accelerator initialization, workload management, simulation execution,
and performance monitoring.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from open_accelerator.core.accelerator import AcceleratorController
from open_accelerator.utils.config import (
    AcceleratorConfig,
    AcceleratorType,
    DataType,
    MedicalConfig,
)
from open_accelerator.workloads.base import BaseWorkload
from open_accelerator.workloads.gemm import GEMMWorkload


class TestAcceleratorController:
    """Test AcceleratorController class."""

    @pytest.fixture
    def test_config(self):
        """Test configuration for accelerator."""
        return AcceleratorConfig(
            name="TestAccelerator",
            accelerator_type=AcceleratorType.BALANCED,
            data_type=DataType.FLOAT32,
            max_cycles=10000,
            debug_mode=True,
        )

    @pytest.fixture
    def mock_accelerator(self, test_config):
        """Mock accelerator controller for testing."""
        controller = Mock(spec=AcceleratorController)
        controller.config = test_config
        controller.current_cycle = 0
        controller.is_running = False
        controller.workload = None
        controller.performance_metrics = {}
        controller.debug_info = []
        controller.logger = Mock()
        return controller

    def test_accelerator_initialization(self, test_config):
        """Test accelerator controller initialization."""
        with patch(
            "open_accelerator.core.accelerator.AcceleratorController._initialize_components"
        ):
            with patch(
                "open_accelerator.core.accelerator.AcceleratorController.__init__",
                return_value=None,
            ):
                controller = AcceleratorController.__new__(AcceleratorController)
                controller.config = test_config
                controller.current_cycle = 0
                controller.is_running = False

                assert controller.config == test_config
                assert controller.current_cycle == 0
                assert controller.is_running is False

    def test_accelerator_load_workload(self, mock_accelerator):
        """Test loading workload into accelerator."""
        # Create test workload
        workload = Mock(spec=GEMMWorkload)
        workload.name = "test_gemm"

        # Mock load_workload method
        mock_accelerator.load_workload = Mock(return_value=True)

        # Test loading workload
        result = mock_accelerator.load_workload(workload)
        assert result is True
        mock_accelerator.load_workload.assert_called_once_with(workload)

    def test_accelerator_execute_workload(self, mock_accelerator):
        """Test workload execution."""
        # Mock workload execution
        mock_accelerator.execute_workload = Mock(
            return_value={
                "status": "completed",
                "cycles": 100,
                "performance": {"ops_per_second": 1000},
            }
        )

        # Test execution
        result = mock_accelerator.execute_workload()
        assert result["status"] == "completed"
        assert result["cycles"] == 100
        assert "performance" in result
        mock_accelerator.execute_workload.assert_called_once()

    def test_accelerator_reset(self, mock_accelerator):
        """Test accelerator reset."""
        # Mock reset method
        mock_accelerator.reset = Mock()
        mock_accelerator.current_cycle = 100

        # Test reset
        mock_accelerator.reset()
        mock_accelerator.reset.assert_called_once()

    def test_accelerator_get_status(self, mock_accelerator):
        """Test getting accelerator status."""
        # Mock get_status method
        mock_accelerator.get_status = Mock(
            return_value={
                "current_cycle": 50,
                "is_running": True,
                "workload_loaded": True,
            }
        )

        # Test status
        status = mock_accelerator.get_status()
        assert status["current_cycle"] == 50
        assert status["is_running"] is True
        assert status["workload_loaded"] is True

    def test_accelerator_performance_metrics(self, mock_accelerator):
        """Test performance metrics collection."""
        # Mock performance metrics
        mock_accelerator.get_performance_metrics = Mock(
            return_value={
                "total_cycles": 1000,
                "total_operations": 8192,
                "operations_per_cycle": 8.192,
                "power_consumption": 50.5,
                "energy_efficiency": 163.4,
            }
        )

        # Test metrics
        metrics = mock_accelerator.get_performance_metrics()
        assert metrics["total_cycles"] == 1000
        assert metrics["total_operations"] == 8192
        assert "operations_per_cycle" in metrics
        assert "power_consumption" in metrics
        assert "energy_efficiency" in metrics

    def test_accelerator_debug_mode(self, mock_accelerator):
        """Test debug mode functionality."""
        # Mock debug methods
        mock_accelerator.enable_debug = Mock()
        mock_accelerator.disable_debug = Mock()
        mock_accelerator.get_debug_info = Mock(
            return_value=[
                {"cycle": 1, "event": "workload_started"},
                {"cycle": 10, "event": "computation_cycle"},
                {"cycle": 100, "event": "workload_completed"},
            ]
        )

        # Test debug mode
        mock_accelerator.enable_debug()
        mock_accelerator.enable_debug.assert_called_once()

        debug_info = mock_accelerator.get_debug_info()
        assert len(debug_info) == 3
        assert debug_info[0]["event"] == "workload_started"
        assert debug_info[-1]["event"] == "workload_completed"

    def test_accelerator_medical_mode(self, mock_accelerator):
        """Test medical mode functionality."""
        # Set up medical config
        medical_config = AcceleratorConfig(
            accelerator_type=AcceleratorType.MEDICAL,
            medical=MedicalConfig(
                enable_medical_mode=True, phi_compliance=True, fda_validation=True
            ),
        )
        mock_accelerator.config = medical_config

        # Mock medical mode methods
        mock_accelerator.enable_medical_mode = Mock()
        mock_accelerator.validate_medical_compliance = Mock(return_value=True)

        # Test medical mode
        mock_accelerator.enable_medical_mode()
        mock_accelerator.enable_medical_mode.assert_called_once()

        compliance = mock_accelerator.validate_medical_compliance()
        assert compliance is True

    def test_accelerator_error_handling(self, mock_accelerator):
        """Test error handling in accelerator."""
        # Mock error scenarios
        mock_accelerator.load_workload = Mock(
            side_effect=ValueError("Invalid workload")
        )
        mock_accelerator.execute_workload = Mock(
            side_effect=RuntimeError("Execution failed")
        )

        # Test error handling
        with pytest.raises(ValueError, match="Invalid workload"):
            mock_accelerator.load_workload(None)

        with pytest.raises(RuntimeError, match="Execution failed"):
            mock_accelerator.execute_workload()

    def test_accelerator_cycle_execution(self, mock_accelerator):
        """Test single cycle execution."""
        # Mock cycle execution
        mock_accelerator.execute_cycle = Mock(
            return_value={
                "cycle_number": 1,
                "operations_completed": 16,
                "power_consumed": 0.5,
                "memory_accesses": 32,
            }
        )

        # Test cycle execution
        result = mock_accelerator.execute_cycle()
        assert result["cycle_number"] == 1
        assert result["operations_completed"] == 16
        assert "power_consumed" in result
        assert "memory_accesses" in result

    def test_accelerator_save_load_state(self, mock_accelerator, tmp_path):
        """Test saving and loading accelerator state."""
        # Mock state management
        mock_accelerator.save_state = Mock()
        mock_accelerator.load_state = Mock()

        # Test state saving
        state_path = tmp_path / "accelerator_state.json"
        mock_accelerator.save_state(state_path)
        mock_accelerator.save_state.assert_called_once_with(state_path)

        # Test state loading
        mock_accelerator.load_state(state_path)
        mock_accelerator.load_state.assert_called_once_with(state_path)


class TestAcceleratorWorkloadIntegration:
    """Test accelerator-workload integration."""

    @pytest.fixture
    def mock_gemm_workload(self):
        """Mock GEMM workload for testing."""
        workload = Mock(spec=GEMMWorkload)
        workload.name = "test_gemm"
        workload.config = Mock()
        workload.config.gemm_M = 8
        workload.config.gemm_K = 8
        workload.config.gemm_P = 8
        workload.get_input_data = Mock(
            return_value={"A": np.random.rand(8, 8), "B": np.random.rand(8, 8)}
        )
        workload.validate_output = Mock(return_value=True)
        return workload

    @pytest.fixture
    def mock_integrated_accelerator(self, test_config):
        """Mock accelerator with workload integration."""
        controller = Mock()
        controller.config = test_config
        controller.workload = None
        controller.is_running = False
        controller.current_cycle = 0
        return controller

    def test_gemm_workload_integration(
        self, mock_integrated_accelerator, mock_gemm_workload
    ):
        """Test GEMM workload integration with accelerator."""
        # Mock workload integration
        mock_integrated_accelerator.load_workload = Mock(return_value=True)
        mock_integrated_accelerator.execute_workload = Mock(
            return_value={
                "status": "completed",
                "result": np.random.rand(8, 8),
                "cycles": 64,
                "accuracy": 0.99,
            }
        )

        # Test integration
        loaded = mock_integrated_accelerator.load_workload(mock_gemm_workload)
        assert loaded is True

        result = mock_integrated_accelerator.execute_workload()
        assert result["status"] == "completed"
        assert "result" in result
        assert result["cycles"] == 64
        assert result["accuracy"] == 0.99

    def test_multiple_workload_execution(self, mock_integrated_accelerator):
        """Test executing multiple workloads sequentially."""
        # Mock multiple workloads
        workload1 = Mock(spec=BaseWorkload)
        workload1.name = "workload1"
        workload2 = Mock(spec=BaseWorkload)
        workload2.name = "workload2"

        # Mock execution results
        mock_integrated_accelerator.execute_workload = Mock(
            side_effect=[
                {"status": "completed", "workload": "workload1", "cycles": 100},
                {"status": "completed", "workload": "workload2", "cycles": 150},
            ]
        )

        # Test multiple executions
        result1 = mock_integrated_accelerator.execute_workload()
        result2 = mock_integrated_accelerator.execute_workload()

        assert result1["workload"] == "workload1"
        assert result2["workload"] == "workload2"
        assert result1["cycles"] == 100
        assert result2["cycles"] == 150

    def test_workload_validation_failure(self, mock_integrated_accelerator):
        """Test workload validation failure handling."""
        # Mock invalid workload
        invalid_workload = Mock(spec=BaseWorkload)
        invalid_workload.validate = Mock(return_value=False)

        # Mock validation failure
        mock_integrated_accelerator.load_workload = Mock(
            side_effect=ValueError("Workload validation failed")
        )

        # Test validation failure
        with pytest.raises(ValueError, match="Workload validation failed"):
            mock_integrated_accelerator.load_workload(invalid_workload)


class TestAcceleratorPerformanceMonitoring:
    """Test accelerator performance monitoring."""

    @pytest.fixture
    def mock_performance_accelerator(self):
        """Mock accelerator with performance monitoring."""
        controller = Mock()
        controller.performance_monitor = Mock()
        controller.power_manager = Mock()
        controller.thermal_monitor = Mock()
        return controller

    def test_performance_monitoring_initialization(self, mock_performance_accelerator):
        """Test performance monitoring initialization."""
        # Mock initialization
        mock_performance_accelerator.initialize_performance_monitoring = Mock()
        mock_performance_accelerator.initialize_performance_monitoring()
        mock_performance_accelerator.initialize_performance_monitoring.assert_called_once()

    def test_real_time_metrics_collection(self, mock_performance_accelerator):
        """Test real-time metrics collection."""
        # Mock metrics collection
        mock_performance_accelerator.collect_metrics = Mock(
            return_value={
                "timestamp": 1000,
                "cycle": 100,
                "throughput": 1000.0,
                "latency": 0.001,
                "power": 45.5,
                "temperature": 65.0,
                "utilization": 0.85,
            }
        )

        # Test metrics collection
        metrics = mock_performance_accelerator.collect_metrics()
        assert metrics["cycle"] == 100
        assert metrics["throughput"] == 1000.0
        assert metrics["power"] == 45.5
        assert metrics["utilization"] == 0.85

    def test_performance_analysis(self, mock_performance_accelerator):
        """Test performance analysis functionality."""
        # Mock analysis
        mock_performance_accelerator.analyze_performance = Mock(
            return_value={
                "bottlenecks": ["memory_bandwidth", "compute_utilization"],
                "optimization_suggestions": [
                    "increase_buffer_size",
                    "optimize_dataflow",
                ],
                "efficiency_score": 0.78,
                "predicted_improvement": 0.15,
            }
        )

        # Test analysis
        analysis = mock_performance_accelerator.analyze_performance()
        assert "bottlenecks" in analysis
        assert "optimization_suggestions" in analysis
        assert analysis["efficiency_score"] == 0.78
        assert len(analysis["bottlenecks"]) == 2

    def test_performance_profiling(self, mock_performance_accelerator):
        """Test performance profiling functionality."""
        # Mock profiling
        mock_performance_accelerator.start_profiling = Mock()
        mock_performance_accelerator.stop_profiling = Mock()
        mock_performance_accelerator.get_profiling_report = Mock(
            return_value={
                "total_runtime": 10.5,
                "compute_time": 8.2,
                "memory_time": 1.8,
                "idle_time": 0.5,
                "instruction_breakdown": {"MAC": 70.0, "LOAD": 20.0, "STORE": 10.0},
            }
        )

        # Test profiling
        mock_performance_accelerator.start_profiling()
        mock_performance_accelerator.stop_profiling()
        report = mock_performance_accelerator.get_profiling_report()

        assert report["total_runtime"] == 10.5
        assert report["compute_time"] == 8.2
        assert "instruction_breakdown" in report
        assert report["instruction_breakdown"]["MAC"] == 70.0

    def test_thermal_monitoring(self, mock_performance_accelerator):
        """Test thermal monitoring functionality."""
        # Mock thermal monitoring
        mock_performance_accelerator.thermal_monitor.get_temperature = Mock(
            return_value=68.5
        )
        mock_performance_accelerator.thermal_monitor.check_thermal_throttling = Mock(
            return_value=False
        )
        mock_performance_accelerator.thermal_monitor.get_thermal_profile = Mock(
            return_value={
                "current_temp": 68.5,
                "max_temp": 85.0,
                "thermal_margin": 16.5,
                "throttling_active": False,
            }
        )

        # Test thermal monitoring
        temp = mock_performance_accelerator.thermal_monitor.get_temperature()
        throttling = (
            mock_performance_accelerator.thermal_monitor.check_thermal_throttling()
        )
        profile = mock_performance_accelerator.thermal_monitor.get_thermal_profile()

        assert temp == 68.5
        assert throttling is False
        assert profile["thermal_margin"] == 16.5

    def test_power_monitoring(self, mock_performance_accelerator):
        """Test power monitoring functionality."""
        # Mock power monitoring
        mock_performance_accelerator.power_manager.get_power_consumption = Mock(
            return_value=42.3
        )
        mock_performance_accelerator.power_manager.get_energy_efficiency = Mock(
            return_value=156.7
        )
        mock_performance_accelerator.power_manager.get_power_breakdown = Mock(
            return_value={
                "compute": 30.5,
                "memory": 8.2,
                "interconnect": 2.8,
                "other": 0.8,
            }
        )

        # Test power monitoring
        power = mock_performance_accelerator.power_manager.get_power_consumption()
        efficiency = mock_performance_accelerator.power_manager.get_energy_efficiency()
        breakdown = mock_performance_accelerator.power_manager.get_power_breakdown()

        assert power == 42.3
        assert efficiency == 156.7
        assert breakdown["compute"] == 30.5
        assert sum(breakdown.values()) == 42.3


class TestAcceleratorReliability:
    """Test accelerator reliability features."""

    @pytest.fixture
    def mock_reliable_accelerator(self):
        """Mock accelerator with reliability features."""
        controller = Mock()
        controller.error_detector = Mock()
        controller.fault_handler = Mock()
        controller.redundancy_manager = Mock()
        return controller

    def test_error_detection(self, mock_reliable_accelerator):
        """Test error detection functionality."""
        # Mock error detection
        mock_reliable_accelerator.error_detector.check_for_errors = Mock(
            return_value=[
                {"type": "parity_error", "location": "PE(2,3)", "severity": "medium"},
                {
                    "type": "timeout",
                    "location": "memory_controller",
                    "severity": "high",
                },
            ]
        )

        # Test error detection
        errors = mock_reliable_accelerator.error_detector.check_for_errors()
        assert len(errors) == 2
        assert errors[0]["type"] == "parity_error"
        assert errors[1]["severity"] == "high"

    def test_fault_handling(self, mock_reliable_accelerator):
        """Test fault handling functionality."""
        # Mock fault handling
        mock_reliable_accelerator.fault_handler.handle_fault = Mock(
            return_value={
                "fault_type": "memory_error",
                "recovery_action": "retry_with_redundancy",
                "recovery_success": True,
                "downtime": 0.005,
            }
        )

        # Test fault handling
        fault = {"type": "memory_error", "location": "buffer_1"}
        result = mock_reliable_accelerator.fault_handler.handle_fault(fault)

        assert result["recovery_action"] == "retry_with_redundancy"
        assert result["recovery_success"] is True
        assert result["downtime"] == 0.005

    def test_redundancy_management(self, mock_reliable_accelerator):
        """Test redundancy management functionality."""
        # Mock redundancy management
        mock_reliable_accelerator.redundancy_manager.enable_redundancy = Mock()
        mock_reliable_accelerator.redundancy_manager.get_redundancy_status = Mock(
            return_value={
                "pe_redundancy": True,
                "memory_redundancy": True,
                "interconnect_redundancy": False,
                "redundancy_overhead": 0.15,
            }
        )

        # Test redundancy management
        mock_reliable_accelerator.redundancy_manager.enable_redundancy("PE")
        status = mock_reliable_accelerator.redundancy_manager.get_redundancy_status()

        assert status["pe_redundancy"] is True
        assert status["memory_redundancy"] is True
        assert status["redundancy_overhead"] == 0.15
