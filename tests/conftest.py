"""
Pytest configuration and fixtures for Open Accelerator tests.

Provides comprehensive test configuration with fixtures for accelerator components,
test data generation, and property-based testing setup with hypothesis.
"""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import hypothesis
from hypothesis import strategies as st

# Configure hypothesis for property-based testing
hypothesis.settings.register_profile("default", max_examples=100, deadline=5000)
hypothesis.settings.load_profile("default")

# Configure pytest
pytest_plugins = ["pytest_mock", "pytest_cov", "pytest_benchmark"]

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


# Basic fixtures
@pytest.fixture
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    with patch("openai.OpenAI") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mock response from OpenAI"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        mock_instance.chat.completions.create.return_value = mock_response
        
        yield mock_instance


# Configuration fixtures
@pytest.fixture
def basic_config() -> Dict[str, Any]:
    """Basic accelerator configuration for testing."""
    return {
        "name": "test_config",
        "array_rows": 4,
        "array_cols": 4,
        "pe_mac_latency": 1,
        "input_buffer_size": 1024,
        "weight_buffer_size": 1024,
        "output_buffer_size": 1024,
        "medical_mode": False,
        "enable_security": False,
        "enable_power_management": False
    }


@pytest.fixture
def medical_config() -> Dict[str, Any]:
    """Medical-grade accelerator configuration for testing."""
    return {
        "name": "medical_test_config",
        "array_rows": 8,
        "array_cols": 8,
        "pe_mac_latency": 1,
        "input_buffer_size": 2048,
        "weight_buffer_size": 2048,
        "output_buffer_size": 2048,
        "medical_mode": True,
        "enable_security": True,
        "enable_power_management": True,
        "precision": "float64",
        "safety_level": "high"
    }


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration alias for compatibility."""
    return {
        "name": "test_config",
        "array_rows": 4,
        "array_cols": 4,
        "pe_mac_latency": 1,
        "input_buffer_size": 1024,
        "weight_buffer_size": 1024,
        "output_buffer_size": 1024,
        "medical_mode": False,
        "enable_security": False,
        "enable_power_management": False
    }


@pytest.fixture
def test_matrices() -> Dict[str, np.ndarray]:
    """Test matrices for GEMM operations."""
    np.random.seed(42)  # Reproducible test data
    
    m, k, p = 16, 12, 8
    
    return {
        "A": np.random.randn(m, k).astype(np.float32),
        "B": np.random.randn(k, p).astype(np.float32),
        "C": np.zeros((m, p), dtype=np.float32),
        "expected_C": np.random.randn(m, k) @ np.random.randn(k, p)
    }


@pytest.fixture
def medical_image_data() -> Dict[str, np.ndarray]:
    """Medical imaging test data."""
    np.random.seed(42)
    
    return {
        "ct_scan": np.random.uniform(0, 1, (512, 512, 64)).astype(np.float32),
        "mri_scan": np.random.uniform(0, 1, (256, 256, 128)).astype(np.float32),
        "xray_image": np.random.uniform(0, 1, (1024, 1024, 1)).astype(np.float32),
        "segmentation_mask": np.random.randint(0, 2, (512, 512, 64)).astype(np.uint8),
        "classification_labels": np.array([0, 1, 0, 1, 0], dtype=np.int32)
    }


# Component fixtures
@pytest.fixture
def mock_processing_element():
    """Mock processing element for testing."""
    from open_accelerator.core.pe import ProcessingElement
    
    with patch.object(ProcessingElement, "__init__", return_value=None):
        pe = MagicMock(spec=ProcessingElement)
        pe.pe_id = (0, 0)
        pe.accumulator = 0.0
        pe.total_mac_ops = 0
        pe.is_computing_this_cycle = False
        
        # Mock methods
        pe.load_inputs = MagicMock()
        pe.cycle = MagicMock()
        pe.get_output_for_propagation = MagicMock(return_value=(None, None))
        pe.get_final_result = MagicMock(return_value=0.0)
        
        yield pe


@pytest.fixture
def mock_memory_buffer():
    """Mock memory buffer for testing."""
    from open_accelerator.core.memory import MemoryBuffer
    
    with patch.object(MemoryBuffer, "__init__", return_value=None):
        buffer = MagicMock(spec=MemoryBuffer)
        buffer.name = "test_buffer"
        buffer.size = 1024
        buffer.bandwidth = 16
        buffer.data = []
        
        # Mock methods
        buffer.write = MagicMock(return_value=True)
        buffer.read = MagicMock(return_value=None)
        buffer.is_full = MagicMock(return_value=False)
        buffer.is_empty = MagicMock(return_value=True)
        
        yield buffer


@pytest.fixture
def mock_accelerator():
    """Mock accelerator for integration testing."""
    from open_accelerator.core.accelerator import AcceleratorController
    
    with patch.object(AcceleratorController, "__init__", return_value=None):
        accelerator = MagicMock(spec=AcceleratorController)
        accelerator.config = MagicMock()
        accelerator.current_cycle = 0
        accelerator.is_running = False
        
        # Mock methods
        accelerator.load_workload = MagicMock(return_value=True)
        accelerator.execute_workload = MagicMock(return_value={"status": "completed"})
        accelerator.reset = MagicMock()
        
        yield accelerator


# Workload fixtures
@pytest.fixture
def gemm_workload():
    """GEMM workload for testing."""
    from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig
    
    config = GEMMWorkloadConfig(
        M=16,
        K=12,
        P=8
    )
    
    workload = GEMMWorkload(config)
    workload.prepare()
    
    return workload


@pytest.fixture
def test_workload_config():
    """Test workload configuration fixture."""
    from open_accelerator.utils.config import WorkloadConfig, WorkloadType
    
    return WorkloadConfig(
        workload_type=WorkloadType.GEMM,
        gemm_M=16,
        gemm_K=12,
        gemm_P=8
    )


@pytest.fixture
def medical_workload():
    """Medical workload for testing."""
    from open_accelerator.workloads.medical import MedicalConvolution, MedicalWorkloadConfig
    from open_accelerator.workloads.medical import MedicalModalityType, MedicalTaskType
    
    config = MedicalWorkloadConfig(
        name="test_medical",
        modality=MedicalModalityType.CT_SCAN,
        task_type=MedicalTaskType.SEGMENTATION,
        image_size=(128, 128, 16),
        batch_size=1,
        precision_level="high"
    )
    
    workload = MedicalConvolution(config)
    workload.generate_medical_ct_data()
    
    return workload


# Hypothesis strategies for property-based testing
@st.composite
def matrix_strategy(draw, min_size=2, max_size=32):
    """Strategy for generating test matrices."""
    rows = draw(st.integers(min_value=min_size, max_value=max_size))
    cols = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate matrix with finite values
    matrix = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=cols,
                max_size=cols
            ),
            min_size=rows,
            max_size=rows
        )
    )
    
    return np.array(matrix, dtype=np.float32)


@st.composite
def gemm_dimensions_strategy(draw):
    """Strategy for generating valid GEMM dimensions."""
    m = draw(st.integers(min_value=1, max_value=32))
    k = draw(st.integers(min_value=1, max_value=32))
    p = draw(st.integers(min_value=1, max_value=32))
    
    return {"m": m, "k": k, "p": p}


@st.composite
def medical_image_strategy(draw):
    """Strategy for generating medical image data."""
    height = draw(st.integers(min_value=64, max_value=512))
    width = draw(st.integers(min_value=64, max_value=512))
    depth = draw(st.integers(min_value=1, max_value=64))
    
    # Generate realistic medical image intensities
    image = draw(
        st.lists(
            st.lists(
                st.lists(
                    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=depth,
                    max_size=depth
                ),
                min_size=width,
                max_size=width
            ),
            min_size=height,
            max_size=height
        )
    )
    
    return np.array(image, dtype=np.float32)


@st.composite
def accelerator_config_strategy(draw):
    """Strategy for generating accelerator configurations."""
    rows = draw(st.integers(min_value=2, max_value=16))
    cols = draw(st.integers(min_value=2, max_value=16))
    
    return {
        "array_rows": rows,
        "array_cols": cols,
        "pe_mac_latency": draw(st.integers(min_value=1, max_value=5)),
        "input_buffer_size": draw(st.integers(min_value=512, max_value=4096)),
        "weight_buffer_size": draw(st.integers(min_value=512, max_value=4096)),
        "output_buffer_size": draw(st.integers(min_value=512, max_value=4096)),
        "medical_mode": draw(st.booleans()),
        "enable_security": draw(st.booleans())
    }


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "warmup": True,
        "sort": "mean",
        "timer": "time.perf_counter"
    }


# Mock file system fixtures
@pytest.fixture
def mock_file_system(tmp_path):
    """Mock file system for testing file operations."""
    # Create test directory structure
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir()
    
    # Create test files
    (test_dir / "config.yaml").write_text("test: config")
    (test_dir / "data.json").write_text('{"test": "data"}')
    
    return test_dir


# Error injection fixtures
@pytest.fixture
def error_injection():
    """Error injection for testing error handling."""
    class ErrorInjector:
        def __init__(self):
            self.errors = {}
        
        def add_error(self, method_name, error_type, error_message):
            self.errors[method_name] = (error_type, error_message)
        
        def get_error(self, method_name):
            if method_name in self.errors:
                error_type, error_message = self.errors[method_name]
                return error_type(error_message)
            return None
    
    return ErrorInjector()


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "medical: mark test as a medical-specific test"
    )
    config.addinivalue_line(
        "markers", "property: mark test as a property-based test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment with proper configuration."""
    # Set test environment variables
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DISABLE_CUDA", "true")
    
    # Mock OpenAI API key to avoid accidental API calls
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    # Set numpy random seed for reproducible tests
    np.random.seed(42)
    
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)


# Clean up fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    
    # Clean up any test artifacts
    import gc
    gc.collect()
    
    # Reset any global state
    if hasattr(np.random, 'seed'):
        np.random.seed(42)


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_matrix_equal(actual, expected, tolerance=1e-6):
        """Assert that two matrices are equal within tolerance."""
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
        assert np.allclose(actual, expected, atol=tolerance), "Matrices are not equal within tolerance"
    
    @staticmethod
    def assert_performance_within_bounds(actual_time, expected_time, tolerance=0.1):
        """Assert that performance is within expected bounds."""
        assert actual_time <= expected_time * (1 + tolerance), f"Performance too slow: {actual_time} > {expected_time * (1 + tolerance)}"
    
    @staticmethod
    def generate_test_data(size, data_type="float32"):
        """Generate test data for testing."""
        if data_type == "float32":
            return np.random.randn(*size).astype(np.float32)
        elif data_type == "int32":
            return np.random.randint(0, 100, size).astype(np.int32)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils 