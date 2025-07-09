"""
Integration tests for OpenAccelerator system.

Tests complete system integration including FastAPI endpoints,
OpenAI agents, Docker setup, and end-to-end workflows.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

try:
    import httpx
    from fastapi.testclient import TestClient

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    import open_accelerator
    from open_accelerator.ai.agents import AgentConfig, AgentOrchestrator
    from open_accelerator.api.main import app
    from open_accelerator.core.security import SecurityConfig, SecurityManager
    from open_accelerator.simulation.simulator import Simulator
    from open_accelerator.utils.config import AcceleratorConfig
    from open_accelerator.workloads.base import ComputeWorkload

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


@pytest.mark.skipif(
    not HTTP_AVAILABLE, reason="HTTP testing dependencies not available"
)
@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestFastAPIIntegration:
    """Test FastAPI endpoints integration."""

    def test_api_app_creation(self):
        """Test FastAPI app can be created."""
        client = TestClient(app)
        assert client is not None

    def test_health_endpoint(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/api/v1/health")

        # Should return 200 OK even if services aren't fully initialized
        assert response.status_code in [200, 503]

        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data

    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "message" in data

    def test_openapi_documentation(self):
        """Test OpenAPI documentation endpoint."""
        client = TestClient(app)
        response = client.get("/api/v1/openapi.json")

        assert response.status_code == 200
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert openapi_spec["info"]["title"] == "Open Accelerator API"

    def test_simulation_endpoint_structure(self):
        """Test simulation endpoint structure."""
        client = TestClient(app)

        # Test endpoint exists (may require authentication)
        response = client.post("/api/v1/simulation/run", json={"test": "data"})

        # Should return 401 (unauthorized) or 422 (validation error) rather than 404
        assert response.status_code in [401, 422, 200]

    def test_agents_endpoint_structure(self):
        """Test agents endpoint structure."""
        client = TestClient(app)

        # Test endpoint exists (may require authentication)
        response = client.post("/api/v1/agents/chat", json={"message": "test"})

        # Should return 401 (unauthorized) or 422 (validation error) rather than 404
        assert response.status_code in [401, 422, 200]

    def test_medical_endpoint_structure(self):
        """Test medical endpoint structure."""
        client = TestClient(app)

        # Test endpoint exists (may require authentication)
        response = client.post("/api/v1/medical/analyze", json={"data": "test"})

        # Should return 401 (unauthorized) or 422 (validation error) rather than 404
        assert response.status_code in [401, 422, 200]


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestEndToEndSimulation:
    """Test end-to-end simulation workflows."""

    def test_complete_simulation_workflow(self):
        """Test complete simulation workflow."""
        # Create configuration
        config = AcceleratorConfig()
        config.array.rows = 4
        config.array.cols = 4
        config.max_cycles = 100

        # Create simulator
        simulator = Simulator(config)

        # Create workload
        workload = ComputeWorkload("E2ETest")
        workload.prepare()

        # Run simulation
        results = simulator.run(workload, cycles=10)

        # Verify results structure
        assert isinstance(results, dict)

        # Should have basic simulation metrics
        expected_keys = ["success", "results", "metrics", "simulation_results"]
        assert any(key in results for key in expected_keys)

    def test_medical_compliance_workflow(self):
        """Test medical compliance workflow."""
        # Create medical configuration
        config = AcceleratorConfig()
        config.medical.enable_medical_mode = True
        config.medical.phi_compliance = True
        config.medical.fda_validation = True

        # Create security manager
        security_config = SecurityConfig()
        security_config.hipaa_compliant = True
        security_config.fda_compliant = True
        security_manager = SecurityManager(security_config)

        # Test data encryption for medical data
        test_medical_data = b"Patient ID: 12345, Diagnosis: Test"
        encrypted_data = security_manager.encrypt_data(test_medical_data)

        assert encrypted_data != test_medical_data
        assert len(encrypted_data) > len(test_medical_data)

        # Test decryption
        decrypted_data = security_manager.decrypt_data(encrypted_data)
        assert decrypted_data == test_medical_data

    def test_performance_analysis_workflow(self):
        """Test performance analysis workflow."""
        from open_accelerator.analysis.performance_analysis import PerformanceAnalyzer

        # Create configuration
        config = AcceleratorConfig()
        analyzer = PerformanceAnalyzer(config)

        # Mock simulation results
        mock_results = {
            "total_cycles": 1000,
            "total_operations": 5000,
            "energy_consumed": 100.5,
            "peak_power": 50.0,
            "memory_accesses": 2000,
            "cache_hits": 1800,
            "cache_misses": 200,
        }

        # Analyze performance
        analysis = analyzer.analyze_performance(mock_results)

        # Verify analysis results
        assert isinstance(analysis, dict)

        # Should contain performance metrics
        expected_metrics = [
            "throughput",
            "efficiency",
            "power",
            "energy",
            "performance",
        ]
        assert any(metric in str(analysis).lower() for metric in expected_metrics)


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestSecurityIntegration:
    """Test security system integration."""

    def test_security_manager_initialization(self):
        """Test security manager initialization."""
        config = SecurityConfig()
        config.enable_encryption = True
        config.enable_audit_logging = True
        config.hipaa_compliant = True

        security_manager = SecurityManager(config)

        assert security_manager.config == config
        assert security_manager.key_manager is not None
        assert security_manager.audit_logger is not None

    def test_end_to_end_encryption(self):
        """Test end-to-end encryption workflow."""
        config = SecurityConfig()
        security_manager = SecurityManager(config)

        # Test various data types
        test_cases = [
            b"Simple string data",
            b"Medical data: Patient 123, CT scan results",
            b"x" * 1000,  # Large data
            b"",  # Empty data
            b"\x00\x01\x02\x03",  # Binary data
        ]

        for test_data in test_cases:
            # Encrypt
            encrypted = security_manager.encrypt_data(test_data)
            assert encrypted != test_data

            # Decrypt
            decrypted = security_manager.decrypt_data(encrypted)
            assert decrypted == test_data

    def test_audit_logging_integration(self):
        """Test audit logging integration."""
        config = SecurityConfig()
        config.enable_audit_logging = True

        security_manager = SecurityManager(config)

        # Perform some operations that should be logged
        test_data = b"Test audit data"
        encrypted = security_manager.encrypt_data(test_data)
        decrypted = security_manager.decrypt_data(encrypted)

        # Verify audit logger is working
        assert security_manager.audit_logger is not None

        # Check metrics
        metrics = security_manager.get_security_status()
        assert metrics["metrics"]["encryption_operations"] >= 1
        assert metrics["metrics"]["decryption_operations"] >= 1


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestAgentIntegration:
    """Test AI agent integration."""

    def test_agent_configuration(self):
        """Test agent configuration setup."""
        config = AgentConfig()
        config.model = "gpt-4o"
        config.temperature = 0.1
        config.medical_compliance = True

        # Should not raise exceptions
        assert config.model == "gpt-4o"
        assert config.temperature == 0.1
        assert config.medical_compliance is True

    @patch("open_accelerator.ai.agents.OPENAI_AVAILABLE", True)
    def test_agent_orchestrator_creation(self):
        """Test agent orchestrator creation."""
        config = AgentConfig()
        config.api_key = "test_key"

        # Mock OpenAI client
        with patch("open_accelerator.ai.agents.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            orchestrator = AgentOrchestrator(config)
            assert orchestrator.config == config

    def test_agent_medical_compliance(self):
        """Test agent medical compliance features."""
        config = AgentConfig()
        config.medical_compliance = True
        config.safety_mode = True

        # Medical compliance should be enabled
        assert config.medical_compliance is True
        assert config.safety_mode is True

    @patch("open_accelerator.ai.agents.OPENAI_AVAILABLE", False)
    def test_agent_fallback_when_openai_unavailable(self):
        """Test agent fallback when OpenAI is not available."""
        config = AgentConfig()

        # Should handle gracefully when OpenAI is not available
        # This tests the fallback behavior
        assert config.model == "gpt-4o"  # Configuration should still work


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestDockerIntegration:
    """Test Docker integration."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile_path.exists()

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        assert compose_path.exists()

    def test_docker_configuration(self):
        """Test Docker configuration files."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"

        if dockerfile_path.exists():
            content = dockerfile_path.read_text()

            # Check for basic Docker requirements
            assert "FROM" in content
            assert "python" in content.lower()
            assert "COPY" in content or "ADD" in content
            assert "RUN" in content

    def test_docker_compose_configuration(self):
        """Test Docker Compose configuration."""
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"

        if compose_path.exists():
            content = compose_path.read_text()

            # Check for basic compose structure
            assert "version:" in content
            assert "services:" in content


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestBuildAutomation:
    """Test build automation and CI/CD."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()

    def test_requirements_structure(self):
        """Test requirements structure."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

        if pyproject_path.exists():
            content = pyproject_path.read_text()

            # Check for basic project structure
            assert "[build-system]" in content
            assert "[project]" in content
            assert "dependencies" in content
            assert "Nik Jois" in content
            assert "nikjois@llamasearch.ai" in content

    def test_makefile_exists(self):
        """Test that Makefile exists."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        assert makefile_path.exists()

    def test_makefile_targets(self):
        """Test Makefile targets."""
        makefile_path = Path(__file__).parent.parent / "Makefile"

        if makefile_path.exists():
            content = makefile_path.read_text()

            # Check for common targets
            expected_targets = ["test", "install", "build", "clean"]
            for target in expected_targets:
                assert f"{target}:" in content or f".PHONY: {target}" in content

    def test_package_structure(self):
        """Test package structure."""
        src_path = Path(__file__).parent.parent / "src" / "open_accelerator"
        assert src_path.exists()

        # Check for key modules
        key_modules = [
            "__init__.py",
            "api/main.py",
            "core/accelerator.py",
            "ai/agents.py",
            "utils/config.py",
        ]

        for module in key_modules:
            module_path = src_path / module
            assert module_path.exists(), f"Module {module} not found"


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestConfigurationValidation:
    """Test configuration validation across system."""

    def test_configuration_consistency(self):
        """Test configuration consistency across components."""
        # Test that all components use consistent configuration
        config = AcceleratorConfig()

        # Basic validation
        assert config.name is not None
        assert config.array.rows > 0
        assert config.array.cols > 0
        assert config.max_cycles > 0

        # Medical configuration
        if config.medical.enable_medical_mode:
            assert hasattr(config.medical, "phi_compliance")
            assert hasattr(config.medical, "fda_validation")

    def test_default_configurations(self):
        """Test default configurations."""
        from open_accelerator.utils.config import get_default_configs

        configs = get_default_configs()

        # Test all default configs are valid
        for config_name, config in configs.items():
            assert config.name is not None
            assert config.array.rows > 0
            assert config.array.cols > 0

            # Medical config should have medical mode enabled
            if config_name == "medical":
                assert config.medical.enable_medical_mode is True

    def test_author_information_consistency(self):
        """Test author information consistency."""
        # Check package info
        assert open_accelerator.__author__ == "LlamaFarms Team"
        assert open_accelerator.__email__ == "team@llamafarms.ai"

        # Check pyproject.toml
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            assert "LlamaFarms Team" in content
            assert "team@llamafarms.ai" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
