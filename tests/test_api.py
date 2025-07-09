"""
Comprehensive tests for API modules.

Tests FastAPI application, middleware, models, and routes for complete coverage.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from open_accelerator.api.main import app
from open_accelerator.api.middleware import (
    CORSMiddleware,
    LoggingMiddleware,
    SecurityMiddleware,
)
from open_accelerator.api.models import (
    AgentRequest,
    AgentResponse,
    ErrorResponse,
    HealthResponse,
    MedicalRequest,
    MedicalResponse,
    SimulationRequest,
    SimulationResponse,
)


class TestAPIMain:
    """Test FastAPI main application."""

    def test_app_initialization(self):
        """Test FastAPI app initialization."""
        assert app is not None
        assert app.title == "OpenAccelerator API"
        assert "1.0.0" in app.version
        assert app.description is not None

    def test_app_routes(self):
        """Test that all routes are registered."""
        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200

        # Test docs endpoint
        response = client.get(
            "/api/v1/docs",
            follow_redirects=False,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200

        # Test OpenAPI spec
        response = client.get(
            "/openapi.json", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200

    def test_cors_configuration(self):
        """Test CORS configuration."""
        client = TestClient(app)

        # Test preflight request
        response = client.options("/api/v1/health/")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

        # Check CORS headers
        response = client.get(
            "/api/v1/health/", headers={"Authorization": "Bearer test-token"}
        )
        assert "access-control-allow-origin" in response.headers

    @patch("open_accelerator.api.main.get_current_user")
    def test_authentication_dependency(self, mock_get_user):
        """Test authentication dependency."""
        mock_get_user.return_value = {"user_id": "test", "role": "admin"}
        client = TestClient(app)
        response = client.get(
            "/api/v1/health/", headers={"Authorization": "Bearer valid-token"}
        )
        assert response.status_code == 200

    def test_error_handling(self):
        """Test global error handling."""
        client = TestClient(app)

        # Test 404 error
        response = client.get("/nonexistent")
        assert response.status_code == 404

        # Test method not allowed
        response = client.post("/health")
        assert response.status_code == 405


class TestAPIMiddleware:
    """Test API middleware components."""

    def test_security_middleware_initialization(self):
        """Test security middleware initialization."""
        mock_app = Mock()
        middleware = SecurityMiddleware(mock_app)
        assert middleware is not None
        assert hasattr(middleware, "dispatch")

    @pytest.mark.asyncio
    async def test_security_middleware_headers(self):
        """Test security middleware adds security headers."""
        mock_app = Mock()
        middleware = SecurityMiddleware(mock_app)

        # Mock request and response
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url.path = "/api/v1/health"

        response = Mock()
        response.headers = {}
        response.status_code = 200

        # Mock call_next
        async def mock_call_next(req):
            return response

        # Process request
        result = await middleware.dispatch(request, mock_call_next)

        # Check security headers
        assert "X-Content-Type-Options" in result.headers
        assert "X-Frame-Options" in result.headers
        assert "X-XSS-Protection" in result.headers

    def test_logging_middleware_initialization(self):
        """Test logging middleware initialization."""
        mock_app = Mock()
        middleware = LoggingMiddleware(mock_app)
        assert middleware is not None
        assert hasattr(middleware, "dispatch")

    @pytest.mark.asyncio
    async def test_logging_middleware_request_logging(self):
        """Test logging middleware logs requests."""
        mock_app = Mock()
        middleware = LoggingMiddleware(mock_app)

        request = Mock()
        request.method = "POST"
        request.url.path = "/api/v1/simulation/run"
        request.client.host = "127.0.0.1"
        request.headers = {}

        with patch("open_accelerator.api.middleware.logging.getLogger") as mock_logger:
            mock_logger.return_value.info = Mock()

            response = Mock()
            response.status_code = 200
            response.headers = {}

            async def mock_call_next(req):
                return response

            await middleware.dispatch(request, mock_call_next)
            mock_logger.assert_called()

    def test_cors_middleware_initialization(self):
        """Test CORS middleware initialization."""
        mock_app = Mock()
        middleware = CORSMiddleware(mock_app, allow_origins=["*"])
        assert middleware is not None


class TestAPIModels:
    """Test API Pydantic models."""

    def test_simulation_request_model(self):
        """Test SimulationRequest model."""
        config_dict = {
            "name": "test_simulation",
            "accelerator_type": "balanced",
            "array": {"rows": 16, "cols": 16},
            "workload": {
                "workload_type": "gemm",
                "name": "test_gemm",
                "gemm_M": 64,
                "gemm_K": 32,
                "gemm_P": 16,
            },
        }

        request = SimulationRequest(**config_dict)
        assert request.name == "test_simulation"
        assert request.accelerator_type == "balanced"
        assert request.array["rows"] == 16
        assert request.workload["workload_type"] == "gemm"

    def test_simulation_response_model(self):
        """Test SimulationResponse model."""
        response_data = {
            "simulation_id": "sim-123",
            "status": "completed",
            "message": "Simulation completed successfully",
            "result": {
                "simulation_id": "sim-123",
                "status": "completed",
                "execution_time_seconds": 10.5,
                "total_cycles": 1000,
                "energy_consumed_joules": 50.0,
                "performance_metrics": {},
            },
        }

        response = SimulationResponse(**response_data)
        assert response.simulation_id == "sim-123"
        assert response.status == "completed"
        assert response.result.total_cycles == 1000

    def test_agent_request_model(self):
        """Test AgentRequest model."""
        request_data = {
            "message": "Optimize my workload",
            "agent_type": "optimization",
            "context": {"workload_type": "gemm", "target_metric": "performance"},
        }

        request = AgentRequest(**request_data)
        assert request.message == "Optimize my workload"
        assert request.agent_type == "optimization"
        assert request.context["workload_type"] == "gemm"

    def test_agent_response_model(self):
        """Test AgentResponse model."""
        response_data = {
            "response": "I recommend increasing array size",
            "agent_type": "optimization",
            "suggestions": [
                "Increase array rows to 32",
                "Use float16 precision",
                "Enable sparsity",
            ],
            "confidence": 0.95,
        }

        response = AgentResponse(**response_data)
        assert response.response == "I recommend increasing array size"
        assert response.agent_type == "optimization"
        assert len(response.suggestions) == 3
        assert response.confidence == 0.95

    def test_medical_request_model(self):
        """Test MedicalRequest model."""
        request_data = {
            "image_data": "base64encodedimage",
            "modality": "CT",
            "analysis_type": "segmentation",
            "patient_id": "patient-123",
            "study_id": "study-456",
        }

        request = MedicalRequest(**request_data)
        assert request.image_data == "base64encodedimage"
        assert request.modality == "CT"
        assert request.analysis_type == "segmentation"
        assert request.patient_id == "patient-123"

    def test_medical_response_model(self):
        """Test MedicalResponse model."""
        response_data = {
            "analysis_id": "analysis-789",
            "results": {
                "segmentation_mask": "base64mask",
                "confidence_scores": [0.9, 0.8, 0.95],
                "detected_regions": 3,
            },
            "compliance": {
                "hipaa_compliant": True,
                "phi_removed": True,
                "audit_trail": "audit-123",
            },
        }

        response = MedicalResponse(**response_data)
        assert response.analysis_id == "analysis-789"
        assert response.results["detected_regions"] == 3
        assert response.compliance["hipaa_compliant"] is True

    def test_health_response_model(self):
        """Test HealthResponse model."""
        health_data = {
            "status": "healthy",
            "timestamp": "2024-01-08T10:00:00Z",
            "version": "1.0.0",
            "components": {
                "database": "healthy",
                "cache": "healthy",
                "ai_agents": "healthy",
            },
        }

        response = HealthResponse(**health_data)
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.components["database"] == "healthy"

    def test_error_response_model(self):
        """Test ErrorResponse model."""
        error_data = {
            "error": "Validation failed",
            "message": "Invalid workload configuration",
            "code": "VALIDATION_ERROR",
            "details": {
                "field": "gemm_M",
                "value": -1,
                "constraint": "must be positive",
            },
        }

        response = ErrorResponse(**error_data)
        assert response.error == "Validation failed"
        assert response.code == "VALIDATION_ERROR"
        assert response.details["field"] == "gemm_M"

    def test_model_validation_errors(self):
        """Test model validation errors."""
        # Test invalid SimulationRequest
        with pytest.raises(ValueError):
            SimulationRequest(
                name="",  # Empty name should fail
                accelerator_type="invalid_type",
            )

        # Test invalid AgentRequest
        with pytest.raises(ValueError):
            AgentRequest(
                message="",  # Empty message should fail
                agent_type="invalid_agent",
            )


class TestAPIRoutes:
    """Test API routes."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health endpoint."""
        client = TestClient(app)
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch("open_accelerator.api.routes.SimulationEngine")
    def test_simulation_run_endpoint(self, mock_engine):
        """Test simulation run endpoint."""
        client = TestClient(app)
        mock_engine.return_value.run_simulation.return_value = {
            "simulation_id": "sim-123",
            "status": "completed",
        }
        request_data = {
            "name": "test_simulation",
            "accelerator_type": "balanced",
        }
        response = client.post(
            "/api/v1/simulation/run",
            json=request_data,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] is not None

    @patch("open_accelerator.ai.agents.AgentOrchestrator")
    def test_agent_chat_endpoint(self, mock_orchestrator):
        """Test agent chat endpoint."""
        client = TestClient(app)
        mock_agent = Mock()

        async def mock_process_message(*args, **kwargs):
            return "Test response"

        mock_agent.process_message = mock_process_message
        mock_orchestrator.return_value.get_agent.return_value = mock_agent

        request_data = {
            "agent_type": "optimization",
            "message": "Hello, agent!",
            "context": {},
        }
        response = client.post(
            "/api/v1/agents/chat",
            json=request_data,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "Test response" in data["response_text"]

    @patch("open_accelerator.medical.workflows.MedicalAnalyzer")
    def test_medical_analyze_endpoint(self, mock_analyzer):
        """Test medical analyze endpoint."""
        client = TestClient(app)
        mock_analyzer.return_value.analyze.return_value = {
            "analysis_id": "analysis-789",
            "results": {"detected_regions": 3},
        }
        request_data = {
            "image_data": "base64encodedimage",
            "modality": "CT",
            "analysis_type": "segmentation",
            "patient_id": "patient-123",
        }
        response = client.post(
            "/api/v1/medical/analyze",
            json=request_data,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["analysis_results"]["findings"] is not None

    def test_simulation_list_endpoint(self):
        """Test simulation list endpoint."""
        client = TestClient(app)
        response = client.get(
            "/api/v1/simulations/",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_unauthorized_access(self):
        """Test unauthorized access to protected endpoints."""
        client = TestClient(app)
        response = client.post("/api/v1/simulation/run", json={"name": "test"})
        assert response.status_code == 401

    def test_input_validation_errors(self):
        """Test input validation errors."""
        client = TestClient(app)
        response = client.post(
            "/api/v1/simulation/run",
            json={"invalid": "data"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422


class TestAPIIntegration:
    """Test API integration scenarios."""

    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        client = TestClient(app)

        with patch("open_accelerator.api.routes.get_current_user") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "admin"}

            with patch("open_accelerator.api.routes.SimulationEngine") as mock_engine:
                # Mock simulation engine responses
                mock_engine.return_value.run_simulation.return_value = {
                    "simulation_id": "sim-123",
                    "status": "completed",
                    "results": {"execution_time": 0.5},
                    "metrics": {"throughput": 1000},
                }

                mock_engine.return_value.get_simulation.return_value = {
                    "simulation_id": "sim-123",
                    "status": "completed",
                    "results": {"execution_time": 0.5},
                }

                # Step 1: Run simulation
                request_data = {
                    "name": "test_simulation",
                    "accelerator_type": "balanced",
                    "array": {"rows": 16, "cols": 16},
                    "workload": {
                        "workload_type": "gemm",
                        "name": "test_gemm",
                        "gemm_M": 64,
                        "gemm_K": 32,
                        "gemm_P": 16,
                    },
                }

                response = client.post(
                    "/api/v1/simulation/run",
                    json=request_data,
                    headers={"Authorization": "Bearer test-token"},
                )

                assert response.status_code == 200
                simulation_id = response.json()["simulation_id"]

                # Step 2: Get simulation status
                response = client.get(
                    f"/api/v1/simulation/{simulation_id}",
                    headers={"Authorization": "Bearer test-token"},
                )

                assert response.status_code == 200
                assert response.json()["status"] == "completed"

    def test_agent_optimization_workflow(self):
        """Test agent optimization workflow."""
        client = TestClient(app)

        with patch("open_accelerator.api.routes.get_current_user") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "user"}

            with patch("open_accelerator.ai.agents.OptimizationAgent") as mock_agent:

                async def mock_process_message(*args, **kwargs):
                    return {
                        "response": "Optimization recommendations",
                        "agent_type": "optimization",
                        "suggestions": ["Increase array size", "Use sparsity"],
                        "confidence": 0.9,
                    }

                mock_agent.return_value.process_message = mock_process_message

                # Step 1: Get optimization suggestions
                request_data = {
                    "message": "How can I optimize my GEMM workload?",
                    "agent_type": "optimization",
                    "context": {"workload_type": "gemm", "current_performance": 0.8},
                }

                response = client.post(
                    "/api/v1/agents/chat",
                    json=request_data,
                    headers={"Authorization": "Bearer test-token"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["agent_type"] == "optimization"
                assert len(data["suggestions"]) == 2

    def test_medical_compliance_workflow(self):
        """Test medical compliance workflow."""
        client = TestClient(app)

        with patch("open_accelerator.api.routes.get_current_user") as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "medical_staff"}

            with patch(
                "open_accelerator.medical.workflows.MedicalAnalyzer"
            ) as mock_analyzer:
                mock_analyzer.return_value.analyze.return_value = {
                    "analysis_id": "analysis-789",
                    "results": {
                        "segmentation_mask": "base64mask",
                        "confidence_scores": [0.9, 0.8, 0.95],
                    },
                    "compliance": {
                        "hipaa_compliant": True,
                        "phi_removed": True,
                        "audit_trail": "audit-123",
                    },
                }

                # Medical analysis request
                request_data = {
                    "image_data": "base64encodedimage",
                    "modality": "CT",
                    "analysis_type": "segmentation",
                    "patient_id": "patient-123",
                    "study_id": "study-456",
                }
                response = client.post(
                    "/api/v1/medical/analyze",
                    json=request_data,
                    headers={"Authorization": "Bearer test-token"},
                )
                assert response.status_code == 200
                data = response.json()
                assert "analysis_results" in data
