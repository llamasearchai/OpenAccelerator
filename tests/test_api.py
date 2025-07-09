"""
Comprehensive tests for API modules.

Tests FastAPI application, middleware, models, and routes for complete coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import json

from open_accelerator.api.main import app, get_current_user
from open_accelerator.api.middleware import SecurityMiddleware, LoggingMiddleware, CORSMiddleware
from open_accelerator.api.models import (
    SimulationRequest, SimulationResponse, AgentRequest, AgentResponse,
    MedicalRequest, MedicalResponse, HealthResponse, ErrorResponse
)
from open_accelerator.api.routes import router
from open_accelerator.utils.config import AcceleratorConfig, WorkloadConfig, WorkloadType


class TestAPIMain:
    """Test FastAPI main application."""

    def test_app_initialization(self):
        """Test FastAPI app initialization."""
        assert app is not None
        assert app.title == "OpenAccelerator API"
        assert app.version == "1.0.0"
        assert app.description is not None

    def test_app_routes(self):
        """Test that all routes are registered."""
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_cors_configuration(self):
        """Test CORS configuration."""
        client = TestClient(app)
        
        # Test preflight request
        response = client.options("/api/v1/health")
        assert response.status_code == 200
        
        # Check CORS headers
        response = client.get("/api/v1/health")
        assert "access-control-allow-origin" in response.headers

    @patch('open_accelerator.api.main.verify_token')
    def test_authentication_dependency(self, mock_verify):
        """Test authentication dependency."""
        mock_verify.return_value = {"user_id": "test", "role": "admin"}
        
        # Test with valid token
        result = get_current_user("Bearer valid-token")
        assert result["user_id"] == "test"
        assert result["role"] == "admin"
        
        # Test with invalid token
        mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")
        with pytest.raises(HTTPException):
            get_current_user("Bearer invalid-token")

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
        middleware = SecurityMiddleware()
        assert middleware is not None
        assert hasattr(middleware, 'process_request')
        assert hasattr(middleware, 'process_response')

    @pytest.mark.asyncio
    async def test_security_middleware_headers(self):
        """Test security middleware adds security headers."""
        middleware = SecurityMiddleware()
        
        # Mock request and response
        request = Mock()
        request.headers = {}
        request.method = "GET"
        request.url.path = "/api/v1/health"
        
        response = Mock()
        response.headers = {}
        
        # Process response
        await middleware.process_response(request, response)
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    def test_logging_middleware_initialization(self):
        """Test logging middleware initialization."""
        middleware = LoggingMiddleware()
        assert middleware is not None
        assert hasattr(middleware, 'log_request')
        assert hasattr(middleware, 'log_response')

    @pytest.mark.asyncio
    async def test_logging_middleware_request_logging(self):
        """Test logging middleware logs requests."""
        middleware = LoggingMiddleware()
        
        request = Mock()
        request.method = "POST"
        request.url.path = "/api/v1/simulation/run"
        request.client.host = "127.0.0.1"
        
        with patch('open_accelerator.api.middleware.logger') as mock_logger:
            await middleware.log_request(request)
            mock_logger.info.assert_called()

    def test_cors_middleware_initialization(self):
        """Test CORS middleware initialization."""
        middleware = CORSMiddleware()
        assert middleware is not None
        assert hasattr(middleware, 'allow_origins')
        assert hasattr(middleware, 'allow_methods')


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
                "gemm_P": 16
            }
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
            "results": {
                "execution_time": 0.5,
                "power_consumption": 10.2,
                "accuracy": 0.99
            },
            "metrics": {
                "throughput": 1000,
                "latency": 0.001,
                "energy_efficiency": 0.95
            }
        }
        
        response = SimulationResponse(**response_data)
        assert response.simulation_id == "sim-123"
        assert response.status == "completed"
        assert response.results["execution_time"] == 0.5
        assert response.metrics["throughput"] == 1000

    def test_agent_request_model(self):
        """Test AgentRequest model."""
        request_data = {
            "message": "Optimize my workload",
            "agent_type": "optimization",
            "context": {
                "workload_type": "gemm",
                "target_metric": "performance"
            }
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
                "Enable sparsity"
            ],
            "confidence": 0.95
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
            "study_id": "study-456"
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
                "detected_regions": 3
            },
            "compliance": {
                "hipaa_compliant": True,
                "phi_removed": True,
                "audit_trail": "audit-123"
            }
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
                "ai_agents": "healthy"
            }
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
                "constraint": "must be positive"
            }
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
                accelerator_type="invalid_type"
            )
        
        # Test invalid AgentRequest
        with pytest.raises(ValueError):
            AgentRequest(
                message="",  # Empty message should fail
                agent_type="invalid_agent"
            )


class TestAPIRoutes:
    """Test API route handlers."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    @patch('open_accelerator.api.routes.SimulationEngine')
    def test_simulation_run_endpoint(self, mock_engine):
        """Test simulation run endpoint."""
        client = TestClient(app)
        
        # Mock simulation engine
        mock_engine.return_value.run_simulation.return_value = {
            "simulation_id": "sim-123",
            "status": "completed",
            "results": {"execution_time": 0.5},
            "metrics": {"throughput": 1000}
        }
        
        request_data = {
            "name": "test_simulation",
            "accelerator_type": "balanced",
            "array": {"rows": 16, "cols": 16},
            "workload": {
                "workload_type": "gemm",
                "name": "test_gemm",
                "gemm_M": 64,
                "gemm_K": 32,
                "gemm_P": 16
            }
        }
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "admin"}
            
            response = client.post(
                "/api/v1/simulation/run",
                json=request_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["simulation_id"] == "sim-123"
            assert data["status"] == "completed"

    @patch('open_accelerator.ai.agents.OptimizationAgent')
    def test_agent_chat_endpoint(self, mock_agent):
        """Test agent chat endpoint."""
        client = TestClient(app)
        
        # Mock agent response
        mock_agent.return_value.process_message.return_value = {
            "response": "I recommend increasing array size",
            "agent_type": "optimization",
            "suggestions": ["Increase array rows"],
            "confidence": 0.95
        }
        
        request_data = {
            "message": "Optimize my workload",
            "agent_type": "optimization",
            "context": {"workload_type": "gemm"}
        }
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "user"}
            
            response = client.post(
                "/api/v1/agents/chat",
                json=request_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "I recommend increasing array size"
            assert data["agent_type"] == "optimization"

    @patch('open_accelerator.medical.workflows.MedicalAnalyzer')
    def test_medical_analyze_endpoint(self, mock_analyzer):
        """Test medical analysis endpoint."""
        client = TestClient(app)
        
        # Mock medical analyzer
        mock_analyzer.return_value.analyze.return_value = {
            "analysis_id": "analysis-789",
            "results": {"detected_regions": 3},
            "compliance": {"hipaa_compliant": True}
        }
        
        request_data = {
            "image_data": "base64encodedimage",
            "modality": "CT",
            "analysis_type": "segmentation",
            "patient_id": "patient-123"
        }
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "medical_staff"}
            
            response = client.post(
                "/api/v1/medical/analyze",
                json=request_data,
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["analysis_id"] == "analysis-789"
            assert data["results"]["detected_regions"] == 3

    def test_simulation_list_endpoint(self):
        """Test simulation list endpoint."""
        client = TestClient(app)
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "user"}
            
            with patch('open_accelerator.api.routes.SimulationEngine') as mock_engine:
                mock_engine.return_value.list_simulations.return_value = [
                    {"simulation_id": "sim-1", "status": "completed"},
                    {"simulation_id": "sim-2", "status": "running"}
                ]
                
                response = client.get(
                    "/api/v1/simulation/list",
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 2
                assert data[0]["simulation_id"] == "sim-1"

    def test_unauthorized_access(self):
        """Test unauthorized access to protected endpoints."""
        client = TestClient(app)
        
        # Test without token
        response = client.post("/api/v1/simulation/run", json={})
        assert response.status_code == 401
        
        # Test with invalid token
        response = client.post(
            "/api/v1/simulation/run",
            json={},
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    def test_input_validation_errors(self):
        """Test input validation errors."""
        client = TestClient(app)
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "admin"}
            
            # Test invalid simulation request
            response = client.post(
                "/api/v1/simulation/run",
                json={"invalid": "data"},
                headers={"Authorization": "Bearer test-token"}
            )
            assert response.status_code == 422
            
            # Test invalid agent request
            response = client.post(
                "/api/v1/agents/chat",
                json={"invalid": "data"},
                headers={"Authorization": "Bearer test-token"}
            )
            assert response.status_code == 422


class TestAPIIntegration:
    """Test API integration scenarios."""

    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        client = TestClient(app)
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "admin"}
            
            with patch('open_accelerator.api.routes.SimulationEngine') as mock_engine:
                # Mock simulation engine responses
                mock_engine.return_value.run_simulation.return_value = {
                    "simulation_id": "sim-123",
                    "status": "completed",
                    "results": {"execution_time": 0.5},
                    "metrics": {"throughput": 1000}
                }
                
                mock_engine.return_value.get_simulation.return_value = {
                    "simulation_id": "sim-123",
                    "status": "completed",
                    "results": {"execution_time": 0.5}
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
                        "gemm_P": 16
                    }
                }
                
                response = client.post(
                    "/api/v1/simulation/run",
                    json=request_data,
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                simulation_id = response.json()["simulation_id"]
                
                # Step 2: Get simulation status
                response = client.get(
                    f"/api/v1/simulation/{simulation_id}",
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                assert response.json()["status"] == "completed"

    def test_agent_optimization_workflow(self):
        """Test agent optimization workflow."""
        client = TestClient(app)
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "user"}
            
            with patch('open_accelerator.ai.agents.OptimizationAgent') as mock_agent:
                mock_agent.return_value.process_message.return_value = {
                    "response": "Optimization recommendations",
                    "agent_type": "optimization",
                    "suggestions": ["Increase array size", "Use sparsity"],
                    "confidence": 0.9
                }
                
                # Step 1: Get optimization suggestions
                request_data = {
                    "message": "How can I optimize my GEMM workload?",
                    "agent_type": "optimization",
                    "context": {
                        "workload_type": "gemm",
                        "current_performance": 0.8
                    }
                }
                
                response = client.post(
                    "/api/v1/agents/chat",
                    json=request_data,
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["agent_type"] == "optimization"
                assert len(data["suggestions"]) == 2

    def test_medical_compliance_workflow(self):
        """Test medical compliance workflow."""
        client = TestClient(app)
        
        with patch('open_accelerator.api.routes.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "role": "medical_staff"}
            
            with patch('open_accelerator.medical.workflows.MedicalAnalyzer') as mock_analyzer:
                mock_analyzer.return_value.analyze.return_value = {
                    "analysis_id": "analysis-789",
                    "results": {
                        "segmentation_mask": "base64mask",
                        "confidence_scores": [0.9, 0.8, 0.95]
                    },
                    "compliance": {
                        "hipaa_compliant": True,
                        "phi_removed": True,
                        "audit_trail": "audit-123"
                    }
                }
                
                # Medical analysis request
                request_data = {
                    "image_data": "base64encodedimage",
                    "modality": "CT",
                    "analysis_type": "segmentation",
                    "patient_id": "patient-123",
                    "study_id": "study-456"
                }
                
                response = client.post(
                    "/api/v1/medical/analyze",
                    json=request_data,
                    headers={"Authorization": "Bearer test-token"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["compliance"]["hipaa_compliant"] is True
                assert data["compliance"]["phi_removed"] is True 