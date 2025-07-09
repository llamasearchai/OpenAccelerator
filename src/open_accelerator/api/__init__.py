"""
FastAPI API module for Open Accelerator simulator.

Provides RESTful endpoints for remote simulation control, agent interaction,
and medical AI workflow orchestration.
"""

from .main import app
from .models import (
    SimulationRequest,
    SimulationResponse,
    AgentRequest,
    AgentResponse,
    HealthResponse,
    MedicalWorkflowRequest,
    MedicalWorkflowResponse,
)
from .middleware import setup_middleware
from .routes import (
    simulation_router,
    agent_router,
    medical_router,
    health_router,
    websocket_router,
)

__all__ = [
    "app",
    "SimulationRequest",
    "SimulationResponse",
    "AgentRequest",
    "AgentResponse",
    "HealthResponse",
    "MedicalWorkflowRequest",
    "MedicalWorkflowResponse",
    "setup_middleware",
    "simulation_router",
    "agent_router",
    "medical_router",
    "health_router",
    "websocket_router",
] 