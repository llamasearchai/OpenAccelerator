"""
FastAPI API module for Open Accelerator simulator.

Provides RESTful endpoints for remote simulation control, agent interaction,
and medical AI workflow orchestration.
"""

from .main import app
from .middleware import setup_middleware
from .models import (
    AgentRequest,
    AgentResponse,
    HealthResponse,
    MedicalWorkflowRequest,
    MedicalWorkflowResponse,
    SimulationRequest,
    SimulationResponse,
)
from .routes import (
    agent_router,
    health_router,
    medical_router,
    simulation_router,
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
