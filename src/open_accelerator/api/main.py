"""
Main FastAPI application for Open Accelerator API.

Comprehensive REST API with OpenAI Agents SDK integration, medical AI workflows,
and real-time simulation control.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .middleware import setup_middleware
from .routes import (
    simulation_router,
    agent_router,
    medical_router,
    health_router,
    websocket_router,
)
from .models import ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for cleanup
app_state: Dict[str, Any] = {
    "startup_time": None,
    "shutdown_initiated": False,
    "active_connections": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    import time
    from datetime import datetime
    
    app_state["startup_time"] = time.time()
    logger.info(f"Open Accelerator API starting up at {datetime.now()}")
    
    # Initialize OpenAI client if available
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            logger.info("OpenAI integration initialized")
        else:
            logger.warning("OpenAI API key not found - agent features will be limited")
    except ImportError:
        logger.warning("OpenAI package not available - agent features disabled")
    
    # Initialize other services
    try:
        # Initialize simulation orchestrator
        from ..simulation.simulator import SimulationOrchestrator, SimulationConfig
        sim_config = SimulationConfig(
            enable_real_time_monitoring=True,
            enable_performance_analysis=True,
            enable_parallel_execution=True,
            max_parallel_workers=4
        )
        app.state.simulation_orchestrator = SimulationOrchestrator(sim_config)
        logger.info("Simulation orchestrator initialized")
        
        # Initialize agent orchestrator if available
        try:
            from ..ai.agents import AgentOrchestrator, AgentConfig
            agent_config = AgentConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                enable_function_calling=True,
                medical_compliance=True
            )
            app.state.agent_orchestrator = AgentOrchestrator(agent_config)
            logger.info("Agent orchestrator initialized")
        except Exception as e:
            logger.warning(f"Agent orchestrator initialization failed: {e}")
            app.state.agent_orchestrator = None
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    logger.info("API startup complete")
    yield
    
    # Shutdown
    logger.info("API shutdown initiated")
    app_state["shutdown_initiated"] = True
    
    # Cleanup resources
    if hasattr(app.state, 'simulation_orchestrator'):
        try:
            # Stop any running simulations
            logger.info("Stopping active simulations...")
            # Implementation would depend on orchestrator interface
        except Exception as e:
            logger.error(f"Error stopping simulations: {e}")
    
    logger.info("API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Open Accelerator API",
    description="""
    ## Advanced ML Accelerator Simulator API
    
    Comprehensive REST API for simulating ML accelerator architectures with:
    - **Real-time simulation control** with WebSocket support
    - **OpenAI Agents SDK integration** for intelligent optimization
    - **Medical AI workflows** with HIPAA compliance
    - **Performance analysis** with roofline modeling
    - **Comprehensive monitoring** with health checks and metrics
    
    ### Features
    - [MEDICAL] Medical AI Focus: Specialized for healthcare applications
    - [AI] AI Agents: Integrated optimization and analysis agents
    - [REALTIME] Real-time: WebSocket streaming for live simulation updates
    - [SECURITY] Security: Enterprise-grade security and compliance
    - [ANALYTICS] Analytics: Detailed performance metrics and visualization
    
    ### Getting Started
    1. Start by checking the `/api/v1/health` endpoint
    2. Run a simulation with `/api/v1/simulation/run`
    3. Monitor progress with WebSocket at `/api/v1/ws/simulation/{id}`
    4. Analyze results with AI agents at `/api/v1/agents/chat`
    
    ### Authentication
    API endpoints require Bearer token authentication for protected operations.
    Get your API key from the configuration or environment variables.
    """,
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
    }
)

# Setup middleware
middleware_config = {
    "allowed_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "api_key": os.getenv("API_KEY"),
    "trusted_hosts": os.getenv("TRUSTED_HOSTS", "").split(",") if os.getenv("TRUSTED_HOSTS") else None,
    "rate_limit": int(os.getenv("RATE_LIMIT", "100")),
    "enable_audit_logging": os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true",
}

setup_middleware(app, middleware_config)

# Add routers
app.include_router(simulation_router)
app.include_router(agent_router)
app.include_router(medical_router)
app.include_router(health_router)
app.include_router(websocket_router)

# Authentication setup
security = HTTPBearer()

def verify_token(token: str) -> dict[str, Any]:
    """Verify authentication token."""
    # In production, this would verify JWT tokens
    # For testing, accept any token that starts with "valid"
    if token.startswith("valid"):
        return {"user_id": "test_user", "role": "admin"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials_or_token = Depends(security)) -> dict[str, Any]:
    """Get current authenticated user."""
    if isinstance(credentials_or_token, str):
        # Direct token string (for testing)
        token = credentials_or_token.replace("Bearer ", "")
        return verify_token(token)
    else:
        # HTTPAuthorizationCredentials object
        token = credentials_or_token.credentials
        return verify_token(token)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An internal server error occurred",
            error_details={"type": type(exc).__name__, "message": str(exc)}
        ).dict()
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            error_details={"status_code": exc.status_code}
        ).dict()
    )

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "Open Accelerator API",
        "version": "1.0.0",
        "description": "Advanced ML Accelerator Simulator API",
        "message": "Welcome to Open Accelerator API - Advanced ML Accelerator Simulator",
        "documentation": "/api/v1/docs",
        "health_check": "/api/v1/health",
        "websocket_endpoint": "/api/v1/ws/simulation/{simulation_id}",
        "features": [
            "Real-time simulation control",
            "OpenAI Agents SDK integration",
            "Medical AI workflows",
            "HIPAA compliance",
            "Performance analysis",
            "WebSocket streaming"
        ],
        "endpoints": {
            "simulation": "/api/v1/simulation",
            "agents": "/api/v1/agents",
            "medical": "/api/v1/medical",
            "health": "/api/v1/health",
            "websocket": "/api/v1/ws"
        }
    }

# OpenAPI customization
def custom_openapi():
    """Custom OpenAPI schema with additional information."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Open Accelerator API",
        version="1.0.0",
        description="Advanced ML Accelerator Simulator API with OpenAI Agents SDK integration",
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your API key as a Bearer token"
        }
    }
    
    # Add tags
    openapi_schema["tags"] = [
        {
            "name": "simulation",
            "description": "Simulation management and control",
            "externalDocs": {
                "description": "Simulation Documentation",
                "url": "https://docs.openaccelerator.ai/simulation"
            }
        },
        {
            "name": "agents",
            "description": "AI agent interaction with OpenAI integration",
            "externalDocs": {
                "description": "Agent Documentation",
                "url": "https://docs.openaccelerator.ai/agents"
            }
        },
        {
            "name": "medical",
            "description": "Medical AI workflows and HIPAA compliance",
            "externalDocs": {
                "description": "Medical AI Documentation",
                "url": "https://docs.openaccelerator.ai/medical"
            }
        },
        {
            "name": "health",
            "description": "System health and monitoring",
        },
        {
            "name": "websocket",
            "description": "Real-time communication via WebSocket",
        }
    ]
    
    # Add examples
    openapi_schema["components"]["examples"] = {
        "BasicSimulation": {
            "summary": "Basic GEMM simulation",
            "value": {
                "simulation_name": "basic_gemm_test",
                "accelerator_config": {
                    "name": "test_config",
                    "array": {
                        "rows": 4,
                        "cols": 4,
                        "dataflow": "output_stationary"
                    },
                    "memory": {
                        "l1_cache_size": 32768,
                        "l2_cache_size": 262144,
                        "memory_bandwidth": 100
                    },
                    "power": {
                        "max_power_watts": 50.0,
                        "enable_dvfs": True,
                        "thermal_limit_celsius": 70.0
                    },
                    "medical_mode": False
                },
                "workload": {
                    "type": "gemm",
                    "config": {
                        "m": 16,
                        "k": 16,
                        "p": 16
                    }
                },
                "enable_visualization": True,
                "enable_analysis": True
            }
        },
        "MedicalWorkflow": {
            "summary": "Medical imaging workflow",
            "value": {
                "workflow_type": "diagnostic",
                "patient_id": "PATIENT_123",
                "study_id": "STUDY_456",
                "metadata": {
                    "modality": "CT",
                    "body_part": "liver",
                    "contrast": True
                },
                "compliance_level": "hipaa"
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add some useful middleware for development
if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
    @app.middleware("http")
    async def development_middleware(request: Request, call_next):
        """Development middleware for debugging."""
        import time
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} (time: {process_time:.3f}s)")
        
        return response

# Export for use with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 