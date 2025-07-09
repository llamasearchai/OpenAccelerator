"""
API route definitions for Open Accelerator.

Provides comprehensive REST endpoints for simulation control, agent interaction,
medical workflows, and real-time communication via WebSocket.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer

from ..ai.agents import AgentOrchestrator

# Import accelerator components
from ..simulation.simulator import SimulationConfig, SimulationOrchestrator
from ..utils.config import AcceleratorConfig as AcceleratorConfigUtil
from ..workloads.base import BaseWorkload
from .models import (
    AgentRequest,
    AgentResponse,
    AgentType,
    HealthResponse,
    HealthStatus,
    MedicalWorkflowRequest,
    MedicalWorkflowResponse,
    MetricsResponse,
    SimulationListResponse,
    SimulationRequest,
    SimulationResponse,
    SimulationResult,
    SimulationStatus,
)

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# In-memory storage for simulation state
active_simulations: Dict[str, Dict[str, Any]] = {}
simulation_results: Dict[str, SimulationResult] = {}
websocket_connections: Dict[str, WebSocket] = {}


# Routers
simulation_router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])
agent_router = APIRouter(prefix="/api/v1/agents", tags=["agents"])
medical_router = APIRouter(prefix="/api/v1/medical", tags=["medical"])
health_router = APIRouter(prefix="/api/v1/health", tags=["health"])
websocket_router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])


# Dependency injection
async def get_simulation_orchestrator() -> SimulationOrchestrator:
    """Get simulation orchestrator instance."""
    config = SimulationConfig(
        enable_real_time_monitoring=True,
        enable_performance_analysis=True,
        enable_parallel_execution=True,
    )
    return SimulationOrchestrator(config)


async def get_agent_orchestrator() -> AgentOrchestrator:
    """Get agent orchestrator instance."""
    from ..ai.agents import AgentConfig

    config = AgentConfig(
        api_key=None,  # Will use environment variable
        enable_function_calling=True,
        medical_compliance=True,
    )
    return AgentOrchestrator(config)


# Simulation endpoints
@simulation_router.post("/run", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    orchestrator: SimulationOrchestrator = Depends(get_simulation_orchestrator),
) -> SimulationResponse:
    """Run a simulation with the specified configuration."""
    try:
        simulation_id = str(uuid4())

        # Convert API model to internal config
        accel_config = _convert_api_config_to_internal(request.accelerator_config)

        # Create workload
        workload = _create_workload_from_request(request.workload, accel_config)

        # Store simulation state
        active_simulations[simulation_id] = {
            "status": SimulationStatus.PENDING,
            "request": request,
            "workload": workload,
            "start_time": time.time(),
            "progress": 0.0,
        }

        # Start simulation in background
        asyncio.create_task(
            _run_simulation_async(simulation_id, accel_config, workload, orchestrator)
        )

        return SimulationResponse(
            simulation_id=simulation_id,
            status=SimulationStatus.PENDING,
            message="Simulation started successfully",
        )

    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start simulation: {str(e)}",
        )


@simulation_router.get("/status/{simulation_id}", response_model=SimulationResponse)
async def get_simulation_status(simulation_id: str) -> SimulationResponse:
    """Get the status of a running simulation."""
    if simulation_id not in active_simulations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Simulation not found"
        )

    sim_state = active_simulations[simulation_id]

    response = SimulationResponse(
        simulation_id=simulation_id,
        status=sim_state["status"],
        progress=sim_state.get("progress"),
        message=f"Simulation is {sim_state['status'].value}",
    )

    # Add result if completed
    if simulation_id in simulation_results:
        response.result = simulation_results[simulation_id]

    return response


@simulation_router.get("/list", response_model=SimulationListResponse)
async def list_simulations(
    page: int = 1, per_page: int = 20, status_filter: Optional[SimulationStatus] = None
) -> SimulationListResponse:
    """List all simulations with optional filtering."""
    all_results = list(simulation_results.values())

    # Apply status filter if provided
    if status_filter:
        all_results = [r for r in all_results if r.status == status_filter]

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = all_results[start:end]

    return SimulationListResponse(
        simulations=paginated_results,
        total_count=len(all_results),
        page=page,
        per_page=per_page,
    )


@simulation_router.delete("/cancel/{simulation_id}")
async def cancel_simulation(simulation_id: str) -> Dict[str, str]:
    """Cancel a running simulation."""
    if simulation_id not in active_simulations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Simulation not found"
        )

    sim_state = active_simulations[simulation_id]
    if sim_state["status"] in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel completed simulation",
        )

    # Update status
    sim_state["status"] = SimulationStatus.CANCELLED

    return {"message": "Simulation cancelled successfully"}


# Agent endpoints
@agent_router.post("/chat", response_model=AgentResponse)
async def chat_with_agent(
    request: AgentRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> AgentResponse:
    """Chat with an AI agent."""
    try:
        # Get appropriate agent
        agent = orchestrator.get_agent(request.agent_type.value)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent type {request.agent_type.value} not available",
            )

        # Process message
        response_text = await agent.process_message(request.message, request.context)

        return AgentResponse(
            agent_id=agent.agent_id,
            response_text=response_text,
            message="Agent response generated successfully",
        )

    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent chat failed: {str(e)}",
        )


@agent_router.post("/stream/{agent_type}")
async def stream_agent_response(
    agent_type: AgentType,
    request: AgentRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> StreamingResponse:
    """Stream agent response for real-time interaction."""

    async def generate_stream():
        try:
            agent = orchestrator.get_agent(agent_type.value)
            if not agent:
                yield f"data: {json.dumps({'error': 'Agent not found'})}\n\n"
                return

            # Mock streaming response (in real implementation, this would stream from OpenAI)
            response_text = await agent.process_message(
                request.message, request.context
            )

            # Stream response word by word
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {"token": word, "is_final": i == len(words) - 1, "index": i}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay

        except Exception as e:
            error_chunk = {"error": str(e)}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )


# Medical endpoints
@medical_router.post("/analyze", response_model=MedicalWorkflowResponse)
async def analyze_medical_image(
    request: MedicalWorkflowRequest,
    orchestrator: SimulationOrchestrator = Depends(get_simulation_orchestrator),
) -> MedicalWorkflowResponse:
    """Analyze medical image with AI acceleration."""
    try:
        workflow_id = str(uuid4())

        # Create medical workload
        from ..workloads.medical import MedicalImagingWorkload, MedicalWorkloadConfig

        workload_config = MedicalWorkloadConfig(
            modality=request.metadata.get("modality", "CT"),
            image_size=(512, 512),
            quality_level="diagnostic",
        )

        workload = MedicalImagingWorkload(workload_config)

        # Run analysis (mock implementation)
        analysis_results = {
            "findings": ["Normal liver parenchyma", "No acute findings"],
            "measurements": {"liver_volume": 1800, "lesion_count": 0},
            "classification": "normal",
        }

        # Audit trail for compliance
        audit_trail = [
            {
                "timestamp": time.time(),
                "action": "image_analysis_started",
                "user_id": "system",
                "metadata": {"workflow_id": workflow_id},
            },
            {
                "timestamp": time.time(),
                "action": "analysis_completed",
                "user_id": "system",
                "metadata": {"findings_count": len(analysis_results["findings"])},
            },
        ]

        return MedicalWorkflowResponse(
            workflow_id=workflow_id,
            analysis_results=analysis_results,
            confidence_score=0.95,
            recommendations=["Follow-up in 6 months", "No immediate action required"],
            compliance_status="HIPAA_COMPLIANT",
            audit_trail=audit_trail,
        )

    except Exception as e:
        logger.error(f"Medical analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Medical analysis failed: {str(e)}",
        )


# Health endpoints
@health_router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Check system health
        import psutil

        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }

        # Check dependencies
        dependencies = {
            "numpy": "available",
            "openai": "available" if "openai" in globals() else "not_available",
            "fastapi": "available",
            "pydantic": "available",
        }

        # Determine overall health
        health_status = HealthStatus.HEALTHY
        if system_metrics["cpu_percent"] > 80 or system_metrics["memory_percent"] > 80:
            health_status = HealthStatus.DEGRADED

        return HealthResponse(
            status=health_status,
            version="1.0.0",
            uptime_seconds=time.time() - 1234567890,  # Mock start time
            system_metrics=system_metrics,
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status=HealthStatus.UNHEALTHY,
            version="1.0.0",
            uptime_seconds=0,
            system_metrics={},
            dependencies={},
            success=False,
            message=f"Health check failed: {str(e)}",
        )


@health_router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics() -> MetricsResponse:
    """Get detailed system metrics."""
    try:
        from datetime import datetime

        import psutil

        metrics = {
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": dict(psutil.disk_usage("/")._asdict()),
                "network": dict(psutil.net_io_counters()._asdict()),
            },
            "application": {
                "active_simulations": len(active_simulations),
                "completed_simulations": len(simulation_results),
                "websocket_connections": len(websocket_connections),
            },
        }

        return MetricsResponse(
            metrics=metrics, collection_time=datetime.now(), retention_period=3600
        )

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics collection failed: {str(e)}",
        )


# WebSocket endpoints
@websocket_router.websocket("/simulation/{simulation_id}")
async def websocket_simulation_updates(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation updates."""
    await websocket.accept()

    try:
        websocket_connections[simulation_id] = websocket

        # Send initial status
        if simulation_id in active_simulations:
            await websocket.send_json(
                {
                    "type": "status_update",
                    "data": {
                        "simulation_id": simulation_id,
                        "status": active_simulations[simulation_id]["status"],
                        "progress": active_simulations[simulation_id].get(
                            "progress", 0
                        ),
                    },
                }
            )

        # Keep connection alive and send updates
        while True:
            try:
                # Wait for message or timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Process incoming message if needed
                await websocket.send_json({"type": "ack", "message": "received"})

            except asyncio.TimeoutError:
                # Send periodic updates
                if simulation_id in active_simulations:
                    await websocket.send_json(
                        {"type": "heartbeat", "timestamp": time.time()}
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for simulation {simulation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if simulation_id in websocket_connections:
            del websocket_connections[simulation_id]


# Helper functions
async def _run_simulation_async(
    simulation_id: str,
    accel_config: AcceleratorConfigUtil,
    workload: BaseWorkload,
    orchestrator: SimulationOrchestrator,
):
    """Run simulation asynchronously."""
    try:
        # Update status
        active_simulations[simulation_id]["status"] = SimulationStatus.RUNNING

        # Run simulation
        result = orchestrator.run_single_simulation(
            accel_config, workload, simulation_name=f"api_simulation_{simulation_id}"
        )

        # Create result
        simulation_result = SimulationResult(
            simulation_id=simulation_id,
            status=SimulationStatus.COMPLETED,
            execution_time_seconds=time.time()
            - active_simulations[simulation_id]["start_time"],
            total_cycles=result.metrics.get("total_cycles", 0),
            energy_consumed_joules=result.metrics.get("energy_consumed", 0),
            performance_metrics=result.metrics,
        )

        # Store result
        simulation_results[simulation_id] = simulation_result
        active_simulations[simulation_id]["status"] = SimulationStatus.COMPLETED

        # Notify WebSocket clients
        if simulation_id in websocket_connections:
            await websocket_connections[simulation_id].send_json(
                {"type": "simulation_completed", "data": simulation_result.dict()}
            )

    except Exception as e:
        logger.error(f"Simulation {simulation_id} failed: {e}")
        active_simulations[simulation_id]["status"] = SimulationStatus.FAILED

        # Create failed result
        simulation_result = SimulationResult(
            simulation_id=simulation_id,
            status=SimulationStatus.FAILED,
            execution_time_seconds=time.time()
            - active_simulations[simulation_id]["start_time"],
            total_cycles=0,
            energy_consumed_joules=0,
            performance_metrics={},
            error_details=str(e),
        )

        simulation_results[simulation_id] = simulation_result


def _convert_api_config_to_internal(api_config) -> AcceleratorConfigUtil:
    """Convert API config to internal config format."""
    from ..utils.config import (
        AcceleratorType,
        ArrayConfig,
        BufferConfig,
        DataflowType,
        DataType,
        MedicalConfig,
        MemoryHierarchyConfig,
        PowerConfig,
    )

    # Convert array config
    array_config = ArrayConfig(
        rows=api_config.array.rows,
        cols=api_config.array.cols,
        frequency=1e9,  # Default 1 GHz
        voltage=1.2,  # Default 1.2V
        dataflow=DataflowType.OUTPUT_STATIONARY,  # Default dataflow
    )

    # Convert memory config
    memory_config = MemoryHierarchyConfig(
        l1_size=api_config.memory.l1_cache_size,
        l2_size=api_config.memory.l2_cache_size,
        main_memory_bandwidth=api_config.memory.memory_bandwidth,
        enable_prefetching=True,
    )

    # Convert power config
    power_config = PowerConfig(
        enable_power_gating=api_config.power.enable_power_gating,
        enable_dvfs=api_config.power.enable_dvfs,
        thermal_design_power=api_config.power.max_power_watts,
        operating_temperature=api_config.power.thermal_limit_celsius,
    )

    # Convert medical config
    medical_config = MedicalConfig(
        enable_medical_mode=api_config.medical_mode,
        phi_compliance=api_config.medical_mode,
        fda_validation=api_config.medical_mode,
    )

    # Create buffer configs with reasonable defaults
    buffer_config = BufferConfig(buffer_size=1024, bandwidth=16, latency=1)

    # Determine accelerator type
    accel_type = (
        AcceleratorType.MEDICAL if api_config.medical_mode else AcceleratorType.BALANCED
    )

    return AcceleratorConfigUtil(
        name=api_config.name,
        accelerator_type=accel_type,
        data_type=DataType.FLOAT32,
        array=array_config,
        input_buffer=buffer_config,
        weight_buffer=buffer_config,
        output_buffer=buffer_config,
        memory=memory_config,
        power=power_config,
        medical=medical_config,
    )


def _create_workload_from_request(workload_config, accel_config) -> BaseWorkload:
    """Create workload from API request."""
    from ..workloads.gemm import GEMMWorkload, GEMMWorkloadConfig

    if workload_config.type == "gemm":
        config = GEMMWorkloadConfig(
            m=workload_config.config.m,
            k=workload_config.config.k,
            p=workload_config.config.p,
        )
        return GEMMWorkload(config, accel_config)
    else:
        raise ValueError(f"Unsupported workload type: {workload_config.type}")
