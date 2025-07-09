"""
Pydantic models for Open Accelerator API.

Defines request and response models with comprehensive validation,
proper typing, and medical AI workflow support.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict

from ..utils.config import WorkloadType


class SimulationStatus(str, Enum):
    """Simulation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Agent type enumeration."""
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    MEDICAL_COMPLIANCE = "medical_compliance"
    PERFORMANCE_TUNING = "performance_tuning"


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


# Configuration Models
class ArrayConfig(BaseModel):
    """Systolic array configuration."""
    rows: int = Field(ge=1, le=256, description="Number of rows in the array")
    cols: int = Field(ge=1, le=256, description="Number of columns in the array")
    dataflow: str = Field(default="output_stationary", description="Dataflow type")
    pe_type: str = Field(default="standard", description="Processing element type")
    enable_double_buffering: bool = Field(default=True, description="Enable double buffering")


class MemoryConfig(BaseModel):
    """Memory hierarchy configuration."""
    l1_cache_size: int = Field(ge=1024, description="L1 cache size in bytes")
    l2_cache_size: int = Field(ge=1024, description="L2 cache size in bytes")
    memory_bandwidth: int = Field(ge=1, description="Memory bandwidth in GB/s")
    enable_compression: bool = Field(default=False, description="Enable memory compression")


class PowerConfig(BaseModel):
    """Power management configuration."""
    max_power_watts: float = Field(ge=1.0, le=500.0, description="Maximum power consumption")
    enable_dvfs: bool = Field(default=True, description="Enable DVFS")
    thermal_limit_celsius: float = Field(ge=20.0, le=100.0, description="Thermal limit")
    enable_power_gating: bool = Field(default=True, description="Enable power gating")


class AcceleratorConfig(BaseModel):
    """Complete accelerator configuration."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Configuration name")
    array: ArrayConfig = Field(description="Array configuration")
    memory: MemoryConfig = Field(description="Memory configuration")
    power: PowerConfig = Field(description="Power configuration")
    medical_mode: bool = Field(default=False, description="Enable medical compliance mode")
    enable_security: bool = Field(default=True, description="Enable security features")


# Workload Models
class GEMMWorkloadConfig(BaseModel):
    """GEMM workload configuration."""
    m: int = Field(ge=1, le=4096, description="Matrix dimension M")
    k: int = Field(ge=1, le=4096, description="Matrix dimension K")
    p: int = Field(ge=1, le=4096, description="Matrix dimension P")
    data_type: str = Field(default="float32", description="Data type")
    use_mixed_precision: bool = Field(default=False, description="Use mixed precision")


class ConvolutionWorkloadConfig(BaseModel):
    """Convolution workload configuration."""
    input_height: int = Field(ge=1, le=4096, description="Input height")
    input_width: int = Field(ge=1, le=4096, description="Input width")
    input_channels: int = Field(ge=1, le=512, description="Input channels")
    kernel_size: int = Field(ge=1, le=32, description="Kernel size")
    stride: int = Field(ge=1, le=16, description="Stride")
    padding: int = Field(ge=0, le=16, description="Padding")
    batch_size: int = Field(ge=1, le=1024, description="Batch size")


class MedicalImagingWorkloadConfig(BaseModel):
    """Medical imaging workload configuration."""
    modality: str = Field(default="CT", description="Imaging modality (CT, MRI, X-ray)")
    image_size: tuple[int, int] = Field(default=(512, 512), description="Image size")
    slice_thickness: float = Field(ge=0.1, le=10.0, default=1.0, description="Slice thickness")
    enhancement_type: str = Field(default="contrast", description="Enhancement type")
    quality_level: str = Field(default="diagnostic", description="Quality level")
    hipaa_compliant: bool = Field(default=True, description="HIPAA compliance required")


class WorkloadConfig(BaseModel):
    """Generic workload configuration."""
    type: WorkloadType = Field(description="Workload type")
    config: Union[GEMMWorkloadConfig, ConvolutionWorkloadConfig, MedicalImagingWorkloadConfig] = Field(
        description="Workload-specific configuration"
    )
    priority: int = Field(ge=1, le=10, default=5, description="Execution priority")
    timeout_seconds: int = Field(ge=1, le=3600, default=300, description="Execution timeout")


# Request Models
class SimulationRequest(BaseModel):
    """Simulation request model."""
    simulation_name: str = Field(description="Simulation name")
    accelerator_config: AcceleratorConfig = Field(description="Accelerator configuration")
    workload: WorkloadConfig = Field(description="Workload configuration")
    enable_visualization: bool = Field(default=False, description="Enable visualization")
    enable_analysis: bool = Field(default=True, description="Enable performance analysis")
    output_format: str = Field(default="json", description="Output format")
    
    @field_validator('simulation_name')
    @classmethod
    def validate_simulation_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Simulation name must be alphanumeric with underscores/hyphens')
        return v


class AgentRequest(BaseModel):
    """Agent interaction request model."""
    agent_type: AgentType = Field(description="Agent type")
    message: str = Field(min_length=1, max_length=10000, description="Message to agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    stream_response: bool = Field(default=False, description="Stream response")
    max_tokens: int = Field(ge=1, le=4096, default=1000, description="Maximum response tokens")
    temperature: float = Field(ge=0.0, le=2.0, default=0.7, description="Response temperature")


class MedicalWorkflowRequest(BaseModel):
    """Medical workflow request model."""
    workflow_type: str = Field(description="Workflow type")
    patient_id: str = Field(description="Patient ID (anonymized)")
    study_id: str = Field(description="Study ID")
    image_data: Optional[str] = Field(default=None, description="Base64 encoded image data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Medical metadata")
    compliance_level: str = Field(default="hipaa", description="Compliance level required")
    
    @field_validator('workflow_type')
    @classmethod
    def validate_workflow_type(cls, v):
        allowed_types = ['diagnostic', 'screening', 'monitoring', 'research']
        if v not in allowed_types:
            raise ValueError(f'Workflow type must be one of {allowed_types}')
        return v
    
    @field_validator('patient_id')
    @classmethod
    def validate_patient_id(cls, v):
        import re
        if not re.match(r'^[A-Z0-9_-]+$', v):
            raise ValueError('Patient ID must contain only uppercase letters, numbers, underscores, and hyphens')
        return v
    
    @field_validator('study_id')
    @classmethod
    def validate_study_id(cls, v):
        import re
        if not re.match(r'^[A-Z0-9_-]+$', v):
            raise ValueError('Study ID must contain only uppercase letters, numbers, underscores, and hyphens')
        return v


# Response Models
class SimulationResult(BaseModel):
    """Simulation result model."""
    simulation_id: str = Field(description="Simulation ID")
    status: SimulationStatus = Field(description="Simulation status")
    execution_time_seconds: float = Field(description="Execution time in seconds")
    total_cycles: int = Field(description="Total simulation cycles")
    energy_consumed_joules: float = Field(description="Energy consumed in joules")
    performance_metrics: Dict[str, Any] = Field(description="Performance metrics")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Output data")
    visualization_urls: Optional[List[str]] = Field(default=None, description="Visualization URLs")
    error_details: Optional[str] = Field(default=None, description="Error details if failed")


class SimulationResponse(BaseResponse):
    """Simulation response model."""
    simulation_id: str = Field(description="Simulation ID")
    status: SimulationStatus = Field(description="Current status")
    result: Optional[SimulationResult] = Field(default=None, description="Result if completed")
    progress: Optional[float] = Field(default=None, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")


class AgentResponse(BaseResponse):
    """Agent response model."""
    agent_id: str = Field(description="Agent ID")
    response_text: str = Field(description="Agent response")
    suggestions: Optional[List[str]] = Field(default=None, description="Optimization suggestions")
    confidence: Optional[float] = Field(default=None, description="Response confidence")
    function_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Function calls made")


class MedicalWorkflowResponse(BaseResponse):
    """Medical workflow response model."""
    workflow_id: str = Field(description="Workflow ID")
    analysis_results: Dict[str, Any] = Field(description="Analysis results")
    confidence_score: float = Field(description="Confidence score")
    recommendations: List[str] = Field(description="Clinical recommendations")
    compliance_status: str = Field(description="Compliance status")
    audit_trail: List[Dict[str, Any]] = Field(description="Audit trail")


class HealthResponse(BaseResponse):
    """Health check response model."""
    status: HealthStatus = Field(description="System health status")
    version: str = Field(description="System version")
    uptime_seconds: float = Field(description="System uptime in seconds")
    system_metrics: Dict[str, Any] = Field(description="System metrics")
    dependencies: Dict[str, str] = Field(description="Dependency status")


class SimulationListResponse(BaseResponse):
    """List of simulations response model."""
    simulations: List[SimulationResult] = Field(description="List of simulations")
    total_count: int = Field(description="Total number of simulations")
    page: int = Field(description="Current page number")
    per_page: int = Field(description="Results per page")


class MetricsResponse(BaseResponse):
    """System metrics response model."""
    metrics: Dict[str, Any] = Field(description="System metrics")
    collection_time: datetime = Field(description="Metrics collection time")
    retention_period: int = Field(description="Metrics retention period in seconds") 