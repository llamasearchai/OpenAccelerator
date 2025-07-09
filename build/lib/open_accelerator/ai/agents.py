"""
OpenAI Agents SDK Integration for Intelligent Accelerator Management.

Provides AI-powered optimization, analysis, and management capabilities
using OpenAI's agents framework for compound AI systems.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import openai
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. AI agents will be disabled.")

from ..analysis.metrics import PerformanceAnalyzer
from ..core.accelerator import AcceleratorController
from ..utils.config import AcceleratorConfig
from ..workloads.base import BaseWorkload

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for AI agents."""

    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000
    enable_function_calling: bool = True
    enable_memory: bool = True
    memory_window: int = 10

    # Agent specialization
    optimization_focus: str = "balanced"  # performance, power, area, balanced
    expertise_level: str = "expert"  # novice, intermediate, expert

    # Safety and compliance
    safety_mode: bool = True
    medical_compliance: bool = False

    # API configuration
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None


@dataclass
class AgentMessage:
    """Message structure for agent communication."""

    role: str  # system, user, assistant, function
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base class for all AI agents."""

    def __init__(self, agent_id: str, config: AgentConfig):
        """Initialize base agent."""
        self.agent_id = agent_id
        self.config = config
        self.client = None
        self.conversation_history: List[AgentMessage] = []
        self.function_registry: Dict[str, Callable] = {}
        self.is_initialized = False

        if OPENAI_AVAILABLE:
            self._initialize_openai_client()

        # System prompt
        self.system_prompt = self._get_system_prompt()
        self.conversation_history.append(
            AgentMessage(
                role="system", content=self.system_prompt, agent_id=self.agent_id
            )
        )

    def _initialize_openai_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not found")
                return

            self.client = OpenAI(
                api_key=api_key,
                base_url=self.config.api_base,
                organization=self.config.organization,
            )

            self.is_initialized = True
            logger.info(f"Agent {self.agent_id} initialized with OpenAI")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return f"""You are {self.agent_id}, an AI agent specialized in accelerator optimization and analysis.

Your expertise includes:
- Systolic array architectures and optimization
- Memory hierarchy design and analysis
- Power management and thermal optimization
- Performance analysis and bottleneck identification
- Workload characterization and mapping
- Medical AI compliance and safety

Your optimization focus is: {self.config.optimization_focus}
Your expertise level is: {self.config.expertise_level}
Safety mode: {'enabled' if self.config.safety_mode else 'disabled'}
Medical compliance: {'required' if self.config.medical_compliance else 'not required'}

Provide accurate, actionable recommendations based on data analysis.
Always consider trade-offs between performance, power, and area.
For medical applications, prioritize safety and regulatory compliance.
"""

    def register_function(self, name: str, func: Callable, description: str = None):
        """Register a function for the agent to call."""
        self.function_registry[name] = {
            "function": func,
            "description": description or f"Function {name}",
            "name": name,
        }
        logger.debug(f"Registered function {name} for agent {self.agent_id}")

    async def process_message(
        self, message: str, context: Dict[str, Any] = None
    ) -> str:
        """Process a message and generate response."""
        if not self.is_initialized:
            return "Agent not initialized. Please check OpenAI configuration."

        # Add user message to history
        user_message = AgentMessage(
            role="user", content=message, agent_id=self.agent_id, metadata=context or {}
        )
        self.conversation_history.append(user_message)

        try:
            # Prepare messages for OpenAI
            messages = self._prepare_messages_for_openai()

            # Prepare function definitions
            functions = self._prepare_function_definitions()

            # Make API call
            response = await self._call_openai_api(messages, functions)

            # Process response
            assistant_message = self._process_openai_response(response)
            self.conversation_history.append(assistant_message)

            return assistant_message.content

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing request: {str(e)}"

    def _prepare_messages_for_openai(self) -> List[Dict[str, str]]:
        """Prepare conversation history for OpenAI API."""
        messages = []

        # Limit conversation history
        recent_messages = self.conversation_history[-self.config.memory_window :]

        for msg in recent_messages:
            openai_msg = {"role": msg.role, "content": msg.content}

            if msg.function_call:
                openai_msg["function_call"] = msg.function_call

            messages.append(openai_msg)

        return messages

    def _prepare_function_definitions(self) -> List[Dict[str, Any]]:
        """Prepare function definitions for OpenAI API."""
        if not self.config.enable_function_calling:
            return []

        functions = []

        for func_name, func_info in self.function_registry.items():
            function_def = {
                "name": func_name,
                "description": func_info["description"],
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
            functions.append(function_def)

        return functions

    async def _call_openai_api(
        self, messages: List[Dict[str, str]], functions: List[Dict[str, Any]]
    ) -> Any:
        """Make API call to OpenAI."""
        try:
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            if functions and self.config.enable_function_calling:
                kwargs["functions"] = functions
                kwargs["function_call"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            return response

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _process_openai_response(self, response: Any) -> AgentMessage:
        """Process OpenAI response and handle function calls."""
        choice = response.choices[0]
        message = choice.message

        # Handle function calls
        if hasattr(message, "function_call") and message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)

            # Execute function
            if function_name in self.function_registry:
                try:
                    function_result = self.function_registry[function_name]["function"](
                        **function_args
                    )
                    content = f"Function {function_name} executed successfully. Result: {function_result}"
                except Exception as e:
                    content = f"Function {function_name} failed: {str(e)}"
            else:
                content = f"Unknown function: {function_name}"

            return AgentMessage(
                role="assistant",
                content=content,
                agent_id=self.agent_id,
                function_call=message.function_call.__dict__,
            )

        # Regular response
        return AgentMessage(
            role="assistant", content=message.content, agent_id=self.agent_id
        )

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history."""
        return {
            "agent_id": self.agent_id,
            "total_messages": len(self.conversation_history),
            "recent_messages": [
                {
                    "role": msg.role,
                    "content": msg.content[:100] + "..."
                    if len(msg.content) > 100
                    else msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in self.conversation_history[-5:]
            ],
        }


class OptimizationAgent(BaseAgent):
    """Agent specialized in accelerator optimization."""

    def __init__(self, config: AgentConfig):
        """Initialize optimization agent."""
        super().__init__("optimization_agent", config)

        # Register optimization functions
        self.register_function(
            "analyze_performance_bottlenecks",
            self._analyze_performance_bottlenecks,
            "Analyze performance bottlenecks in accelerator execution",
        )

        self.register_function(
            "suggest_configuration_changes",
            self._suggest_configuration_changes,
            "Suggest configuration changes for optimization",
        )

        self.register_function(
            "estimate_optimization_impact",
            self._estimate_optimization_impact,
            "Estimate impact of proposed optimizations",
        )

    def _analyze_performance_bottlenecks(
        self, performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        bottlenecks = {
            "compute_utilization": performance_data.get("pe_utilization", 0.0),
            "memory_utilization": performance_data.get("memory_utilization", 0.0),
            "power_efficiency": performance_data.get(
                "energy_efficiency_tops_per_watt", 0.0
            ),
            "primary_bottleneck": "unknown",
            "recommendations": [],
        }

        # Analyze bottlenecks
        if bottlenecks["compute_utilization"] < 0.5:
            bottlenecks["primary_bottleneck"] = "compute_underutilization"
            bottlenecks["recommendations"].append(
                "Low compute utilization detected. Consider workload optimization or array size reduction."
            )

        if bottlenecks["memory_utilization"] > 0.9:
            bottlenecks["primary_bottleneck"] = "memory_bound"
            bottlenecks["recommendations"].append(
                "High memory utilization. Consider increasing buffer sizes or optimizing memory access patterns."
            )

        if bottlenecks["power_efficiency"] < 10:
            bottlenecks["recommendations"].append(
                "Low power efficiency. Consider enabling power management features."
            )

        return bottlenecks

    def _suggest_configuration_changes(
        self, current_config: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest configuration changes."""
        suggestions = []

        # Array size optimization
        pe_util = performance_data.get("pe_utilization", 0.0)
        if pe_util < 0.3:
            suggestions.append(
                {
                    "parameter": "array_size",
                    "current_value": f"{current_config.get('array_rows', 16)}x{current_config.get('array_cols', 16)}",
                    "suggested_value": f"{current_config.get('array_rows', 16)//2}x{current_config.get('array_cols', 16)//2}",
                    "reason": "Reduce array size to improve utilization",
                    "expected_impact": "Improved efficiency, reduced power consumption",
                }
            )
        elif pe_util > 0.9:
            suggestions.append(
                {
                    "parameter": "array_size",
                    "current_value": f"{current_config.get('array_rows', 16)}x{current_config.get('array_cols', 16)}",
                    "suggested_value": f"{current_config.get('array_rows', 16)*2}x{current_config.get('array_cols', 16)*2}",
                    "reason": "Increase array size for higher throughput",
                    "expected_impact": "Higher throughput, increased power consumption",
                }
            )

        # Memory optimization
        mem_util = performance_data.get("memory_utilization", 0.0)
        if mem_util > 0.8:
            suggestions.append(
                {
                    "parameter": "buffer_sizes",
                    "current_value": "Current buffer configuration",
                    "suggested_value": "Increase buffer sizes by 50%",
                    "reason": "High memory utilization detected",
                    "expected_impact": "Reduced memory stalls, improved performance",
                }
            )

        # Frequency optimization
        power_efficiency = performance_data.get("energy_efficiency_tops_per_watt", 0.0)
        if power_efficiency < 10:
            suggestions.append(
                {
                    "parameter": "frequency",
                    "current_value": f"{current_config.get('frequency', 1e9)/1e9:.1f} GHz",
                    "suggested_value": f"{current_config.get('frequency', 1e9)*0.8/1e9:.1f} GHz",
                    "reason": "Low power efficiency",
                    "expected_impact": "Improved power efficiency, reduced performance",
                }
            )

        return suggestions

    def _estimate_optimization_impact(
        self, optimization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate impact of optimization."""
        impact = {
            "performance_change": 0.0,
            "power_change": 0.0,
            "area_change": 0.0,
            "confidence": 0.8,
            "trade_offs": [],
        }

        opt_type = optimization.get("parameter", "")

        if opt_type == "array_size":
            if "increase" in optimization.get("reason", "").lower():
                impact["performance_change"] = 0.5  # 50% improvement
                impact["power_change"] = 0.8  # 80% increase
                impact["area_change"] = 1.0  # 100% increase
                impact["trade_offs"].append(
                    "Higher performance but increased power and area"
                )
            else:
                impact["performance_change"] = -0.3  # 30% reduction
                impact["power_change"] = -0.5  # 50% reduction
                impact["area_change"] = -0.5  # 50% reduction
                impact["trade_offs"].append(
                    "Lower performance but reduced power and area"
                )

        elif opt_type == "frequency":
            if "increase" in optimization.get("reason", "").lower():
                impact["performance_change"] = 0.2  # 20% improvement
                impact["power_change"] = 0.4  # 40% increase
                impact["trade_offs"].append(
                    "Higher performance but increased power consumption"
                )
            else:
                impact["performance_change"] = -0.2  # 20% reduction
                impact["power_change"] = -0.3  # 30% reduction
                impact["trade_offs"].append(
                    "Lower performance but improved power efficiency"
                )

        return impact


class AnalysisAgent(BaseAgent):
    """Agent specialized in performance analysis."""

    def __init__(self, config: AgentConfig):
        """Initialize analysis agent."""
        super().__init__("analysis_agent", config)

        # Register analysis functions
        self.register_function(
            "analyze_workload_characteristics",
            self._analyze_workload_characteristics,
            "Analyze workload characteristics and requirements",
        )

        self.register_function(
            "generate_performance_report",
            self._generate_performance_report,
            "Generate comprehensive performance analysis report",
        )

        self.register_function(
            "compare_configurations",
            self._compare_configurations,
            "Compare multiple accelerator configurations",
        )

    def _analyze_workload_characteristics(
        self, workload_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze workload characteristics."""
        analysis = {
            "workload_type": workload_info.get("type", "unknown"),
            "computational_intensity": "medium",
            "memory_requirements": "medium",
            "parallelism_potential": "high",
            "optimization_opportunities": [],
        }

        # Analyze based on workload type
        if analysis["workload_type"] == "gemm":
            analysis["computational_intensity"] = "high"
            analysis["memory_requirements"] = "medium"
            analysis["parallelism_potential"] = "very_high"
            analysis["optimization_opportunities"] = [
                "Matrix blocking for cache efficiency",
                "Data layout optimization",
                "Sparsity exploitation",
            ]

        elif analysis["workload_type"] == "convolution":
            analysis["computational_intensity"] = "high"
            analysis["memory_requirements"] = "high"
            analysis["parallelism_potential"] = "high"
            analysis["optimization_opportunities"] = [
                "Im2col transformation",
                "Winograd convolution",
                "Channel-wise parallelization",
            ]

        # Add complexity metrics
        complexity = workload_info.get("complexity", {})
        analysis["flop_count"] = complexity.get("total_operations", 0)
        analysis["memory_footprint"] = complexity.get("memory_footprint_bytes", 0)

        return analysis

    def _generate_performance_report(
        self, simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "executive_summary": {},
            "detailed_metrics": {},
            "bottleneck_analysis": {},
            "recommendations": [],
            "benchmarking": {},
        }

        # Executive summary
        exec_summary = simulation_results.get("execution_summary", {})
        report["executive_summary"] = {
            "total_cycles": exec_summary.get("total_cycles", 0),
            "throughput": exec_summary.get("throughput_ops_per_second", 0.0),
            "efficiency": simulation_results.get("performance_metrics", {}).get(
                "energy_efficiency_tops_per_watt", 0.0
            ),
            "utilization": simulation_results.get("performance_metrics", {}).get(
                "pe_utilization", 0.0
            ),
        }

        # Detailed metrics
        perf_metrics = simulation_results.get("performance_metrics", {})
        report["detailed_metrics"] = {
            "compute_metrics": {
                "pe_utilization": perf_metrics.get("pe_utilization", 0.0),
                "total_operations": perf_metrics.get("total_operations", 0),
                "peak_throughput": perf_metrics.get("throughput_ops_per_second", 0.0),
            },
            "memory_metrics": {
                "memory_utilization": perf_metrics.get("memory_utilization", 0.0),
                "memory_bandwidth_utilization": 0.0,  # Would need to calculate
            },
            "power_metrics": {
                "average_power": perf_metrics.get("average_power_watts", 0.0),
                "peak_power": perf_metrics.get("peak_power_watts", 0.0),
                "energy_efficiency": perf_metrics.get(
                    "energy_efficiency_tops_per_watt", 0.0
                ),
            },
        }

        # Bottleneck analysis
        bottleneck_data = simulation_results.get("bottleneck_analysis", {})
        report["bottleneck_analysis"] = {
            "primary_bottleneck": bottleneck_data.get("primary_bottleneck", "unknown"),
            "severity": bottleneck_data.get("bottleneck_severity", "low"),
            "impact": "Performance limited by "
            + bottleneck_data.get("primary_bottleneck", "unknown factors"),
        }

        # Recommendations
        report["recommendations"] = simulation_results.get(
            "optimization_recommendations", []
        )

        return report

    def _compare_configurations(
        self, config_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple configurations."""
        if not config_results:
            return {"error": "No configurations to compare"}

        comparison = {
            "configuration_count": len(config_results),
            "best_performance": {},
            "best_efficiency": {},
            "best_power": {},
            "trade_off_analysis": {},
            "recommendations": [],
        }

        # Find best in each category
        best_perf = max(
            config_results, key=lambda x: x.get("throughput_ops_per_second", 0)
        )
        best_eff = max(
            config_results, key=lambda x: x.get("energy_efficiency_tops_per_watt", 0)
        )
        best_power = min(
            config_results, key=lambda x: x.get("average_power_watts", float("inf"))
        )

        comparison["best_performance"] = {
            "config": best_perf.get("accelerator_config", "unknown"),
            "throughput": best_perf.get("throughput_ops_per_second", 0.0),
        }

        comparison["best_efficiency"] = {
            "config": best_eff.get("accelerator_config", "unknown"),
            "efficiency": best_eff.get("energy_efficiency_tops_per_watt", 0.0),
        }

        comparison["best_power"] = {
            "config": best_power.get("accelerator_config", "unknown"),
            "power": best_power.get("average_power_watts", 0.0),
        }

        # Trade-off analysis
        comparison["trade_off_analysis"] = {
            "performance_vs_power": "Higher performance configurations consume more power",
            "efficiency_vs_area": "More efficient configurations may require larger arrays",
            "optimization_space": "Multiple Pareto-optimal configurations exist",
        }

        # Recommendations
        comparison["recommendations"] = [
            f"For maximum performance: {best_perf.get('accelerator_config', 'unknown')}",
            f"For maximum efficiency: {best_eff.get('accelerator_config', 'unknown')}",
            f"For minimum power: {best_power.get('accelerator_config', 'unknown')}",
        ]

        return comparison


class MedicalComplianceAgent(BaseAgent):
    """Agent specialized in medical AI compliance and safety."""

    def __init__(self, config: AgentConfig):
        """Initialize medical compliance agent."""
        config.medical_compliance = True
        config.safety_mode = True
        super().__init__("medical_compliance_agent", config)

        # Register medical compliance functions
        self.register_function(
            "check_medical_compliance",
            self._check_medical_compliance,
            "Check medical AI compliance requirements",
        )

        self.register_function(
            "validate_safety_requirements",
            self._validate_safety_requirements,
            "Validate safety requirements for medical applications",
        )

        self.register_function(
            "generate_compliance_report",
            self._generate_compliance_report,
            "Generate medical compliance report",
        )

    def _check_medical_compliance(
        self, system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check medical compliance requirements."""
        compliance = {
            "overall_compliance": True,
            "requirements_met": [],
            "requirements_failed": [],
            "recommendations": [],
        }

        # Check precision requirements
        precision = system_config.get("precision", "float32")
        if precision in ["float64", "float32"]:
            compliance["requirements_met"].append("Precision requirements (float32/64)")
        else:
            compliance["requirements_failed"].append(
                "Insufficient precision for medical applications"
            )
            compliance["overall_compliance"] = False

        # Check reliability features
        reliability_enabled = system_config.get("enable_reliability", False)
        if reliability_enabled:
            compliance["requirements_met"].append("Reliability features enabled")
        else:
            compliance["requirements_failed"].append("Reliability features not enabled")
            compliance["overall_compliance"] = False

        # Check security features
        security_enabled = system_config.get("enable_security", False)
        if security_enabled:
            compliance["requirements_met"].append("Security features enabled")
        else:
            compliance["requirements_failed"].append("Security features not enabled")
            compliance["overall_compliance"] = False

        # Generate recommendations
        if not compliance["overall_compliance"]:
            compliance["recommendations"] = [
                "Enable medical-grade precision (float32 minimum)",
                "Implement error correction and detection",
                "Enable data encryption and secure processing",
                "Implement safety monitoring and shutdown mechanisms",
            ]

        return compliance

    def _validate_safety_requirements(
        self, performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate safety requirements."""
        safety = {
            "safety_level": "unknown",
            "safety_violations": [],
            "safety_metrics": {},
            "recommendations": [],
        }

        # Check error rates
        reliability_metrics = performance_data.get("reliability_metrics", {})
        error_rate = reliability_metrics.get("error_rate", 0.0)

        if error_rate < 1e-9:
            safety["safety_level"] = "medical_grade"
            safety["safety_metrics"]["error_rate"] = error_rate
        elif error_rate < 1e-6:
            safety["safety_level"] = "industrial_grade"
            safety["safety_violations"].append(
                "Error rate too high for medical applications"
            )
        else:
            safety["safety_level"] = "consumer_grade"
            safety["safety_violations"].append(
                "Error rate unacceptable for medical applications"
            )

        # Check availability
        availability = reliability_metrics.get("availability", 1.0)
        if availability < 0.9999:
            safety["safety_violations"].append(
                "Availability below medical requirements (99.99%)"
            )

        # Generate recommendations
        if safety["safety_violations"]:
            safety["recommendations"] = [
                "Implement triple modular redundancy",
                "Add comprehensive error detection and correction",
                "Implement graceful degradation mechanisms",
                "Add real-time health monitoring",
            ]

        return safety

    def _generate_compliance_report(
        self, full_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            "compliance_summary": {},
            "regulatory_assessment": {},
            "safety_assessment": {},
            "recommendations": [],
            "certification_readiness": "not_ready",
        }

        # Compliance summary
        compliance_data = full_analysis.get("medical_compliance", {})
        report["compliance_summary"] = {
            "overall_score": compliance_data.get("regulatory_compliance_score", 0.0),
            "safety_events": compliance_data.get("safety_events", 0),
            "precision_violations": compliance_data.get("precision_violations", 0),
        }

        # Regulatory assessment
        report["regulatory_assessment"] = {
            "fda_510k_readiness": "requires_additional_validation",
            "iec_62304_compliance": "partial",
            "iso_13485_compliance": "not_assessed",
        }

        # Safety assessment
        report["safety_assessment"] = {
            "safety_integrity_level": "SIL-2",
            "fault_tolerance": "single_point_of_failure_protected",
            "failure_modes_analyzed": True,
        }

        # Certification readiness
        if (
            report["compliance_summary"]["overall_score"] > 0.95
            and report["compliance_summary"]["safety_events"] == 0
        ):
            report["certification_readiness"] = "ready_for_pre_submission"
        elif report["compliance_summary"]["overall_score"] > 0.8:
            report["certification_readiness"] = "requires_minor_improvements"
        else:
            report["certification_readiness"] = "requires_major_improvements"

        return report


class AgentOrchestrator:
    """Orchestrates multiple AI agents for comprehensive analysis."""

    def __init__(self, config: AgentConfig):
        """Initialize agent orchestrator."""
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents."""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available. Agents will not be functional.")
            return

        # Optimization agent
        self.agents["optimization"] = OptimizationAgent(self.config)

        # Analysis agent
        self.agents["analysis"] = AnalysisAgent(self.config)

        # Medical compliance agent (if needed)
        if self.config.medical_compliance:
            self.agents["medical"] = MedicalComplianceAgent(self.config)

        logger.info(f"Initialized {len(self.agents)} AI agents")

    async def comprehensive_analysis(
        self,
        simulation_results: Dict[str, Any],
        accelerator_config: AcceleratorConfig,
        workload: BaseWorkload,
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis using all agents."""
        if not self.agents:
            return {"error": "No agents available"}

        analysis_results = {}

        # Optimization analysis
        if "optimization" in self.agents:
            opt_prompt = f"""
            Analyze the following accelerator simulation results and provide optimization recommendations:

            Simulation Results: {json.dumps(simulation_results, indent=2, default=str)}
            Accelerator Config: {accelerator_config.name}
            Workload: {workload.get_name()}

            Focus on: {self.config.optimization_focus}

            Provide specific, actionable recommendations for improving performance, power efficiency, and overall system optimization.
            """

            opt_response = await self.agents["optimization"].process_message(opt_prompt)
            analysis_results["optimization"] = opt_response

        # Performance analysis
        if "analysis" in self.agents:
            analysis_prompt = f"""
            Generate a comprehensive performance analysis report for the following simulation:

            Results: {json.dumps(simulation_results, indent=2, default=str)}

            Include:
            1. Executive summary of key metrics
            2. Detailed performance breakdown
            3. Bottleneck identification
            4. Comparison with theoretical limits
            5. Recommendations for improvement
            """

            analysis_response = await self.agents["analysis"].process_message(
                analysis_prompt
            )
            analysis_results["performance_analysis"] = analysis_response

        # Medical compliance analysis (if applicable)
        if "medical" in self.agents and self.config.medical_compliance:
            medical_prompt = f"""
            Assess the medical compliance and safety aspects of this accelerator system:

            System Configuration: {accelerator_config.__dict__}
            Simulation Results: {json.dumps(simulation_results, indent=2, default=str)}

            Evaluate:
            1. Medical-grade precision and accuracy
            2. Safety and reliability requirements
            3. Regulatory compliance readiness
            4. Risk assessment and mitigation
            """

            medical_response = await self.agents["medical"].process_message(
                medical_prompt
            )
            analysis_results["medical_compliance"] = medical_response

        return analysis_results

    async def interactive_consultation(
        self, user_query: str, context: Dict[str, Any] = None
    ) -> str:
        """Provide interactive consultation using appropriate agent."""
        if not self.agents:
            return "No agents available for consultation."

        # Determine which agent to use based on query
        query_lower = user_query.lower()

        if any(
            word in query_lower
            for word in ["optimize", "improve", "performance", "speed"]
        ):
            agent = self.agents.get("optimization")
        elif any(
            word in query_lower
            for word in ["analyze", "report", "metrics", "bottleneck"]
        ):
            agent = self.agents.get("analysis")
        elif any(
            word in query_lower
            for word in ["medical", "safety", "compliance", "regulatory"]
        ):
            agent = self.agents.get("medical")
        else:
            # Default to optimization agent
            agent = self.agents.get("optimization")

        if not agent:
            return "No appropriate agent available for this query."

        return await agent.process_message(user_query, context)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {
            "total_agents": len(self.agents),
            "available_agents": list(self.agents.keys()),
            "openai_available": OPENAI_AVAILABLE,
            "agents_initialized": sum(
                1 for agent in self.agents.values() if agent.is_initialized
            ),
        }

        for agent_name, agent in self.agents.items():
            status[f"{agent_name}_status"] = {
                "initialized": agent.is_initialized,
                "conversation_length": len(agent.conversation_history),
                "functions_registered": len(agent.function_registry),
            }

        return status


# Convenience functions


def create_agent_orchestrator(
    optimization_focus: str = "balanced", medical_compliance: bool = False
) -> AgentOrchestrator:
    """Create agent orchestrator with default configuration."""
    config = AgentConfig(
        optimization_focus=optimization_focus, medical_compliance=medical_compliance
    )

    return AgentOrchestrator(config)


async def quick_optimization_advice(
    simulation_results: Dict[str, Any], accelerator_config: AcceleratorConfig
) -> str:
    """Get quick optimization advice from AI agent."""
    orchestrator = create_agent_orchestrator()

    if not orchestrator.agents:
        return "AI agents not available. Please check OpenAI configuration."

    query = f"""
    Based on these simulation results, what are the top 3 optimization recommendations?

    Performance Summary:
    - Throughput: {simulation_results.get('execution_summary', {}).get('throughput_ops_per_second', 0):.2f} ops/sec
    - PE Utilization: {simulation_results.get('performance_metrics', {}).get('pe_utilization', 0):.1%}
    - Energy Efficiency: {simulation_results.get('performance_metrics', {}).get('energy_efficiency_tops_per_watt', 0):.1f} TOPS/W

    Accelerator: {accelerator_config.name}

    Provide concise, actionable recommendations.
    """

    return await orchestrator.interactive_consultation(query)
