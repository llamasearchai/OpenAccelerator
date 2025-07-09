"""
Analysis and visualization tools for the Open Accelerator simulator.

This module provides performance analysis, bottleneck identification, and
visualization capabilities for accelerator simulation results.
"""

from .bottleneck_analysis import *
from .performance_analysis import *
from .reporting import *
from .visualization import *

__version__ = "0.1.0"

__all__ = [
    # Performance Analysis exports
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "LatencyAnalyzer",
    "ThroughputAnalyzer",
    "UtilizationAnalyzer",
    "EnergyAnalyzer",
    # Bottleneck Analysis exports
    "BottleneckAnalyzer",
    "CriticalPathAnalyzer",
    "ResourceContentionAnalyzer",
    "MemoryBottleneckAnalyzer",
    "ComputeBottleneckAnalyzer",
    # Visualization exports
    "plot_performance_metrics",
    "plot_utilization_heatmap",
    "plot_power_consumption",
    "plot_memory_hierarchy",
    "plot_dataflow_graph",
    "create_performance_dashboard",
    # Reporting exports
    "SimulationReport",
    "ComparisonReport",
    "OptimizationReport",
    "generate_performance_report",
    "generate_power_report",
    "generate_comparison_report",
]
