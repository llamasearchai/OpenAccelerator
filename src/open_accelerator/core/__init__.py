"""
Core functionality for the Open Accelerator simulator.

This module provides the fundamental building blocks for accelerator simulation
including processing elements, memory systems, interconnects, and power management.
"""

from .accelerator import AcceleratorController, AcceleratorMetrics, create_medical_accelerator, create_edge_accelerator, create_datacenter_accelerator
from .interconnect import *
from .memory_system import *
from .power_management import *
from .processing_element import *

# Version information
__version__ = "0.1.0"
__author__ = "Open Accelerator Team"

# Core module exports
__all__ = [
    # Accelerator Controller exports
    "AcceleratorController",
    "AcceleratorMetrics", 
    "create_medical_accelerator",
    "create_edge_accelerator",
    "create_datacenter_accelerator",
    # Processing Element exports
    "ProcessingElement",
    "PEConfig",
    "PEState",
    "MACUnit",
    "RegisterFile",
    "create_simple_pe",
    "create_advanced_pe",
    "create_vector_pe",
    # Memory System exports
    "Buffer",
    "MemoryHierarchy",
    "CacheLevel",
    "MemoryController",
    "MemoryConfig",
    "MemoryRequest",
    "MemoryResponse",
    "CachePolicy",
    "ReplacementPolicy",
    "create_simple_memory_hierarchy",
    "create_automotive_memory_hierarchy",
    "create_edge_memory_hierarchy",
    "create_datacenter_memory_hierarchy",
    # Interconnect exports
    "InterconnectConfig",
    "NetworkTopology",
    "RoutingAlgorithm",
    "FlowControl",
    "Router",
    "NetworkInterface",
    "Link",
    "Packet",
    "NetworkMessage",
    "NoC",
    "CrossbarSwitch",
    "create_mesh_noc",
    "create_torus_noc",
    "create_tree_noc",
    "create_crossbar_interconnect",
    # Power Management exports
    "PowerState",
    "VoltageLevel",
    "FrequencyLevel",
    "ThermalState",
    "PowerConfig",
    "ComponentPowerProfile",
    "PowerMetrics",
    "PowerModel",
    "SimplePowerModel",
    "DetailedPowerModel",
    "DVFSController",
    "ThermalManager",
    "PowerGatingController",
    "ClockGatingController",
    "PowerManager",
    "PowerOptimizer",
    "PowerAnalyzer",
    "PowerBudgetManager",
    "PowerLogger",
    "create_medical_power_config",
    "create_automotive_power_config",
    "create_edge_power_config",
    "create_datacenter_power_config",
    "integrate_power_management",
    "create_power_report",
]
