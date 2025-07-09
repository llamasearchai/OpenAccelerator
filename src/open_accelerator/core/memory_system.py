"""
Memory system module that provides compatibility layer for core imports.

This module exports all memory-related functionality from the main memory module.
"""

from .memory import (
    MemoryBuffer,
    CacheBuffer,
    MemoryHierarchy,
    MemoryState,
    MemoryMetrics,
)

# Create aliases for compatibility
Buffer = MemoryBuffer
CacheLevel = CacheBuffer

# Factory functions for memory system creation
def create_simple_memory_hierarchy():
    """Create a simple memory hierarchy configuration."""
    from ..utils.config import AcceleratorConfig
    return MemoryHierarchy(AcceleratorConfig())

def create_automotive_memory_hierarchy():
    """Create automotive memory hierarchy configuration."""
    from ..utils.config import AcceleratorConfig
    return MemoryHierarchy(AcceleratorConfig())

def create_edge_memory_hierarchy():
    """Create edge memory hierarchy configuration."""
    from ..utils.config import AcceleratorConfig
    return MemoryHierarchy(AcceleratorConfig())

def create_datacenter_memory_hierarchy():
    """Create datacenter memory hierarchy configuration."""
    from ..utils.config import AcceleratorConfig
    return MemoryHierarchy(AcceleratorConfig())

# Re-export everything from memory module
__all__ = [
    "Buffer",
    "MemoryBuffer", 
    "CacheBuffer",
    "MemoryHierarchy", 
    "CacheLevel",
    "MemoryState",
    "MemoryMetrics",
    "create_simple_memory_hierarchy",
    "create_automotive_memory_hierarchy",
    "create_edge_memory_hierarchy",
    "create_datacenter_memory_hierarchy",
] 