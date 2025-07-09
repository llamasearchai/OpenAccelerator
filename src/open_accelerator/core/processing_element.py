"""
Processing element module that provides compatibility layer for core imports.

This module exports all processing element functionality from the main pe module.
"""

from .pe import (
    ProcessingElement,
    PEState,
    PEMetrics,
)

# Create aliases for compatibility
from ..utils.config import PEConfig

# Factory functions for processing element creation
def create_simple_pe():
    """Create a simple processing element configuration."""
    config = PEConfig()
    return ProcessingElement((0, 0), config)

def create_advanced_pe():
    """Create an advanced processing element configuration."""
    config = PEConfig(enable_sparsity=True, power_gating=True)
    return ProcessingElement((0, 0), config)

def create_vector_pe():
    """Create a vector processing element configuration."""
    config = PEConfig(enable_sparsity=True)
    return ProcessingElement((0, 0), config)

# Create classes for compatibility
class MACUnit:
    """Multiply-accumulate unit for processing elements."""
    
    def __init__(self, precision: str = "float32"):
        """Initialize MAC unit."""
        self.precision = precision
        self.operations_count = 0
        self.energy_consumed = 0.0
        
    def multiply_accumulate(self, a: float, b: float, c: float) -> float:
        """Perform multiply-accumulate operation: a * b + c."""
        self.operations_count += 1
        self.energy_consumed += 0.1  # pJ per operation
        return a * b + c
    
    def reset(self):
        """Reset MAC unit state."""
        self.operations_count = 0
        self.energy_consumed = 0.0

class RegisterFile:
    """Register file for processing elements."""
    
    def __init__(self, size: int = 16, width: int = 32):
        """Initialize register file."""
        self.size = size
        self.width = width
        self.registers = [0] * size
        self.read_count = 0
        self.write_count = 0
        
    def read(self, address: int) -> int:
        """Read from register file."""
        if 0 <= address < self.size:
            self.read_count += 1
            return self.registers[address]
        else:
            raise IndexError(f"Register address {address} out of range")
    
    def write(self, address: int, value: int):
        """Write to register file."""
        if 0 <= address < self.size:
            self.write_count += 1
            self.registers[address] = value
        else:
            raise IndexError(f"Register address {address} out of range")
    
    def reset(self):
        """Reset register file."""
        self.registers = [0] * self.size
        self.read_count = 0
        self.write_count = 0

# Re-export everything from pe module
__all__ = [
    "ProcessingElement",
    "PEConfig", 
    "PEState",
    "PEMetrics",
    "MACUnit",
    "RegisterFile",
    "create_simple_pe",
    "create_advanced_pe",
    "create_vector_pe",
] 