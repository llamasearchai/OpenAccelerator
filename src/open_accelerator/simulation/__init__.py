"""Simulation sub-package public interface.

Re-exports the high-level *Simulator* API and helper functions for
convenience so that user code can simply do::

    from open_accelerator.simulation import Simulator

without having to import the internal module path.
"""

from __future__ import annotations

from .simulator import (
    SimulationConfig,
    SimulationOrchestrator,
    SimulationResult,
    Simulator,
    run_comparison_study,
    run_quick_simulation,
)

__all__: list[str] = [
    "Simulator",
    "SimulationConfig",
    "SimulationResult",
    "SimulationOrchestrator",
    "run_quick_simulation",
    "run_comparison_study",
]
