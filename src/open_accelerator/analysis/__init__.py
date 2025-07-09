"""OpenAccelerator analysis sub-package.

Only the performance-analysis utilities are currently implemented.
Additional advanced analysis modules (bottleneck, power, visualisation, etc.)
will be added incrementally. For now we expose a stable public surface that
does not import missing modules so that importing :pymod:`open_accelerator.analysis`
never fails.
"""

from __future__ import annotations

# Keep version in sync with package root
__version__: str = "0.1.0"

from .performance_analysis import PerformanceAnalyzer, PerformanceMetrics  # noqa: F401

# Public export list – will be extended later by the compatibility layer
__all__: list[str] = [
    "PerformanceAnalyzer",
    "PerformanceMetrics",
]

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Dict, TYPE_CHECKING

import numpy as np

# ---------------------------------------------------------------------------
# Legacy compatibility layer
# ---------------------------------------------------------------------------

@dataclass
class LegacyPerformanceMetrics:  # noqa: D401 – simple container
    """Legacy metrics container expected by early examples."""

    total_cycles: int
    total_mac_operations: int
    average_pe_utilization: float
    macs_per_cycle: float
    theoretical_peak_macs: int
    roofline_utilization: float
    pe_activity_map: Optional[np.ndarray] = None


def analyze_simulation_results(
    simulation_stats: Dict[str, Any],
    accel_config: Any | None = None,  # *Any* to avoid cyclical imports
    workload: Any | None = None,
) -> LegacyPerformanceMetrics:  # noqa: D401 – keeping signature stable
    """Compute high-level metrics from raw *simulation_stats*.

    The implementation purposefully stays **very lightweight** so that it can be
    called from any environment without additional dependencies.  It mirrors the
    behaviour of the original helper shipped with the first OSS release so that
    existing tutorials such as *examples/comprehensive_simulation.py* continue
    to run unchanged.
    """

    total_cycles: int = int(simulation_stats.get("total_cycles") or simulation_stats.get("results", {}).get("total_cycles", 0))
    total_mac_operations: int = int(simulation_stats.get("total_mac_operations") or simulation_stats.get("results", {}).get("total_mac_operations", 0))

    # Utilisation
    pe_activity_map = simulation_stats.get("pe_activity_map_over_time")
    if isinstance(pe_activity_map, np.ndarray) and pe_activity_map.size and total_cycles > 0:
        average_pe_utilization: float = float(np.mean(pe_activity_map))
    else:
        average_pe_utilization = 0.0

    macs_per_cycle: float = 0.0 if total_cycles == 0 else total_mac_operations / total_cycles

    # Theoretical peak MACs per cycle
    theoretical_peak_macs: int = 0
    if accel_config is not None:
        # Try the property helpers first (for modern config classes)
        theoretical_peak_macs = getattr(accel_config, "array_rows", 0) * getattr(accel_config, "array_cols", 0)
        if theoretical_peak_macs == 0 and hasattr(accel_config, "array"):
            theoretical_peak_macs = accel_config.array.rows * accel_config.array.cols
    if theoretical_peak_macs == 0 and isinstance(pe_activity_map, np.ndarray) and pe_activity_map.size:
        theoretical_peak_macs = pe_activity_map.shape[0] * pe_activity_map.shape[1]

    roofline_utilization: float = 0.0 if theoretical_peak_macs == 0 else macs_per_cycle / theoretical_peak_macs

    return LegacyPerformanceMetrics(
        total_cycles=total_cycles,
        total_mac_operations=total_mac_operations,
        average_pe_utilization=average_pe_utilization,
        macs_per_cycle=macs_per_cycle,
        theoretical_peak_macs=theoretical_peak_macs,
        roofline_utilization=roofline_utilization,
        pe_activity_map=pe_activity_map if isinstance(pe_activity_map, np.ndarray) else None,
    )


# Expose in public namespace for import *
__all__.extend([
    "LegacyPerformanceMetrics",
    "analyze_simulation_results",
])
