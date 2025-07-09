from __future__ import annotations

"""Performance analysis utilities for OpenAccelerator.

This module provides a simple, **self-contained** implementation that converts
raw simulation statistics (as returned by :py:meth:`open_accelerator.simulation.simulator.Simulator.run`)
into a convenient summary dictionary as well as a dataclass for programmatic
consumption by other subsystems such as the CLI or dashboards.

The implementation purposefully stays lightweight so that it does **not**
introduce new heavy runtime dependencies. It is therefore safe to import from
any environment—including headless or minimal Docker images.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(slots=True)
class PerformanceMetrics:
    """Container for common accelerator performance figures."""

    total_cycles: int
    total_macs: int
    macs_per_cycle: float
    efficiency: float  # Achieved / peak MACs-per-cycle
    pe_utilization: float  # Average utilisation across all PEs (0-1)

    # Optional extended metrics
    throughput: float | None = None  # TOPS (trillion operations per second)
    power: float | None = None  # Watts
    energy: float | None = None  # Joules


class PerformanceAnalyzer:
    """Compute high-level metrics from raw simulation statistics."""

    _TERA: float = 1e12

    def __init__(self, sim_stats: Dict[str, Any]):
        self._stats = sim_stats

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def compute_metrics(self) -> Dict[str, Any]:
        """Return a *dict* compatible with the expectations of the CLI."""
        metrics = self._compute()
        # Convert the dataclass to a plain dict so that it can be serialised
        # easily by the CLI / FastAPI.
        return {
            "total_cycles": metrics.total_cycles,
            "total_macs": metrics.total_macs,
            "macs_per_cycle": metrics.macs_per_cycle,
            "efficiency": metrics.efficiency,
            "pe_utilization": metrics.pe_utilization,
            "throughput": metrics.throughput,
            "power": metrics.power,
            "energy": metrics.energy,
        }

    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance from simulation results (test compatibility method)."""
        # Update internal stats with new results
        if hasattr(self._stats, 'update'):
            self._stats.update(results)
        else:
            # Handle case where _stats might not be a dict
            self._stats = results
        return self.compute_metrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute(self) -> PerformanceMetrics:
        cycles: int = int(self._stats.get("total_cycles", 0))
        total_macs: int = int(self._stats.get("total_mac_operations", 0))

        if cycles == 0:
            macs_per_cycle = 0.0
        else:
            macs_per_cycle = total_macs / cycles

        # Theoretical peak = array_rows × array_cols
        activity_map = self._stats.get("pe_activity_map_over_time")
        if isinstance(activity_map, np.ndarray) and activity_map.size:
            rows, cols = activity_map.shape
            peak_macs_per_cycle = rows * cols
            pe_util = float(np.mean(activity_map))
        else:
            # Fallback: derive from output matrix shape
            output_matrix = self._stats.get("output_matrix")
            if isinstance(output_matrix, np.ndarray):
                rows, cols = output_matrix.shape
                peak_macs_per_cycle = rows * cols
            else:
                peak_macs_per_cycle = 1  # Prevent div-by-zero
            pe_util = 0.0

        efficiency = 0.0 if peak_macs_per_cycle == 0 else macs_per_cycle / peak_macs_per_cycle

        # Optional throughput calculation (assume 1 GHz clock for rough TOPS)
        throughput_tops = (macs_per_cycle * 1e9) / self._TERA  # cycles/s → TOPS

        return PerformanceMetrics(
            total_cycles=cycles,
            total_macs=total_macs,
            macs_per_cycle=macs_per_cycle,
            efficiency=efficiency,
            pe_utilization=pe_util,
            throughput=throughput_tops,
        ) 