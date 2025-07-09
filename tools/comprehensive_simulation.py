#!/usr/bin/env python3
"""
Comprehensive Simulation Runner for Open Accelerator

Orchestrates large-scale simulation campaigns with intelligent scheduling,
progress tracking, and result aggregation.
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psutil

# Add the open_accelerator package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_accelerator.analysis import analyze_simulation_results
from open_accelerator.simulation import Simulator
from open_accelerator.utils import AcceleratorConfig, WorkloadConfig
from open_accelerator.workloads import GEMMWorkload


@dataclass
class SimulationTask:
    """Represents a single simulation task."""

    task_id: str
    workload_spec: dict[str, Any]
    config_spec: dict[str, Any]
    priority: int = 1
    estimated_runtime_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 2
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class SimulationCampaign:
    """Represents a collection of simulation tasks."""

    campaign_id: str
    name: str
    description: str
    tasks: list[SimulationTask]
    created_time: float
    total_estimated_runtime: float
    priority: int = 1


class ResourceMonitor:
    """Monitors system resources during simulation campaigns."""

    def __init__(self):
        self.cpu_threshold = 90.0  # CPU usage threshold
        self.memory_threshold = 85.0  # Memory usage threshold
        self.monitoring = False
        self.stats_history = []

    def get_system_stats(self) -> dict[str, float]:
        """Get current system resource statistics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "load_average": os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
        }

    def should_throttle(self) -> bool:
        """Check if system resources are under stress."""
        stats = self.get_system_stats()
        return (
            stats["cpu_percent"] > self.cpu_threshold
            or stats["memory_percent"] > self.memory_threshold
        )

    def estimate_optimal_workers(self) -> int:
        """Estimate optimal number of worker processes."""
        stats = self.get_system_stats()
        cpu_count = stats["cpu_count"]

        # Conservative approach: use fewer workers if memory is limited
        if stats["available_memory_gb"] < 4:
            return max(1, cpu_count // 4)
        elif stats["available_memory_gb"] < 8:
            return max(1, cpu_count // 2)
        else:
            return max(1, cpu_count - 1)  # Leave one core for system


class ProgressTracker:
    """Tracks and displays progress of simulation campaigns."""

    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.completed_tasks = 0
        self.total_tasks = 0
        self.failed_tasks = 0
        self.current_task = ""

    def update(
        self, completed: int, total: int, failed: int = 0, current_task: str = ""
    ):
        """Update progress statistics."""
        self.completed_tasks = completed
        self.total_tasks = total
        self.failed_tasks = failed
        self.current_task = current_task
        self.last_update = time.time()

    def get_progress_report(self) -> dict[str, Any]:
        """Get current progress report."""
        elapsed = time.time() - self.start_time
        completion_rate = (
            self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0
        )

        # Estimate remaining time
        if completion_rate > 0:
            estimated_total_time = elapsed / completion_rate
            estimated_remaining = estimated_total_time - elapsed
        else:
            estimated_remaining = 0

        return {
            "completed": self.completed_tasks,
            "total": self.total_tasks,
            "failed": self.failed_tasks,
            "completion_percentage": completion_rate * 100,
            "elapsed_time_minutes": elapsed / 60,
            "estimated_remaining_minutes": estimated_remaining / 60,
            "current_task": self.current_task,
            "tasks_per_minute": self.completed_tasks / (elapsed / 60)
            if elapsed > 0
            else 0,
        }

    def print_progress(self):
        """Print current progress to console."""
        report = self.get_progress_report()

        print(
            f"\rProgress: {report['completed']}/{report['total']} "
            f"({report['completion_percentage']:.1f}%) | "
            f"Failed: {report['failed']} | "
            f"Elapsed: {report['elapsed_time_minutes']:.1f}m | "
            f"ETA: {report['estimated_remaining_minutes']:.1f}m | "
            f"Rate: {report['tasks_per_minute']:.1f} tasks/min",
            end="",
        )


def run_single_simulation(task: SimulationTask) -> SimulationTask:
    """Run a single simulation task."""
    task.status = "running"
    task.start_time = time.time()

    try:
        # Create configuration objects
        accel_config = AcceleratorConfig(
            array_rows=task.config_spec["array_rows"],
            array_cols=task.config_spec["array_cols"],
            pe_mac_latency=task.config_spec["pe_mac_latency"],
            input_buffer_size=task.config_spec["input_buffer_size"],
            weight_buffer_size=task.config_spec["weight_buffer_size"],
            output_buffer_size=task.config_spec["output_buffer_size"],
            data_type=np.int32,
        )

        workload_config = WorkloadConfig(
            gemm_M=task.workload_spec["M"],
            gemm_K=task.workload_spec["K"],
            gemm_P=task.workload_spec["P"],
        )

        # Check compatibility for Output Stationary
        if (
            task.workload_spec["M"] != task.config_spec["array_rows"]
            or task.workload_spec["P"] != task.config_spec["array_cols"]
        ):
            raise ValueError(
                f"Incompatible workload-configuration pair: "
                f"M={task.workload_spec['M']}, P={task.workload_spec['P']} "
                f"vs array {task.config_spec['array_rows']}x{task.config_spec['array_cols']}"
            )

        # Create and run simulation
        workload = GEMMWorkload(workload_config, accel_config)
        workload.generate_data(seed=42)

        simulator = Simulator(accel_config, workload)
        simulation_stats = simulator.run()

        # Analyze results
        metrics = analyze_simulation_results(simulation_stats, accel_config, workload)

        # Store results
        task.result = {
            "simulation_name": task.task_id,
            "workload_name": task.workload_spec["name"],
            "workload_category": task.workload_spec["category"],
            "workload_M": task.workload_spec["M"],
            "workload_K": task.workload_spec["K"],
            "workload_P": task.workload_spec["P"],
            "config_name": task.config_spec["name"],
            "config_category": task.config_spec["category"],
            "array_rows": task.config_spec["array_rows"],
            "array_cols": task.config_spec["array_cols"],
            "array_size": task.config_spec["array_rows"]
            * task.config_spec["array_cols"],
            "pe_mac_latency": task.config_spec["pe_mac_latency"],
            "total_cycles": metrics.total_cycles,
            "total_mac_operations": metrics.total_mac_operations,
            "macs_per_cycle": metrics.macs_per_cycle,
            "pe_utilization": metrics.average_pe_utilization,
            "roofline_utilization": metrics.roofline_utilization,
            "theoretical_peak_macs": metrics.theoretical_peak_macs,
            "actual_runtime_seconds": time.time() - task.start_time,
        }

        task.status = "completed"
        task.end_time = time.time()

    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.end_time = time.time()

        # Retry logic
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = "pending"  # Reset for retry

    return task


class ComprehensiveSimulator:
    """Main class for orchestrating comprehensive simulation campaigns."""

    def __init__(self, max_workers: Optional[int] = None):
        self.resource_monitor = ResourceMonitor()
        self.progress_tracker = ProgressTracker()
        self.campaigns: list[SimulationCampaign] = []
        self.max_workers = (
            max_workers or self.resource_monitor.estimate_optimal_workers()
        )
        self.results_cache = {}

    def load_benchmark_suite(self, benchmark_file: str) -> dict[str, Any]:
        """Load benchmark suite from JSON file."""
        try:
            with open(benchmark_file) as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading benchmark suite: {e}")

    def create_campaign_from_suite(
        self, benchmark_suite: dict[str, Any], suite_name: str
    ) -> SimulationCampaign:
        """Create a simulation campaign from a benchmark suite."""
        if suite_name not in benchmark_suite["test_suites"]:
            raise ValueError(f"Test suite '{suite_name}' not found in benchmark suite")

        suite_info = benchmark_suite["test_suites"][suite_name]

        # Get workloads and configurations
        if suite_info["workloads"] == "all":
            workloads = benchmark_suite["workloads"]
        else:
            workloads = [
                w
                for w in benchmark_suite["workloads"]
                if w["name"] in suite_info["workloads"]
            ]

        if suite_info["configurations"] == "all":
            configurations = benchmark_suite["configurations"]
        else:
            configurations = [
                c
                for c in benchmark_suite["configurations"]
                if c["name"] in suite_info["configurations"]
            ]

        # Create tasks
        tasks = []
        total_estimated_runtime = 0.0

        for workload in workloads:
            for config in configurations:
                # Check compatibility
                if (
                    workload["M"] == config["array_rows"]
                    and workload["P"] == config["array_cols"]
                ):
                    task_id = f"{workload['name']}_{config['name']}"
                    estimated_runtime = self._estimate_task_runtime(workload, config)

                    task = SimulationTask(
                        task_id=task_id,
                        workload_spec=workload,
                        config_spec=config,
                        estimated_runtime_seconds=estimated_runtime,
                    )

                    tasks.append(task)
                    total_estimated_runtime += estimated_runtime

        campaign = SimulationCampaign(
            campaign_id=f"{suite_name}_{int(time.time())}",
            name=suite_name,
            description=suite_info["description"],
            tasks=tasks,
            created_time=time.time(),
            total_estimated_runtime=total_estimated_runtime,
        )

        self.campaigns.append(campaign)
        return campaign

    def _estimate_task_runtime(
        self, workload: dict[str, Any], config: dict[str, Any]
    ) -> float:
        """Estimate runtime for a simulation task."""
        # Simple heuristic based on computational complexity
        total_ops = workload["M"] * workload["K"] * workload["P"]
        array_size = config["array_rows"] * config["array_cols"]

        # Base time + complexity factor
        base_time = 5.0  # seconds
        complexity_factor = (total_ops / 1000) * 0.1  # Scale with operations
        array_factor = max(1.0, array_size / 100)  # Larger arrays take more time

        return base_time + complexity_factor + array_factor

    def run_campaign(
        self,
        campaign: SimulationCampaign,
        save_intermediate: bool = True,
        output_dir: str = "simulation_results",
    ) -> list[dict[str, Any]]:
        """Run a complete simulation campaign."""
        print(f"Starting campaign: {campaign.name}")
        print(f"Description: {campaign.description}")
        print(f"Total tasks: {len(campaign.tasks)}")
        print(f"Estimated runtime: {campaign.total_estimated_runtime/3600:.1f} hours")
        print(f"Using {self.max_workers} worker processes")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # Initialize progress tracking
        self.progress_tracker = ProgressTracker()
        self.progress_tracker.update(0, len(campaign.tasks))

        # Filter out already completed tasks (for resume functionality)
        pending_tasks = [task for task in campaign.tasks if task.status == "pending"]

        results = []
        completed_count = len(campaign.tasks) - len(pending_tasks)
        failed_count = len([task for task in campaign.tasks if task.status == "failed"])

        # Run simulations with process pool
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_single_simulation, task): task
                for task in pending_tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    completed_task = future.result()

                    if completed_task.status == "completed":
                        results.append(completed_task.result)
                        completed_count += 1
                    elif completed_task.status == "failed":
                        failed_count += 1
                        print(
                            f"\nTask failed: {completed_task.task_id} - {completed_task.error_message}"
                        )

                    # Update progress
                    self.progress_tracker.update(
                        completed_count,
                        len(campaign.tasks),
                        failed_count,
                        completed_task.task_id,
                    )
                    self.progress_tracker.print_progress()

                    # Save intermediate results
                    if save_intermediate and len(results) % 10 == 0:
                        self._save_intermediate_results(
                            results, output_dir, campaign.campaign_id
                        )

                    # Check system resources and throttle if necessary
                    if self.resource_monitor.should_throttle():
                        print(
                            "\nSystem resources under stress. Pausing for 30 seconds..."
                        )
                        time.sleep(30)

                except Exception as e:
                    print(f"\nUnexpected error processing task {task.task_id}: {e}")
                    failed_count += 1

        print("\n\nCampaign completed!")
        print(f"Successful simulations: {completed_count}")
        print(f"Failed simulations: {failed_count}")
        print(
            f"Total runtime: {(time.time() - self.progress_tracker.start_time)/3600:.2f} hours"
        )

        return results

    def _save_intermediate_results(
        self, results: list[dict[str, Any]], output_dir: str, campaign_id: str
    ):
        """Save intermediate results to avoid data loss."""
        intermediate_file = os.path.join(output_dir, f"{campaign_id}_intermediate.csv")
        try:
            df = pd.DataFrame(results)
            df.to_csv(intermediate_file, index=False)
        except Exception as e:
            print(f"\nWarning: Could not save intermediate results: {e}")

    def save_campaign_results(
        self,
        results: list[dict[str, Any]],
        campaign: SimulationCampaign,
        output_dir: str = "simulation_results",
    ) -> str:
        """Save campaign results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results as CSV
        results_file = os.path.join(output_dir, f"{campaign.campaign_id}_results.csv")
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)

        # Save campaign summary
        summary = {
            "campaign_id": campaign.campaign_id,
            "name": campaign.name,
            "description": campaign.description,
            "total_tasks": len(campaign.tasks),
            "successful_tasks": len(results),
            "failed_tasks": len(campaign.tasks) - len(results),
            "total_runtime_hours": (time.time() - campaign.created_time) / 3600,
            "created_time": datetime.fromtimestamp(campaign.created_time).isoformat(),
            "completed_time": datetime.now().isoformat(),
            "results_file": results_file,
        }

        summary_file = os.path.join(output_dir, f"{campaign.campaign_id}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")

        return results_file

    def resume_campaign(self, campaign_file: str) -> SimulationCampaign:
        """Resume a previously interrupted campaign."""
        try:
            with open(campaign_file, "rb") as f:
                campaign = pickle.load(f)

            print(f"Resuming campaign: {campaign.name}")
            pending_tasks = len(
                [task for task in campaign.tasks if task.status == "pending"]
            )
            print(f"Pending tasks: {pending_tasks}")

            return campaign
        except Exception as e:
            raise ValueError(f"Error resuming campaign: {e}")

    def save_campaign_state(
        self, campaign: SimulationCampaign, output_dir: str = "simulation_results"
    ):
        """Save campaign state for resumption."""
        state_file = os.path.join(output_dir, f"{campaign.campaign_id}_state.pkl")
        try:
            with open(state_file, "wb") as f:
                pickle.dump(campaign, f)
            print(f"Campaign state saved to: {state_file}")
        except Exception as e:
            print(f"Warning: Could not save campaign state: {e}")

    def generate_campaign_report(
        self, results: list[dict[str, Any]], output_dir: str = "simulation_results"
    ) -> str:
        """Generate a comprehensive campaign report."""
        if not results:
            print("No results to generate report from.")
            return ""

        df = pd.DataFrame(results)
        report_lines = [
            "# Comprehensive Simulation Campaign Report",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total simulations: {len(results)}",
            "",
            "## Summary Statistics",
            "",
        ]

        # Overall statistics
        total_cycles = df["total_cycles"].sum()
        total_mac_ops = df["total_mac_operations"].sum()
        avg_pe_utilization = df["pe_utilization"].mean()
        avg_roofline_utilization = df["roofline_utilization"].mean()

        report_lines.extend(
            [
                f"- Total computation cycles: {total_cycles:,}",
                f"- Total MAC operations: {total_mac_ops:,}",
                f"- Average PE utilization: {avg_pe_utilization:.1%}",
                f"- Average roofline utilization: {avg_roofline_utilization:.1%}",
                f"- Average throughput: {df['macs_per_cycle'].mean():.2f} MACs/cycle",
                "",
            ]
        )

        # Performance by category
        if "workload_category" in df.columns:
            report_lines.extend(["## Performance by Workload Category", ""])

            category_stats = (
                df.groupby("workload_category")
                .agg(
                    {
                        "macs_per_cycle": ["mean", "std"],
                        "pe_utilization": ["mean", "std"],
                        "total_cycles": "mean",
                    }
                )
                .round(3)
            )

            for category in category_stats.index:
                report_lines.extend(
                    [
                        f"### {category.title()}",
                        f"- Average throughput: {category_stats.loc[category, ('macs_per_cycle', 'mean')]:.2f} ± {category_stats.loc[category, ('macs_per_cycle', 'std')]:.2f} MACs/cycle",
                        f"- Average PE utilization: {category_stats.loc[category, ('pe_utilization', 'mean')]:.1%} ± {category_stats.loc[category, ('pe_utilization', 'std')]:.1%}",
                        f"- Average cycles: {category_stats.loc[category, ('total_cycles', 'mean')]:.0f}",
                        "",
                    ]
                )

        # Performance by configuration
        if "config_category" in df.columns:
            report_lines.extend(["## Performance by Configuration Category", ""])

            config_stats = (
                df.groupby("config_category")
                .agg(
                    {
                        "macs_per_cycle": ["mean", "std"],
                        "pe_utilization": ["mean", "std"],
                        "roofline_utilization": ["mean", "std"],
                    }
                )
                .round(3)
            )

            for config in config_stats.index:
                report_lines.extend(
                    [
                        f"### {config.title()}",
                        f"- Average throughput: {config_stats.loc[config, ('macs_per_cycle', 'mean')]:.2f} ± {config_stats.loc[config, ('macs_per_cycle', 'std')]:.2f} MACs/cycle",
                        f"- Average PE utilization: {config_stats.loc[config, ('pe_utilization', 'mean')]:.1%} ± {config_stats.loc[config, ('pe_utilization', 'std')]:.1%}",
                        f"- Average roofline utilization: {config_stats.loc[config, ('roofline_utilization', 'mean')]:.1%} ± {config_stats.loc[config, ('roofline_utilization', 'std')]:.1%}",
                        "",
                    ]
                )

        # Top performers
        report_lines.extend(
            ["## Top Performing Configurations", "", "### Highest Throughput", ""]
        )

        top_throughput = df.nlargest(5, "macs_per_cycle")[
            ["simulation_name", "macs_per_cycle", "pe_utilization"]
        ]
        for _, row in top_throughput.iterrows():
            report_lines.append(
                f"- {row['simulation_name']}: {row['macs_per_cycle']:.2f} MACs/cycle ({row['pe_utilization']:.1%} PE util)"
            )

        report_lines.extend(["", "### Highest PE Utilization", ""])

        top_utilization = df.nlargest(5, "pe_utilization")[
            ["simulation_name", "pe_utilization", "macs_per_cycle"]
        ]
        for _, row in top_utilization.iterrows():
            report_lines.append(
                f"- {row['simulation_name']}: {row['pe_utilization']:.1%} PE util ({row['macs_per_cycle']:.2f} MACs/cycle)"
            )

        # Scaling analysis
        if "array_size" in df.columns:
            report_lines.extend(["", "## Scaling Analysis", ""])

            scaling_stats = (
                df.groupby("array_size")
                .agg(
                    {
                        "macs_per_cycle": "mean",
                        "pe_utilization": "mean",
                        "roofline_utilization": "mean",
                    }
                )
                .round(3)
            )

            for array_size in sorted(scaling_stats.index):
                stats = scaling_stats.loc[array_size]
                report_lines.append(
                    f"- {array_size} PEs: {stats['macs_per_cycle']:.2f} MACs/cycle, {stats['pe_utilization']:.1%} PE util, {stats['roofline_utilization']:.1%} roofline util"
                )

        # Save report
        report_content = "\n".join(report_lines)
        report_file = os.path.join(output_dir, "campaign_report.md")

        try:
            with open(report_file, "w") as f:
                f.write(report_content)
            print(f"Campaign report saved to: {report_file}")
            return report_file
        except Exception as e:
            print(f"Error saving campaign report: {e}")
            return ""


def main():
    """Main function for comprehensive simulation runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Simulation Runner for Open Accelerator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick validation suite
  python comprehensive_simulation.py benchmarks/benchmark_suite.json --suite quick_validation

  # Run performance benchmarking with 8 workers
  python comprehensive_simulation.py benchmarks/benchmark_suite.json --suite performance_benchmarking --workers 8

  # Run comprehensive suite with intermediate saves
  python comprehensive_simulation.py benchmarks/benchmark_suite.json --suite comprehensive --save-intermediate

  # Resume interrupted campaign
  python comprehensive_simulation.py --resume simulation_results/campaign_12345_state.pkl
        """,
    )

    parser.add_argument(
        "benchmark_file", nargs="?", help="Path to benchmark suite JSON file"
    )

    parser.add_argument("--suite", required=False, help="Name of test suite to run")

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of worker processes (default: auto-detect)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="simulation_results",
        help="Output directory for results (default: simulation_results)",
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results every 10 completed simulations",
    )

    parser.add_argument("--resume", help="Resume campaign from state file")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print campaign info without running simulations",
    )

    parser.add_argument(
        "--list-suites", action="store_true", help="List available test suites and exit"
    )

    parser.add_argument(
        "--system-info", action="store_true", help="Display system information and exit"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # System info
    if args.system_info:
        monitor = ResourceMonitor()
        stats = monitor.get_system_stats()
        optimal_workers = monitor.estimate_optimal_workers()

        print("System Information:")
        print(f"  CPU cores: {stats['cpu_count']}")
        print(f"  CPU usage: {stats['cpu_percent']:.1f}%")
        print(f"  Memory usage: {stats['memory_percent']:.1f}%")
        print(f"  Available memory: {stats['available_memory_gb']:.1f} GB")
        print(f"  Load average: {stats['load_average']:.2f}")
        print(f"  Recommended workers: {optimal_workers}")
        return 0

    try:
        simulator = ComprehensiveSimulator(max_workers=args.workers)

        # Resume campaign
        if args.resume:
            print(f"Resuming campaign from: {args.resume}")
            campaign = simulator.resume_campaign(args.resume)
            results = simulator.run_campaign(
                campaign, args.save_intermediate, args.output_dir
            )

            # Save results and generate report
            simulator.save_campaign_results(results, campaign, args.output_dir)
            simulator.generate_campaign_report(results, args.output_dir)
            return 0

        # Load benchmark suite
        if not args.benchmark_file:
            print("Error: benchmark file required (or use --resume)")
            parser.print_help()
            return 1

        if not os.path.exists(args.benchmark_file):
            print(f"Error: benchmark file not found: {args.benchmark_file}")
            return 1

        benchmark_suite = simulator.load_benchmark_suite(args.benchmark_file)

        # List available suites
        if args.list_suites:
            print("Available test suites:")
            for suite_name, suite_info in benchmark_suite["test_suites"].items():
                print(f"  {suite_name}: {suite_info['description']}")
                print(
                    f"    Estimated runtime: {suite_info['estimated_runtime_minutes']} minutes"
                )
            return 0

        if not args.suite:
            print(
                "Error: --suite required (or use --list-suites to see available options)"
            )
            return 1

        # Create and run campaign
        print(f"Creating campaign for suite: {args.suite}")
        campaign = simulator.create_campaign_from_suite(benchmark_suite, args.suite)

        if args.dry_run:
            print("\nDry run - campaign details:")
            print(f"  Campaign ID: {campaign.campaign_id}")
            print(f"  Name: {campaign.name}")
            print(f"  Description: {campaign.description}")
            print(f"  Total tasks: {len(campaign.tasks)}")
            print(
                f"  Estimated runtime: {campaign.total_estimated_runtime/3600:.2f} hours"
            )
            print(f"  Max workers: {simulator.max_workers}")

            # Show task breakdown
            workload_counts = {}
            config_counts = {}
            for task in campaign.tasks:
                wl_name = task.workload_spec["name"]
                cfg_name = task.config_spec["name"]
                workload_counts[wl_name] = workload_counts.get(wl_name, 0) + 1
                config_counts[cfg_name] = config_counts.get(cfg_name, 0) + 1

            print(f"\n  Workloads ({len(workload_counts)}):")
            for wl, count in sorted(workload_counts.items()):
                print(f"    {wl}: {count} tasks")

            print(f"\n  Configurations ({len(config_counts)}):")
            for cfg, count in sorted(config_counts.items()):
                print(f"    {cfg}: {count} tasks")

            return 0

        # Save campaign state before starting
        simulator.save_campaign_state(campaign, args.output_dir)

        # Run campaign
        results = simulator.run_campaign(
            campaign,
            save_intermediate=args.save_intermediate,
            output_dir=args.output_dir,
        )

        # Save final results and generate report
        results_file = simulator.save_campaign_results(
            results, campaign, args.output_dir
        )
        simulator.generate_campaign_report(results, args.output_dir)

        print(f"\nCampaign complete! Results available in: {args.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\nSimulation campaign interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during simulation campaign: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
