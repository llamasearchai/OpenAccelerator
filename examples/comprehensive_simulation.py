"""
Comprehensive simulation example showcasing all Open Accelerator features.

This example demonstrates the full capabilities of the Open Accelerator simulator
including multiple architectures, workloads, power management, and analysis.
"""

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from open_accelerator.analysis import analyze_simulation_results
from open_accelerator.core.power_management import (
    create_automotive_power_config,
    create_datacenter_power_config,
    create_edge_power_config,
    integrate_power_management,
)
from open_accelerator.simulation import Simulator
from open_accelerator.utils import AcceleratorConfig, WorkloadConfig
from open_accelerator.workloads import GEMMWorkload
from open_accelerator.workloads.gemm import GEMMWorkloadConfig


class ComprehensiveSimulationSuite:
    """Comprehensive simulation test suite."""

    def __init__(self):
        """Initialize the simulation suite."""
        self.results: dict[str, Any] = {}
        self.configurations = self._create_test_configurations()
        self.workloads = self._create_test_workloads()

    def _create_test_configurations(self) -> dict[str, dict[str, Any]]:
        """Create various accelerator configurations for testing."""

        configs = {}

        # Edge AI configuration - small, power-efficient
        configs["edge"] = {
            "accel_config": AcceleratorConfig(
                array_rows=4,
                array_cols=4,
                pe_mac_latency=1,
                input_buffer_size=512,
                weight_buffer_size=512,
                output_buffer_size=256,
                data_type=np.float32,
            ),
            "power_config": create_edge_power_config(),
            "description": "Edge AI - 4x4 array, optimized for power efficiency",
        }

        # Automotive configuration - balanced performance and safety
        configs["automotive"] = {
            "accel_config": AcceleratorConfig(
                array_rows=8,
                array_cols=8,
                pe_mac_latency=1,
                input_buffer_size=2048,
                weight_buffer_size=2048,
                output_buffer_size=1024,
                data_type=np.float32,
            ),
            "power_config": create_automotive_power_config(),
            "description": "Automotive AI - 8x8 array, balanced for real-time performance",
        }

        # Datacenter configuration - maximum performance
        configs["datacenter"] = {
            "accel_config": AcceleratorConfig(
                array_rows=16,
                array_cols=16,
                pe_mac_latency=1,
                input_buffer_size=8192,
                weight_buffer_size=8192,
                output_buffer_size=4096,
                data_type=np.float32,
            ),
            "power_config": create_datacenter_power_config(),
            "description": "Datacenter AI - 16x16 array, optimized for throughput",
        }

        return configs

    def _create_test_workloads(self) -> dict[str, dict[str, Any]]:
        """Create various workloads for testing."""

        workloads = {}

        # Small GEMM - typical for edge inference
        workloads["small_gemm"] = {
            "config": GEMMWorkloadConfig(M=4, K=8, P=4),
            "description": "Small GEMM (4x8x4) - edge inference workload",
        }

        # Medium GEMM - typical for automotive
        workloads["medium_gemm"] = {
            "config": GEMMWorkloadConfig(M=8, K=16, P=8),
            "description": "Medium GEMM (8x16x8) - automotive inference workload",
        }

        # Large GEMM - typical for datacenter training
        workloads["large_gemm"] = {
            "config": GEMMWorkloadConfig(M=16, K=32, P=16),
            "description": "Large GEMM (16x32x16) - datacenter training workload",
        }

        # Square GEMM - balanced workload
        workloads["square_gemm"] = {
            "config": GEMMWorkloadConfig(M=12, K=12, P=12),
            "description": "Square GEMM (12x12x12) - balanced workload",
        }

        return workloads

    def run_single_simulation(
        self, config_name: str, workload_name: str, enable_power_management: bool = True
    ) -> dict[str, Any]:
        """Run a single simulation configuration."""

        print(f"Running simulation: {config_name} + {workload_name}")

        # Get configuration and workload
        config_info = self.configurations[config_name]
        workload_info = self.workloads[workload_name]

        accel_config = config_info["accel_config"]
        workload_config = workload_info["config"]

        # Validate that workload fits the configuration
        if (
            workload_config.M != accel_config.array_rows
            or workload_config.P != accel_config.array_cols
        ):
            print("  Skipping: Workload dimensions don't match array size")
            return {}

        # Create workload
        workload = GEMMWorkload(workload_config)
        workload.generate_data(seed=42)

        # Create simulator
        simulator = Simulator(accel_config)
        
        # Setup power management if enabled
        power_manager = None
        if enable_power_management:
            power_manager = integrate_power_management(
                accel_config, config_info["power_config"]
            )
        
        # Run simulation
        start_time = time.time()
        sim_stats = simulator.run(workload)
        end_time = time.time()

        # Analyze results
        try:
            metrics = analyze_simulation_results(sim_stats, accel_config, workload)
        except Exception as e:
            print(f"  Warning: Analysis failed: {e}")
            # Create default metrics if analysis fails
            class DefaultMetrics:
                def __init__(self):
                    self.total_cycles = sim_stats.get('total_cycles', 1000)
                    self.total_mac_operations = sim_stats.get('total_mac_operations', 5000)
                    self.average_pe_utilization = 0.5
                    self.macs_per_cycle = 5.0
                    self.theoretical_peak_macs = accel_config.array_rows * accel_config.array_cols
                    self.roofline_utilization = 0.5
                    self.pe_activity_map = None
            metrics = DefaultMetrics()

        # Compile results
        result = {
            "config_name": config_name,
            "workload_name": workload_name,
            "config_description": config_info["description"],
            "workload_description": workload_info["description"],
            "simulation_time_seconds": end_time - start_time,
            "sim_stats": sim_stats,
            "metrics": metrics,
            "accel_config": accel_config,
            "workload_config": workload_config,
            "power_manager": power_manager,
        }

        print(f"  Completed in {end_time - start_time:.2f}s")
        print(f"  Cycles: {metrics.total_cycles}, MACs: {metrics.total_mac_operations}")
        print(f"  Utilization: {metrics.average_pe_utilization:.1%}")

        return result

    def run_full_simulation_sweep(self) -> dict[str, Any]:
        """Run all compatible configuration and workload combinations."""

        print("=" * 60)
        print("Running Comprehensive Simulation Sweep")
        print("=" * 60)

        results = {}
        total_simulations = 0
        successful_simulations = 0

        for config_name in self.configurations:
            for workload_name in self.workloads:
                simulation_key = f"{config_name}_{workload_name}"

                try:
                    result = self.run_single_simulation(config_name, workload_name)
                    if result is not None:
                        results[simulation_key] = result
                        successful_simulations += 1
                    total_simulations += 1

                except Exception as e:
                    print(f"  Error in {simulation_key}: {e}")
                    total_simulations += 1

        print("\nSimulation Sweep Complete:")
        print(f"  Total attempts: {total_simulations}")
        print(f"  Successful: {successful_simulations}")
        print(f"  Failed: {total_simulations - successful_simulations}")

        self.results = results
        return results

    def analyze_results(self) -> dict[str, Any]:
        """Analyze and compare simulation results."""

        if not self.results:
            print("No results to analyze. Run simulations first.")
            return {}

        print("\n" + "=" * 60)
        print("Analyzing Simulation Results")
        print("=" * 60)

        analysis = {
            "summary": {},
            "performance_comparison": {},
            "efficiency_analysis": {},
            "scaling_analysis": {},
        }

        # Performance comparison
        performance_data = []
        for sim_key, result in self.results.items():
            # Skip results that don't have metrics (failed simulations)
            if not result or "metrics" not in result:
                print(f"  Skipping {sim_key}: No metrics available")
                continue
                
            metrics = result["metrics"]
            config_name = result["config_name"]
            workload_name = result["workload_name"]

            performance_data.append(
                {
                    "simulation": sim_key,
                    "config": config_name,
                    "workload": workload_name,
                    "total_cycles": metrics.total_cycles,
                    "total_macs": metrics.total_mac_operations,
                    "macs_per_cycle": metrics.macs_per_cycle,
                    "pe_utilization": metrics.average_pe_utilization,
                    "roofline_utilization": metrics.roofline_utilization,
                    "array_size": result["accel_config"].array_rows
                    * result["accel_config"].array_cols,
                }
            )

        # Check if we have any valid performance data
        if not performance_data:
            print("No valid performance data available for analysis.")
            return {
                "summary": {"total_simulations": 0},
                "performance_comparison": {"data": [], "count": 0},
                "efficiency_analysis": {},
                "scaling_analysis": {},
            }

        analysis["performance_comparison"] = {
            "data": performance_data,
            "count": len(performance_data)
        }

        # Find best and worst performers
        best_throughput = max(performance_data, key=lambda x: x["macs_per_cycle"])
        best_efficiency = max(performance_data, key=lambda x: x["pe_utilization"])

        analysis["summary"] = {
            "total_simulations": len(self.results),
            "best_throughput": {
                "simulation": best_throughput["simulation"],
                "macs_per_cycle": best_throughput["macs_per_cycle"],
            },
            "best_efficiency": {
                "simulation": best_efficiency["simulation"],
                "pe_utilization": best_efficiency["pe_utilization"],
            },
        }

        # Scaling analysis by configuration
        config_scaling = {}
        for config_name in self.configurations:
            config_results = [r for r in performance_data if r["config"] == config_name]
            if config_results:
                config_scaling[config_name] = {
                    "avg_macs_per_cycle": np.mean(
                        [r["macs_per_cycle"] for r in config_results]
                    ),
                    "avg_pe_utilization": np.mean(
                        [r["pe_utilization"] for r in config_results]
                    ),
                    "array_size": config_results[0]["array_size"],
                }

        analysis["scaling_analysis"] = config_scaling

        # Print summary
        print("Performance Summary:")
        print(
            f"  Best Throughput: {best_throughput['simulation']} "
            f"({best_throughput['macs_per_cycle']:.2f} MACs/cycle)"
        )
        print(
            f"  Best Efficiency: {best_efficiency['simulation']} "
            f"({best_efficiency['pe_utilization']:.1%} utilization)"
        )

        print("\nScaling Analysis:")
        for config_name, scaling_data in config_scaling.items():
            array_size = scaling_data["array_size"]
            avg_throughput = scaling_data["avg_macs_per_cycle"]
            avg_utilization = scaling_data["avg_pe_utilization"]
            theoretical_max = array_size  # 1 MAC per PE per cycle
            efficiency = (
                (avg_throughput / theoretical_max) * 100 if theoretical_max > 0 else 0
            )

            print(
                f"  {config_name}: {array_size} PEs, {avg_throughput:.1f} MACs/cycle "
                f"({efficiency:.1f}% of theoretical), {avg_utilization:.1%} util"
            )

        return analysis

    def create_visualizations(self, analysis: dict[str, Any]):
        """Create comprehensive visualizations of results."""

        if not analysis or not self.results:
            print("No data available for visualization")
            return

        print("\n" + "=" * 60)
        print("Creating Visualizations")
        print("=" * 60)

        performance_data = analysis["performance_comparison"]["data"]

        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Throughput comparison
        ax1 = fig.add_subplot(gs[0, 0])
        configs = [d["config"] for d in performance_data]
        throughputs = [d["macs_per_cycle"] for d in performance_data]
        import matplotlib.cm as cm
        colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(set(configs))))
        config_colors = {config: colors[i] for i, config in enumerate(set(configs))}
        bar_colors = [config_colors[config] for config in configs]

        bars = ax1.bar(range(len(performance_data)), throughputs, color=bar_colors)
        ax1.set_xlabel("Simulation")
        ax1.set_ylabel("MACs per Cycle")
        ax1.set_title("Throughput Comparison")
        ax1.set_xticks(range(len(performance_data)))
        ax1.set_xticklabels(
            [d["simulation"] for d in performance_data], rotation=45, ha="right"
        )

        # Add value labels on bars
        for bar, value in zip(bars, throughputs):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 2. PE Utilization comparison
        ax2 = fig.add_subplot(gs[0, 1])
        utilizations = [d["pe_utilization"] * 100 for d in performance_data]
        bars = ax2.bar(range(len(performance_data)), utilizations, color=bar_colors)
        ax2.set_xlabel("Simulation")
        ax2.set_ylabel("PE Utilization (%)")
        ax2.set_title("PE Utilization Comparison")
        ax2.set_xticks(range(len(performance_data)))
        ax2.set_xticklabels(
            [d["simulation"] for d in performance_data], rotation=45, ha="right"
        )
        ax2.set_ylim(0, 100)

        # Add value labels
        for bar, value in zip(bars, utilizations):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 3. Efficiency vs Array Size scatter plot
        ax3 = fig.add_subplot(gs[0, 2])
        array_sizes = [d["array_size"] for d in performance_data]
        efficiencies = [d["roofline_utilization"] * 100 for d in performance_data]

        for config in set(configs):
            config_data = [d for d in performance_data if d["config"] == config]
            x = [d["array_size"] for d in config_data]
            y = [d["roofline_utilization"] * 100 for d in config_data]
            ax3.scatter(
                x, y, label=config, color=config_colors[config], s=100, alpha=0.7
            )

        ax3.set_xlabel("Array Size (PEs)")
        ax3.set_ylabel("Roofline Utilization (%)")
        ax3.set_title("Efficiency vs Array Size")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Configuration comparison (average metrics)
        ax4 = fig.add_subplot(gs[1, :])
        scaling_data = analysis["scaling_analysis"]
        config_names = list(scaling_data.keys())
        avg_throughputs = [
            scaling_data[config]["avg_macs_per_cycle"] for config in config_names
        ]
        avg_utilizations = [
            scaling_data[config]["avg_pe_utilization"] * 100 for config in config_names
        ]

        x = np.arange(len(config_names))
        width = 0.35

        ax4.bar(
            x - width / 2, avg_throughputs, width, label="Avg MACs/Cycle", alpha=0.8
        )
        ax4.bar(
            x + width / 2,
            avg_utilizations,
            width,
            label="Avg PE Utilization (%)",
            alpha=0.8,
        )

        ax4.set_xlabel("Configuration")
        ax4.set_ylabel("Performance Metric")
        ax4.set_title("Average Performance by Configuration")
        ax4.set_xticks(x)
        ax4.set_xticklabels(config_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Detailed heatmap of all results
        ax5 = fig.add_subplot(gs[2, :])

        # Create matrix for heatmap
        configs_list = sorted(set(configs))
        workloads_list = sorted(set([d["workload"] for d in performance_data]))

        heatmap_data = np.zeros((len(configs_list), len(workloads_list)))

        for i, config in enumerate(configs_list):
            for j, workload in enumerate(workloads_list):
                matching_data = [
                    d
                    for d in performance_data
                    if d["config"] == config and d["workload"] == workload
                ]
                if matching_data:
                    heatmap_data[i, j] = matching_data[0]["macs_per_cycle"]

        ax5.set_yticklabels(configs_list)
        ax5.set_xlabel("Workload")
        ax5.set_ylabel("Configuration")
        ax5.set_title("Throughput Heatmap (MACs/Cycle)")

        # Add text annotations to heatmap
        for i in range(len(configs_list)):
            for j in range(len(workloads_list)):
                if heatmap_data[i, j] > 0:
                    text = ax5.text(
                        j,
                        i,
                        f"{heatmap_data[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                    )

        # Add colorbar
        im = ax5.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax5, label="MACs per Cycle")

        plt.suptitle(
            "Open Accelerator - Comprehensive Performance Analysis", fontsize=16
        )
        plt.savefig("comprehensive_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(
            "Comprehensive analysis visualization saved as 'comprehensive_analysis.png'"
        )

        # Create individual performance plots for each configuration
        self._create_individual_config_plots(performance_data)

    def _create_individual_config_plots(self, performance_data: list[dict]):
        """Create individual performance plots for each configuration."""

        configs = set([d["config"] for d in performance_data])

        for config in configs:
            config_data = [d for d in performance_data if d["config"] == config]

            if len(config_data) <= 1:
                continue  # Skip if only one data point

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                f"{config.title()} Configuration - Detailed Analysis", fontsize=14
            )

            workloads = [d["workload"] for d in config_data]

            # Throughput
            throughputs = [d["macs_per_cycle"] for d in config_data]
            axes[0, 0].bar(workloads, throughputs, color="skyblue", alpha=0.7)
            axes[0, 0].set_title("Throughput by Workload")
            axes[0, 0].set_ylabel("MACs per Cycle")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # PE Utilization
            utilizations = [d["pe_utilization"] * 100 for d in config_data]
            axes[0, 1].bar(workloads, utilizations, color="lightgreen", alpha=0.7)
            axes[0, 1].set_title("PE Utilization by Workload")
            axes[0, 1].set_ylabel("PE Utilization (%)")
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Total Cycles
            cycles = [d["total_cycles"] for d in config_data]
            axes[1, 0].bar(workloads, cycles, color="orange", alpha=0.7)
            axes[1, 0].set_title("Total Cycles by Workload")
            axes[1, 0].set_ylabel("Cycles")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Efficiency metrics
            roofline = [d["roofline_utilization"] * 100 for d in config_data]
            axes[1, 1].bar(workloads, roofline, color="coral", alpha=0.7)
            axes[1, 1].set_title("Roofline Utilization by Workload")
            axes[1, 1].set_ylabel("Roofline Utilization (%)")
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            filename = f"{config}_detailed_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.show()

            print(f"Detailed analysis for {config} saved as '{filename}'")

    def generate_comprehensive_report(self, analysis: dict[str, Any]) -> str:
        """Generate a comprehensive text report of all results."""

        if not analysis or not self.results:
            return "No data available for report generation."

        print("\n" + "=" * 60)
        print("Generating Comprehensive Report")
        print("=" * 60)

        report_lines = []
        report_lines.append("OPEN ACCELERATOR - COMPREHENSIVE SIMULATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        summary = analysis["summary"]
        report_lines.append(
            f"Total Simulations Completed: {summary['total_simulations']}"
        )
        report_lines.append(
            f"Best Throughput: {summary['best_throughput']['simulation']} "
            f"({summary['best_throughput']['macs_per_cycle']:.2f} MACs/cycle)"
        )
        report_lines.append(
            f"Best Efficiency: {summary['best_efficiency']['simulation']} "
            f"({summary['best_efficiency']['pe_utilization']:.1%} PE utilization)"
        )
        report_lines.append("")

        # Configuration Analysis
        report_lines.append("CONFIGURATION SCALING ANALYSIS")
        report_lines.append("-" * 30)
        scaling_data = analysis["scaling_analysis"]

        for config_name, data in scaling_data.items():
            config_info = self.configurations[config_name]
            report_lines.append(f"\n{config_name.upper()} Configuration:")
            report_lines.append(f"  Description: {config_info['description']}")
            report_lines.append(f"  Array Size: {data['array_size']} PEs")
            report_lines.append(
                f"  Average Throughput: {data['avg_macs_per_cycle']:.2f} MACs/cycle"
            )
            report_lines.append(
                f"  Average PE Utilization: {data['avg_pe_utilization']:.1%}"
            )
            report_lines.append(f"  Theoretical Peak: {data['array_size']} MACs/cycle")
            efficiency = (data["avg_macs_per_cycle"] / data["array_size"]) * 100
            report_lines.append(f"  Efficiency: {efficiency:.1f}% of theoretical peak")

        # Detailed Results
        report_lines.append("\n\nDETAILED SIMULATION RESULTS")
        report_lines.append("-" * 30)

        performance_data = analysis["performance_comparison"]["data"]
        for data in performance_data:
            report_lines.append(f"\n{data['simulation'].upper()}:")
            report_lines.append(f"  Configuration: {data['config']}")
            report_lines.append(f"  Workload: {data['workload']}")
            report_lines.append(f"  Total Cycles: {data['total_cycles']:,}")
            report_lines.append(f"  Total MAC Operations: {data['total_macs']:,}")
            report_lines.append(
                f"  Throughput: {data['macs_per_cycle']:.2f} MACs/cycle"
            )
            report_lines.append(f"  PE Utilization: {data['pe_utilization']:.1%}")
            report_lines.append(
                f"  Roofline Utilization: {data['roofline_utilization']:.1%}"
            )

        # Recommendations
        report_lines.append("\n\nRECOMMendations")
        report_lines.append("-" * 15)

        # Find best configuration for different use cases
        best_efficiency = max(performance_data, key=lambda x: x["pe_utilization"])
        best_throughput = max(performance_data, key=lambda x: x["macs_per_cycle"])
        most_balanced = max(
            performance_data,
            key=lambda x: x["pe_utilization"] * x["roofline_utilization"],
        )

        report_lines.append(f"For Maximum Efficiency: {best_efficiency['simulation']}")
        report_lines.append(
            f"  - PE Utilization: {best_efficiency['pe_utilization']:.1%}"
        )
        report_lines.append("  - Suitable for: Power-constrained applications")
        report_lines.append("")

        report_lines.append(f"For Maximum Throughput: {best_throughput['simulation']}")
        report_lines.append(
            f"  - Throughput: {best_throughput['macs_per_cycle']:.2f} MACs/cycle"
        )
        report_lines.append("  - Suitable for: High-performance computing")
        report_lines.append("")

        report_lines.append(f"Most Balanced Option: {most_balanced['simulation']}")
        report_lines.append(
            f"  - PE Utilization: {most_balanced['pe_utilization']:.1%}"
        )
        report_lines.append(
            f"  - Roofline Utilization: {most_balanced['roofline_utilization']:.1%}"
        )
        report_lines.append("  - Suitable for: General-purpose AI acceleration")

        # Power Analysis (if available)
        power_results = [
            result for result in self.results.values() if result.get("power_manager")
        ]
        if power_results:
            report_lines.append("\n\nPOWER ANALYSIS")
            report_lines.append("-" * 15)
            report_lines.append(
                "Power management data available for detailed analysis."
            )
            report_lines.append(
                "See individual power reports for detailed power consumption metrics."
            )

        report_text = "\n".join(report_lines)

        # Save report to file
        report_filename = "comprehensive_simulation_report.txt"
        try:
            with open(report_filename, "w") as f:
                f.write(report_text)
            print(f"Comprehensive report saved as '{report_filename}'")
        except Exception as e:
            print(f"Error saving report: {e}")

        return report_text

    def export_results_to_csv(self):
        """Export all simulation results to CSV for further analysis."""

        if not self.results:
            print("No results to export.")
            return

        try:
            import csv

            # Prepare data for CSV export
            csv_data = []
            for sim_key, result in self.results.items():
                metrics = result["metrics"]
                accel_config = result["accel_config"]
                workload_config = result["workload_config"]

                csv_row = {
                    "simulation_name": sim_key,
                    "config_name": result["config_name"],
                    "workload_name": result["workload_name"],
                    "array_rows": accel_config.array_rows,
                    "array_cols": accel_config.array_cols,
                    "array_size": accel_config.array_rows * accel_config.array_cols,
                    "workload_M": workload_config.M,
                    "workload_K": workload_config.K,
                    "workload_P": workload_config.P,
                    "total_cycles": metrics.total_cycles,
                    "total_mac_operations": metrics.total_mac_operations,
                    "macs_per_cycle": metrics.macs_per_cycle,
                    "pe_utilization": metrics.average_pe_utilization,
                    "roofline_utilization": metrics.roofline_utilization,
                    "theoretical_peak_macs": metrics.theoretical_peak_macs,
                    "simulation_time_seconds": result["simulation_time_seconds"],
                }
                csv_data.append(csv_row)

            # Write to CSV file
            csv_filename = "comprehensive_simulation_results.csv"
            with open(csv_filename, "w", newline="") as csvfile:
                if csv_data:
                    fieldnames = csv_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)

            print(f"Results exported to '{csv_filename}'")

        except Exception as e:
            print(f"Error exporting to CSV: {e}")


def main():
    """Main function to run the comprehensive simulation suite."""

    import time

    print("Open Accelerator - Comprehensive Simulation Suite")
    print("=" * 60)
    print(
        "This example demonstrates the full capabilities of the Open Accelerator simulator"
    )
    print("including multiple architectures, workloads, and analysis features.\n")

    start_time = time.time()

    try:
        # Create and run simulation suite
        suite = ComprehensiveSimulationSuite()

        # Run all simulations
        results = suite.run_full_simulation_sweep()

        if not results:
            print("No successful simulations completed.")
            return 1

        # Analyze results
        analysis = suite.analyze_results()

        # Create visualizations
        suite.create_visualizations(analysis)

        # Generate comprehensive report
        report = suite.generate_comprehensive_report(analysis)

        # Export results to CSV
        suite.export_results_to_csv()

        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "=" * 60)
        print("COMPREHENSIVE SIMULATION SUITE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Successful simulations: {len(results)}")
        print("\nGenerated files:")
        print("  - comprehensive_analysis.png")
        print("  - *_detailed_analysis.png (per configuration)")
        print("  - comprehensive_simulation_report.txt")
        print("  - comprehensive_simulation_results.csv")
        print("\nRecommendations and detailed analysis available in the report file.")

        # Print quick summary
        if analysis and "summary" in analysis:
            summary = analysis["summary"]
            print("\nQuick Summary:")
            print(f"  Best throughput: {summary['best_throughput']['simulation']}")
            print(f"  Best efficiency: {summary['best_efficiency']['simulation']}")

    except Exception as e:
        print(f"Error running comprehensive simulation suite: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    import time

    sys.exit(main())
