#!/usr/bin/env python3
"""
Performance Analyzer for Open Accelerator Simulation Results

Analyzes simulation results from benchmark campaigns, generates insights,
visualizations, and performance reports.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")


class PerformanceAnalyzer:
    """Analyzes simulation results and generates performance insights."""

    def __init__(self, results_file: str):
        """Initialize analyzer with results file."""
        self.results_file = results_file
        self.df = self._load_results()
        self.output_dir = "analysis_results"

    def _load_results(self) -> pd.DataFrame:
        """Load results from CSV file."""
        try:
            df = pd.read_csv(self.results_file)
            print(f"Loaded {len(df)} simulation results from {self.results_file}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading results file: {e}")

    def set_output_dir(self, output_dir: str):
        """Set output directory for analysis results."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def basic_statistics(self) -> dict[str, Any]:
        """Generate basic statistics about the simulation results."""
        stats = {
            "total_simulations": len(self.df),
            "unique_workloads": self.df["workload_name"].nunique()
            if "workload_name" in self.df.columns
            else 0,
            "unique_configurations": self.df["config_name"].nunique()
            if "config_name" in self.df.columns
            else 0,
            "total_cycles": self.df["total_cycles"].sum(),
            "total_mac_operations": self.df["total_mac_operations"].sum(),
            "throughput_stats": {
                "mean_macs_per_cycle": self.df["macs_per_cycle"].mean(),
                "median_macs_per_cycle": self.df["macs_per_cycle"].median(),
                "max_macs_per_cycle": self.df["macs_per_cycle"].max(),
                "min_macs_per_cycle": self.df["macs_per_cycle"].min(),
                "std_macs_per_cycle": self.df["macs_per_cycle"].std(),
            },
            "utilization_stats": {
                "mean_pe_utilization": self.df["pe_utilization"].mean(),
                "median_pe_utilization": self.df["pe_utilization"].median(),
                "max_pe_utilization": self.df["pe_utilization"].max(),
                "min_pe_utilization": self.df["pe_utilization"].min(),
                "std_pe_utilization": self.df["pe_utilization"].std(),
            },
            "efficiency_stats": {
                "mean_roofline_utilization": self.df["roofline_utilization"].mean(),
                "median_roofline_utilization": self.df["roofline_utilization"].median(),
                "max_roofline_utilization": self.df["roofline_utilization"].max(),
                "min_roofline_utilization": self.df["roofline_utilization"].min(),
                "std_roofline_utilization": self.df["roofline_utilization"].std(),
            },
        }
        return stats

    def analyze_scaling_behavior(self) -> dict[str, Any]:
        """Analyze how performance scales with array size."""
        if "array_size" not in self.df.columns:
            return {"error": "Array size information not available"}

        scaling_analysis = {}

        # Group by array size
        array_groups = self.df.groupby("array_size")

        scaling_stats = array_groups.agg(
            {
                "macs_per_cycle": ["count", "mean", "std", "min", "max"],
                "pe_utilization": ["mean", "std"],
                "roofline_utilization": ["mean", "std"],
                "total_cycles": "mean",
            }
        ).round(3)

        scaling_analysis["by_array_size"] = scaling_stats.to_dict()

        # Calculate scaling efficiency
        array_sizes = sorted(self.df["array_size"].unique())
        if len(array_sizes) > 1:
            baseline_size = array_sizes[0]
            baseline_throughput = array_groups.get_group(baseline_size)[
                "macs_per_cycle"
            ].mean()

            scaling_efficiency = {}
            for size in array_sizes:
                group_throughput = array_groups.get_group(size)["macs_per_cycle"].mean()
                theoretical_speedup = size / baseline_size
                actual_speedup = group_throughput / baseline_throughput
                efficiency = (
                    actual_speedup / theoretical_speedup
                    if theoretical_speedup > 0
                    else 0
                )

                scaling_efficiency[size] = {
                    "theoretical_speedup": theoretical_speedup,
                    "actual_speedup": actual_speedup,
                    "scaling_efficiency": efficiency,
                }

            scaling_analysis["scaling_efficiency"] = scaling_efficiency

        return scaling_analysis

    def analyze_workload_characteristics(self) -> dict[str, Any]:
        """Analyze performance characteristics by workload type."""
        workload_analysis = {}

        if "workload_category" in self.df.columns:
            category_stats = (
                self.df.groupby("workload_category")
                .agg(
                    {
                        "macs_per_cycle": ["count", "mean", "std", "min", "max"],
                        "pe_utilization": ["mean", "std"],
                        "roofline_utilization": ["mean", "std"],
                        "total_cycles": ["mean", "std"],
                        "total_mac_operations": "mean",
                    }
                )
                .round(3)
            )

            workload_analysis["by_category"] = category_stats.to_dict()

        if "workload_name" in self.df.columns:
            workload_stats = (
                self.df.groupby("workload_name")
                .agg(
                    {
                        "macs_per_cycle": ["mean", "std"],
                        "pe_utilization": ["mean", "std"],
                        "roofline_utilization": ["mean"],
                    }
                )
                .round(3)
            )

            # Find best and worst performing workloads
            best_throughput = workload_stats["macs_per_cycle"]["mean"].idxmax()
            worst_throughput = workload_stats["macs_per_cycle"]["mean"].idxmin()
            best_utilization = workload_stats["pe_utilization"]["mean"].idxmax()
            worst_utilization = workload_stats["pe_utilization"]["mean"].idxmin()

            workload_analysis["performance_ranking"] = {
                "best_throughput": best_throughput,
                "worst_throughput": worst_throughput,
                "best_utilization": best_utilization,
                "worst_utilization": worst_utilization,
            }

            workload_analysis["by_workload"] = workload_stats.to_dict()

        return workload_analysis

    def analyze_configuration_impact(self) -> dict[str, Any]:
        """Analyze impact of different configuration parameters."""
        config_analysis = {}

        if "config_category" in self.df.columns:
            config_stats = (
                self.df.groupby("config_category")
                .agg(
                    {
                        "macs_per_cycle": ["count", "mean", "std"],
                        "pe_utilization": ["mean", "std"],
                        "roofline_utilization": ["mean", "std"],
                    }
                )
                .round(3)
            )

            config_analysis["by_category"] = config_stats.to_dict()

        # Analyze parameter correlations
        numeric_cols = [
            "array_rows",
            "array_cols",
            "array_size",
            "pe_mac_latency",
            "macs_per_cycle",
            "pe_utilization",
            "roofline_utilization",
        ]
        available_cols = [col for col in numeric_cols if col in self.df.columns]

        if len(available_cols) > 2:
            correlation_matrix = self.df[available_cols].corr()
            config_analysis["parameter_correlations"] = correlation_matrix.to_dict()

        return config_analysis

    def identify_performance_bottlenecks(self) -> dict[str, Any]:
        """Identify potential performance bottlenecks."""
        bottlenecks = {}

        # Low utilization cases
        low_pe_threshold = 0.5  # 50% PE utilization
        low_roofline_threshold = 0.3  # 30% roofline utilization

        low_pe_util = self.df[self.df["pe_utilization"] < low_pe_threshold]
        low_roofline_util = self.df[
            self.df["roofline_utilization"] < low_roofline_threshold
        ]

        bottlenecks["low_pe_utilization"] = {
            "count": len(low_pe_util),
            "percentage": len(low_pe_util) / len(self.df) * 100,
            "examples": low_pe_util.nsmallest(5, "pe_utilization")[
                ["simulation_name", "pe_utilization"]
            ].to_dict("records"),
        }

        bottlenecks["low_roofline_utilization"] = {
            "count": len(low_roofline_util),
            "percentage": len(low_roofline_util) / len(self.df) * 100,
            "examples": low_roofline_util.nsmallest(5, "roofline_utilization")[
                ["simulation_name", "roofline_utilization"]
            ].to_dict("records"),
        }

        # Identify configurations with consistently poor performance
        if "config_name" in self.df.columns:
            config_performance = self.df.groupby("config_name")[
                "roofline_utilization"
            ].mean()
            poor_configs = config_performance[
                config_performance < low_roofline_threshold
            ]

            bottlenecks["poor_performing_configs"] = {
                "configs": poor_configs.to_dict(),
                "count": len(poor_configs),
            }

        return bottlenecks

    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up figure parameters
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

        # 1. Throughput distribution
        self._plot_throughput_distribution()

        # 2. Utilization analysis
        self._plot_utilization_analysis()

        # 3. Scaling analysis
        if "array_size" in self.df.columns:
            self._plot_scaling_analysis()

        # 4. Workload comparison
        if "workload_category" in self.df.columns or "workload_name" in self.df.columns:
            self._plot_workload_comparison()

        # 5. Configuration comparison
        if "config_category" in self.df.columns:
            self._plot_configuration_comparison()

        # 6. Performance correlation heatmap
        self._plot_correlation_heatmap()

        # 7. Performance vs Array Size scatter plot
        if "array_size" in self.df.columns:
            self._plot_performance_vs_array_size()

        # 8. Efficiency analysis
        self._plot_efficiency_analysis()

        print(f"Visualizations saved to: {self.output_dir}")

    def _plot_throughput_distribution(self):
        """Plot throughput distribution histogram."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(self.df["macs_per_cycle"], bins=30, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("MACs per Cycle")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Throughput (MACs/Cycle)")
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(self.df["macs_per_cycle"])
        ax2.set_ylabel("MACs per Cycle")
        ax2.set_title("Throughput Distribution (Box Plot)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "throughput_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_utilization_analysis(self):
        """Plot PE and roofline utilization analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # PE Utilization histogram
        axes[0, 0].hist(
            self.df["pe_utilization"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 0].set_xlabel("PE Utilization")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("PE Utilization Distribution")
        axes[0, 0].grid(True, alpha=0.3)

        # Roofline utilization histogram
        axes[0, 1].hist(
            self.df["roofline_utilization"],
            bins=30,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 1].set_xlabel("Roofline Utilization")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Roofline Utilization Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # PE vs Roofline utilization scatter
        axes[1, 0].scatter(
            self.df["pe_utilization"], self.df["roofline_utilization"], alpha=0.6
        )
        axes[1, 0].set_xlabel("PE Utilization")
        axes[1, 0].set_ylabel("Roofline Utilization")
        axes[1, 0].set_title("PE vs Roofline Utilization")
        axes[1, 0].grid(True, alpha=0.3)

        # Utilization vs Throughput
        axes[1, 1].scatter(
            self.df["pe_utilization"],
            self.df["macs_per_cycle"],
            alpha=0.6,
            color="green",
        )
        axes[1, 1].set_xlabel("PE Utilization")
        axes[1, 1].set_ylabel("MACs per Cycle")
        axes[1, 1].set_title("PE Utilization vs Throughput")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "utilization_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_scaling_analysis(self):
        """Plot scaling behavior analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Group by array size
        array_groups = self.df.groupby("array_size")
        array_sizes = sorted(self.df["array_size"].unique())

        # Throughput vs Array Size
        throughput_means = [
            array_groups.get_group(size)["macs_per_cycle"].mean()
            for size in array_sizes
        ]
        throughput_stds = [
            array_groups.get_group(size)["macs_per_cycle"].std() for size in array_sizes
        ]

        axes[0, 0].errorbar(
            array_sizes, throughput_means, yerr=throughput_stds, marker="o", capsize=5
        )
        axes[0, 0].set_xlabel("Array Size (PEs)")
        axes[0, 0].set_ylabel("Average MACs per Cycle")
        axes[0, 0].set_title("Throughput vs Array Size")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale("log")

        # PE Utilization vs Array Size
        pe_util_means = [
            array_groups.get_group(size)["pe_utilization"].mean()
            for size in array_sizes
        ]
        pe_util_stds = [
            array_groups.get_group(size)["pe_utilization"].std() for size in array_sizes
        ]

        axes[0, 1].errorbar(
            array_sizes,
            pe_util_means,
            yerr=pe_util_stds,
            marker="s",
            capsize=5,
            color="orange",
        )
        axes[0, 1].set_xlabel("Array Size (PEs)")
        axes[0, 1].set_ylabel("Average PE Utilization")
        axes[0, 1].set_title("PE Utilization vs Array Size")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale("log")

        # Scaling efficiency
        if len(array_sizes) > 1:
            baseline_throughput = throughput_means[0]
            baseline_size = array_sizes[0]

            scaling_efficiencies = []
            for i, size in enumerate(array_sizes):
                theoretical_speedup = size / baseline_size
                actual_speedup = throughput_means[i] / baseline_throughput
                efficiency = (
                    actual_speedup / theoretical_speedup
                    if theoretical_speedup > 0
                    else 0
                )
                scaling_efficiencies.append(efficiency)

            axes[1, 0].plot(
                array_sizes, scaling_efficiencies, marker="^", linewidth=2, markersize=8
            )
            axes[1, 0].axhline(
                y=1.0, color="red", linestyle="--", alpha=0.7, label="Perfect Scaling"
            )
            axes[1, 0].set_xlabel("Array Size (PEs)")
            axes[1, 0].set_ylabel("Scaling Efficiency")
            axes[1, 0].set_title("Scaling Efficiency vs Array Size")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xscale("log")
            axes[1, 0].legend()

        # Throughput per PE
        throughput_per_pe = [
            throughput_means[i] / array_sizes[i] for i in range(len(array_sizes))
        ]
        axes[1, 1].plot(
            array_sizes,
            throughput_per_pe,
            marker="d",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        axes[1, 1].set_xlabel("Array Size (PEs)")
        axes[1, 1].set_ylabel("MACs per Cycle per PE")
        axes[1, 1].set_title("Per-PE Throughput vs Array Size")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale("log")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "scaling_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_workload_comparison(self):
        """Plot workload performance comparison."""
        if "workload_category" in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Throughput by category
            category_throughput = self.df.groupby("workload_category")[
                "macs_per_cycle"
            ].agg(["mean", "std"])
            axes[0, 0].bar(
                category_throughput.index,
                category_throughput["mean"],
                yerr=category_throughput["std"],
                capsize=5,
                alpha=0.7,
            )
            axes[0, 0].set_xlabel("Workload Category")
            axes[0, 0].set_ylabel("Average MACs per Cycle")
            axes[0, 0].set_title("Throughput by Workload Category")
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # PE Utilization by category
            category_pe_util = self.df.groupby("workload_category")[
                "pe_utilization"
            ].agg(["mean", "std"])
            axes[0, 1].bar(
                category_pe_util.index,
                category_pe_util["mean"],
                yerr=category_pe_util["std"],
                capsize=5,
                alpha=0.7,
                color="orange",
            )
            axes[0, 1].set_xlabel("Workload Category")
            axes[0, 1].set_ylabel("Average PE Utilization")
            axes[0, 1].set_title("PE Utilization by Workload Category")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Box plot of throughput by category
            self.df.boxplot(
                column="macs_per_cycle", by="workload_category", ax=axes[1, 0]
            )
            axes[1, 0].set_xlabel("Workload Category")
            axes[1, 0].set_ylabel("MACs per Cycle")
            axes[1, 0].set_title("Throughput Distribution by Category")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Box plot of PE utilization by category
            self.df.boxplot(
                column="pe_utilization", by="workload_category", ax=axes[1, 1]
            )
            axes[1, 1].set_xlabel("Workload Category")
            axes[1, 1].set_ylabel("PE Utilization")
            axes[1, 1].set_title("PE Utilization Distribution by Category")
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "workload_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _plot_configuration_comparison(self):
        """Plot configuration performance comparison."""
        if "config_category" in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Throughput by config category
            config_throughput = self.df.groupby("config_category")[
                "macs_per_cycle"
            ].agg(["mean", "std"])
            axes[0, 0].bar(
                config_throughput.index,
                config_throughput["mean"],
                yerr=config_throughput["std"],
                capsize=5,
                alpha=0.7,
                color="lightblue",
            )
            axes[0, 0].set_xlabel("Configuration Category")
            axes[0, 0].set_ylabel("Average MACs per Cycle")
            axes[0, 0].set_title("Throughput by Configuration Category")
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Roofline utilization by config category
            config_roofline = self.df.groupby("config_category")[
                "roofline_utilization"
            ].agg(["mean", "std"])
            axes[0, 1].bar(
                config_roofline.index,
                config_roofline["mean"],
                yerr=config_roofline["std"],
                capsize=5,
                alpha=0.7,
                color="lightcoral",
            )
            axes[0, 1].set_xlabel("Configuration Category")
            axes[0, 1].set_ylabel("Average Roofline Utilization")
            axes[0, 1].set_title("Roofline Utilization by Configuration Category")
            axes[0, 1].tick_params(axis="x", rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Configuration efficiency comparison
            self.df.boxplot(
                column="roofline_utilization", by="config_category", ax=axes[1, 0]
            )
            axes[1, 0].set_xlabel("Configuration Category")
            axes[1, 0].set_ylabel("Roofline Utilization")
            axes[1, 0].set_title("Efficiency Distribution by Configuration")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Throughput variance by configuration
            config_variance = self.df.groupby("config_category")["macs_per_cycle"].var()
            axes[1, 1].bar(
                config_variance.index,
                config_variance.values,
                alpha=0.7,
                color="lightgreen",
            )
            axes[1, 1].set_xlabel("Configuration Category")
            axes[1, 1].set_ylabel("Throughput Variance")
            axes[1, 1].set_title("Throughput Consistency by Configuration")
            axes[1, 1].tick_params(axis="x", rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "configuration_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap of performance metrics."""
        numeric_cols = [
            "array_rows",
            "array_cols",
            "array_size",
            "pe_mac_latency",
            "total_cycles",
            "total_mac_operations",
            "macs_per_cycle",
            "pe_utilization",
            "roofline_utilization",
        ]
        available_cols = [col for col in numeric_cols if col in self.df.columns]

        if len(available_cols) > 2:
            correlation_matrix = self.df[available_cols].corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            plt.title("Performance Metrics Correlation Heatmap", fontsize=16)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "correlation_heatmap.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _plot_performance_vs_array_size(self):
        """Plot performance metrics vs array size scatter plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Throughput vs Array Size
        axes[0, 0].scatter(
            self.df["array_size"], self.df["macs_per_cycle"], alpha=0.6, s=50
        )
        axes[0, 0].set_xlabel("Array Size (PEs)")
        axes[0, 0].set_ylabel("MACs per Cycle")
        axes[0, 0].set_title("Throughput vs Array Size")
        axes[0, 0].set_xscale("log")
        axes[0, 0].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(np.log(self.df["array_size"]), self.df["macs_per_cycle"], 1)
        p = np.poly1d(z)
        x_trend = np.logspace(
            np.log10(self.df["array_size"].min()),
            np.log10(self.df["array_size"].max()),
            100,
        )
        axes[0, 0].plot(x_trend, p(np.log(x_trend)), "r--", alpha=0.8, linewidth=2)

        # PE Utilization vs Array Size
        axes[0, 1].scatter(
            self.df["array_size"],
            self.df["pe_utilization"],
            alpha=0.6,
            s=50,
            color="orange",
        )
        axes[0, 1].set_xlabel("Array Size (PEs)")
        axes[0, 1].set_ylabel("PE Utilization")
        axes[0, 1].set_title("PE Utilization vs Array Size")
        axes[0, 1].set_xscale("log")
        axes[0, 1].grid(True, alpha=0.3)

        # Roofline Utilization vs Array Size
        axes[1, 0].scatter(
            self.df["array_size"],
            self.df["roofline_utilization"],
            alpha=0.6,
            s=50,
            color="red",
        )
        axes[1, 0].set_xlabel("Array Size (PEs)")
        axes[1, 0].set_ylabel("Roofline Utilization")
        axes[1, 0].set_title("Roofline Utilization vs Array Size")
        axes[1, 0].set_xscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Cycles vs Array Size
        axes[1, 1].scatter(
            self.df["array_size"],
            self.df["total_cycles"],
            alpha=0.6,
            s=50,
            color="green",
        )
        axes[1, 1].set_xlabel("Array Size (PEs)")
        axes[1, 1].set_ylabel("Total Cycles")
        axes[1, 1].set_title("Computation Cycles vs Array Size")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "performance_vs_array_size.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_efficiency_analysis(self):
        """Plot efficiency analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Efficiency vs Throughput
        axes[0, 0].scatter(
            self.df["macs_per_cycle"], self.df["roofline_utilization"], alpha=0.6, s=50
        )
        axes[0, 0].set_xlabel("MACs per Cycle")
        axes[0, 0].set_ylabel("Roofline Utilization")
        axes[0, 0].set_title("Efficiency vs Throughput")
        axes[0, 0].grid(True, alpha=0.3)

        # Efficiency distribution
        axes[0, 1].hist(
            self.df["roofline_utilization"],
            bins=20,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 1].axvline(
            self.df["roofline_utilization"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {self.df["roofline_utilization"].mean():.3f}',
        )
        axes[0, 1].axvline(
            self.df["roofline_utilization"].median(),
            color="blue",
            linestyle="--",
            label=f'Median: {self.df["roofline_utilization"].median():.3f}',
        )
        axes[0, 1].set_xlabel("Roofline Utilization")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Efficiency Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Performance efficiency quadrants
        median_throughput = self.df["macs_per_cycle"].median()
        median_efficiency = self.df["roofline_utilization"].median()

        colors = []
        for _, row in self.df.iterrows():
            if (
                row["macs_per_cycle"] >= median_throughput
                and row["roofline_utilization"] >= median_efficiency
            ):
                colors.append("green")  # High throughput, high efficiency
            elif (
                row["macs_per_cycle"] >= median_throughput
                and row["roofline_utilization"] < median_efficiency
            ):
                colors.append("orange")  # High throughput, low efficiency
            elif (
                row["macs_per_cycle"] < median_throughput
                and row["roofline_utilization"] >= median_efficiency
            ):
                colors.append("blue")  # Low throughput, high efficiency
            else:
                colors.append("red")  # Low throughput, low efficiency

        axes[1, 0].scatter(
            self.df["macs_per_cycle"],
            self.df["roofline_utilization"],
            c=colors,
            alpha=0.6,
            s=50,
        )
        axes[1, 0].axvline(median_throughput, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].axhline(median_efficiency, color="black", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("MACs per Cycle")
        axes[1, 0].set_ylabel("Roofline Utilization")
        axes[1, 0].set_title("Performance-Efficiency Quadrants")
        axes[1, 0].grid(True, alpha=0.3)

        # Add quadrant labels
        axes[1, 0].text(
            0.75 * axes[1, 0].get_xlim()[1],
            0.75 * axes[1, 0].get_ylim()[1],
            "High Perf\nHigh Eff",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

        # Runtime vs Performance
        if "actual_runtime_seconds" in self.df.columns:
            axes[1, 1].scatter(
                self.df["actual_runtime_seconds"],
                self.df["macs_per_cycle"],
                alpha=0.6,
                s=50,
                color="purple",
            )
            axes[1, 1].set_xlabel("Simulation Runtime (seconds)")
            axes[1, 1].set_ylabel("MACs per Cycle")
            axes[1, 1].set_title("Simulation Runtime vs Performance")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Runtime data\nnot available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
            )
            axes[1, 1].set_title("Runtime Analysis (No Data)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "efficiency_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report_lines = [
            "# Open Accelerator Performance Analysis Report",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Results file: {self.results_file}",
            "",
            "## Executive Summary",
            "",
        ]

        # Basic statistics
        stats = self.basic_statistics()

        report_lines.extend(
            [
                f"- **Total Simulations**: {stats['total_simulations']:,}",
                f"- **Unique Workloads**: {stats['unique_workloads']}",
                f"- **Unique Configurations**: {stats['unique_configurations']}",
                f"- **Total Computation Cycles**: {stats['total_cycles']:,}",
                f"- **Total MAC Operations**: {stats['total_mac_operations']:,}",
                "",
                "### Performance Overview",
                "",
                f"- **Average Throughput**: {stats['throughput_stats']['mean_macs_per_cycle']:.2f} MACs/cycle",
                f"- **Peak Throughput**: {stats['throughput_stats']['max_macs_per_cycle']:.2f} MACs/cycle",
                f"- **Average PE Utilization**: {stats['utilization_stats']['mean_pe_utilization']:.1%}",
                f"- **Peak PE Utilization**: {stats['utilization_stats']['max_pe_utilization']:.1%}",
                f"- **Average Roofline Utilization**: {stats['efficiency_stats']['mean_roofline_utilization']:.1%}",
                f"- **Peak Roofline Utilization**: {stats['efficiency_stats']['max_roofline_utilization']:.1%}",
                "",
            ]
        )

        # Scaling analysis
        scaling_analysis = self.analyze_scaling_behavior()
        if "scaling_efficiency" in scaling_analysis:
            report_lines.extend(["## Scaling Analysis", ""])

            for array_size, metrics in scaling_analysis["scaling_efficiency"].items():
                report_lines.extend(
                    [
                        f"### Array Size: {array_size} PEs",
                        f"- Theoretical Speedup: {metrics['theoretical_speedup']:.2f}x",
                        f"- Actual Speedup: {metrics['actual_speedup']:.2f}x",
                        f"- Scaling Efficiency: {metrics['scaling_efficiency']:.1%}",
                        "",
                    ]
                )

        # Workload analysis
        workload_analysis = self.analyze_workload_characteristics()
        if "performance_ranking" in workload_analysis:
            report_lines.extend(
                [
                    "## Workload Performance Ranking",
                    "",
                    f"- **Best Throughput**: {workload_analysis['performance_ranking']['best_throughput']}",
                    f"- **Worst Throughput**: {workload_analysis['performance_ranking']['worst_throughput']}",
                    f"- **Best Utilization**: {workload_analysis['performance_ranking']['best_utilization']}",
                    f"- **Worst Utilization**: {workload_analysis['performance_ranking']['worst_utilization']}",
                    "",
                ]
            )

        # Bottleneck analysis
        bottlenecks = self.identify_performance_bottlenecks()
        report_lines.extend(
            [
                "## Performance Bottleneck Analysis",
                "",
                "### Low PE Utilization Issues",
                f"- **Count**: {bottlenecks['low_pe_utilization']['count']} simulations ({bottlenecks['low_pe_utilization']['percentage']:.1f}%)",
                "",
            ]
        )

        if bottlenecks["low_pe_utilization"]["examples"]:
            report_lines.append("**Examples of low PE utilization:**")
            for example in bottlenecks["low_pe_utilization"]["examples"]:
                report_lines.append(
                    f"- {example['simulation_name']}: {example['pe_utilization']:.1%} PE utilization"
                )
            report_lines.append("")

        report_lines.extend(
            [
                "### Low Roofline Utilization Issues",
                f"- **Count**: {bottlenecks['low_roofline_utilization']['count']} simulations ({bottlenecks['low_roofline_utilization']['percentage']:.1f}%)",
                "",
            ]
        )

        if bottlenecks["low_roofline_utilization"]["examples"]:
            report_lines.append("**Examples of low roofline utilization:**")
            for example in bottlenecks["low_roofline_utilization"]["examples"]:
                report_lines.append(
                    f"- {example['simulation_name']}: {example['roofline_utilization']:.1%} roofline utilization"
                )
            report_lines.append("")

        # Configuration analysis
        config_analysis = self.analyze_configuration_impact()
        if "parameter_correlations" in config_analysis:
            report_lines.extend(["## Key Parameter Correlations", ""])

            correlations = config_analysis["parameter_correlations"]
            if "macs_per_cycle" in correlations:
                throughput_corrs = correlations["macs_per_cycle"]
                sorted_corrs = sorted(
                    [
                        (k, v)
                        for k, v in throughput_corrs.items()
                        if k != "macs_per_cycle"
                    ],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )

                report_lines.append("**Factors most correlated with throughput:**")
                for param, corr in sorted_corrs[:5]:
                    direction = "positively" if corr > 0 else "negatively"
                    report_lines.append(
                        f"- {param}: {direction} correlated ({corr:.3f})"
                    )
                report_lines.append("")

        # Recommendations
        report_lines.extend(
            ["## Recommendations", "", "### Performance Optimization", ""]
        )

        # Generate recommendations based on analysis
        if stats["utilization_stats"]["mean_pe_utilization"] < 0.7:
            report_lines.append(
                "- **Low PE Utilization**: Consider optimizing data flow patterns or workload mapping to improve PE utilization"
            )

        if stats["efficiency_stats"]["mean_roofline_utilization"] < 0.5:
            report_lines.append(
                "- **Low Roofline Utilization**: System is not reaching theoretical peak performance; investigate memory bandwidth or dataflow bottlenecks"
            )

        if "scaling_efficiency" in scaling_analysis:
            avg_scaling_eff = np.mean(
                [
                    metrics["scaling_efficiency"]
                    for metrics in scaling_analysis["scaling_efficiency"].values()
                ]
            )
            if avg_scaling_eff < 0.8:
                report_lines.append(
                    "- **Poor Scaling**: Scaling efficiency is below 80%; consider architectural improvements for larger arrays"
                )

        report_lines.extend(["", "### Configuration Recommendations", ""])

        if bottlenecks["poor_performing_configs"]["count"] > 0:
            report_lines.append(
                "- **Poor Performing Configurations**: The following configurations consistently underperform:"
            )
            for config, util in bottlenecks["poor_performing_configs"][
                "configs"
            ].items():
                report_lines.append(
                    f"  - {config}: {util:.1%} average roofline utilization"
                )
            report_lines.append("")

        # Best practices
        report_lines.extend(["### Best Practices Identified", ""])

        # Find top performing configurations
        top_performers = self.df.nlargest(3, "roofline_utilization")
        if len(top_performers) > 0:
            report_lines.append("**Top performing configurations:**")
            for _, row in top_performers.iterrows():
                if "config_name" in row:
                    report_lines.append(
                        f"- {row['config_name']}: {row['roofline_utilization']:.1%} roofline utilization, {row['macs_per_cycle']:.2f} MACs/cycle"
                    )
            report_lines.append("")

        # Data quality notes
        report_lines.extend(
            [
                "## Data Quality Notes",
                "",
                f"- Results loaded from: `{os.path.basename(self.results_file)}`",
                f"- Analysis performed on {len(self.df)} simulation results",
                f"- Visualizations saved to: `{self.output_dir}/`",
                "",
            ]
        )

        # Methodology
        report_lines.extend(
            [
                "## Methodology",
                "",
                "### Metrics Definitions",
                "",
                "- **PE Utilization**: Fraction of cycles where processing elements are actively computing",
                "- **Roofline Utilization**: Achieved performance as fraction of theoretical peak performance",
                "- **MACs per Cycle**: Multiply-accumulate operations performed per clock cycle",
                "- **Scaling Efficiency**: Actual speedup divided by theoretical speedup for larger arrays",
                "",
                "### Analysis Approach",
                "",
                "1. Statistical analysis of performance distributions",
                "2. Correlation analysis between configuration parameters and performance",
                "3. Scaling behavior analysis across different array sizes",
                "4. Workload characterization and performance ranking",
                "5. Bottleneck identification through threshold analysis",
                "",
            ]
        )

        # Save report
        report_content = "\n".join(report_lines)
        report_file = os.path.join(self.output_dir, "performance_analysis_report.md")

        try:
            with open(report_file, "w") as f:
                f.write(report_content)
            print(f"Comprehensive analysis report saved to: {report_file}")
            return report_file
        except Exception as e:
            print(f"Error saving analysis report: {e}")
            return ""

    def export_summary_data(self) -> str:
        """Export summary statistics to JSON for further processing."""
        summary_data = {
            "basic_statistics": self.basic_statistics(),
            "scaling_analysis": self.analyze_scaling_behavior(),
            "workload_analysis": self.analyze_workload_characteristics(),
            "configuration_analysis": self.analyze_configuration_impact(),
            "bottleneck_analysis": self.identify_performance_bottlenecks(),
            "metadata": {
                "results_file": self.results_file,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_simulations": len(self.df),
                "analysis_version": "1.0",
            },
        }

        summary_file = os.path.join(self.output_dir, "analysis_summary.json")

        try:
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2, default=str)
            print(f"Summary data exported to: {summary_file}")
            return summary_file
        except Exception as e:
            print(f"Error exporting summary data: {e}")
            return ""

    def compare_configurations(self, config1: str, config2: str) -> dict[str, Any]:
        """Compare performance between two specific configurations."""
        if "config_name" not in self.df.columns:
            return {"error": "Configuration names not available in data"}

        config1_data = self.df[self.df["config_name"] == config1]
        config2_data = self.df[self.df["config_name"] == config2]

        if len(config1_data) == 0:
            return {"error": f"Configuration '{config1}' not found"}
        if len(config2_data) == 0:
            return {"error": f"Configuration '{config2}' not found"}

        comparison = {
            "config1": {
                "name": config1,
                "count": len(config1_data),
                "throughput": {
                    "mean": config1_data["macs_per_cycle"].mean(),
                    "std": config1_data["macs_per_cycle"].std(),
                    "max": config1_data["macs_per_cycle"].max(),
                    "min": config1_data["macs_per_cycle"].min(),
                },
                "pe_utilization": {
                    "mean": config1_data["pe_utilization"].mean(),
                    "std": config1_data["pe_utilization"].std(),
                },
                "roofline_utilization": {
                    "mean": config1_data["roofline_utilization"].mean(),
                    "std": config1_data["roofline_utilization"].std(),
                },
            },
            "config2": {
                "name": config2,
                "count": len(config2_data),
                "throughput": {
                    "mean": config2_data["macs_per_cycle"].mean(),
                    "std": config2_data["macs_per_cycle"].std(),
                    "max": config2_data["macs_per_cycle"].max(),
                    "min": config2_data["macs_per_cycle"].min(),
                },
                "pe_utilization": {
                    "mean": config2_data["pe_utilization"].mean(),
                    "std": config2_data["pe_utilization"].std(),
                },
                "roofline_utilization": {
                    "mean": config2_data["roofline_utilization"].mean(),
                    "std": config2_data["roofline_utilization"].std(),
                },
            },
        }

        # Statistical significance tests
        throughput_ttest = stats.ttest_ind(
            config1_data["macs_per_cycle"], config2_data["macs_per_cycle"]
        )
        pe_util_ttest = stats.ttest_ind(
            config1_data["pe_utilization"], config2_data["pe_utilization"]
        )

        comparison["statistical_tests"] = {
            "throughput_ttest": {
                "statistic": throughput_ttest.statistic,
                "pvalue": throughput_ttest.pvalue,
                "significant": throughput_ttest.pvalue < 0.05,
            },
            "pe_utilization_ttest": {
                "statistic": pe_util_ttest.statistic,
                "pvalue": pe_util_ttest.pvalue,
                "significant": pe_util_ttest.pvalue < 0.05,
            },
        }

        # Performance comparison
        throughput_improvement = (
            comparison["config2"]["throughput"]["mean"]
            - comparison["config1"]["throughput"]["mean"]
        ) / comparison["config1"]["throughput"]["mean"]

        comparison["performance_comparison"] = {
            "throughput_improvement": throughput_improvement,
            "better_config": config2 if throughput_improvement > 0 else config1,
            "improvement_percentage": abs(throughput_improvement) * 100,
        }

        return comparison


def main():
    """Main function for performance analyzer."""
    parser = argparse.ArgumentParser(
        description="Performance Analyzer for Open Accelerator Simulation Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with visualizations
  python performance_analyzer.py simulation_results/campaign_results.csv

  # Analysis with custom output directory
  python performance_analyzer.py results.csv --output-dir my_analysis

  # Generate only report without visualizations
  python performance_analyzer.py results.csv --report-only

  # Compare two specific configurations
  python performance_analyzer.py results.csv --compare config1 config2

  # Export summary data for further processing
  python performance_analyzer.py results.csv --export-summary
        """,
    )

    parser.add_argument("results_file", help="Path to simulation results CSV file")

    parser.add_argument(
        "--output-dir",
        "-o",
        default="analysis_results",
        help="Output directory for analysis results (default: analysis_results)",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate only the analysis report without visualizations",
    )

    parser.add_argument(
        "--no-visualizations", action="store_true", help="Skip visualization generation"
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG1", "CONFIG2"),
        help="Compare performance between two specific configurations",
    )

    parser.add_argument(
        "--export-summary",
        action="store_true",
        help="Export summary statistics to JSON",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit",
    )

    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="List available workloads and exit",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return 1

    try:
        # Initialize analyzer
        analyzer = PerformanceAnalyzer(args.results_file)
        analyzer.set_output_dir(args.output_dir)

        print(f"Analyzing simulation results from: {args.results_file}")
        print(f"Output directory: {args.output_dir}")

        # List configurations
        if args.list_configs:
            if "config_name" in analyzer.df.columns:
                configs = analyzer.df["config_name"].unique()
                print(f"Available configurations ({len(configs)}):")
                for config in sorted(configs):
                    count = len(analyzer.df[analyzer.df["config_name"] == config])
                    print(f"  {config}: {count} simulations")
            else:
                print("No configuration information available in results")
            return 0

        # List workloads
        if args.list_workloads:
            if "workload_name" in analyzer.df.columns:
                workloads = analyzer.df["workload_name"].unique()
                print(f"Available workloads ({len(workloads)}):")
                for workload in sorted(workloads):
                    count = len(analyzer.df[analyzer.df["workload_name"] == workload])
                    print(f"  {workload}: {count} simulations")
            else:
                print("No workload information available in results")
            return 0

        # Compare configurations
        if args.compare:
            config1, config2 = args.compare
            print(f"Comparing configurations: {config1} vs {config2}")

            comparison = analyzer.compare_configurations(config1, config2)

            if "error" in comparison:
                print(f"Error: {comparison['error']}")
                return 1

            # Print comparison results
            print(f"\n{config1}:")
            print(f"  Simulations: {comparison['config1']['count']}")
            print(
                f"  Avg Throughput: {comparison['config1']['throughput']['mean']:.2f} ± {comparison['config1']['throughput']['std']:.2f} MACs/cycle"
            )
            print(
                f"  Avg PE Utilization: {comparison['config1']['pe_utilization']['mean']:.1%} ± {comparison['config1']['pe_utilization']['std']:.1%}"
            )
            print(
                f"  Avg Roofline Utilization: {comparison['config1']['roofline_utilization']['mean']:.1%}"
            )

            print(f"\n{config2}:")
            print(f"  Simulations: {comparison['config2']['count']}")
            print(
                f"  Avg Throughput: {comparison['config2']['throughput']['mean']:.2f} ± {comparison['config2']['throughput']['std']:.2f} MACs/cycle"
            )
            print(
                f"  Avg PE Utilization: {comparison['config2']['pe_utilization']['mean']:.1%} ± {comparison['config2']['pe_utilization']['std']:.1%}"
            )
            print(
                f"  Avg Roofline Utilization: {comparison['config2']['roofline_utilization']['mean']:.1%}"
            )

            print("\nComparison:")
            print(
                f"  Better Configuration: {comparison['performance_comparison']['better_config']}"
            )
            print(
                f"  Throughput Improvement: {comparison['performance_comparison']['improvement_percentage']:.1f}%"
            )

            if comparison["statistical_tests"]["throughput_ttest"]["significant"]:
                print(
                    f"  Throughput difference is statistically significant (p={comparison['statistical_tests']['throughput_ttest']['pvalue']:.4f})"
                )
            else:
                print(
                    f"  Throughput difference is not statistically significant (p={comparison['statistical_tests']['throughput_ttest']['pvalue']:.4f})"
                )

            return 0

        # Generate basic statistics
        if args.verbose:
            print("\nGenerating basic statistics...")

        stats = analyzer.basic_statistics()
        print("\nBasic Statistics:")
        print(f"  Total simulations: {stats['total_simulations']:,}")
        print(f"  Unique workloads: {stats['unique_workloads']}")
        print(f"  Unique configurations: {stats['unique_configurations']}")
        print(
            f"  Average throughput: {stats['throughput_stats']['mean_macs_per_cycle']:.2f} MACs/cycle"
        )
        print(
            f"  Average PE utilization: {stats['utilization_stats']['mean_pe_utilization']:.1%}"
        )
        print(
            f"  Average roofline utilization: {stats['efficiency_stats']['mean_roofline_utilization']:.1%}"
        )

        # Generate visualizations
        if not args.no_visualizations and not args.report_only:
            if args.verbose:
                print("\nGenerating visualizations...")
            analyzer.generate_visualizations()

        # Generate comprehensive report
        if args.verbose:
            print("\nGenerating comprehensive report...")

        report_file = analyzer.generate_comprehensive_report()

        # Export summary data
        if args.export_summary:
            if args.verbose:
                print("\nExporting summary data...")
            summary_file = analyzer.export_summary_data()
            if summary_file:
                print(f"Summary data exported to: {summary_file}")

        print(f"\nAnalysis complete! Results available in: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
