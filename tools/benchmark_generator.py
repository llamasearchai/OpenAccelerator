#!/usr/bin/env python3
"""
Benchmark Generator for Open Accelerator

Generates standardized benchmark workloads and configurations
for systematic performance evaluation.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class BenchmarkWorkload:
    """Represents a benchmark workload configuration."""

    name: str
    description: str
    M: int
    K: int
    P: int
    category: str
    typical_use_case: str


@dataclass
class BenchmarkConfiguration:
    """Represents a benchmark accelerator configuration."""

    name: str
    description: str
    array_rows: int
    array_cols: int
    pe_mac_latency: int
    input_buffer_size: int
    weight_buffer_size: int
    output_buffer_size: int
    category: str


class BenchmarkGenerator:
    """Generates benchmark suites for accelerator evaluation."""

    def __init__(self):
        self.workloads: List[BenchmarkWorkload] = []
        self.configurations: List[BenchmarkConfiguration] = []

    def generate_ml_workloads(self) -> List[BenchmarkWorkload]:
        """Generate ML-inspired benchmark workloads."""
        workloads = [
            # Small workloads (edge/mobile)
            BenchmarkWorkload(
                name="tiny_conv",
                description="Tiny convolution layer (mobile/edge)",
                M=8,
                K=16,
                P=8,
                category="edge",
                typical_use_case="Mobile object detection",
            ),
            BenchmarkWorkload(
                name="small_fc",
                description="Small fully connected layer",
                M=32,
                K=64,
                P=16,
                category="edge",
                typical_use_case="Lightweight classification",
            ),
            # Medium workloads (server/datacenter)
            BenchmarkWorkload(
                name="medium_conv",
                description="Medium convolution layer (server)",
                M=64,
                K=128,
                P=64,
                category="server",
                typical_use_case="Image classification backbone",
            ),
            BenchmarkWorkload(
                name="medium_fc",
                description="Medium fully connected layer",
                M=128,
                K=256,
                P=128,
                category="server",
                typical_use_case="Neural network hidden layer",
            ),
            BenchmarkWorkload(
                name="attention_qk",
                description="Attention Query-Key multiplication",
                M=64,
                K=64,
                P=512,
                category="transformer",
                typical_use_case="Transformer attention mechanism",
            ),
            # Large workloads (high-performance)
            BenchmarkWorkload(
                name="large_conv",
                description="Large convolution layer (high-perf)",
                M=256,
                K=512,
                P=256,
                category="high_performance",
                typical_use_case="Large-scale image processing",
            ),
            BenchmarkWorkload(
                name="large_fc",
                description="Large fully connected layer",
                M=512,
                K=1024,
                P=512,
                category="high_performance",
                typical_use_case="Large neural network layer",
            ),
            BenchmarkWorkload(
                name="attention_vo",
                description="Attention Value-Output multiplication",
                M=512,
                K=64,
                P=512,
                category="transformer",
                typical_use_case="Transformer value projection",
            ),
            # Extreme workloads (research/future)
            BenchmarkWorkload(
                name="huge_gemm",
                description="Huge GEMM operation",
                M=1024,
                K=2048,
                P=1024,
                category="extreme",
                typical_use_case="Large language model layers",
            ),
            # Irregular workloads
            BenchmarkWorkload(
                name="irregular_narrow",
                description="Narrow matrix multiplication",
                M=256,
                K=8,
                P=256,
                category="irregular",
                typical_use_case="Embedding lookup layers",
            ),
            BenchmarkWorkload(
                name="irregular_wide",
                description="Wide matrix multiplication",
                M=16,
                K=512,
                P=16,
                category="irregular",
                typical_use_case="Feature reduction layers",
            ),
            BenchmarkWorkload(
                name="square_small",
                description="Small square matrices",
                M=32,
                K=32,
                P=32,
                category="balanced",
                typical_use_case="Balanced computation",
            ),
            BenchmarkWorkload(
                name="square_large",
                description="Large square matrices",
                M=128,
                K=128,
                P=128,
                category="balanced",
                typical_use_case="Large balanced computation",
            ),
        ]

        self.workloads.extend(workloads)
        return workloads

    def generate_synthetic_workloads(self) -> List[BenchmarkWorkload]:
        """Generate synthetic workloads for stress testing."""
        workloads = [
            # Power-of-2 dimensions
            BenchmarkWorkload(
                name="pow2_small",
                description="Power-of-2 small workload",
                M=16,
                K=16,
                P=16,
                category="synthetic",
                typical_use_case="Algorithm verification",
            ),
            BenchmarkWorkload(
                name="pow2_medium",
                description="Power-of-2 medium workload",
                M=64,
                K=64,
                P=64,
                category="synthetic",
                typical_use_case="Performance benchmarking",
            ),
            BenchmarkWorkload(
                name="pow2_large",
                description="Power-of-2 large workload",
                M=256,
                K=256,
                P=256,
                category="synthetic",
                typical_use_case="Scaling analysis",
            ),
            # Prime number dimensions (stress test)
            BenchmarkWorkload(
                name="prime_small",
                description="Prime number dimensions (small)",
                M=17,
                K=19,
                P=23,
                category="stress",
                typical_use_case="Edge case testing",
            ),
            BenchmarkWorkload(
                name="prime_medium",
                description="Prime number dimensions (medium)",
                M=67,
                K=71,
                P=73,
                category="stress",
                typical_use_case="Irregular access patterns",
            ),
            # Single dimension tests
            BenchmarkWorkload(
                name="vector_vector",
                description="Vector-vector outer product",
                M=1,
                K=256,
                P=1,
                category="degenerate",
                typical_use_case="Vector operations",
            ),
            BenchmarkWorkload(
                name="matrix_vector",
                description="Matrix-vector multiplication",
                M=128,
                K=128,
                P=1,
                category="degenerate",
                typical_use_case="Linear system solving",
            ),
            BenchmarkWorkload(
                name="vector_matrix",
                description="Vector-matrix multiplication",
                M=1,
                K=128,
                P=128,
                category="degenerate",
                typical_use_case="Feature transformation",
            ),
        ]

        self.workloads.extend(workloads)
        return workloads

    def generate_accelerator_configurations(self) -> List[BenchmarkConfiguration]:
        """Generate various accelerator configurations."""
        configurations = [
            # Small configurations (edge/mobile)
            BenchmarkConfiguration(
                name="edge_tiny",
                description="Tiny edge accelerator (4x4)",
                array_rows=4,
                array_cols=4,
                pe_mac_latency=1,
                input_buffer_size=256,
                weight_buffer_size=256,
                output_buffer_size=256,
                category="edge",
            ),
            BenchmarkConfiguration(
                name="edge_small",
                description="Small edge accelerator (8x8)",
                array_rows=8,
                array_cols=8,
                pe_mac_latency=1,
                input_buffer_size=512,
                weight_buffer_size=512,
                output_buffer_size=512,
                category="edge",
            ),
            # Medium configurations (server/datacenter)
            BenchmarkConfiguration(
                name="server_medium",
                description="Medium server accelerator (16x16)",
                array_rows=16,
                array_cols=16,
                pe_mac_latency=1,
                input_buffer_size=1024,
                weight_buffer_size=1024,
                output_buffer_size=1024,
                category="server",
            ),
            BenchmarkConfiguration(
                name="server_large",
                description="Large server accelerator (32x32)",
                array_rows=32,
                array_cols=32,
                pe_mac_latency=2,
                input_buffer_size=2048,
                weight_buffer_size=2048,
                output_buffer_size=2048,
                category="server",
            ),
            # High-performance configurations
            BenchmarkConfiguration(
                name="hpc_large",
                description="Large HPC accelerator (64x64)",
                array_rows=64,
                array_cols=64,
                pe_mac_latency=1,
                input_buffer_size=4096,
                weight_buffer_size=4096,
                output_buffer_size=4096,
                category="high_performance",
            ),
            BenchmarkConfiguration(
                name="hpc_huge",
                description="Huge HPC accelerator (128x128)",
                array_rows=128,
                array_cols=128,
                pe_mac_latency=1,
                input_buffer_size=8192,
                weight_buffer_size=8192,
                output_buffer_size=8192,
                category="high_performance",
            ),
            # Configurations with varying buffer sizes
            BenchmarkConfiguration(
                name="small_buffers",
                description="Accelerator with small buffers",
                array_rows=16,
                array_cols=16,
                pe_mac_latency=1,
                input_buffer_size=128,
                weight_buffer_size=128,
                output_buffer_size=128,
                category="memory_bound",
            ),
            BenchmarkConfiguration(
                name="large_buffers",
                description="Accelerator with large buffers",
                array_rows=16,
                array_cols=16,
                pe_mac_latency=1,
                input_buffer_size=4096,
                weight_buffer_size=4096,
                output_buffer_size=4096,
                category="memory_bound",
            ),
            # Configurations with varying latencies
            BenchmarkConfiguration(
                name="high_latency",
                description="Accelerator with high PE latency",
                array_rows=16,
                array_cols=16,
                pe_mac_latency=4,
                input_buffer_size=1024,
                weight_buffer_size=1024,
                output_buffer_size=1024,
                category="latency_bound",
            ),
        ]

        self.configurations.extend(configurations)
        return configurations

    def generate_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark suite."""
        self.generate_ml_workloads()
        self.generate_synthetic_workloads()
        self.generate_accelerator_configurations()

        recommended_combinations = self._generate_recommended_combinations()
        test_suites = self._generate_test_suites()

        return {
            "metadata": {
                "name": "Open Accelerator Comprehensive Benchmark Suite",
                "version": "1.0",
                "total_workloads": len(self.workloads),
                "total_configurations": len(self.configurations),
                "total_combinations": len(self.workloads) * len(self.configurations),
            },
            "workloads": [asdict(w) for w in self.workloads],
            "configurations": [asdict(c) for c in self.configurations],
            "recommended_combinations": recommended_combinations,
            "test_suites": test_suites,
        }

    def _generate_recommended_combinations(self) -> List[Dict[str, Any]]:
        """Generate recommended workload-configuration combinations."""
        recommendations = []
        for workload in self.workloads:
            for config in self.configurations:
                if self._is_compatible(workload, config):
                    recommendations.append(
                        {
                            "workload": workload.name,
                            "configuration": config.name,
                            "reason": "Good match for architecture",
                        }
                    )
        return recommendations

    def _is_compatible(
        self, workload: BenchmarkWorkload, config: BenchmarkConfiguration
    ) -> bool:
        """Check if a workload is compatible with a configuration."""
        # Example compatibility logic
        workload_size = workload.M * workload.K * workload.P
        config_size = config.array_rows * config.array_cols

        if workload.category == "edge" and config.category != "edge":
            return False
        if workload_size > 1_000_000 and config_size < 256:
            return False
        return True

    def _generate_test_suites(self) -> Dict[str, Any]:
        """Generate test suites for different purposes."""
        return {
            "quick_validation": {
                "description": "Quick validation suite for development and CI.",
                "estimated_runtime_minutes": 5,
                "workloads": ["tiny_conv", "small_fc", "pow2_small"],
                "configurations": ["edge_tiny", "server_medium"],
            },
            "performance_benchmarking": {
                "description": "Comprehensive suite for performance benchmarking.",
                "estimated_runtime_minutes": 60,
                "workloads": "all",
                "configurations": "all",
            },
            "scaling_analysis": {
                "description": "Suite for analyzing performance scaling.",
                "estimated_runtime_minutes": 30,
                "workloads": ["pow2_small", "pow2_medium", "pow2_large"],
                "configurations": ["edge_small", "server_medium", "hpc_large"],
            },
            "irregular_workloads": {
                "description": "Suite for testing irregular workloads.",
                "estimated_runtime_minutes": 20,
                "workloads": [
                    "irregular_narrow",
                    "irregular_wide",
                    "prime_small",
                    "prime_medium",
                ],
                "configurations": "all",
            },
        }

    def generate_simulation_scripts(
        self, output_dir: str, benchmark_suite: Dict[str, Any]
    ):
        """Generate executable simulation scripts for each test suite."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for suite_name, suite_info in benchmark_suite["test_suites"].items():
            script_content = self._create_simulation_script(
                suite_name, suite_info, benchmark_suite
            )
            script_file = os.path.join(output_dir, f"run_{suite_name}_suite.py")
            with open(script_file, "w") as f:
                f.write(script_content)

            # Make script executable
            os.chmod(script_file, 0o755)

            print(f"Generated simulation script: {script_file}")

    def _create_simulation_script(
        self,
        suite_name: str,
        suite_info: Dict[str, Any],
        benchmark_suite: Dict[str, Any],
    ) -> str:
        """Create a simulation script for a specific test suite."""
        script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            f"Simulation script for {suite_name} test suite",
            f'Description: {suite_info["description"]}',
            f'Estimated runtime: {suite_info["estimated_runtime_minutes"]} minutes',
            '"""',
            "",
            "import os",
            "import sys",
            "import time",
            "from pathlib import Path",
            "",
            "# Add the open_accelerator package to path",
            "sys.path.insert(0, str(Path(__file__).parent.parent))",
            "",
            "from open_accelerator.utils import AcceleratorConfig, WorkloadConfig",
            "from open_accelerator.workloads import GEMMWorkload",
            "from open_accelerator.simulation import Simulator",
            "from open_accelerator.analysis import analyze_simulation_results",
            "import numpy as np",
            "import pandas as pd",
            "",
            "def main():",
            f'    """Run {suite_name} test suite."""',
            f'    print("Starting {suite_name} test suite...")',
            '    print("=" * 50)',
            "",
            "    results = []",
            "    start_time = time.time()",
            "",
        ]
        # Get workloads and configurations for this suite
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

        # Add simulation logic
        script_lines.extend(["    # Test configurations", "    workloads = ["])

        for workload in workloads:
            script_lines.append(f"        {asdict(workload)},")

        script_lines.extend(["    ]", "", "    configurations = ["])

        for config in configurations:
            script_lines.append(f"        {asdict(config)},")

        script_lines.extend(
            [
                "    ]",
                "",
                "    total_combinations = len(workloads) * len(configurations)",
                "    current_combination = 0",
                "",
                "    for workload_spec in workloads:",
                "        for config_spec in configurations:",
                "            current_combination += 1",
                "            ",
                "            # Check compatibility for Output Stationary",
                '            if workload_spec["M"] != config_spec["array_rows"] or workload_spec["P"] != config_spec["array_cols"]:',
                "                print(f\"Skipping incompatible combination: {workload_spec['name']} + {config_spec['name']}\")",
                "                continue",
                "",
                '            print(f"\\nRunning combination {current_combination}/{total_combinations}:")',
                "            print(f\"  Workload: {workload_spec['name']} ({workload_spec['description']})\")",
                "            print(f\"  Configuration: {config_spec['name']} ({config_spec['description']})\")",
                "",
                "            # Create configuration objects",
                "            accel_config = AcceleratorConfig(",
                '                array_rows=config_spec["array_rows"],',
                '                array_cols=config_spec["array_cols"],',
                '                pe_mac_latency=config_spec["pe_mac_latency"],',
                '                input_buffer_size=config_spec["input_buffer_size"],',
                '                weight_buffer_size=config_spec["weight_buffer_size"],',
                '                output_buffer_size=config_spec["output_buffer_size"],',
                "                data_type=np.int32",
                "            )",
                "",
                "            workload_config = WorkloadConfig(",
                '                gemm_M=workload_spec["M"],',
                '                gemm_K=workload_spec["K"],',
                '                gemm_P=workload_spec["P"]',
                "            )",
                "",
                "            # Create and run simulation",
                "            workload = GEMMWorkload(workload_config, accel_config)",
                "            workload.generate_data(seed=42)",
                "",
                "            simulator = Simulator(accel_config, workload)",
                "            sim_start = time.time()",
                "            simulation_stats = simulator.run()",
                "            sim_time = time.time() - sim_start",
                "",
                "            # Analyze results",
                "            metrics = analyze_simulation_results(simulation_stats, accel_config, workload)",
                "",
                "            # Store results",
                "            result = {",
                "                \"simulation_name\": f\"{workload_spec['name']}_{config_spec['name']}\",",
                '                "workload_name": workload_spec["name"],',
                '                "workload_category": workload_spec["category"],',
                '                "workload_M": workload_spec["M"],',
                '                "work_K": workload_spec["K"],',
                '                "workload_P": workload_spec["P"],',
                '                "config_name": config_spec["name"],',
                '                "config_category": config_spec["category"],',
                '                "array_rows": config_spec["array_rows"],',
                '                "array_cols": config_spec["array_cols"],',
                '                "array_size": config_spec["array_rows"] * config_spec["array_cols"],',
                '                "pe_mac_latency": config_spec["pe_mac_latency"],',
                '                "total_cycles": metrics.total_cycles,',
                '                "total_mac_operations": metrics.total_mac_operations,',
                '                "macs_per_cycle": metrics.macs_per_cycle,',
                '                "pe_utilization": metrics.average_pe_utilization,',
                '                "roofline_utilization": metrics.roofline_utilization,',
                '                "simulation_time_seconds": sim_time',
                "            }",
                "",
                "            results.append(result)",
                "",
                '            print(f"    Cycles: {metrics.total_cycles}, Throughput: {metrics.macs_per_cycle:.2f} MACs/cycle")',
                '            print(f"    PE Utilization: {metrics.average_pe_utilization:.1%}, Sim Time: {sim_time:.2f}s")',
                "",
                "    # Save results",
                "    results_df = pd.DataFrame(results)",
                f'    output_file = "{suite_name}_results.csv"',
                "    results_df.to_csv(output_file, index=False)",
                "",
                "    total_time = time.time() - start_time",
                f'    print(f"\\n{suite_name} test suite completed!")',
                '    print(f"Total runtime: {total_time/60:.1f} minutes")',
                '    print(f"Results saved to: {output_file}")',
                '    print(f"Total simulations: {len(results)}")',
                "",
                'if __name__ == "__main__":',
                "    main()",
            ]
        )

        return "\n".join(script_lines)

    def save_benchmark_suite(self, benchmark_suite: Dict[str, Any], output_file: str):
        """Save benchmark suite to JSON file."""
        try:
            with open(output_file, "w") as f:
                json.dump(benchmark_suite, f, indent=2)
            print(f"Benchmark suite saved to: {output_file}")
        except Exception as e:
            print(f"Error saving benchmark suite: {e}")

    def generate_documentation(self, benchmark_suite: Dict[str, Any], output_file: str):
        """Generate documentation for the benchmark suite."""
        doc_lines = [
            "# Open Accelerator Benchmark Suite Documentation",
            "",
            "This document describes the comprehensive benchmark suite for the Open Accelerator simulator.",
            "",
            "## Overview",
            "",
            f"- **Total Workloads**: {benchmark_suite['metadata']['total_workloads']}",
            f"- **Total Configurations**: {benchmark_suite['metadata']['total_configurations']}",
            f"- **Total Combinations**: {benchmark_suite['metadata']['total_combinations']}",
            "",
            "## Workload Categories",
            "",
        ]

        # Document workload categories
        workloads_by_category = {}
        for workload in benchmark_suite["workloads"]:
            category = workload["category"]
            if category not in workloads_by_category:
                workloads_by_category[category] = []
            workloads_by_category[category].append(workload)

        for category, workloads in workloads_by_category.items():
            doc_lines.extend([f"### {category.title()} Workloads", ""])

            for workload in workloads:
                doc_lines.extend(
                    [
                        f"**{workload['name']}**: {workload['description']}",
                        f"- Dimensions: {workload['M']}×{workload['K']}×{workload['P']} (M×K×P)",
                        f"- Use Case: {workload['typical_use_case']}",
                        "",
                    ]
                )

        # Document configuration categories
        doc_lines.extend(["## Configuration Categories", ""])

        configs_by_category = {}
        for config in benchmark_suite["configurations"]:
            category = config["category"]
            if category not in configs_by_category:
                configs_by_category[category] = []
            configs_by_category[category].append(config)

        for category, configs in configs_by_category.items():
            doc_lines.extend([f"### {category.title()} Configurations", ""])

            for config in configs:
                doc_lines.extend(
                    [
                        f"**{config['name']}**: {config['description']}",
                        f"- Array Size: {config['array_rows']}×{config['array_cols']} PEs",
                        f"- MAC Latency: {config['pe_mac_latency']} cycle(s)",
                        f"- Buffer Sizes: {config['input_buffer_size']} elements each",
                        "",
                    ]
                )

        # Document test suites
        doc_lines.extend(["## Test Suites", ""])

        for suite_name, suite_info in benchmark_suite["test_suites"].items():
            doc_lines.extend(
                [
                    f"### {suite_name.title().replace('_', ' ')}",
                    "",
                    f"**Description**: {suite_info['description']}",
                    f"**Estimated Runtime**: {suite_info['estimated_runtime_minutes']} minutes",
                    "",
                ]
            )

            if suite_info["workloads"] != "all":
                doc_lines.extend(["**Workloads**:", ""])
                for workload_name in suite_info["workloads"]:
                    workload = next(
                        w
                        for w in benchmark_suite["workloads"]
                        if w["name"] == workload_name
                    )
                    doc_lines.append(f"- {workload['name']}: {workload['description']}")
                doc_lines.append("")

            if suite_info["configurations"] != "all":
                doc_lines.extend(["**Configurations**:", ""])
                for config_name in suite_info["configurations"]:
                    config = next(
                        c
                        for c in benchmark_suite["configurations"]
                        if c["name"] == config_name
                    )
                    doc_lines.append(f"- {config['name']}: {config['description']}")
                doc_lines.append("")

        # Usage instructions
        doc_lines.extend(
            [
                "## Usage Instructions",
                "",
                "### Running Individual Test Suites",
                "",
                "Each test suite has its own generated script:",
                "",
                "",
                "# Quick validation (development)",
                "python run_quick_validation_suite.py",
                "",
                "# Performance benchmarking",
                "python run_performance_benchmarking_suite.py",
                "",
                "# Scaling analysis",
                "python run_scaling_analysis_suite.py",
                "",
                "",
                "### Running All Benchmarks",
                "",
                "",
                "A `Makefile` is provided for convenience:",
                "",
                "```bash",
                "# Run all benchmark suites",
                "make benchmark",
                "",
                "# Run a specific suite",
                "make benchmark-quick",
                "make benchmark-performance",
                "```",
            ]
        )

        try:
            with open(output_file, "w") as f:
                f.write("\n".join(doc_lines))
            print(f"Documentation saved to: {output_file}")
        except Exception as e:
            print(f"Error generating documentation: {e}")

    def create_makefile(self, output_dir: str, benchmark_suite: Dict[str, Any]):
        """Create a Makefile for running benchmarks."""
        makefile_lines = [
            ".PHONY: all benchmark clean",
            "",
            "all: benchmark",
            "",
            "benchmark:",
        ]

        for suite_name in benchmark_suite["test_suites"]:
            makefile_lines.append(f"\t@echo 'Running {suite_name} benchmark suite...'")
            makefile_lines.append(f"\t@python {output_dir}/run_{suite_name}_suite.py")

        makefile_lines.extend(
            [
                "",
                "clean:",
                "\t@echo 'Cleaning up benchmark results...'",
                "\t@rm -f *.csv",
                "\t@rm -f *.json",
                "\t@rm -f *.md",
            ]
        )

        # Add individual targets for each suite
        for suite_name in benchmark_suite["test_suites"]:
            makefile_lines.extend(
                [
                    "",
                    f".PHONY: benchmark-{suite_name.replace('_', '-')}",
                    f"benchmark-{suite_name.replace('_', '-')}:",
                    f"\t@echo 'Running {suite_name} benchmark suite...'",
                    f"\t@python {output_dir}/run_{suite_name}_suite.py",
                ]
            )

        try:
            with open("Makefile.benchmarks", "w") as f:
                f.write("\n".join(makefile_lines))
            print("Benchmark Makefile created: Makefile.benchmarks")
        except Exception as e:
            print(f"Error creating Makefile: {e}")

    def generate_all_outputs(self, output_dir: str = "benchmarks"):
        """Generate all benchmark artifacts."""
        print("Generating comprehensive benchmark suite...")
        benchmark_suite = self.generate_comprehensive_benchmark_suite()
        print("Benchmark suite generated.")

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save benchmark suite to file
        suite_file = os.path.join(output_dir, "benchmark_suite.json")
        self.save_benchmark_suite(benchmark_suite, suite_file)

        # Generate simulation scripts
        print("\nGenerating simulation scripts...")
        self.generate_simulation_scripts(output_dir, benchmark_suite)
        print("Simulation scripts generated.")

        # Generate documentation
        print("\nGenerating documentation...")
        doc_file = os.path.join(output_dir, "BENCHMARK_SUITE.md")
        self.generate_documentation(benchmark_suite, doc_file)
        print("Documentation generated.")

        # Create Makefile
        print("\nCreating Makefile...")
        self.create_makefile(output_dir, benchmark_suite)
        print("Makefile created.")

        print("\n[SUCCESS] All benchmark artifacts generated successfully!")
        print(f"Output directory: {output_dir}")


def main():
    """Main function to generate benchmarks."""
    parser = argparse.ArgumentParser(
        description="Generate Open Accelerator benchmarks."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="benchmarks",
        help="Output directory for generated artifacts.",
    )
    args = parser.parse_args()

    generator = BenchmarkGenerator()
    generator.generate_all_outputs(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
