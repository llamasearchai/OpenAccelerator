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
from typing import Any, Dict, List, Tuple

import numpy as np


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
                M=8, K=16, P=8,
                category="edge",
                typical_use_case="Mobile object detection"
            ),
            BenchmarkWorkload(
                name="small_fc",
                description="Small fully connected layer",
                M=32, K=64, P=16,
                category="edge",
                typical_use_case="Lightweight classification"
            ),

            # Medium workloads (server/datacenter)
            BenchmarkWorkload(
                name="medium_conv",
                description="Medium convolution layer (server)",
                M=64, K=128, P=64,
                category="server",
                typical_use_case="Image classification backbone"
            ),
            BenchmarkWorkload(
                name="medium_fc",
                description="Medium fully connected layer",
                M=128, K=256, P=128,
                category="server",
                typical_use_case="Neural network hidden layer"
            ),
            BenchmarkWorkload(
                name="attention_qk",
                description="Attention Query-Key multiplication",
                M=64, K=64, P=512,
                category="transformer",
                typical_use_case="Transformer attention mechanism"
            ),

            # Large workloads (high-performance)
            BenchmarkWorkload(
                name="large_conv",
                description="Large convolution layer (high-perf)",
                M=256, K=512, P=256,
                category="high_performance",
                typical_use_case="Large-scale image processing"
            ),
            BenchmarkWorkload(
                name="large_fc",
                description="Large fully connected layer",
                M=512, K=1024, P=512,
                category="high_performance",
                typical_use_case="Large neural network layer"
            ),
            BenchmarkWorkload(
                name="attention_vo",
                description="Attention Value-Output multiplication",
                M=512, K=64, P=512,
                category="transformer",
                typical_use_case="Transformer value projection"
            ),

            # Extreme workloads (research/future)
            BenchmarkWorkload(
                name="huge_gemm",
                description="Huge GEMM operation",
                M=1024, K=2048, P=1024,
                category="extreme",
                typical_use_case="Large language model layers"
            ),

            # Irregular workloads
            BenchmarkWorkload(
                name="irregular_narrow",
                description="Narrow matrix multiplication",
                M=256, K=8, P=256,
                category="irregular",
                typical_use_case="Embedding lookup layers"
            ),
            BenchmarkWorkload(
                name="irregular_wide",
                description="Wide matrix multiplication",
                M=16, K=512, P=16,
                category="irregular",
                typical_use_case="Feature reduction layers"
            ),
            BenchmarkWorkload(
                name="square_small",
                description="Small square matrices",
                M=32, K=32, P=32,
                category="balanced",
                typical_use_case="Balanced computation"
            ),
            BenchmarkWorkload(
                name="square_large",
                description="Large square matrices",
                M=128, K=128, P=128,
                category="balanced",
                typical_use_case="Large balanced computation"
            )
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
                M=16, K=16, P=16,
                category="synthetic",
                typical_use_case="Algorithm verification"
            ),
            BenchmarkWorkload(
                name="pow2_medium",
                description="Power-of-2 medium workload",
                M=64, K=64, P=64,
                category="synthetic",
                typical_use_case="Performance benchmarking"
            ),
            BenchmarkWorkload(
                name="pow2_large",
                description="Power-of-2 large workload",
                M=256, K=256, P=256,
                category="synthetic",
                typical_use_case="Scaling analysis"
            ),

            # Prime number dimensions (stress test)
            BenchmarkWorkload(
                name="prime_small",
                description="Prime number dimensions (small)",
                M=17, K=19, P=23,
                category="stress",
                typical_use_case="Edge case testing"
            ),
            BenchmarkWorkload(
                name="prime_medium",
                description="Prime number dimensions (medium)",
                M=67, K=71, P=73,
                category="stress",
                typical_use_case="Irregular access patterns"
            ),

            # Single dimension tests
            BenchmarkWorkload(
                name="vector_vector",
                description="Vector-vector outer product",
                M=1, K=256, P=1,
                category="degenerate",
                typical_use_case="Vector operations"
            ),
            BenchmarkWorkload(
                name="matrix_vector",
                description="Matrix-vector multiplication",
                M=128, K=128, P=1,
                category="degenerate",
                typical_use_case="Linear system solving"
            ),
            BenchmarkWorkload(
                name="vector_matrix",
                description="Vector-matrix multiplication",
                M=1, K=128, P=128,
                category="degenerate",
                typical_use_case="Feature transformation"
            )
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
                array_rows=4, array_cols=4,
                pe_mac_latency=1,
                input_buffer_size=256,
                weight_buffer_size=256,
                output_buffer_size=256,
                category="edge"
            ),
            BenchmarkConfiguration(
                name="edge_small",
                description="Small edge accelerator (8x8)",
                array_rows=8, array_cols=8,
                pe_mac_latency=1,
                input_buffer_size=512,
                weight_buffer_size=512,
                output_buffer_size=512,
                category="edge"
            ),

            # Medium configurations (server/datacenter)
            BenchmarkConfiguration(
                name="server_small",
                description="Small server accelerator (16x16)",
                array_rows=16, array_cols=16,
                pe_mac_latency=1,
                input_buffer_size=2048,
                weight_buffer_size=2048,
                output_buffer_size=2048,
                category="server"
            ),
            BenchmarkConfiguration(
                name="server_medium",
                description="Medium server accelerator (32x32)",
                array_rows=32, array_cols=32,
                pe_mac_latency=1,
                input_buffer_size=8192,
                weight_buffer_size=8192,
                output_buffer_size=8192,
                category="server"
            ),

            # Large configurations (high-performance)
            BenchmarkConfiguration(
                name="hpc_medium",
                description="HPC medium accelerator (64x64)",
                array_rows=64, array_cols=64,
                pe_mac_latency=1,
                input_buffer_size=32768,
                weight_buffer_size=32768,
                output_buffer_size=32768,
                category="high_performance"
            ),
            BenchmarkConfiguration(
                name="hpc_large",
                description="HPC large accelerator (128x128)",
                array_rows=128, array_cols=128,
                pe_mac_latency=1,
                input_buffer_size=131072,
                weight_buffer_size=131072,
                output_buffer_size=131072,
                category="high_performance"
            ),

            # Rectangular configurations
            BenchmarkConfiguration(
                name="rect_wide",
                description="Wide rectangular accelerator (16x64)",
                array_rows=16, array_cols=64,
                pe_mac_latency=1,
                input_buffer_size=4096,
                weight_buffer_size=4096,
                output_buffer_size=4096,
                category="specialized"
            ),
            BenchmarkConfiguration(
                name="rect_tall",
                description="Tall rectangular accelerator (64x16)",
                array_rows=64, array_cols=16,
                pe_mac_latency=1,
                input_buffer_size=4096,
                weight_buffer_size=4096,
                output_buffer_size=4096,
                category="specialized"
            ),

            # Memory-constrained configurations
            BenchmarkConfiguration(
                name="mem_limited",
                description="Memory-limited accelerator (32x32)",
                array_rows=32, array_cols=32,
                pe_mac_latency=1,
                input_buffer_size=1024,  # Smaller buffers
                weight_buffer_size=1024,
                output_buffer_size=1024,
                category="memory_limited"
            ),

            # High-latency configurations
            BenchmarkConfiguration(
                name="high_latency",
                description="High-latency MAC accelerator (16x16)",
                array_rows=16, array_cols=16,
                pe_mac_latency=3,  # Higher MAC latency
                input_buffer_size=2048,
                weight_buffer_size=2048,
                output_buffer_size=2048,
                category="high_latency"
            )
        ]

        self.configurations.extend(configurations)
        return configurations

    def generate_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Generate a comprehensive benchmark suite."""
        print("Generating comprehensive benchmark suite...")

        # Generate all workloads and configurations
        ml_workloads = self.generate_ml_workloads()
        synthetic_workloads = self.generate_synthetic_workloads()
        configurations = self.generate_accelerator_configurations()

        # Create benchmark combinations
        benchmark_suite = {
            "metadata": {
                "generator_version": "1.0",
                "total_workloads": len(self.workloads),
                "total_configurations": len(self.configurations),
                "total_combinations": len(self.workloads) * len(self.configurations),
                "categories": {
                    "workloads": list(set(w.category for w in self.workloads)),
                    "configurations": list(set(c.category for c in self.configurations))
                }
            },
            "workloads": [asdict(w) for w in self.workloads],
            "configurations": [asdict(c) for c in self.configurations],
            "recommended_combinations": self._generate_recommended_combinations(),
            "test_suites": self._generate_test_suites()
        }

        return benchmark_suite

    def _generate_recommended_combinations(self) -> List[Dict[str, Any]]:
        """Generate recommended workload-configuration combinations."""
        recommendations = []

        # Edge device combinations
        edge_workloads = [w for w in self.workloads if w.category == "edge"]
        edge_configs = [c for c in self.configurations if c.category == "edge"]

        for workload in edge_workloads:
            for config in edge_configs:
                if self._is_compatible(workload, config):
                    recommendations.append({
                        "suite_name": "edge_deployment",
                        "workload": workload.name,
                        "configuration": config.name,
                        "rationale": "Edge deployment scenario",
                        "expected_characteristics": "Low power, moderate performance"
                    })

        # Server combinations
        server_workloads = [w for w in self.workloads if w.category == "server"]
        server_configs = [c for c in self.configurations if c.category == "server"]

        for workload in server_workloads:
            for config in server_configs:
                if self._is_compatible(workload, config):
                    recommendations.append({
                        "suite_name": "server_deployment",
                        "workload": workload.name,
                        "configuration": config.name,
                        "rationale": "Server deployment scenario",
                        "expected_characteristics": "High throughput, balanced efficiency"
                    })

        # High-performance combinations
        hpc_workloads = [w for w in self.workloads if w.category == "high_performance"]
        hpc_configs = [c for c in self.configurations if c.category == "high_performance"]

        for workload in hpc_workloads:
            for config in hpc_configs:
                if self._is_compatible(workload, config):
                    recommendations.append({
                        "suite_name": "hpc_deployment",
                        "workload": workload.name,
                        "configuration": config.name,
                        "rationale": "High-performance computing scenario",
                        "expected_characteristics": "Maximum throughput, scaling analysis"
                    })

        # Stress test combinations
        stress_workloads = [w for w in self.workloads if w.category == "stress"]

        for workload in stress_workloads:
            for config in self.configurations[:3]:  # Test with first few configs
                recommendations.append({
                    "suite_name": "stress_testing",
                    "workload": workload.name,
                    "configuration": config.name,
                    "rationale": "Stress testing with irregular dimensions",
                    "expected_characteristics": "Edge case behavior analysis"
                })

        return recommendations

    def _is_compatible(self, workload: BenchmarkWorkload, config: BenchmarkConfiguration) -> bool:
        """Check if workload and configuration are compatible."""
        # For Output Stationary, array dimensions must match M and P
        return (workload.M == config.array_rows and
                workload.P == config.array_cols)

    def _generate_test_suites(self) -> Dict[str, Any]:
        """Generate organized test suites."""
        test_suites = {
            "quick_validation": {
                "description": "Quick validation suite for development",
                "workloads": ["tiny_conv", "small_fc", "square_small"],
                "configurations": ["edge_tiny", "edge_small"],
                "estimated_runtime_minutes": 5
            },
            "performance_benchmarking": {
                "description": "Performance benchmarking suite",
                "workloads": ["medium_conv", "medium_fc", "large_conv", "square_large"],
                "configurations": ["server_small", "server_medium", "hpc_medium"],
                "estimated_runtime_minutes": 30
            },
            "scaling_analysis": {
                "description": "Scaling analysis across array sizes",
                "workloads": ["pow2_small", "pow2_medium", "pow2_large"],
                "configurations": ["edge_small", "server_small", "server_medium", "hpc_medium"],
                "estimated_runtime_minutes": 45
            },
            "efficiency_analysis": {
                "description": "PE utilization and efficiency analysis",
                "workloads": ["irregular_narrow", "irregular_wide", "attention_qk", "attention_vo"],
                "configurations": ["rect_wide", "rect_tall", "server_medium"],
                "estimated_runtime_minutes": 25
            },
            "stress_testing": {
                "description": "Stress testing with edge cases",
                "workloads": ["prime_small", "prime_medium", "vector_vector", "matrix_vector"],
                "configurations": ["edge_small", "server_small", "mem_limited"],
                "estimated_runtime_minutes": 20
            },
            "memory_analysis": {
                "description": "Memory system analysis",
                "workloads": ["large_fc", "huge_gemm"],
                "configurations": ["mem_limited", "server_medium", "hpc_medium"],
                "estimated_runtime_minutes": 40
            },
            "comprehensive": {
                "description": "Comprehensive evaluation suite",
                "workloads": "all",
                "configurations": "all",
                "estimated_runtime_minutes": 180
            }
        }

        return test_suites

    def generate_simulation_scripts(self, output_dir: str, benchmark_suite: Dict[str, Any]):
        """Generate simulation scripts for different test suites."""
        os.makedirs(output_dir, exist_ok=True)

        for suite_name, suite_info in benchmark_suite["test_suites"].items():
            script_content = self._create_simulation_script(suite_name, suite_info, benchmark_suite)

            script_file = os.path.join(output_dir, f"run_{suite_name}_suite.py")
            with open(script_file, 'w') as f:
                f.write(script_content)

            # Make script executable
            os.chmod(script_file, 0o755)

            print(f"Generated simulation script: {script_file}")

    def _create_simulation_script(self, suite_name: str, suite_info: Dict[str, Any],
                                benchmark_suite: Dict[str, Any]) -> str:
        """Create a simulation script for a specific test suite."""

        script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            f'Simulation script for {suite_name} test suite',
            f'Description: {suite_info["description"]}',
            f'Estimated runtime: {suite_info["estimated_runtime_minutes"]} minutes',
            '"""',
            '',
            'import os',
            'import sys',
            'import time',
            'from pathlib import Path',
            '',
            '# Add the open_accelerator package to path',
            'sys.path.insert(0, str(Path(__file__).parent.parent))',
            '',
            'from open_accelerator.utils import AcceleratorConfig, WorkloadConfig',
            'from open_accelerator.workloads import GEMMWorkload',
            'from open_accelerator.simulation import Simulator',
            'from open_accelerator.analysis import analyze_simulation_results',
            'import numpy as np',
            'import pandas as pd',
            '',
            'def main():',
            f'    """Run {suite_name} test suite."""',
            f'    print("Starting {suite_name} test suite...")',
            '    print("=" * 50)',
            '',
            '    results = []',
            '    start_time = time.time()',
            ''
        ]

        # Get workloads and configurations for this suite
        if suite_info["workloads"] == "all":
            workloads = benchmark_suite["workloads"]
        else:
            workloads = [w for w in benchmark_suite["workloads"] if w["name"] in suite_info["workloads"]]

        if suite_info["configurations"] == "all":
            configurations = benchmark_suite["configurations"]
        else:
            configurations = [c for c in benchmark_suite["configurations"] if c["name"] in suite_info["configurations"]]

        # Add simulation logic
        script_lines.extend([
            '    # Test configurations',
            '    workloads = ['
        ])

        for workload in workloads:
            script_lines.append(f'        {workload},')

        script_lines.extend([
            '    ]',
            '',
            '    configurations = ['
        ])

        for config in configurations:
            script_lines.append(f'        {config},')

        script_lines.extend([
            '    ]',
            '',
            '    total_combinations = len(workloads) * len(configurations)',
            '    current_combination = 0',
            '',
            '    for workload_spec in workloads:',
            '        for config_spec in configurations:',
            '            current_combination += 1',
            '            ',
            '            # Check compatibility for Output Stationary',
            '            if workload_spec["M"] != config_spec["array_rows"] or workload_spec["P"] != config_spec["array_cols"]:',
            '                print(f"Skipping incompatible combination: {workload_spec[\'name\']} + {config_spec[\'name\']}")',
            '                continue',
            '',
            '            print(f"\\nRunning combination {current_combination}/{total_combinations}:")',
            '            print(f"  Workload: {workload_spec[\'name\']} ({workload_spec[\'description\']})")',
            '            print(f"  Configuration: {config_spec[\'name\']} ({config_spec[\'description\']})")',
            '',
            '            # Create configuration objects',
            '            accel_config = AcceleratorConfig(',
            '                array_rows=config_spec["array_rows"],',
            '                array_cols=config_spec["array_cols"],',
            '                pe_mac_latency=config_spec["pe_mac_latency"],',
            '                input_buffer_size=config_spec["input_buffer_size"],',
            '                weight_buffer_size=config_spec["weight_buffer_size"],',
            '                output_buffer_size=config_spec["output_buffer_size"],',
            '                data_type=np.int32',
            '            )',
            '',
            '            workload_config = WorkloadConfig(',
            '                gemm_M=workload_spec["M"],',
            '                gemm_K=workload_spec["K"],',
            '                gemm_P=workload_spec["P"]',
            '            )',
            '',
            '            # Create and run simulation',
            '            workload = GEMMWorkload(workload_config, accel_config)',
            '            workload.generate_data(seed=42)',
            '',
            '            simulator = Simulator(accel_config, workload)',
            '            sim_start = time.time()',
            '            simulation_stats = simulator.run()',
            '            sim_time = time.time() - sim_start',
            '',
            '            # Analyze results',
            '            metrics = analyze_simulation_results(simulation_stats, accel_config, workload)',
            '',
            '            # Store results',
            '            result = {',
            '                "simulation_name": f"{workload_spec[\'name\']}_{config_spec[\'name\']}",',
            '                "workload_name": workload_spec["name"],',
            '                "workload_category": workload_spec["category"],',
            '                "workload_M": workload_spec["M"],',
            '                "workload_K": workload_spec["K"],',
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
            '            }',
            '',
            '            results.append(result)',
            '',
            '            print(f"    Cycles: {metrics.total_cycles}, Throughput: {metrics.macs_per_cycle:.2f} MACs/cycle")',
            '            print(f"    PE Utilization: {metrics.average_pe_utilization:.1%}, Sim Time: {sim_time:.2f}s")',
            '',
            '    # Save results',
            f'    results_df = pd.DataFrame(results)',
            f'    output_file = "{suite_name}_results.csv"',
            '    results_df.to_csv(output_file, index=False)',
            '',
            '    total_time = time.time() - start_time',
            f'    print(f"\\n{suite_name} test suite completed!")',
            '    print(f"Total runtime: {total_time/60:.1f} minutes")',
            '    print(f"Results saved to: {output_file}")',
            '    print(f"Total simulations: {len(results)}")',
            '',
            'if __name__ == "__main__":',
            '    main()'
        ]

        return '\n'.join(script_lines)

    def save_benchmark_suite(self, benchmark_suite: Dict[str, Any], output_file: str):
        """Save benchmark suite to JSON file."""
        try:
            with open(output_file, 'w') as f:
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
            ""
        ]

        # Document workload categories
        workloads_by_category = {}
        for workload in benchmark_suite["workloads"]:
            category = workload["category"]
            if category not in workloads_by_category:
                workloads_by_category[category] = []
            workloads_by_category[category].append(workload)

        for category, workloads in workloads_by_category.items():
            doc_lines.extend([
                f"### {category.title()} Workloads",
                ""
            ])

            for workload in workloads:
                doc_lines.extend([
                    f"**{workload['name']}**: {workload['description']}",
                    f"- Dimensions: {workload['M']}×{workload['K']}×{workload['P']} (M×K×P)",
                    f"- Use Case: {workload['typical_use_case']}",
                    ""
                ])

        # Document configuration categories
        doc_lines.extend([
            "## Configuration Categories",
            ""
        ])

        configs_by_category = {}
        for config in benchmark_suite["configurations"]:
            category = config["category"]
            if category not in configs_by_category:
                configs_by_category[category] = []
            configs_by_category[category].append(config)

        for category, configs in configs_by_category.items():
            doc_lines.extend([
                f"### {category.title()} Configurations",
                ""
            ])

            for config in configs:
                doc_lines.extend([
                    f"**{config['name']}**: {config['description']}",
                    f"- Array Size: {config['array_rows']}×{config['array_cols']} PEs",
                    f"- MAC Latency: {config['pe_mac_latency']} cycle(s)",
                    f"- Buffer Sizes: {config['input_buffer_size']} elements each",
                    ""
                ])

        # Document test suites
        doc_lines.extend([
            "## Test Suites",
            ""
        ])

        for suite_name, suite_info in benchmark_suite["test_suites"].items():
            doc_lines.extend([
                f"### {suite_name.title().replace('_', ' ')}",
                "",
                f"**Description**: {suite_info['description']}",
                f"**Estimated Runtime**: {suite_info['estimated_runtime_minutes']} minutes",
                ""
            ])

            if suite_info["workloads"] != "all":
                doc_lines.extend([
                    "**Workloads**:",
                    ""
                ])
                for workload_name in suite_info["workloads"]:
                    workload = next(w for w in benchmark_suite["workloads"] if w["name"] == workload_name)
                    doc_lines.append(f"- {workload['name']}: {workload['description']}")
                doc_lines.append("")

            if suite_info["configurations"] != "all":
                doc_lines.extend([
                    "**Configurations**:",
                    ""
                ])
                for config_name in suite_info["configurations"]:
                    config = next(c for c in benchmark_suite["configurations"] if c["name"] == config_name)
                    doc_lines.append(f"- {config['name']}: {config['description']}")
                doc_lines.append("")

        # Usage instructions
        doc_lines.extend([
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
            "python run_comprehensive_suite.py",
            "",
            "",
            "### Analyzing Results",
            "",
            "",
            "python ../tools/performance_analyzer.py suite_results.csv",
            "",
            "",
            "## Expected Results",
            "",
            "### Performance Characteristics by Category",
            "",
            "**Edge Workloads**:",
            "- Lower absolute throughput but good efficiency",
            "- Suitable for power-constrained environments",
            "- PE utilization typically 60-80%",
            "",
            "**Server Workloads**:",
            "- Balanced throughput and efficiency",
            "- Good scaling characteristics",
            "- PE utilization typically 70-90%",
            "",
            "**High-Performance Workloads**:",
            "- Maximum throughput emphasis",
            "- May show scaling bottlenecks",
            "- PE utilization varies widely (50-95%)",
            "",
            "**Stress Test Workloads**:",
            "- May show lower efficiency due to irregular access patterns",
            "- Useful for identifying edge cases",
            "- PE utilization typically 30-60%",
            "",
            "### Configuration Performance Trends",
            "",
            "**Small Arrays (4x4, 8x8)**:",
            "- Good for small workloads",
            "- May underutilize larger workloads",
            "- Lower absolute throughput",
            "",
            "**Medium Arrays (16x16, 32x32)**:",
            "- Balanced performance across workload sizes",
            "- Good efficiency for most applications",
            "- Moderate throughput",
            "",
            "**Large Arrays (64x64, 128x128)**:",
            "- High throughput for large workloads",
            "- May show inefficiency on small workloads",
            "- Memory bandwidth may become limiting",
            "",
            "## Troubleshooting",
            "",
            "### Common Issues",
            "",
            "1. **Incompatible Workload-Configuration Pairs**:",
            "   - For Output Stationary dataflow, workload M must equal array_rows and workload P must equal array_cols",
            "   - Scripts will automatically skip incompatible combinations",
            "",
            "2. **Long Simulation Times**:",
            "   - Large workloads on large arrays can take significant time",
            "   - Consider running smaller test suites first",
            "   - Use the quick_validation suite for development",
            "",
            "3. **Memory Issues**:",
            "   - Very large arrays may require significant memory",
            "   - Monitor system resources during comprehensive runs",
            "",
            "### Performance Optimization Tips",
            "",
            "1. **Workload Selection**:",
            "   - Match workload size to array capabilities",
            "   - Use square or balanced dimensions when possible",
            "",
            "2. **Configuration Tuning**:",
            "   - Buffer sizes should accommodate workload data",
            "   - Consider memory bandwidth limitations",
            "",
            "## References",
            "",
            "- Open Accelerator Documentation",
            "- Systolic Array Literature",
            "- ML Accelerator Benchmarking Best Practices"
        ])

        doc_content = '\n'.join(doc_lines)

        try:
            with open(output_file, 'w') as f:
                f.write(doc_content)
            print(f"Documentation saved to: {output_file}")
        except Exception as e:
            print(f"Error saving documentation: {e}")

    def create_makefile(self, output_dir: str, benchmark_suite: Dict[str, Any]):
        """Create a Makefile for easy benchmark execution."""
        makefile_content = [
            "# Open Accelerator Benchmark Suite Makefile",
            "",
            "# Default target",
            ".PHONY: all clean help quick perf scaling efficiency stress memory comprehensive",
            "",
            "all: help",
            "",
            "help:",
            "\t@echo 'Open Accelerator Benchmark Suite'",
            "\t@echo ''",
            "\t@echo 'Available targets:'",
            "\t@echo '  quick      - Quick validation suite (~5 min)'",
            "\t@echo '  perf       - Performance benchmarking suite (~30 min)'",
            "\t@echo '  scaling    - Scaling analysis suite (~45 min)'",
            "\t@echo '  efficiency - Efficiency analysis suite (~25 min)'",
            "\t@echo '  stress     - Stress testing suite (~20 min)'",
            "\t@echo '  memory     - Memory analysis suite (~40 min)'",
            "\t@echo '  comprehensive - All benchmarks (~180 min)'",
            "\t@echo '  analyze    - Analyze results from last run'",
            "\t@echo '  clean      - Clean up result files'",
            "\t@echo ''",
            "",
            "quick:",
            "\t@echo 'Running quick validation suite...'",
            "\tpython run_quick_validation_suite.py",
            "\t@echo 'Quick validation complete. Results in quick_validation_results.csv'",
            "",
            "perf:",
            "\t@echo 'Running performance benchmarking suite...'",
            "\tpython run_performance_benchmarking_suite.py",
            "\t@echo 'Performance benchmarking complete. Results in performance_benchmarking_results.csv'",
            "",
            "scaling:",
            "\t@echo 'Running scaling analysis suite...'",
            "\tpython run_scaling_analysis_suite.py",
            "\t@echo 'Scaling analysis complete. Results in scaling_analysis_results.csv'",
            "",
            "efficiency:",
            "\t@echo 'Running efficiency analysis suite...'",
            "\tpython run_efficiency_analysis_suite.py",
            "\t@echo 'Efficiency analysis complete. Results in efficiency_analysis_results.csv'",
            "",
            "stress:",
            "\t@echo 'Running stress testing suite...'",
            "\tpython run_stress_testing_suite.py",
            "\t@echo 'Stress testing complete. Results in stress_testing_results.csv'",
            "",
            "memory:",
            "\t@echo 'Running memory analysis suite...'",
            "\tpython run_memory_analysis_suite.py",
            "\t@echo 'Memory analysis complete. Results in memory_analysis_results.csv'",
            "",
            "comprehensive:",
            "\t@echo 'Running comprehensive benchmark suite...'",
            "\t@echo 'This will take approximately 3 hours...'",
            "\tpython run_comprehensive_suite.py",
            "\t@echo 'Comprehensive benchmarking complete. Results in comprehensive_results.csv'",
            "",
            "analyze:",
            "\t@echo 'Analyzing most recent results...'",
            "\t@if [ -f comprehensive_results.csv ]; then \\",
            "\t\tpython ../tools/performance_analyzer.py comprehensive_results.csv -o analysis_results; \\",
            "\telif [ -f performance_benchmarking_results.csv ]; then \\",
            "\t\tpython ../tools/performance_analyzer.py performance_benchmarking_results.csv -o analysis_results; \\",
            "\telif [ -f quick_validation_results.csv ]; then \\",
            "\t\tpython ../tools/performance_analyzer.py quick_validation_results.csv -o analysis_results; \\",
            "\telse \\",
            "\t\techo 'No results files found. Run a benchmark first.'; \\",
            "\tfi",
            "",
            "clean:",
            "\t@echo 'Cleaning up result files...'",
            "\trm -f *_results.csv",
            "\trm -rf analysis_results/",
            "\t@echo 'Cleanup complete.'",
            "",
            "# Individual suite targets with timing",
            "quick-timed:",
            "\t@echo 'Starting timed quick validation suite...'",
            "\ttime python run_quick_validation_suite.py",
            "",
            "perf-timed:",
            "\t@echo 'Starting timed performance benchmarking suite...'",
            "\ttime python run_performance_benchmarking_suite.py",
            "",
            "# Parallel execution (if system supports it)",
            "parallel-quick:",
            "\t@echo 'Running quick suites in parallel...'",
            "\tpython run_quick_validation_suite.py & \\",
            "\tpython run_stress_testing_suite.py & \\",
            "\twait",
            "\t@echo 'Parallel quick suites complete.'"
        ]

        makefile_path = os.path.join(output_dir, "Makefile")
        try:
            with open(makefile_path, 'w') as f:
                f.write('\n'.join(makefile_content))
            print(f"Makefile created: {makefile_path}")
        except Exception as e:
            print(f"Error creating Makefile: {e}")

    def generate_all_outputs(self, output_dir: str = "benchmarks"):
        """Generate all benchmark outputs."""
        print("Generating comprehensive benchmark suite...")
        print("=" * 50)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate benchmark suite
        benchmark_suite = self.generate_comprehensive_benchmark_suite()

        # Save benchmark suite JSON
        suite_file = os.path.join(output_dir, "benchmark_suite.json")
        self.save_benchmark_suite(benchmark_suite, suite_file)

        # Generate simulation scripts
        print("Generating simulation scripts...")
        self.generate_simulation_scripts(output_dir, benchmark_suite)

        # Generate documentation
        print("Generating documentation...")
        doc_file = os.path.join(output_dir, "README.md")
        self.generate_documentation(benchmark_suite, doc_file)

        # Create Makefile
        print("Creating Makefile...")
        self.create_makefile(output_dir, benchmark_suite)

        print(f"\nBenchmark suite generation complete!")
        print(f"Output directory: {output_dir}")
        print(f"Generated files:")
        print(f"  - benchmark_suite.json (benchmark definitions)")
        print(f"  - README.md (documentation)")
        print(f"  - Makefile (easy execution)")
        print(f"  - run_*_suite.py (simulation scripts)")

        print(f"\nTo get started:")
        print(f"  cd {output_dir}")
        print(f"  make help")
        print(f"  make quick    # Run quick validation suite")

def main():
    """Main function for the benchmark generator tool."""
    parser = argparse.ArgumentParser(
        description="Open Accelerator Benchmark Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate complete benchmark suite
  python benchmark_generator.py

  # Generate to specific directory
  python benchmark_generator.py --output-dir my_benchmarks

  # Generate only workloads
  python benchmark_generator.py --workloads-only

  # Generate only configurations
  python benchmark_generator.py --configs-only
        """
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='benchmarks',
        help='Output directory for benchmark suite (default: benchmarks)'
    )

    parser.add_argument(
        '--workloads-only',
        action='store_true',
        help='Generate only workload definitions'
    )

    parser.add_argument(
        '--configs-only',
        action='store_true',
        help='Generate only configuration definitions'
    )

    parser.add_argument(
        '--no-scripts',
        action='store_true',
        help='Skip generation of simulation scripts'
    )

    parser.add_argument(
        '--no-docs',
        action='store_true',
        help='Skip generation of documentation'
    )

    parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'csv'],
        default='json',
        help='Output format for benchmark definitions (default: json)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    try:
        generator = BenchmarkGenerator()

        if args.workloads_only:
            print("Generating workload definitions only...")
            ml_workloads = generator.generate_ml_workloads()
            synthetic_workloads = generator.generate_synthetic_workloads()

            os.makedirs(args.output_dir, exist_ok=True)

            workloads_data = {
                "ml_workloads": [asdict(w) for w in ml_workloads],
                "synthetic_workloads": [asdict(w) for w in synthetic_workloads],
                "total_workloads": len(generator.workloads)
            }

            output_file = os.path.join(args.output_dir, f"workloads.{args.format}")

            if args.format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(workloads_data, f, indent=2)
            elif args.format == 'csv':
                import pandas as pd
                df = pd.DataFrame([asdict(w) for w in generator.workloads])
                df.to_csv(output_file, index=False)

            print(f"Workloads saved to: {output_file}")

        elif args.configs_only:
            print("Generating configuration definitions only...")
            configurations = generator.generate_accelerator_configurations()

            os.makedirs(args.output_dir, exist_ok=True)

            configs_data = {
                "configurations": [asdict(c) for c in configurations],
                "total_configurations": len(configurations)
            }

            output_file = os.path.join(args.output_dir, f"configurations.{args.format}")

            if args.format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(configs_data, f, indent=2)
            elif args.format == 'csv':
                import pandas as pd
                df = pd.DataFrame([asdict(c) for c in configurations])
                df.to_csv(output_file, index=False)

            print(f"Configurations saved to: {output_file}")

        else:
            # Generate complete benchmark suite
            generator.generate_all_outputs(args.output_dir)

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark generation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during benchmark generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
