#!/usr/bin/env python3
"""
Configuration Generator for Open Accelerator

Generates configuration files for accelerator architectures and workloads
for systematic design space exploration.
"""

import argparse
import json
import os
from typing import Any, Optional

import numpy as np


class ConfigurationGenerator:
    """Generates configuration files for accelerator design space exploration."""

    def __init__(self):
        self.output_dir = "generated_configs"
        self.template_dir = "config_templates"

    def set_output_dir(self, output_dir: str):
        """Set output directory for generated configurations."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_array_size_sweep(
        self,
        min_size: int = 2,
        max_size: int = 64,
        step_type: str = "exponential",
        aspect_ratios: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Generate array size sweep configurations."""

        if aspect_ratios is None:
            aspect_ratios = [1.0, 0.5, 2.0]  # Square, tall, wide

        configurations = []

        if step_type == "exponential":
            # Generate powers of 2
            sizes = [
                2**i
                for i in range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
            ]
        elif step_type == "linear":
            # Generate linear steps
            step = max(1, (max_size - min_size) // 10)
            sizes = list(range(min_size, max_size + 1, step))
        else:
            raise ValueError(f"Unknown step_type: {step_type}")

        config_id = 0
        for total_size in sizes:
            for aspect_ratio in aspect_ratios:
                # Calculate rows and cols based on aspect ratio
                # aspect_ratio = cols / rows
                # total_size = rows * cols
                # rows = sqrt(total_size / aspect_ratio)
                # cols = sqrt(total_size * aspect_ratio)

                rows = int(np.sqrt(total_size / aspect_ratio))
                cols = int(total_size / rows)

                # Ensure we get close to the target size
                actual_size = rows * cols
                if actual_size < min_size or actual_size > max_size * 1.2:
                    continue

                config = {
                    "name": f"array_size_sweep_{config_id:03d}",
                    "description": f"Array size sweep: {rows}x{cols} = {actual_size} PEs (aspect ratio {aspect_ratio})",
                    "category": "array_size_sweep",
                    "parameters": {
                        "array_rows": rows,
                        "array_cols": cols,
                        "array_size": actual_size,
                        "aspect_ratio": aspect_ratio,
                        "pe_mac_latency": 1,
                        "input_buffer_size": max(1024, actual_size * 4),
                        "weight_buffer_size": max(1024, actual_size * 4),
                        "output_buffer_size": max(1024, actual_size * 2),
                        "input_buffer_bandwidth": min(64, actual_size // 4),
                        "weight_buffer_bandwidth": min(64, actual_size // 4),
                        "output_buffer_bandwidth": min(32, actual_size // 8),
                        "data_type": "float32",
                    },
                }
                configurations.append(config)
                config_id += 1

        return configurations

    def generate_buffer_size_sweep(
        self,
        base_array_size: int = 16,
        buffer_size_multipliers: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Generate buffer size sweep configurations."""

        if buffer_size_multipliers is None:
            buffer_size_multipliers = [0.5, 1.0, 2.0, 4.0, 8.0]

        base_rows = int(np.sqrt(base_array_size))
        base_cols = base_array_size // base_rows

        configurations = []

        for i, multiplier in enumerate(buffer_size_multipliers):
            base_input_size = base_array_size * 4
            base_weight_size = base_array_size * 4
            base_output_size = base_array_size * 2

            config = {
                "name": f"buffer_size_sweep_{i:03d}",
                "description": f"Buffer size sweep: {multiplier}x base size",
                "category": "buffer_size_sweep",
                "parameters": {
                    "array_rows": base_rows,
                    "array_cols": base_cols,
                    "array_size": base_array_size,
                    "pe_mac_latency": 1,
                    "input_buffer_size": int(base_input_size * multiplier),
                    "weight_buffer_size": int(base_weight_size * multiplier),
                    "output_buffer_size": int(base_output_size * multiplier),
                    "input_buffer_bandwidth": 16,
                    "weight_buffer_bandwidth": 16,
                    "output_buffer_bandwidth": 8,
                    "buffer_size_multiplier": multiplier,
                    "data_type": "float32",
                },
            }
            configurations.append(config)

        return configurations

    def generate_bandwidth_sweep(
        self,
        base_array_size: int = 16,
        bandwidth_multipliers: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Generate memory bandwidth sweep configurations."""

        if bandwidth_multipliers is None:
            bandwidth_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]

        base_rows = int(np.sqrt(base_array_size))
        base_cols = base_array_size // base_rows

        configurations = []

        for i, multiplier in enumerate(bandwidth_multipliers):
            base_bandwidth = 16

            config = {
                "name": f"bandwidth_sweep_{i:03d}",
                "description": f"Bandwidth sweep: {multiplier}x base bandwidth",
                "category": "bandwidth_sweep",
                "parameters": {
                    "array_rows": base_rows,
                    "array_cols": base_cols,
                    "array_size": base_array_size,
                    "pe_mac_latency": 1,
                    "input_buffer_size": base_array_size * 4,
                    "weight_buffer_size": base_array_size * 4,
                    "output_buffer_size": base_array_size * 2,
                    "input_buffer_bandwidth": max(1, int(base_bandwidth * multiplier)),
                    "weight_buffer_bandwidth": max(1, int(base_bandwidth * multiplier)),
                    "output_buffer_bandwidth": max(
                        1, int(base_bandwidth * multiplier * 0.5)
                    ),
                    "bandwidth_multiplier": multiplier,
                    "data_type": "float32",
                },
            }
            configurations.append(config)

        return configurations

    def generate_latency_sweep(
        self, base_array_size: int = 16, mac_latencies: Optional[list[int]] = None
    ) -> list[dict[str, Any]]:
        """Generate MAC latency sweep configurations."""

        if mac_latencies is None:
            mac_latencies = [1, 2, 3, 4, 5]

        base_rows = int(np.sqrt(base_array_size))
        base_cols = base_array_size // base_rows

        configurations = []

        for i, latency in enumerate(mac_latencies):
            config = {
                "name": f"latency_sweep_{i:03d}",
                "description": f"MAC latency sweep: {latency} cycles",
                "category": "latency_sweep",
                "parameters": {
                    "array_rows": base_rows,
                    "array_cols": base_cols,
                    "array_size": base_array_size,
                    "pe_mac_latency": latency,
                    "input_buffer_size": base_array_size * 4,
                    "weight_buffer_size": base_array_size * 4,
                    "output_buffer_size": base_array_size * 2,
                    "input_buffer_bandwidth": 16,
                    "weight_buffer_bandwidth": 16,
                    "output_buffer_bandwidth": 8,
                    "data_type": "float32",
                },
            }
            configurations.append(config)

        return configurations

    def generate_workload_suite(
        self,
        complexity_levels: Optional[list[str]] = None,
        problem_sizes: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Generate workload suite configurations."""

        if complexity_levels is None:
            complexity_levels = ["small", "medium", "large"]

        if problem_sizes is None:
            problem_sizes = ["square", "tall", "wide"]

        workload_templates = {
            "small": {
                "square": {"M": 4, "K": 4, "P": 4},
                "tall": {"M": 8, "K": 4, "P": 4},
                "wide": {"M": 4, "K": 4, "P": 8},
            },
            "medium": {
                "square": {"M": 16, "K": 16, "P": 16},
                "tall": {"M": 32, "K": 16, "P": 16},
                "wide": {"M": 16, "K": 16, "P": 32},
            },
            "large": {
                "square": {"M": 64, "K": 64, "P": 64},
                "tall": {"M": 128, "K": 64, "P": 64},
                "wide": {"M": 64, "K": 64, "P": 128},
            },
        }

        workloads = []
        workload_id = 0

        for complexity in complexity_levels:
            for size_type in problem_sizes:
                dims = workload_templates[complexity][size_type]

                workload = {
                    "name": f"gemm_{complexity}_{size_type}_{workload_id:03d}",
                    "description": f"GEMM {complexity} {size_type}: {dims['M']}x{dims['K']} * {dims['K']}x{dims['P']}",
                    "category": f"gemm_{complexity}",
                    "type": "GEMM",
                    "parameters": {
                        "gemm_M": dims["M"],
                        "gemm_K": dims["K"],
                        "gemm_P": dims["P"],
                        "data_generation_seed": 42 + workload_id,
                        "complexity_level": complexity,
                        "size_type": size_type,
                    },
                }
                workloads.append(workload)
                workload_id += 1

        return workloads

    def generate_neural_network_workloads(self) -> list[dict[str, Any]]:
        """Generate neural network-inspired workload configurations."""

        # Common layer sizes from popular networks
        layer_configs = [
            # AlexNet-inspired
            {"name": "alexnet_conv1", "M": 55 * 55, "K": 11 * 11 * 3, "P": 96},
            {"name": "alexnet_conv2", "M": 27 * 27, "K": 5 * 5 * 96, "P": 256},
            {"name": "alexnet_fc1", "M": 1, "K": 9216, "P": 4096},
            {"name": "alexnet_fc2", "M": 1, "K": 4096, "P": 4096},
            # VGG-inspired
            {"name": "vgg_conv1", "M": 224 * 224, "K": 3 * 3 * 3, "P": 64},
            {"name": "vgg_conv2", "M": 112 * 112, "K": 3 * 3 * 64, "P": 128},
            {"name": "vgg_conv3", "M": 56 * 56, "K": 3 * 3 * 128, "P": 256},
            # ResNet-inspired
            {"name": "resnet_conv1", "M": 56 * 56, "K": 1 * 1 * 64, "P": 64},
            {"name": "resnet_conv2", "M": 56 * 56, "K": 3 * 3 * 64, "P": 64},
            {"name": "resnet_conv3", "M": 28 * 28, "K": 1 * 1 * 128, "P": 128},
            {"name": "resnet_conv4", "M": 14 * 14, "K": 3 * 3 * 256, "P": 256},
            # MobileNet-inspired (depthwise separable)
            {"name": "mobilenet_dw1", "M": 112 * 112, "K": 3 * 3 * 1, "P": 32},
            {"name": "mobilenet_pw1", "M": 112 * 112, "K": 1 * 1 * 32, "P": 64},
            {"name": "mobilenet_dw2", "M": 56 * 56, "K": 3 * 3 * 1, "P": 64},
            {"name": "mobilenet_pw2", "M": 56 * 56, "K": 1 * 1 * 64, "P": 128},
            # Transformer-inspired (scaled down)
            {
                "name": "transformer_qkv",
                "M": 512,
                "K": 768,
                "P": 2304,
            },  # Q, K, V projection
            {
                "name": "transformer_attn",
                "M": 512,
                "K": 512,
                "P": 768,
            },  # Attention output
            {
                "name": "transformer_ffn1",
                "M": 512,
                "K": 768,
                "P": 3072,
            },  # Feed-forward 1
            {
                "name": "transformer_ffn2",
                "M": 512,
                "K": 3072,
                "P": 768,
            },  # Feed-forward 2
        ]

        workloads = []

        for config in layer_configs:
            # Scale down large dimensions to be manageable for simulation
            M = min(config["M"], 1024)  # Cap at 1024
            K = min(config["K"], 1024)
            P = min(config["P"], 1024)

            # Ensure dimensions are reasonable for array mapping
            M = max(M, 1)
            K = max(K, 1)
            P = max(P, 1)

            workload = {
                "name": f"nn_{config['name']}",
                "description": f"Neural Network layer: {config['name']} ({M}x{K} * {K}x{P})",
                "category": "neural_network",
                "type": "GEMM",
                "parameters": {
                    "gemm_M": M,
                    "gemm_K": K,
                    "gemm_P": P,
                    "data_generation_seed": 42,
                    "layer_type": config["name"],
                    "original_M": config["M"],
                    "original_K": config["K"],
                    "original_P": config["P"],
                },
            }
            workloads.append(workload)

        return workloads

    def generate_design_space_exploration(
        self,
        array_sizes: Optional[list[int]] = None,
        workload_complexities: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Generate comprehensive design space exploration configurations."""

        if array_sizes is None:
            array_sizes = [4, 16, 64]  # Small, medium, large arrays

        if workload_complexities is None:
            workload_complexities = ["small", "medium", "large"]

        # Generate combinations of array architectures and workloads
        arch_configs = []
        workload_configs = []

        # Generate architecture configurations
        for size in array_sizes:
            rows = int(np.sqrt(size))
            cols = size // rows

            # Base configuration
            base_config = {
                "array_rows": rows,
                "array_cols": cols,
                "array_size": size,
                "pe_mac_latency": 1,
                "input_buffer_size": size * 4,
                "weight_buffer_size": size * 4,
                "output_buffer_size": size * 2,
                "input_buffer_bandwidth": min(32, size // 2),
                "weight_buffer_bandwidth": min(32, size // 2),
                "output_buffer_bandwidth": min(16, size // 4),
                "data_type": "float32",
            }

            # Variants with different buffer sizes and bandwidths
            variants = [
                {"suffix": "baseline", "multiplier": 1.0},
                {"suffix": "large_buffers", "multiplier": 4.0},
                {"suffix": "small_buffers", "multiplier": 0.5},
            ]

            for variant in variants:
                config = base_config.copy()
                mult = variant["multiplier"]

                config.update(
                    {
                        "input_buffer_size": int(config["input_buffer_size"] * mult),
                        "weight_buffer_size": int(config["weight_buffer_size"] * mult),
                        "output_buffer_size": int(config["output_buffer_size"] * mult),
                    }
                )

                arch_config = {
                    "name": f"arch_{size}pe_{variant['suffix']}",
                    "description": f"Architecture: {rows}x{cols} array, {variant['suffix']} configuration",
                    "category": "dse_architecture",
                    "parameters": config,
                }
                arch_configs.append(arch_config)

        # Generate workload configurations
        workload_templates = {
            "small": [
                {"M": 4, "K": 4, "P": 4},
                {"M": 8, "K": 4, "P": 4},
                {"M": 4, "K": 8, "P": 4},
            ],
            "medium": [
                {"M": 16, "K": 16, "P": 16},
                {"M": 32, "K": 16, "P": 16},
                {"M": 16, "K": 32, "P": 16},
            ],
            "large": [
                {"M": 64, "K": 64, "P": 64},
                {"M": 128, "K": 64, "P": 64},
                {"M": 64, "K": 128, "P": 64},
            ],
        }

        for complexity in workload_complexities:
            for i, dims in enumerate(workload_templates[complexity]):
                workload = {
                    "name": f"workload_{complexity}_{i}",
                    "description": f"Workload: {complexity} GEMM {dims['M']}x{dims['K']}x{dims['P']}",
                    "category": "dse_workload",
                    "type": "GEMM",
                    "parameters": {
                        "gemm_M": dims["M"],
                        "gemm_K": dims["K"],
                        "gemm_P": dims["P"],
                        "data_generation_seed": 42,
                        "complexity": complexity,
                    },
                }
                workload_configs.append(workload)

        return {"architectures": arch_configs, "workloads": workload_configs}

    def generate_optimization_study_configs(self) -> list[dict[str, Any]]:
        """Generate configurations for optimization studies (Pareto analysis)."""

        # Define optimization objectives and constraints
        optimization_points = [
            # High performance (large arrays, high bandwidth)
            {
                "name": "high_performance",
                "array_size": 64,
                "buffer_multiplier": 4.0,
                "bandwidth_multiplier": 4.0,
                "objective": "maximize_throughput",
            },
            # Balanced (medium arrays, balanced resources)
            {
                "name": "balanced",
                "array_size": 16,
                "buffer_multiplier": 2.0,
                "bandwidth_multiplier": 2.0,
                "objective": "balance_performance_efficiency",
            },
            # Efficient (smaller arrays, optimized utilization)
            {
                "name": "high_efficiency",
                "array_size": 8,
                "buffer_multiplier": 1.0,
                "bandwidth_multiplier": 1.0,
                "objective": "maximize_efficiency",
            },
            # Resource constrained
            {
                "name": "resource_constrained",
                "array_size": 4,
                "buffer_multiplier": 0.5,
                "bandwidth_multiplier": 0.5,
                "objective": "minimize_resources",
            },
        ]

        configurations = []

        for point in optimization_points:
            rows = int(np.sqrt(point["array_size"]))
            cols = point["array_size"] // rows

            base_buffer_size = point["array_size"] * 4
            base_bandwidth = 16

            config = {
                "name": f"optimization_{point['name']}",
                "description": f"Optimization study: {point['objective']} ({rows}x{cols} array)",
                "category": "optimization_study",
                "objective": point["objective"],
                "parameters": {
                    "array_rows": rows,
                    "array_cols": cols,
                    "array_size": point["array_size"],
                    "pe_mac_latency": 1,
                    "input_buffer_size": int(
                        base_buffer_size * point["buffer_multiplier"]
                    ),
                    "weight_buffer_size": int(
                        base_buffer_size * point["buffer_multiplier"]
                    ),
                    "output_buffer_size": int(
                        base_buffer_size * point["buffer_multiplier"] * 0.5
                    ),
                    "input_buffer_bandwidth": int(
                        base_bandwidth * point["bandwidth_multiplier"]
                    ),
                    "weight_buffer_bandwidth": int(
                        base_bandwidth * point["bandwidth_multiplier"]
                    ),
                    "output_buffer_bandwidth": int(
                        base_bandwidth * point["bandwidth_multiplier"] * 0.5
                    ),
                    "data_type": "float32",
                },
            }
            configurations.append(config)

        return configurations

    def save_configurations(self, configurations: list[dict[str, Any]], filename: str):
        """Save configurations to JSON file."""
        output_path = os.path.join(self.output_dir, filename)

        try:
            with open(output_path, "w") as f:
                json.dump(configurations, f, indent=2)
            print(f"Saved {len(configurations)} configurations to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving configurations: {e}")
            return None

    def save_combined_configuration(
        self,
        arch_configs: list[dict[str, Any]],
        workload_configs: list[dict[str, Any]],
        filename: str,
    ):
        """Save combined architecture and workload configurations."""
        combined_config = {
            "metadata": {
                "generator_version": "1.0",
                "generation_timestamp": str(np.datetime64("now")),
                "total_architectures": len(arch_configs),
                "total_workloads": len(workload_configs),
                "total_combinations": len(arch_configs) * len(workload_configs),
            },
            "architectures": arch_configs,
            "workloads": workload_configs,
        }

        output_path = os.path.join(self.output_dir, filename)

        try:
            with open(output_path, "w") as f:
                json.dump(combined_config, f, indent=2)
            print(f"Saved combined configuration to: {output_path}")
            print(f"  Architectures: {len(arch_configs)}")
            print(f"  Workloads: {len(workload_configs)}")
            print(f"  Total combinations: {len(arch_configs) * len(workload_configs)}")
            return output_path
        except Exception as e:
            print(f"Error saving combined configuration: {e}")
            return None

    def generate_template_configs(self):
        """Generate template configuration files for common use cases."""

        templates = {
            "quick_test": {
                "description": "Quick test configuration for development",
                "architectures": self.generate_array_size_sweep(2, 8, "linear", [1.0]),
                "workloads": self.generate_workload_suite(["small"], ["square"]),
            },
            "performance_sweep": {
                "description": "Performance-focused parameter sweep",
                "architectures": self.generate_array_size_sweep(4, 64, "exponential"),
                "workloads": self.generate_workload_suite(
                    ["small", "medium"], ["square", "tall", "wide"]
                ),
            },
            "memory_analysis": {
                "description": "Memory subsystem analysis",
                "architectures": (
                    self.generate_buffer_size_sweep() + self.generate_bandwidth_sweep()
                ),
                "workloads": self.generate_workload_suite(["medium"], ["square"]),
            },
            "neural_network_study": {
                "description": "Neural network workload characterization",
                "architectures": self.generate_array_size_sweep(
                    8, 32, "exponential", [1.0, 0.5, 2.0]
                ),
                "workloads": self.generate_neural_network_workloads(),
            },
            "design_space_exploration": {
                "description": "Comprehensive design space exploration",
                "combined": self.generate_design_space_exploration(),
            },
            "optimization_study": {
                "description": "Multi-objective optimization study",
                "architectures": self.generate_optimization_study_configs(),
                "workloads": self.generate_workload_suite(
                    ["small", "medium"], ["square"]
                ),
            },
        }

        for template_name, template_data in templates.items():
            if "combined" in template_data:
                # Special handling for combined configs
                combined = template_data["combined"]
                self.save_combined_configuration(
                    combined["architectures"],
                    combined["workloads"],
                    f"{template_name}.json",
                )
            else:
                # Separate architecture and workload configs
                self.save_combined_configuration(
                    template_data["architectures"],
                    template_data["workloads"],
                    f"{template_name}.json",
                )

    def validate_configuration(self, config: dict[str, Any]) -> list[str]:
        """Validate a configuration and return list of issues."""
        issues = []

        if "parameters" not in config:
            issues.append("Missing 'parameters' section")
            return issues

        params = config["parameters"]

        # Check required parameters
        required_params = ["array_rows", "array_cols"]
        for param in required_params:
            if param not in params:
                issues.append(f"Missing required parameter: {param}")

        # Check parameter ranges
        if "array_rows" in params and params["array_rows"] <= 0:
            issues.append("array_rows must be positive")

        if "array_cols" in params and params["array_cols"] <= 0:
            issues.append("array_cols must be positive")

        if "pe_mac_latency" in params and params["pe_mac_latency"] <= 0:
            issues.append("pe_mac_latency must be positive")

        # Check buffer sizes
        buffer_params = [
            "input_buffer_size",
            "weight_buffer_size",
            "output_buffer_size",
        ]
        for param in buffer_params:
            if param in params and params[param] <= 0:
                issues.append(f"{param} must be positive")

        # Check bandwidth parameters
        bandwidth_params = [
            "input_buffer_bandwidth",
            "weight_buffer_bandwidth",
            "output_buffer_bandwidth",
        ]
        for param in bandwidth_params:
            if param in params and params[param] <= 0:
                issues.append(f"{param} must be positive")

        # Check data type
        if "data_type" in params:
            valid_types = ["float32", "float64", "int32", "int64"]
            if params["data_type"] not in valid_types:
                issues.append(f"data_type must be one of: {valid_types}")

        # Check logical consistency
        if "array_rows" in params and "array_cols" in params and "array_size" in params:
            expected_size = params["array_rows"] * params["array_cols"]
            if params["array_size"] != expected_size:
                issues.append(
                    f"array_size ({params['array_size']}) doesn't match array_rows * array_cols ({expected_size})"
                )

        return issues

    def validate_workload(self, workload: dict[str, Any]) -> list[str]:
        """Validate a workload configuration."""
        issues = []

        if "parameters" not in workload:
            issues.append("Missing 'parameters' section")
            return issues

        params = workload["parameters"]

        if workload.get("type") == "GEMM":
            # Check GEMM-specific parameters
            gemm_params = ["gemm_M", "gemm_K", "gemm_P"]
            for param in gemm_params:
                if param not in params:
                    issues.append(f"Missing GEMM parameter: {param}")
                elif params[param] <= 0:
                    issues.append(f"GEMM parameter {param} must be positive")

        return issues


def main():
    """Main function for configuration generator."""
    parser = argparse.ArgumentParser(
        description="Configuration Generator for Open Accelerator Design Space Exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all template configurations
  python config_generator.py --generate-templates

  # Generate array size sweep
  python config_generator.py --array-sweep --min-size 4 --max-size 64

  # Generate buffer size sweep
  python config_generator.py --buffer-sweep --base-array-size 16

  # Generate neural network workloads
  python config_generator.py --neural-network-workloads

  # Generate comprehensive design space exploration
  python config_generator.py --design-space-exploration

  # Generate custom configuration with specific parameters
  python config_generator.py --custom --array-sizes 4,16,64 --workload-complexities small,medium
        """,
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="generated_configs",
        help="Output directory for generated configurations (default: generated_configs)",
    )

    # Generation modes
    parser.add_argument(
        "--generate-templates",
        action="store_true",
        help="Generate all template configurations",
    )

    parser.add_argument(
        "--array-sweep",
        action="store_true",
        help="Generate array size sweep configurations",
    )

    parser.add_argument(
        "--buffer-sweep",
        action="store_true",
        help="Generate buffer size sweep configurations",
    )

    parser.add_argument(
        "--bandwidth-sweep",
        action="store_true",
        help="Generate memory bandwidth sweep configurations",
    )

    parser.add_argument(
        "--latency-sweep",
        action="store_true",
        help="Generate MAC latency sweep configurations",
    )

    parser.add_argument(
        "--neural-network-workloads",
        action="store_true",
        help="Generate neural network workload configurations",
    )

    parser.add_argument(
        "--design-space-exploration",
        action="store_true",
        help="Generate comprehensive design space exploration configurations",
    )

    parser.add_argument(
        "--optimization-study",
        action="store_true",
        help="Generate multi-objective optimization study configurations",
    )

    parser.add_argument(
        "--custom",
        action="store_true",
        help="Generate custom configuration based on specified parameters",
    )

    # Array sweep parameters
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Minimum array size for sweep (default: 2)",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=64,
        help="Maximum array size for sweep (default: 64)",
    )

    parser.add_argument(
        "--step-type",
        choices=["linear", "exponential"],
        default="exponential",
        help="Step type for array size sweep (default: exponential)",
    )

    parser.add_argument(
        "--aspect-ratios",
        type=str,
        default="1.0,0.5,2.0",
        help="Comma-separated aspect ratios for arrays (default: 1.0,0.5,2.0)",
    )

    # Buffer sweep parameters
    parser.add_argument(
        "--base-array-size",
        type=int,
        default=16,
        help="Base array size for buffer/bandwidth sweeps (default: 16)",
    )

    parser.add_argument(
        "--buffer-multipliers",
        type=str,
        default="0.5,1.0,2.0,4.0,8.0",
        help="Comma-separated buffer size multipliers (default: 0.5,1.0,2.0,4.0,8.0)",
    )

    parser.add_argument(
        "--bandwidth-multipliers",
        type=str,
        default="0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated bandwidth multipliers (default: 0.25,0.5,1.0,2.0,4.0)",
    )

    # Latency sweep parameters
    parser.add_argument(
        "--mac-latencies",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated MAC latencies in cycles (default: 1,2,3,4,5)",
    )

    # Custom configuration parameters
    parser.add_argument(
        "--array-sizes",
        type=str,
        help="Comma-separated array sizes for custom configuration",
    )

    parser.add_argument(
        "--workload-complexities",
        type=str,
        default="small,medium,large",
        help="Comma-separated workload complexities (default: small,medium,large)",
    )

    # Validation and utility options
    parser.add_argument(
        "--validate", type=str, help="Validate an existing configuration file"
    )

    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available template configurations",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = ConfigurationGenerator()
    generator.set_output_dir(args.output_dir)

    if args.verbose:
        print("Configuration Generator initialized")
        print(f"Output directory: {args.output_dir}")

    # List templates
    if args.list_templates:
        print("Available template configurations:")
        templates = [
            "quick_test - Quick test configuration for development",
            "performance_sweep - Performance-focused parameter sweep",
            "memory_analysis - Memory subsystem analysis",
            "neural_network_study - Neural network workload characterization",
            "design_space_exploration - Comprehensive design space exploration",
            "optimization_study - Multi-objective optimization study",
        ]
        for template in templates:
            print(f"  {template}")
        return 0

    # Validate existing configuration
    if args.validate:
        if not os.path.exists(args.validate):
            print(f"Error: Configuration file not found: {args.validate}")
            return 1

        try:
            with open(args.validate) as f:
                config_data = json.load(f)

            print(f"Validating configuration file: {args.validate}")

            if isinstance(config_data, list):
                # Multiple configurations
                total_issues = 0
                for i, config in enumerate(config_data):
                    issues = generator.validate_configuration(config)
                    if issues:
                        print(f"Configuration {i}: {len(issues)} issues")
                        for issue in issues:
                            print(f"  - {issue}")
                        total_issues += len(issues)
                    elif args.verbose:
                        print(f"Configuration {i}: OK")

                print(f"Validation complete: {total_issues} total issues found")

            elif isinstance(config_data, dict):
                if "architectures" in config_data:
                    # Combined configuration file
                    print("Validating architectures...")
                    arch_issues = 0
                    for i, arch in enumerate(config_data["architectures"]):
                        issues = generator.validate_configuration(arch)
                        arch_issues += len(issues)
                        if issues and args.verbose:
                            print(f"Architecture {i}: {len(issues)} issues")
                            for issue in issues:
                                print(f"  - {issue}")

                    print("Validating workloads...")
                    workload_issues = 0
                    for i, workload in enumerate(config_data.get("workloads", [])):
                        issues = generator.validate_workload(workload)
                        workload_issues += len(issues)
                        if issues and args.verbose:
                            print(f"Workload {i}: {len(issues)} issues")
                            for issue in issues:
                                print(f"  - {issue}")

                    print(
                        f"Validation complete: {arch_issues} architecture issues, {workload_issues} workload issues"
                    )

                else:
                    # Single configuration
                    issues = generator.validate_configuration(config_data)
                    if issues:
                        print(f"Configuration has {len(issues)} issues:")
                        for issue in issues:
                            print(f"  - {issue}")
                    else:
                        print("Configuration is valid")

            return 0

        except Exception as e:
            print(f"Error validating configuration: {e}")
            return 1

    # Parse comma-separated parameters
    def parse_float_list(s):
        return [float(x.strip()) for x in s.split(",")]

    def parse_int_list(s):
        return [int(x.strip()) for x in s.split(",")]

    def parse_str_list(s):
        return [x.strip() for x in s.split(",")]

    configurations_generated = []

    # Generate configurations based on selected modes
    if args.generate_templates:
        if args.verbose:
            print("Generating template configurations...")
        generator.generate_template_configs()
        print("All template configurations generated")
        return 0

    if args.array_sweep:
        if args.verbose:
            print("Generating array size sweep configurations...")

        aspect_ratios = parse_float_list(args.aspect_ratios)
        configs = generator.generate_array_size_sweep(
            args.min_size, args.max_size, args.step_type, aspect_ratios
        )
        configurations_generated.extend(configs)
        print(f"Generated {len(configs)} array size sweep configurations")

    if args.buffer_sweep:
        if args.verbose:
            print("Generating buffer size sweep configurations...")

        buffer_multipliers = parse_float_list(args.buffer_multipliers)
        configs = generator.generate_buffer_size_sweep(
            args.base_array_size, buffer_multipliers
        )
        configurations_generated.extend(configs)
        print(f"Generated {len(configs)} buffer size sweep configurations")

    if args.bandwidth_sweep:
        if args.verbose:
            print("Generating bandwidth sweep configurations...")

        bandwidth_multipliers = parse_float_list(args.bandwidth_multipliers)
        configs = generator.generate_bandwidth_sweep(
            args.base_array_size, bandwidth_multipliers
        )
        configurations_generated.extend(configs)
        print(f"Generated {len(configs)} bandwidth sweep configurations")

    if args.latency_sweep:
        if args.verbose:
            print("Generating MAC latency sweep configurations...")

        mac_latencies = parse_int_list(args.mac_latencies)
        configs = generator.generate_latency_sweep(args.base_array_size, mac_latencies)
        configurations_generated.extend(configs)
        print(f"Generated {len(configs)} MAC latency sweep configurations")

    if args.neural_network_workloads:
        if args.verbose:
            print("Generating neural network workload configurations...")

        workloads = generator.generate_neural_network_workloads()
        generator.save_configurations(workloads, "neural_network_workloads.json")
        print(f"Generated {len(workloads)} neural network workload configurations")

    if args.design_space_exploration:
        if args.verbose:
            print("Generating design space exploration configurations...")

        dse_configs = generator.generate_design_space_exploration()
        generator.save_combined_configuration(
            dse_configs["architectures"],
            dse_configs["workloads"],
            "design_space_exploration.json",
        )
        print("Generated design space exploration configurations")

    if args.optimization_study:
        if args.verbose:
            print("Generating optimization study configurations...")

        opt_configs = generator.generate_optimization_study_configs()
        workloads = generator.generate_workload_suite(["small", "medium"], ["square"])
        generator.save_combined_configuration(
            opt_configs, workloads, "optimization_study.json"
        )
        print("Generated optimization study configurations")

    if args.custom:
        if args.verbose:
            print("Generating custom configuration...")

        array_sizes = parse_int_list(args.array_sizes)
        workload_complexities = parse_str_list(args.workload_complexities)

        dse_configs = generator.generate_design_space_exploration(
            array_sizes, workload_complexities
        )
        generator.save_combined_configuration(
            dse_configs["architectures"], dse_configs["workloads"], "custom_config.json"
        )
        print("Generated custom configuration")
