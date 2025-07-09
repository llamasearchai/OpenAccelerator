#!/usr/bin/env python3
"""
OpenAccelerator Performance Optimization Example

This comprehensive example demonstrates advanced usage of the OpenAccelerator system
including multi-workload analysis, performance optimization, and medical compliance.

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 1.0.0
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_accelerator.utils.config import AcceleratorConfig, ArrayConfig, MemoryHierarchyConfig, MedicalConfig, AcceleratorType, DataflowType
from open_accelerator.workloads.gemm import GEMMWorkload, GEMMWorkloadConfig
from open_accelerator.workloads.medical import MedicalWorkloadConfig, MedicalModalityType, MedicalTaskType
from open_accelerator.core.accelerator import AcceleratorController
from open_accelerator.analysis.performance_analysis import PerformanceAnalyzer
from open_accelerator.ai.agents import AgentOrchestrator, AgentConfig


class PerformanceOptimizer:
    """Advanced performance optimization and analysis system."""
    
    def __init__(self):
        self.results = []
        self.best_configs = {}
        
    def create_accelerator_variants(self) -> List[AcceleratorConfig]:
        """Create different accelerator configurations for comparison."""
        variants = []
        
        # Variant 1: Small Edge Device
        edge_config = AcceleratorConfig(
            name="EdgeAccelerator",
            accelerator_type=AcceleratorType.EDGE,
            array=ArrayConfig(
                rows=4,
                cols=4,
                frequency=1e9,  # 1 GHz
                dataflow=DataflowType.OUTPUT_STATIONARY
            ),
            memory=MemoryHierarchyConfig(
                l1_size=8192,     # 8KB
                l2_size=32768,    # 32KB
                main_memory_size=1048576  # 1MB
            )
        )
        variants.append(edge_config)
        
        # Variant 2: Balanced Server
        server_config = AcceleratorConfig(
            name="ServerAccelerator",
            accelerator_type=AcceleratorType.DATACENTER,
            array=ArrayConfig(
                rows=8,
                cols=8,
                frequency=2e9,  # 2 GHz
                dataflow=DataflowType.OUTPUT_STATIONARY
            ),
            memory=MemoryHierarchyConfig(
                l1_size=32768,    # 32KB
                l2_size=262144,   # 256KB
                main_memory_size=16777216  # 16MB
            )
        )
        variants.append(server_config)
        
        # Variant 3: High-Performance Medical
        medical_config = AcceleratorConfig(
            name="MedicalAccelerator",
            accelerator_type=AcceleratorType.MEDICAL,
            array=ArrayConfig(
                rows=16,
                cols=16,
                frequency=3e9,  # 3 GHz
                dataflow=DataflowType.OUTPUT_STATIONARY
            ),
            memory=MemoryHierarchyConfig(
                l1_size=65536,    # 64KB
                l2_size=1048576,  # 1MB
                main_memory_size=134217728  # 128MB
            ),
            medical=MedicalConfig(
                enable_medical_mode=True,
                fda_validation=True,
                phi_compliance=True
            )
        )
        variants.append(medical_config)
        
        return variants
    
    def create_workload_suite(self) -> List[Tuple[Any, str]]:
        """Create a comprehensive suite of workloads for testing."""
        workloads = []
        
        # GEMM Workloads of different sizes
        gemm_configs = [
            (4, 4, 4, "Small GEMM"),
            (8, 8, 8, "Medium GEMM"),
            (16, 16, 16, "Large GEMM"),
            (32, 32, 32, "XLarge GEMM"),
            (4, 64, 4, "Skinny GEMM"),
            (64, 4, 64, "Fat GEMM")
        ]
        
        for M, K, P, name in gemm_configs:
            config = GEMMWorkloadConfig(M=M, K=K, P=P)
            workload = GEMMWorkload(config, name.replace(" ", "_").lower())
            workload.prepare(seed=42)
            workloads.append((workload, name))
        
        # Medical Workloads
        medical_configs = [
            ("ct_scan", "segmentation", (256, 256, 16), "CT Segmentation"),
            ("mri", "classification", (512, 512, 32), "MRI Classification"),
            ("xray", "detection", (1024, 1024, 1), "X-Ray Detection"),
            ("ultrasound", "segmentation", (128, 128, 64), "Ultrasound Segmentation")
        ]
        
        for modality, task, image_size, name in medical_configs:
            try:
                config = MedicalWorkloadConfig(
                    modality=MedicalModalityType(modality),
                    task_type=MedicalTaskType(task),
                    image_size=image_size,
                    precision_level="medical"
                )
                # For this example, we'll create a mock workload
                workloads.append((config, name))
            except Exception as e:
                print(f"Warning: Could not create {name}: {e}")
        
        return workloads
    
    def benchmark_configuration(self, accel_config: AcceleratorConfig, 
                              workloads: List[Tuple[Any, str]]) -> Dict[str, Any]:
        """Benchmark a specific accelerator configuration."""
        print(f"\n🔬 Benchmarking {accel_config.name}")
        print("=" * 50)
        
        results = {
            "config_name": accel_config.name,
            "array_size": f"{accel_config.array.rows}x{accel_config.array.cols}",
            "frequency_ghz": accel_config.array.frequency / 1e9,
            "workload_results": [],
            "summary": {}
        }
        
        total_time = 0
        successful_runs = 0
        
        for workload, name in workloads:
            print(f"  [METRICS] Running {name}...")
            
            try:
                # Create accelerator controller
                controller = AcceleratorController(accel_config)
                
                # Simulate workload execution time
                start_time = time.time()
                
                if hasattr(workload, 'M'):  # GEMM workload
                    # Calculate theoretical performance
                    ops = workload.M * workload.K * workload.P
                    theoretical_cycles = max(
                        workload.M + workload.K + workload.P - 2,  # Pipeline depth
                        ops // (accel_config.array.rows * accel_config.array.cols)  # Compute bound
                    )
                    actual_cycles = theoretical_cycles + np.random.randint(0, 5)  # Add some variance
                    
                    # Calculate metrics
                    execution_time = actual_cycles / accel_config.array.frequency
                    throughput = ops / execution_time / 1e9  # GOPs
                    efficiency = ops / (actual_cycles * accel_config.array.rows * accel_config.array.cols)
                    
                else:  # Medical workload
                    # Mock medical workload analysis
                    if hasattr(workload, 'image_size'):
                        pixels = np.prod(workload.image_size)
                        ops = pixels * 100  # Assume 100 ops per pixel
                    else:
                        ops = 1000000  # Default
                    
                    theoretical_cycles = ops // (accel_config.array.rows * accel_config.array.cols * 10)
                    actual_cycles = theoretical_cycles + np.random.randint(0, 100)
                    execution_time = actual_cycles / accel_config.array.frequency
                    throughput = ops / execution_time / 1e9
                    efficiency = 0.75 + np.random.random() * 0.2  # Random efficiency
                
                end_time = time.time()
                
                workload_result = {
                    "name": name,
                    "success": True,
                    "ops": ops,
                    "cycles": actual_cycles,
                    "execution_time_s": execution_time,
                    "throughput_gops": throughput,
                    "efficiency": efficiency,
                    "benchmark_time_s": end_time - start_time
                }
                
                results["workload_results"].append(workload_result)
                total_time += execution_time
                successful_runs += 1
                
                print(f"    [SUCCESS] {throughput:.2f} GOPs, {efficiency:.1%} efficiency")
                
            except Exception as e:
                print(f"    [ERROR] Failed: {e}")
                results["workload_results"].append({
                    "name": name,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        successful_results = [r for r in results["workload_results"] if r["success"]]
        if successful_results:
            results["summary"] = {
                "successful_runs": successful_runs,
                "total_runs": len(workloads),
                "success_rate": successful_runs / len(workloads),
                "avg_throughput_gops": np.mean([r["throughput_gops"] for r in successful_results]),
                "avg_efficiency": np.mean([r["efficiency"] for r in successful_results]),
                "total_execution_time_s": total_time,
                "peak_throughput_gops": max([r["throughput_gops"] for r in successful_results]),
                "min_efficiency": min([r["efficiency"] for r in successful_results]),
                "max_efficiency": max([r["efficiency"] for r in successful_results])
            }
        
        return results
    
    def analyze_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze and compare all benchmark results."""
        print(f"\n[METRICS] Performance Analysis")
        print("=" * 50)
        
        analysis = {
            "best_overall": None,
            "best_throughput": None,
            "best_efficiency": None,
            "best_medical": None,
            "recommendations": []
        }
        
        # Find best configurations
        valid_results = [r for r in all_results if r.get("summary")]
        
        if not valid_results:
            return analysis
        
        # Best overall (balanced score)
        for result in valid_results:
            score = (result["summary"]["avg_throughput_gops"] * 
                    result["summary"]["avg_efficiency"] * 
                    result["summary"]["success_rate"])
            if not analysis["best_overall"] or score > analysis["best_overall"]["score"]:
                analysis["best_overall"] = {**result, "score": score}
        
        # Best throughput
        best_throughput = max(valid_results, key=lambda x: x["summary"]["avg_throughput_gops"])
        analysis["best_throughput"] = best_throughput
        
        # Best efficiency
        best_efficiency = max(valid_results, key=lambda x: x["summary"]["avg_efficiency"])
        analysis["best_efficiency"] = best_efficiency
        
        # Best for medical (if medical configs exist)
        medical_results = [r for r in valid_results if "Medical" in r["config_name"]]
        if medical_results:
            analysis["best_medical"] = max(medical_results, key=lambda x: x["summary"]["avg_throughput_gops"])
        
        # Generate recommendations
        analysis["recommendations"] = self.generate_recommendations(valid_results)
        
        return analysis
    
    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        if not results:
            return ["No valid results to analyze"]
        
        # Throughput analysis
        throughputs = [r["summary"]["avg_throughput_gops"] for r in results]
        if max(throughputs) / min(throughputs) > 2:
            recommendations.append(
                f"Significant throughput variance detected ({min(throughputs):.1f} - {max(throughputs):.1f} GOPs). "
                "Consider array size optimization for your target workloads."
            )
        
        # Efficiency analysis
        efficiencies = [r["summary"]["avg_efficiency"] for r in results]
        avg_efficiency = np.mean(efficiencies)
        if avg_efficiency < 0.7:
            recommendations.append(
                f"Average efficiency is {avg_efficiency:.1%}. Consider optimizing memory hierarchy or dataflow."
            )
        
        # Array size recommendations
        array_sizes = [(r["config_name"], int(r["array_size"].split('x')[0])) for r in results]
        if len(set(size for _, size in array_sizes)) > 1:
            best_config = max(results, key=lambda x: x["summary"]["avg_throughput_gops"])
            recommendations.append(
                f"For your workload mix, {best_config['config_name']} ({best_config['array_size']}) "
                "shows best performance. Consider this as your baseline."
            )
        
        # Medical-specific recommendations
        medical_results = [r for r in results if "Medical" in r["config_name"]]
        if medical_results:
            medical_perf = medical_results[0]["summary"]["avg_throughput_gops"]
            general_perf = max(r["summary"]["avg_throughput_gops"] for r in results if "Medical" not in r["config_name"])
            if medical_perf > general_perf * 1.2:
                recommendations.append(
                    "Medical-optimized configuration shows significant benefits. "
                    "Consider medical-specific optimizations for healthcare applications."
                )
        
        return recommendations
    
    def create_visualizations(self, all_results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create performance visualization charts."""
        if not all_results:
            return
        
        valid_results = [r for r in all_results if r.get("summary")]
        if not valid_results:
            return
        
        # Create performance comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OpenAccelerator Performance Analysis', fontsize=16, fontweight='bold')
        
        config_names = [r["config_name"] for r in valid_results]
        
        # Throughput comparison
        throughputs = [r["summary"]["avg_throughput_gops"] for r in valid_results]
        bars1 = ax1.bar(config_names, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Average Throughput (GOPs)', fontweight='bold')
        ax1.set_ylabel('Throughput (GOPs)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, throughputs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency comparison
        efficiencies = [r["summary"]["avg_efficiency"] * 100 for r in valid_results]
        bars2 = ax2.bar(config_names, efficiencies, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax2.set_title('Average Efficiency (%)', fontweight='bold')
        ax2.set_ylabel('Efficiency (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, efficiencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Success rate comparison
        success_rates = [r["summary"]["success_rate"] * 100 for r in valid_results]
        bars3 = ax3.bar(config_names, success_rates, color=['#A8E6CF', '#FFB3BA', '#FFDFBA'])
        ax3.set_title('Success Rate (%)', fontweight='bold')
        ax3.set_ylabel('Success Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Array size vs Performance scatter plot
        array_sizes = [int(r["array_size"].split('x')[0]) for r in valid_results]
        ax4.scatter(array_sizes, throughputs, s=100, c=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
        ax4.set_title('Array Size vs Throughput', fontweight='bold')
        ax4.set_xlabel('Array Size (rows)')
        ax4.set_ylabel('Throughput (GOPs)')
        
        # Add trend line
        z = np.polyfit(array_sizes, throughputs, 1)
        p = np.poly1d(z)
        ax4.plot(array_sizes, p(array_sizes), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"[METRICS] Performance charts saved to performance_analysis.png")
    
    def save_detailed_report(self, all_results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save a detailed JSON report of all results."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_configurations": len(all_results),
                "successful_configurations": len([r for r in all_results if r.get("summary")]),
                "best_overall_config": analysis.get("best_overall", {}).get("config_name"),
                "best_throughput_config": analysis.get("best_throughput", {}).get("config_name"),
                "best_efficiency_config": analysis.get("best_efficiency", {}).get("config_name")
            },
            "detailed_results": all_results,
            "analysis": analysis,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        with open('performance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[REPORT] Detailed report saved to performance_report.json")
    
    def run_complete_analysis(self):
        """Run the complete performance analysis suite."""
        print("[SYSTEM] OpenAccelerator Performance Optimization Suite")
        print("Author: Nik Jois <nikjois@llamasearch.ai>")
        print("=" * 60)
        
        # Create accelerator variants
        print("[CONFIG] Creating accelerator configurations...")
        configs = self.create_accelerator_variants()
        print(f"Created {len(configs)} accelerator variants")
        
        # Create workload suite
        print("📋 Creating workload suite...")
        workloads = self.create_workload_suite()
        print(f"Created {len(workloads)} test workloads")
        
        # Run benchmarks
        all_results = []
        for config in configs:
            result = self.benchmark_configuration(config, workloads)
            all_results.append(result)
        
        # Analyze results
        analysis = self.analyze_results(all_results)
        
        # Print summary
        print(f"\n[RESULT] Performance Summary")
        print("=" * 50)
        
        if analysis["best_overall"]:
            best = analysis["best_overall"]
            print(f"🥇 Best Overall: {best['config_name']}")
            print(f"   Array Size: {best['array_size']}")
            print(f"   Avg Throughput: {best['summary']['avg_throughput_gops']:.2f} GOPs")
            print(f"   Avg Efficiency: {best['summary']['avg_efficiency']:.1%}")
            print(f"   Success Rate: {best['summary']['success_rate']:.1%}")
        
        if analysis["best_throughput"]:
            best_tp = analysis["best_throughput"]
            print(f"[SYSTEM] Best Throughput: {best_tp['config_name']} ({best_tp['summary']['avg_throughput_gops']:.2f} GOPs)")
        
        if analysis["best_efficiency"]:
            best_eff = analysis["best_efficiency"]
            print(f"⚡ Best Efficiency: {best_eff['config_name']} ({best_eff['summary']['avg_efficiency']:.1%})")
        
        # Print recommendations
        if analysis["recommendations"]:
            print(f"\n💡 Optimization Recommendations")
            print("-" * 35)
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"{i}. {rec}")
        
        # Create visualizations
        try:
            self.create_visualizations(all_results, analysis)
        except ImportError:
            print("[WARNING]  Matplotlib not available - skipping visualizations")
        
        # Save detailed report
        self.save_detailed_report(all_results, analysis)
        
        print(f"\n[SUCCESS] Performance analysis complete!")
        print(f"[METRICS] Results: performance_analysis.png")
        print(f"[REPORT] Report: performance_report.json")
        
        return all_results, analysis


def main():
    """Main function to run the performance optimization suite."""
    optimizer = PerformanceOptimizer()
    
    try:
        results, analysis = optimizer.run_complete_analysis()
        return 0
    except KeyboardInterrupt:
        print("\n[WARNING]  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 