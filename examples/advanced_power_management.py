"""
Advanced power management example for the Open Accelerator simulator.

This example demonstrates comprehensive power management features including
DVFS, thermal management, power gating, and power optimization.
"""

import matplotlib.pyplot as plt
import numpy as np

from open_accelerator.core.power_management import (
    PowerAnalyzer,
    PowerLogger,
    PowerOptimizer,
    create_automotive_power_config,
    create_edge_power_config,
    create_power_report,
    integrate_power_management,
)
from open_accelerator.simulation import Simulator
from open_accelerator.utils import AcceleratorConfig, WorkloadConfig
from open_accelerator.workloads import GEMMWorkload


def run_power_management_comparison():
    """Compare different power management configurations."""

    print("=== Advanced Power Management Example ===\n")

    # Define test configurations
    configs = {
        "automotive": create_automotive_power_config(),
        "edge": create_edge_power_config(),
    }

    # Test workload: Medium-sized GEMM
    M, K, P = 8, 16, 8
    accel_config = AcceleratorConfig(
        array_rows=M,
        array_cols=P,
        pe_mac_latency=1,
        input_buffer_size=M * K * 4,
        weight_buffer_size=K * P * 4,
        output_buffer_size=M * P * 4,
        data_type=np.float32,
    )

    workload_config = WorkloadConfig(gemm_M=M, gemm_K=K, gemm_P=P)
    workload = GEMMWorkload(workload_config, accel_config)
    workload.generate_data(seed=42)

    results = {}

    for config_name, power_config in configs.items():
        print(f"--- Testing {config_name.title()} Power Configuration ---")

        # Create power manager
        power_manager = integrate_power_management(accel_config, power_config)

        # Create power analyzer and logger
        power_analyzer = PowerAnalyzer()
        power_logger = PowerLogger(f"power_log_{config_name}.csv")

        # Run simulation with power management
        simulator = Simulator(accel_config, workload)

        # Simulate with power monitoring
        print("Running simulation with power monitoring...")
        cycle = 0
        while simulator.step():
            # Update component utilizations (example values)
            systolic_util = (
                min(1.0, cycle / 50.0)
                if cycle < 100
                else max(0.1, 1.0 - (cycle - 100) / 50.0)
            )
            memory_util = 0.6 if cycle % 10 < 7 else 0.2
            control_util = 0.3
            io_util = 0.1 if cycle % 20 < 5 else 0.05

            power_manager.update_component_utilization("systolic_array", systolic_util)
            power_manager.update_component_utilization("memory", memory_util)
            power_manager.update_component_utilization("control", control_util)
            power_manager.update_component_utilization("io", io_util)

            # Update power management
            power_manager.cycle_update()

            # Log power state
            power_logger.log_power_state(power_manager, cycle)

            # Add sample to analyzer
            metrics = power_manager.get_power_metrics()
            avg_util = (systolic_util + memory_util + control_util + io_util) / 4
            power_analyzer.add_sample(
                metrics.current_power_watts, avg_util, metrics.current_temperature_c
            )

            cycle += 1

        # Finalize logging
        power_logger.finalize()

        # Get results
        final_status = power_manager.get_power_status()
        power_stats = power_analyzer.get_power_statistics()
        efficiency_report = power_analyzer.get_power_efficiency_report()

        results[config_name] = {
            "final_status": final_status,
            "power_stats": power_stats,
            "efficiency_report": efficiency_report,
            "power_manager": power_manager,
        }

        # Print summary
        print(f"  Average Power: {power_stats['power_stats']['mean_watts']:.2f}W")
        print(f"  Peak Power: {power_stats['power_stats']['max_watts']:.2f}W")
        print(
            f"  Energy Efficiency: {final_status['current_metrics']['energy_efficiency_tops_per_watt']:.2f} TOPS/W"
        )
        print(
            f"  Power Utilization: {final_status['current_metrics']['power_watts']:.2f}W "
            f"({final_status['budget_status']['budget_utilization_percent']:.1f}% of budget)"
        )
        print()

    # Generate comparison report
    print("--- Power Management Comparison ---")

    for config_name, result in results.items():
        power_stats = result["power_stats"]
        final_status = result["final_status"]

        print(f"\n{config_name.title()} Configuration:")
        print(
            f"  Total Energy: {final_status['current_metrics']['energy_joules']:.3f}J"
        )
        print(f"  Average Power: {power_stats['power_stats']['mean_watts']:.2f}W")
        print(
            f"  Power Efficiency: {final_status['current_metrics']['energy_efficiency_tops_per_watt']:.1f} TOPS/W"
        )
        print(f"  DVFS Transitions: {final_status['dvfs_status']['transition_count']}")
        print(
            f"  Power Gate Events: {final_status['power_gating_status']['gate_events']}"
        )
        print(f"  Thermal State: {final_status['thermal_status']['thermal_state']}")

    # Create detailed reports
    for config_name, result in results.items():
        report_file = f"power_report_{config_name}.json"
        power_report = create_power_report(result["power_manager"], report_file)
        print(f"Detailed power report saved to {report_file}")

    return results


def demonstrate_power_optimization():
    """Demonstrate power optimization capabilities."""

    print("\n=== Power Optimization Demonstration ===\n")

    # Create automotive power configuration
    power_config = create_automotive_power_config()

    # Create accelerator configuration
    accel_config = AcceleratorConfig(
        array_rows=16, array_cols=16, pe_mac_latency=1, data_type=np.float32
    )

    # Create power manager and optimizer
    power_manager = integrate_power_management(accel_config, power_config)
    power_optimizer = PowerOptimizer(power_config)

    # Example workload characteristics
    workload_chars = {
        "compute_intensity": 0.7,
        "memory_intensity": 0.4,
        "parallelism": 0.8,
        "duration_seconds": 5.0,
    }

    # Optimize for energy efficiency
    energy_recommendations = power_optimizer.optimize_for_energy_efficiency(
        workload_chars
    )
    print("Energy Efficiency Optimization Recommendations:")
    for key, value in energy_recommendations.items():
        print(f"  {key}: {value}")

    # Performance requirements
    perf_requirements = {
        "target_throughput_tops": 200.0,
        "max_latency_ms": 5.0,
        "power_budget_watts": 400.0,
    }

    # Optimize for performance
    performance_recommendations = power_optimizer.optimize_for_performance(
        perf_requirements
    )
    print("\nPerformance Optimization Recommendations:")
    for key, value in performance_recommendations.items():
        print(f"  {key}: {value}")

    # Show optimization history
    optimization_history = power_optimizer.get_optimization_history()
    print(f"\nOptimization History: {len(optimization_history)} decisions recorded")


def visualize_power_consumption():
    """Create power consumption visualizations."""

    print("\n=== Power Consumption Visualization ===\n")

    # Simulate power consumption over time
    time_points = np.linspace(0, 10, 100)

    # Simulate different components' power consumption
    systolic_power = (
        100 + 50 * np.sin(0.5 * time_points) + 10 * np.random.randn(len(time_points))
    )
    memory_power = (
        30 + 15 * np.cos(0.3 * time_points) + 5 * np.random.randn(len(time_points))
    )
    control_power = (
        20 + 5 * np.sin(0.8 * time_points) + 2 * np.random.randn(len(time_points))
    )
    io_power = (
        10 + 3 * np.cos(1.2 * time_points) + 1 * np.random.randn(len(time_points))
    )

    total_power = systolic_power + memory_power + control_power + io_power

    # Create comprehensive power visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Power Consumption Analysis", fontsize=16)

    # Component power breakdown over time
    axes[0, 0].plot(time_points, systolic_power, label="Systolic Array", linewidth=2)
    axes[0, 0].plot(time_points, memory_power, label="Memory", linewidth=2)
    axes[0, 0].plot(time_points, control_power, label="Control", linewidth=2)
    axes[0, 0].plot(time_points, io_power, label="I/O", linewidth=2)
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Power (Watts)")
    axes[0, 0].set_title("Component Power Consumption")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Total power consumption
    axes[0, 1].plot(time_points, total_power, "b-", linewidth=2, label="Total Power")
    axes[0, 1].axhline(
        y=np.mean(total_power),
        color="r",
        linestyle="--",
        label=f"Average ({np.mean(total_power):.1f}W)",
    )
    axes[0, 1].axhline(
        y=np.max(total_power),
        color="orange",
        linestyle="--",
        label=f"Peak ({np.max(total_power):.1f}W)",
    )
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Power (Watts)")
    axes[0, 1].set_title("Total Power Consumption")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Power distribution pie chart
    avg_powers = [
        np.mean(systolic_power),
        np.mean(memory_power),
        np.mean(control_power),
        np.mean(io_power),
    ]
    labels = ["Systolic Array", "Memory", "Control", "I/O"]
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

    axes[1, 0].pie(
        avg_powers, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    axes[1, 0].set_title("Average Power Distribution")

    # Power efficiency over time (simulated utilization)
    utilization = (
        0.5 + 0.3 * np.sin(0.4 * time_points) + 0.1 * np.random.randn(len(time_points))
    )
    utilization = np.clip(utilization, 0, 1)  # Keep in valid range
    power_efficiency = utilization / (total_power / 100)  # Normalize for visualization

    axes[1, 1].plot(
        time_points, utilization * 100, "g-", linewidth=2, label="Utilization (%)"
    )
    axes[1, 1].plot(
        time_points,
        power_efficiency * 50,
        "r--",
        linewidth=2,
        label="Efficiency (scaled)",
    )
    axes[1, 1].set_xlabel("Time (seconds)")
    axes[1, 1].set_ylabel("Percentage / Efficiency")
    axes[1, 1].set_title("Utilization vs Power Efficiency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("power_consumption_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Power consumption visualization saved as 'power_consumption_analysis.png'")

    # Create thermal visualization
    plt.figure(figsize=(12, 6))

    # Simulate temperature data
    ambient_temp = 25
    temperature = (
        ambient_temp
        + (total_power - np.min(total_power)) * 0.3
        + 5 * np.random.randn(len(time_points)) * 0.1
    )
    temperature = np.clip(
        temperature, ambient_temp, 100
    )  # Reasonable temperature range

    plt.subplot(1, 2, 1)
    plt.plot(time_points, temperature, "r-", linewidth=2, label="Temperature")
    plt.axhline(y=70, color="orange", linestyle="--", label="Target (70°C)")
    plt.axhline(y=85, color="red", linestyle="--", label="Critical (85°C)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temperature (°C)")
    plt.title("Thermal Profile")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(total_power, temperature, alpha=0.6, c=time_points, cmap="viridis")
    plt.xlabel("Total Power (Watts)")
    plt.ylabel("Temperature (°C)")
    plt.title("Power vs Temperature Correlation")
    plt.colorbar(label="Time (seconds)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("thermal_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Thermal analysis visualization saved as 'thermal_analysis.png'")


def demonstrate_power_budgeting():
    """Demonstrate power budgeting and allocation."""

    print("\n=== Power Budgeting Demonstration ===\n")

    from open_accelerator.core.power_management import PowerBudgetManager

    # Create power budget manager with 500W total budget
    budget_manager = PowerBudgetManager(500.0)

    # Set component budgets
    budget_manager.set_component_budget("systolic_array", 300.0)
    budget_manager.set_component_budget("memory", 120.0)
    budget_manager.set_component_budget("control", 50.0)
    budget_manager.set_component_budget("io", 30.0)

    print("Initial Budget Allocation:")
    status = budget_manager.get_budget_status()
    for component, info in status["component_status"].items():
        print(f"  {component}: {info['budget_watts']}W budget")

    print(f"\nTotal Budget Utilization: {status['budget_utilization_percent']:.1f}%")
    print(f"Remaining Budget: {status['remaining_budget_watts']}W")

    # Simulate power allocation over time
    print("\nSimulating Power Allocation:")

    allocation_history = []
    for cycle in range(20):
        # Simulate varying power needs
        systolic_need = 200 + 50 * np.sin(cycle * 0.3)
        memory_need = 80 + 20 * np.cos(cycle * 0.2)
        control_need = 30 + 10 * np.sin(cycle * 0.5)
        io_need = 15 + 5 * np.cos(cycle * 0.4)

        # Allocate power
        systolic_allocated = budget_manager.allocate_power(
            "systolic_array", systolic_need
        )
        memory_allocated = budget_manager.allocate_power("memory", memory_need)
        control_allocated = budget_manager.allocate_power("control", control_need)
        io_allocated = budget_manager.allocate_power("io", io_need)

        total_allocated = (
            systolic_allocated + memory_allocated + control_allocated + io_allocated
        )

        allocation_history.append(
            {
                "cycle": cycle,
                "total_allocated": total_allocated,
                "systolic": systolic_allocated,
                "memory": memory_allocated,
                "control": control_allocated,
                "io": io_allocated,
            }
        )

        if cycle % 5 == 0:
            print(
                f"  Cycle {cycle}: {total_allocated:.1f}W allocated "
                f"({total_allocated/500*100:.1f}% of budget)"
            )

    # Show final budget status
    final_status = budget_manager.get_budget_status()
    print("\nFinal Allocation Summary:")
    print(f"  Total Allocations Made: {len(budget_manager.budget_history)}")
    print(f"  Current Total Allocation: {final_status['total_allocated_watts']:.1f}W")

    # Demonstrate budget rebalancing
    print("\nDemonstrating Budget Rebalancing:")
    new_budgets = budget_manager.rebalance_budgets(target_utilization=0.85)

    print("Recommended Budget Rebalancing:")
    for component, new_budget in new_budgets.items():
        old_budget = status["component_status"][component]["budget_watts"]
        change = new_budget - old_budget
        print(f"  {component}: {old_budget:.1f}W → {new_budget:.1f}W ({change:+.1f}W)")


def main():
    """Run all power management examples."""

    print("Open Accelerator - Advanced Power Management Examples")
    print("=" * 60)

    try:
        # Run power management comparison
        results = run_power_management_comparison()

        # Demonstrate power optimization
        demonstrate_power_optimization()

        # Create visualizations
        visualize_power_consumption()

        # Demonstrate power budgeting
        demonstrate_power_budgeting()

        print("\n" + "=" * 60)
        print("All power management examples completed successfully!")
        print("\nGenerated files:")
        print("  - power_log_automotive.csv")
        print("  - power_log_edge.csv")
        print("  - power_report_automotive.json")
        print("  - power_report_edge.json")
        print("  - power_consumption_analysis.png")
        print("  - thermal_analysis.png")

    except Exception as e:
        print(f"Error running power management examples: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
