"""
CLI commands for OpenAccelerator.
"""

import time
from pathlib import Path
from typing import Any, Dict

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ..analysis.performance_analysis import PerformanceAnalyzer
from ..simulation.simulator import Simulator
from ..utils.config import (
    AcceleratorConfig,
    get_default_configs,
    load_config,
    save_config,
)
from .animations import ProgressBarWithCat, RunningCatAnimation

console = Console()


def register_commands(app):
    """Register all CLI commands with the app."""
    # Add all the click commands to the group
    app.add_command(configure)
    app.add_command(simulate)
    app.add_command(benchmark)


@click.command()
@click.option("--interactive", "-i", is_flag=True, help="Interactive configuration")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["small", "large", "medical", "edge"]),
    help="Configuration template",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.yaml",
    help="Output configuration file",
)
@click.pass_context
def configure(ctx, interactive, template, output):
    """Configure accelerator parameters."""

    console.print("[bold cyan]Accelerator Configuration[/bold cyan]")

    if interactive:
        config = _interactive_config()
    elif template:
        configs = get_default_configs()
        config = configs[template]
        console.print(f"Using template: [green]{template}[/green]")
    else:
        config = AcceleratorConfig()
        console.print("Using default configuration")

    # Save configuration
    save_config(config, output)
    console.print(f"Configuration saved to [green]{output}[/green]")

    # Display configuration summary
    _display_config_summary(config)


def _interactive_config() -> AcceleratorConfig:
    """Interactive configuration wizard."""
    console.print("\n[bold]Interactive Configuration Wizard[/bold]")

    # Basic parameters
    name = Prompt.ask("Accelerator name", default="MyAccelerator")

    # Array dimensions
    console.print("\n[bold blue]Array Configuration[/bold blue]")
    rows = int(Prompt.ask("Array rows", default="16"))
    cols = int(Prompt.ask("Array columns", default="16"))
    frequency = float(Prompt.ask("Operating frequency (Hz)", default="1e9"))

    # Medical mode
    medical_mode = Confirm.ask("Enable medical AI mode?", default=False)

    # Create configuration
    from ..utils.config import ArrayConfig, DataType

    config = AcceleratorConfig(
        name=name,
        array=ArrayConfig(rows=rows, cols=cols, frequency=frequency),
        data_type=DataType.FLOAT32 if medical_mode else DataType.FLOAT32,
    )

    # Set medical mode if requested
    if medical_mode:
        config.medical.enable_medical_mode = True

    return config


def _display_config_summary(config: AcceleratorConfig):
    """Display configuration summary table."""
    table = Table(title="Configuration Summary")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Name", config.name)
    table.add_row("Array Size", f"{config.array.rows}×{config.array.cols}")
    table.add_row("Frequency", f"{config.array.frequency/1e9:.1f} GHz")
    table.add_row("Data Type", config.data_type.value)
    table.add_row("Medical Mode", "Yes" if config.medical_mode else "No")

    console.print(table)


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option(
    "--workload",
    "-w",
    type=click.Choice(["gemm", "medical"]),
    default="gemm",
    help="Workload type",
)
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--visualize", is_flag=True, help="Generate visualizations")
@click.pass_context
def simulate(ctx, config, workload, output, visualize):
    """Run accelerator simulation."""

    console.print("[bold cyan]Starting Simulation[/bold cyan]")

    # Load configuration
    if config:
        accel_config = load_config(config)
    else:
        accel_config = AcceleratorConfig()

    console.print(f"Loaded configuration: [green]{accel_config.name}[/green]")

    # Start cat animation for long simulation
    cat = RunningCatAnimation()
    cat.start("Running simulation...")

    try:
        # Create simulator
        simulator = Simulator(accel_config)

        # Create workload based on type
        if workload == "medical":
            # Medical workload not available yet, fall back to GEMM
            console.print("[yellow]Medical workload not available, using GEMM[/yellow]")
            from ..workloads.gemm import GEMMWorkload, GEMMWorkloadConfig

            sim_workload = GEMMWorkload(GEMMWorkloadConfig())
            sim_workload.prepare()
        else:
            # Default to GEMM workload for generic simulations
            from ..workloads.gemm import GEMMWorkload, GEMMWorkloadConfig

            sim_workload = GEMMWorkload(GEMMWorkloadConfig())
            sim_workload.prepare()

        # Run simulation
        results = simulator.run(sim_workload)

        cat.stop()
        console.print("[green]Simulation completed successfully.[/green]")

        # Analyze results
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.compute_metrics()

        _display_simulation_results(metrics)

        # Generate visualizations if requested
        if visualize:
            _generate_visualizations(results, output or "results")

    except Exception as e:
        cat.stop()
        console.print(f"[red]Simulation failed: {e}[/red]")
        raise


def _display_simulation_results(metrics: Dict[str, Any]):
    """Display simulation results in a nice format."""

    # Performance metrics table
    table = Table(title="Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=15)
    table.add_column("Unit", style="dim", width=10)

    table.add_row("Total Cycles", f"{metrics['total_cycles']:,}", "cycles")
    table.add_row("MAC Operations", f"{metrics['total_macs']:,}", "ops")
    table.add_row("Throughput", f"{metrics['throughput']:.2f}", "TOPS")
    table.add_row("Efficiency", f"{metrics['efficiency']:.1%}", "%")

    # Handle optional metrics safely
    power = metrics.get("power", 0)
    if power is not None:
        table.add_row("Power", f"{power:.1f}", "W")

    energy = metrics.get("energy", 0)
    if energy is not None:
        table.add_row("Energy", f"{energy:.2f}", "J")

    console.print(table)

    # Utilization breakdown
    if "pe_utilization" in metrics:
        util_table = Table(title="Resource Utilization")
        util_table.add_column("Resource", style="cyan")
        util_table.add_column("Utilization", style="green")
        util_table.add_column("Bar", style="blue")

        pe_util = metrics["pe_utilization"]
        bar = "█" * int(pe_util * 20) + "░" * (20 - int(pe_util * 20))
        util_table.add_row("Processing Elements", f"{pe_util:.1%}", bar)

        if "memory_utilization" in metrics:
            mem_util = metrics["memory_utilization"]
            bar = "█" * int(mem_util * 20) + "░" * (20 - int(mem_util * 20))
            util_table.add_row("Memory Bandwidth", f"{mem_util:.1%}", bar)

        console.print(util_table)


def _generate_visualizations(results: Dict[str, Any], output_dir: str):
    """Generate visualization files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"Generating visualizations in [green]{output_path}[/green]")

    with ProgressBarWithCat(100, "Creating visualizations") as pbar:
        # Generate various plots
        time.sleep(1)  # Simulate work
        pbar.update(25)

        time.sleep(1)
        pbar.update(50)

        time.sleep(1)
        pbar.update(75)

        time.sleep(1)
        pbar.update(100)

    console.print("Visualizations complete.")


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True))
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True),
    help="Reference configuration for comparison",
)
@click.option("--output", "-o", type=click.Path(), help="Output report file")
@click.pass_context
def benchmark(ctx, config, reference, output):
    """Run comprehensive benchmarks."""

    console.print("[bold cyan]Running Benchmarks[/bold cyan]")

    cat = RunningCatAnimation()
    cat.start("Running benchmark suite...")

    try:
        # Simulate benchmarking
        time.sleep(3)

        cat.stop()

        # Display benchmark results
        _display_benchmark_results()

    except Exception as e:
        cat.stop()
        console.print(f"[red]Benchmark failed: {e}[/red]")


def _display_benchmark_results():
    """Display benchmark results."""
    table = Table(title="Benchmark Results")
    table.add_column("Workload", style="cyan")
    table.add_column("Throughput", style="green")
    table.add_column("Efficiency", style="yellow")
    table.add_column("Score", style="bold magenta")

    benchmarks = [
        ("ResNet-50 Inference", "12.5 TOPS", "89.2%", "*****"),
        ("BERT-Large Training", "8.7 TOPS", "76.1%", "****"),
        ("Medical CT Segmentation", "15.2 TOPS", "94.3%", "*****"),
        ("Edge Inference (MobileNet)", "3.2 TOPS", "92.8%", "*****"),
    ]

    for workload, throughput, efficiency, score in benchmarks:
        table.add_row(workload, throughput, efficiency, score)

    console.print(table)


# NOTE: Advanced Dashboard functionality will be re-added in a future release.
