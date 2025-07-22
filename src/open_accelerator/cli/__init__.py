"""Advanced CLI interface with animations and rich UI."""

import click
from rich.console import Console
from rich.panel import Panel

from .commands import register_commands

console = Console()


@click.group()
@click.version_option(version="1.0.3", prog_name="OpenAccelerator")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--config", type=click.Path(), help="Configuration file path")
@click.pass_context
def cli(ctx, debug, config):
    """
    OpenAccelerator – Advanced ML Accelerator Simulator

    Specialized for medical AI and healthcare applications.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["config"] = config

    # Display welcome banner
    console.print(
        Panel.fit(
            "[bold blue]OpenAccelerator v1.0.3[/bold blue]\n"
            "[cyan]Advanced ML Accelerator Simulator[/cyan]\n"
            "[dim]Specialized for Medical AI Applications[/dim]",
            title="Medical AI Accelerator",
            border_style="blue",
        )
    )


# Register all commands
register_commands(cli)


# Expose Typer/click app for console_scripts entry-point
app = cli


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
