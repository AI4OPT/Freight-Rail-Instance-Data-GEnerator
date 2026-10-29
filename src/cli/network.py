import os

import typer
from pydantic import ValidationError
from rich import print
from rich.console import Console
from rich.table import Table

from railgen.constants import (
    ALL_DISTANCES_GLOB,
    BDC_KEY,
    GRAPH_FILE,
    RAIL_NETWORK_DIR,
    YARDS_DICT_FILE,
)
from railgen.models import yds_dict_adapter
from railgen.rail_network_gen.graph_generation import generate_or_load_network


def build_network():
    """Build a graph representation of the United States rail network."""
    graph_path = RAIL_NETWORK_DIR / GRAPH_FILE
    if graph_path.exists() and typer.confirm(
        "Rail network already exists. Would you like to regenerate it?",
        default=False,
        abort=True,
    ):
        graph_path.unlink()
        # Computed from the graph and only regenerated when missing
        for cache in RAIL_NETWORK_DIR.glob(ALL_DISTANCES_GLOB):
            cache.unlink()

    if not BDC_KEY:
        print("[red]BDC_KEY environment variable is not set.[/red]")
        print(
            "Please obtain a free API key from https://www.bigdatacloud.com and set it in your evironment.",
        )
        raise typer.Exit(1)
    generate_or_load_network()


def list_yards():
    """List yards with their city and state in an interactive pager."""
    yds_dict_path = RAIL_NETWORK_DIR / YARDS_DICT_FILE
    try:
        yds = yds_dict_adapter.validate_json(yds_dict_path.read_text())
    except (OSError, ValidationError) as e:
        print(
            f"[red]Failed to read[/red] [blue]{yds_dict_path}[/blue], regenerate it with `uv run fridge build-network`: {e}"
        )
        raise typer.Abort() from e

    table = Table("Yard", "City", "State", box=None)
    for key, yd in yds.items():
        table.add_row(key, yd.city, yd.state)
    console = Console()
    if console.is_terminal:
        # Allow case-insensitive search
        os.environ.setdefault("PAGER", "less -I")
        with console.pager(styles=True):
            console.print(table)
    else:
        # Rich falls back to 80 columns when piped and wraps cells which breaks grep
        console.width = 1000
        console.print(table)
