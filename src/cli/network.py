import os

import typer
from pydantic import ValidationError
from rich import print
from rich.console import Console
from rich.table import Table

from railgen.constants import (
    ALL_DISTANCES_GLOB,
    GRAPH_FILE,
    RAIL_NETWORK_DIR,
    YARD_GROUPS_DICT_FILE,
)
from railgen.models import ygs_dict_adapter
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
    generate_or_load_network()


def list_yard_groups():
    """List yard groups with their city and state in an interactive pager."""
    ygs_dict_path = RAIL_NETWORK_DIR / YARD_GROUPS_DICT_FILE
    try:
        ygs = ygs_dict_adapter.validate_json(ygs_dict_path.read_text())
    except (OSError, ValidationError) as e:
        print(
            f"[red]Failed to read[/red] [blue]{ygs_dict_path}[/blue], regenerate it with `uv run fridge build-network`: {e}"
        )
        raise typer.Abort() from e

    table = Table("Yard Group", "City", "State", box=None)
    for key, yg in ygs.items():
        table.add_row(key, yg.city, yg.state)
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
