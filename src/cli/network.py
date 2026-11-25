import typer
from rich import print

from railgen.constants import (
    ALL_DISTANCES_GLOB,
    BDC_KEY,
    GRAPH_FILE,
    RAIL_NETWORK_DIR,
)
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
