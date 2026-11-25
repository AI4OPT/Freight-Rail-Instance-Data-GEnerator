import os
from typing import Annotated

import typer
from haversine.haversine import Unit, haversine
from pydantic import ValidationError
from rich import print
from rich.console import Console
from rich.table import Table

from railgen.constants import RAIL_NETWORK_DIR, YARDS_DICT_FILE
from railgen.models import Coord, Yard, YdKey, yds_dict_adapter
from railgen.rail_network_gen.graph_utils import forward_geocode

yards_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect rail yards.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@yards_app.command("list")
def _list_yards():
    """View all yards with their city and state in an interactive pager."""
    yds = _load_yards()
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


@yards_app.command("matching")
def _matching(
    city: Annotated[str, typer.Option()], state: Annotated[str, typer.Option()]
):
    """List yards located at the given city and state."""
    city = city.lower()
    state = state.upper()
    yds = _load_yards()
    found_any = False
    for key, yd in yds.items():
        if yd.city.lower() == city and yd.state.upper() == state:
            found_any = True
            print(key)
    if not found_any:
        raise typer.Exit(1)


@yards_app.command("within-radius")
def _within_radius(
    radius: Annotated[float, typer.Option(help="Radius in kilometers")],
    city: Annotated[str | None, typer.Option(help="City of search center")] = None,
    state: Annotated[str | None, typer.Option(help="State of search center")] = None,
    lat: Annotated[float | None, typer.Option(help="Latitude of search center")] = None,
    lon: Annotated[
        float | None, typer.Option(help="Longitude of search center")
    ] = None,
):
    """List yards within radius of a given center location. Center can be specified by city/state _or_ lat/long."""
    loc_provided = city is not None or state is not None
    coord_provided = lat is not None or lon is not None
    if (loc_provided and coord_provided) or (not loc_provided and not coord_provided):
        raise typer.BadParameter(
            "Location must be specified by either city/state or lat/long, not both."
        )

    coord: Coord | None = None
    if coord_provided:
        if lat is None or lon is None:
            raise typer.BadParameter(
                "Both latitude and longitude must be provided when specifying a coordinate."
            )
        coord = Coord(lat=lat, lon=lon)
    if loc_provided:
        if city is None or state is None:
            raise typer.BadParameter(
                "Both city and state must be provided when specifying a location."
            )
        coord = forward_geocode(city, state)
        if coord is None:
            print(f"[red]Failed to geocode location[/red] [blue]{city}, {state}[/blue]")
            raise typer.Abort()
    assert coord is not None, "coord should be set by this point"

    yds = _load_yards()
    found_any = False
    for key, yd in yds.items():
        if (
            haversine(
                (coord.lat, coord.lon),
                (yd.coord.lat, yd.coord.lon),
                unit=Unit.KILOMETERS,
            )
            <= radius
        ):
            print(key, yd.city, yd.state, sep="\t")
            found_any = True

    if not found_any:
        raise typer.Exit(1)


def _load_yards() -> dict[YdKey, Yard]:
    yds_dict_path = RAIL_NETWORK_DIR / YARDS_DICT_FILE
    try:
        return yds_dict_adapter.validate_json(yds_dict_path.read_text())
    except (OSError, ValidationError) as e:
        print(
            f"[red]Failed to read[/red] [blue]{yds_dict_path}[/blue], regenerate it with `uv run fridge build-network`: {e}"
        )
        raise typer.Abort() from e
