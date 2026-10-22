import csv
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich import print
from rich.progress import track

from railgen.constants import TRAIN_SCHEDULE_FILE_NAME, ZONE_YARD_GROUPS_FILE
from railgen.models import TrainCategory, YgKey
from railgen.rail_network import RailNetwork
from railgen.rail_network_gen.graph_generation import write_zone_yard_groups
from railgen.schedule_gen.attributes import TrainSymbol
from railgen.schedule_gen.build_distance_matrix import (
    DistanceOutputRow,
    generate_output_yg_distances,
)
from railgen.schedule_gen.schedule_config import CONFIG_FILE_PATH, load_config
from railgen.schedule_gen.schedule_output import DataOutputRow
from railgen.schedule_gen.solve_by_category import SOLUTION_DIR
from railgen.schedule_gen.train_generation import generate_trains
from railgen.stats import ScheduleGenerationStats
from railgen.utils import directory_timestamp, get_progress_desc_with_padding


def generate_schedule(
    config_file: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to schedule generation config file.",
            show_default=True,
            rich_help_panel="Options",
        ),
    ] = CONFIG_FILE_PATH,
    train_seed: Annotated[
        int | None,
        typer.Option(
            "--train-seed",
            "-t",
            help="Value used to seed random train generation. Overrides config.general_settings.train_seed.",
            show_default=False,
            rich_help_panel="Options",
        ),
    ] = None,
    output_parent_dir_name: Annotated[
        str | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save generated data. Overrides config.general_settings.data_output_dir.",
            show_default=False,
            rich_help_panel="Options",
        ),
    ] = None,
    region: Annotated[
        str | None,
        typer.Option(
            help="full, east or west. Overrides config.service_design_settings.region.instance.",
            show_default=False,
            rich_help_panel="Options",
        ),
    ] = None,
    skip_distances: Annotated[
        bool,
        typer.Option(
            "--skip-distances",
            help="Skip yard group distance generation.",
            show_default=False,
        ),
    ] = False,
):
    """Generate a train schedule for a single week."""
    if not config_file.is_file():
        print(
            f"[red]No config file found at[/red] [blue]{config_file}[/blue].",
            "\nPlease create one or download the default config from the project repository.",
        )
        raise typer.Exit(1)
    try:
        config = load_config(config_file)
        print(f"[green]Loaded config from[/green] [blue]{config_file}[/blue]")
    except (FileNotFoundError, ValidationError) as e:
        print(f"[red]Invalid configuration file:[/red] {e}")
        raise typer.Abort() from e
    if region is not None:
        sd = config.service_design_settings
        try:
            new_region = type(sd.region).model_validate(
                {**sd.region.model_dump(), "instance": region}
            )
        except ValidationError as e:
            print(f"[red]Invalid --region:[/red] {e}")
            raise typer.Abort() from e
        config = config.model_copy(
            update={
                "service_design_settings": sd.model_copy(update={"region": new_region})
            }
        )

    # An explicit --train-seed wins, then general_settings.train_seed, then a fresh random seed.
    seed = train_seed if train_seed is not None else config.general_settings.train_seed
    if seed is None:
        seed = secrets.randbits(32)
    print(f"[green]Train seed:[/green] {seed}")
    # Saved with the run's config, so that file alone reproduces the run.
    config = config.model_copy(
        update={
            "general_settings": config.general_settings.model_copy(
                update={"train_seed": seed}
            )
        }
    )
    stats = ScheduleGenerationStats(generation_rng_seed=seed)

    # Get the yard groups graph and yard groups dict
    network = RailNetwork(config)
    stats.report_od_assignment(network.od_assignment)

    output_parent_dir = Path(
        output_parent_dir_name or config.general_settings.data_output_dir
    )
    if output_parent_dir.exists() and not output_parent_dir.is_dir():
        print(
            f"[red]The specified output directory[/red] [blue]{output_parent_dir}[/blue] [red]already exists, but is not a directory.[/red]"
        )
        raise typer.Abort()

    # The run's solutions and Gurobi logs go to a folder of the same name under SOLUTION_DIR
    timestamp = directory_timestamp()
    output_dir = output_parent_dir / timestamp
    if output_dir.exists():
        print(
            f"[red]Target output directory[/red] [blue]{output_dir}[/blue] [red]already exists.[/red]"
        )
        print("[yellow]Please try again.[/yellow]")
        raise typer.Abort()
    output_dir.mkdir(parents=True, exist_ok=False)

    # Generate data set
    trains = generate_trains(
        seed=seed,
        network=network,
        output_dir=output_dir,
        solution_dir=SOLUTION_DIR / timestamp,
    )

    rows: list[DataOutputRow] = []
    output_ygs: set[YgKey] = set()
    for train in trains:
        for trip in train.route:
            row = DataOutputRow.new(train, trip)
            output_ygs.add(trip.bgn_yg)
            output_ygs.add(trip.end_yg)
            rows.append(row)

    stats.report_totals(
        route_od_pairs=len(
            {(train.route[0].bgn_yg, train.route[-1].end_yg) for train in trains}
        ),
        segment_od_pairs=len({(r.BGN_YG, r.END_YG) for r in rows}),
        yg_count=len(output_ygs),
        pu_count=sum(r.IS_PU for r in rows),
        so_count=sum(r.IS_SO for r in rows),
        record_count=len(rows),
    )

    # Per-category stats. Locomotive counts get one sample per leg per running day.
    symbols_per_cat: dict[TrainCategory, set[TrainSymbol]] = defaultdict(set)
    pus_per_cat: dict[TrainCategory, int] = defaultdict(int)
    sos_per_cat: dict[TrainCategory, int] = defaultdict(int)
    locos_per_cat: dict[TrainCategory, list[int]] = defaultdict(list)
    for r in rows:
        cat = r.TRN_CATG
        symbols_per_cat[cat].add(r.TRN_SYMB)
        pus_per_cat[cat] += r.IS_PU
        sos_per_cat[cat] += r.IS_SO
        for day, req in (
            (r.RUNS_MON, r.MON_REQ_LOC),
            (r.RUNS_TUE, r.TUE_REQ_LOC),
            (r.RUNS_WED, r.WED_REQ_LOC),
            (r.RUNS_THU, r.THU_REQ_LOC),
            (r.RUNS_FRI, r.FRI_REQ_LOC),
            (r.RUNS_SAT, r.SAT_REQ_LOC),
            (r.RUNS_SUN, r.SUN_REQ_LOC),
        ):
            if day:
                locos_per_cat[cat].append(req)

    print("[green]Per-category statistics:[/green]")
    for cat in sorted(symbols_per_cat):
        stats.report_category_trains(
            cat,
            symbols=len(symbols_per_cat[cat]),
            pick_ups=pus_per_cat[cat],
            set_outs=sos_per_cat[cat],
            locos=locos_per_cat[cat],
        )

    TRAIN_DATA_PATH = output_dir / TRAIN_SCHEDULE_FILE_NAME
    # Write generated train data
    with open(TRAIN_DATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DataOutputRow.get_headers())
        writer.writeheader()

        for row in track(
            rows,
            description=get_progress_desc_with_padding("Writing train data..."),
            # Only display progress bar for large files
            disable=len(rows) < 200_000,
        ):
            writer.writerow(row.to_dict())

    if not skip_distances:
        yg_distances = generate_output_yg_distances(network, output_ygs)

        DISTANCES_PATH = output_dir / "distance_matrix.csv"
        # Write yard group distances
        with open(DISTANCES_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DistanceOutputRow.get_headers())
            writer.writeheader()

            for dist_row in track(
                yg_distances,
                description=get_progress_desc_with_padding(
                    "Writing yard group distances..."
                ),
                # Only display progress bar for large files
                disable=len(yg_distances) < 200_000,
            ):
                writer.writerow(dist_row.to_dict())

        print(
            f"[green]Wrote distances for[/green] {len(output_ygs)} [green]yard groups to[/green] [blue]{DISTANCES_PATH}[/blue]"
        )

    zone_yards_path = output_dir / ZONE_YARD_GROUPS_FILE
    write_zone_yard_groups(
        network.ygs_dict, network.hub_yard_groups_by_city(), zone_yards_path
    )
    print(
        f"[green]Wrote the yard groups of each FAF zone, with the hubs, to[/green] [blue]{zone_yards_path}[/blue]"
    )

    generation_config_path = output_dir / "schedule-generation-config.yaml"
    try:
        generation_config_path.write_text(config.to_yaml())
        print(
            f"[green]Wrote configuration settings used for schedule generation to[/green] [blue]{generation_config_path}[/blue]"
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[red]Failed to write generation config to[/red] [blue]{generation_config_path}[/blue]: {e}"
        )

    stats_output_path = output_dir / "schedule-generation-stats.json"
    try:
        stats_output_path.write_text(stats.to_json())
        print(
            f"[green]Wrote generation statistics to[/green] [blue]{stats_output_path}[/blue]"
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[red]Failed to write generation stats to[/red] [blue]{stats_output_path}[/blue]: {e}"
        )

    print(
        f"[green]Successfully generated[/green] {len(rows)} [green]records to[/green] [blue]{TRAIN_DATA_PATH}[/blue]"
    )
