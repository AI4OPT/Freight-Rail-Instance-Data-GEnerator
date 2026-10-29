import dataclasses
import json
import secrets
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich import print

from railgen.constants import RAIL_NETWORK_DIR, YARDS_DICT_FILE
from railgen.models import yds_dict_adapter
from railgen.scenario_gen.scenario_config import (
    SCENARIO_CONFIG_FILE_PATH,
    load_scenario_config,
)
from railgen.scenario_gen.scenario_generation import generate_scenario
from railgen.scenario_gen.scenario_report import write_report
from railgen.scenario_gen.schedule_input import parse_schedule
from railgen.scenario_gen.yard_resolution import (
    geocode_yd_regions,
    resolve_yard_distributions,
)
from railgen.utils import directory_timestamp


def generate_scenarios(
    target_schedule_dir: Annotated[
        Path,
        typer.Option(
            "--schedule-dir",
            "-d",
            help="Directory containing the schedule to use as the basis for scenario generation.",
        ),
    ],
    config_file: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to scenario generation config file.",
            show_default=True,
            rich_help_panel="Options",
        ),
    ] = SCENARIO_CONFIG_FILE_PATH,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            "-s",
            help="Value used to seed random scenario generation. Overrides config.general_settings.seed.",
            show_default=False,
            rich_help_panel="Options",
        ),
    ] = None,
    scenario_count: Annotated[
        int | None,
        typer.Option(
            "--scenario-count",
            "-n",
            help="Number of scenarios to generate. Overrides config.general_settings.scenario_count.",
            show_default=False,
            rich_help_panel="Options",
        ),
    ] = None,
):
    """Generate scenario variations of a previously generated train schedule."""
    if not config_file.is_file():
        print(
            f"[red]No config file found at[/red] [blue]{config_file}[/blue].",
            "\nPlease create one or download the default config from the project repository.",
        )
        raise typer.Exit(1)
    try:
        config = load_scenario_config(config_file)
        print(f"[green]Loaded config from[/green] [blue]{config_file}[/blue]")
    except (FileNotFoundError, ValidationError) as e:
        print(f"[red]Invalid configuration file:[/red] {e}")
        raise typer.Abort() from e

    # An explicit --seed wins, then general_settings.seed, then a fresh random seed.
    seed = seed if seed is not None else config.general_settings.seed
    if seed is None:
        seed = secrets.randbits(32)
    # Record the resolved seed so it's visible in the dumped config file
    config.general_settings.seed = seed
    print(f"[green]Scenario seed:[/green] {seed}")

    if scenario_count is not None:
        config.general_settings.scenario_count = scenario_count
    print(
        f"[green]Scenarios to generate:[/green] {config.general_settings.scenario_count}"
    )

    try:
        trains = parse_schedule(target_schedule_dir)
    except (OSError, ValueError) as e:
        print(
            f"[red]Failed to read schedule from[/red] [blue]{target_schedule_dir}[/blue]: {e}"
        )
        raise typer.Abort() from e

    print(
        f"[green]Loaded[/green] {len(trains)} [green]train symbols,[/green] "
        f"{sum(len(train.trips) for train in trains)} [green]trips from[/green] [blue]{target_schedule_dir}[/blue]"
    )

    yds_dict_path = RAIL_NETWORK_DIR / YARDS_DICT_FILE
    try:
        yds_dict = (
            yds_dict_adapter.validate_json(yds_dict_path.read_text())
            if config.region_distributions  # We only need to load yards if we have to geocode regions
            else {}
        )
    except (OSError, ValidationError) as e:
        print(
            f"[red]Failed to read[/red] [blue]{yds_dict_path}[/blue], regenerate it with `uv run fridge build-network`: {e}"
        )
        raise typer.Abort() from e
    try:
        geocode_yd_regions(config)
    except ValueError as e:
        print(f"[red]Failed to geocode region center[/red] {e}")
        raise typer.Abort() from e
    try:
        yd_dists = resolve_yard_distributions(config, yds_dict)
    except Exception as e:
        print(f"[red]Invalid region distributions:[/red] {e}")
        raise typer.Abort() from e

    results = generate_scenario(config, trains, yd_dists)
    print(
        f"[green]Generated[/green] {config.general_settings.scenario_count} [green]scenarios for[/green] {len(trains)} [green]train symbols,[/green] {len(results)} [green]trips.[/green]"
    )

    output_json = {
        row_id: {
            day.name[:3].lower(): [dataclasses.asdict(sample) for sample in samples]
            for day, samples in week
            if samples is not None
        }
        for row_id, week in results.items()
    }

    scenario_dir = (
        target_schedule_dir / "scenarios" / f"scenario-{directory_timestamp()}"
    )
    if scenario_dir.exists():
        print(
            f"[red]Scenario directory[/red] [blue]{scenario_dir}[/blue] [red]already exists.[/red]"
        )
        print("[yellow]Please try again.[/yellow]")
        raise typer.Abort()
    scenario_dir.mkdir(parents=True)

    scenario_output_path = scenario_dir / "scenario.json"
    try:
        with open(scenario_output_path, "w") as f:
            json.dump(output_json, f, indent=2)
    except Exception as e:
        print(
            f"[red]Failed to write scenario results to[/red] [blue]{scenario_output_path}[/blue]: {e}"
        )
        raise typer.Abort() from e
    BYTES_TO_MB = 1e-6
    print(
        f"[green]Wrote scenario results to[/green] [blue]{scenario_output_path}[/blue] [yellow]({scenario_output_path.stat().st_size * BYTES_TO_MB:.1f}MB)[/yellow]"
    )

    generation_config_path = scenario_dir / "scenario-generation-config.yaml"
    try:
        generation_config_path.write_text(config.to_yaml())
        print(
            f"[green]Wrote configuration settings used for scenario generation to[/green] [blue]{generation_config_path}[/blue]"
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[red]Failed to write generation config to[/red] [blue]{generation_config_path}[/blue]: {e}"
        )

    report_path = write_report(scenario_dir, config, trains, results)
    print(f"[green]Wrote scenario report to[/green] [blue]{report_path}[/blue]")
