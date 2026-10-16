import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from railgen.scenario_gen.scenario_config import (
    SCENARIO_CONFIG_FILE_PATH,
    SCENARIO_SCHEMA_PATH,
    ScenarioGenerationConfig,
)
from railgen.schedule_gen.schedule_config import (
    CONFIG_FILE_PATH,
    CONFIGS_DIR,
    SCHEMA_PATH,
    Config,
)

config_cli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect and regenerate configuration files.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class ConfigKind(StrEnum):
    Schedule = "schedule"
    Scenario = "scenario"


# Model, default config path, default schema path
_CONFIGS: dict[
    ConfigKind, tuple[type[Config | ScenarioGenerationConfig], Path, Path]
] = {
    ConfigKind.Schedule: (Config, CONFIG_FILE_PATH, SCHEMA_PATH),
    ConfigKind.Scenario: (
        ScenarioGenerationConfig,
        SCENARIO_CONFIG_FILE_PATH,
        SCENARIO_SCHEMA_PATH,
    ),
}


@config_cli.command("default")
def config_dump(
    kind: Annotated[ConfigKind, typer.Argument(help="Which config to generate.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=f"Where to write the config file. Defaults to {CONFIGS_DIR}/<kind>_config.yaml.",
            show_default=False,
        ),
    ] = None,
):
    """Generate a fully-populated config file with default values."""
    model, default_path, _ = _CONFIGS[kind]
    path = output or default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model().to_yaml())
    print(f"[green]Wrote default {kind} config to[/green] [blue]{path}[/blue]")


@config_cli.command("schema")
def config_schema(
    kind: Annotated[
        ConfigKind | None,
        typer.Argument(help="Which schema to generate. Both are generated if omitted."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Where to write the JSON Schema. Requires an explicit config kind.",
            show_default=False,
        ),
    ] = None,
):
    """Generate the JSON Schema for FRIDGE's configs. Used by language servers to provide editor support."""
    if output is not None and kind is None:
        print(
            "[red]--output requires an explicit config kind[/red] (schedule or scenario)."
        )
        raise typer.Abort()

    for config_kind in [kind] if kind is not None else list(ConfigKind):
        model, _, default_path = _CONFIGS[config_kind]
        path = output or default_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(model.model_json_schema(by_alias=True), indent=2) + "\n"
        )
        print(
            f"[green]Wrote {config_kind} config schema to[/green] [blue]{path}[/blue]"
        )
