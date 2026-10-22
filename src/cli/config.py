import json

from rich import print

from railgen.scenario_gen.scenario_config import (
    SCENARIO_SCHEMA_FILE,
    ScenarioGenerationConfig,
)
from railgen.schedule_gen.schedule_config import (
    SCHEDULE_SCHEMA_FILE,
    SCHEMA_DIR,
    Config,
)

_CONFIGS = [
    (Config, SCHEDULE_SCHEMA_FILE),
    (ScenarioGenerationConfig, SCENARIO_SCHEMA_FILE),
]


def config_schema():
    """Regenerate config file JSON Schemas used by yaml language server."""

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for config, schema_path in _CONFIGS:
        (SCHEMA_DIR / schema_path).write_text(
            json.dumps(config.model_json_schema(by_alias=True), indent=2) + "\n"
        )
    print(f"[green]Wrote config schemas to[/green] [blue]{SCHEMA_DIR}/[/blue]")
