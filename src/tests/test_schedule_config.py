import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from cli.main import app
from railgen.scenario_gen.scenario_config import SCENARIO_SCHEMA_PATH
from railgen.schedule_gen.schedule_config import (
    CONFIG_FILE_NAME,
    SCHEMA_PATH,
    Config,
    load_config,
)

REPO_ROOT = Path(__file__).parent.parent.parent


def test_config_round_trips():
    cfg = Config()
    assert Config.model_validate(yaml.safe_load(cfg.to_yaml())) == cfg


def test_sparse_config_fills_defaults(tmp_path: Path):
    path = tmp_path / CONFIG_FILE_NAME
    _ = path.write_text("train_settings:\n  speed_km_h: 120\n")
    _, cfg = load_config(path)
    assert cfg.train_settings.speed_km_h == 120
    # Untouched fields fall back to defaults, at every nesting level.
    assert cfg.general_settings.median_f == 72.0
    assert cfg.service_design_settings.locomotive_power.locomotive_hp == 4400.0
    assert len(cfg.network_settings.hub_locations) == 20


def test_empty_config_is_all_defaults(tmp_path: Path):
    path = tmp_path / CONFIG_FILE_NAME
    _ = path.write_text("")
    _, cfg = load_config(path)
    assert cfg == Config()


def test_missing_default_config_falls_back_to_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, cfg = load_config()
    assert cfg == Config()


def test_explicitly_requested_missing_config_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _ = load_config(tmp_path / "nope.yaml")


def test_lambda_is_aliased_for_yaml():
    assert (
        load_yaml(
            "general_settings:\n  lambda: 0.9\n"
        ).general_settings.normalized_seg_arc_cost_exponent_lambda
        == 0.9
    )


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ = load_yaml("general_settings:\n  train_sed: 3\n")


def test_repo_config_file_is_valid():
    """The `configs/schedule_config.yaml` in the repo loads, so no key in it is stale."""
    _, cfg = load_config(REPO_ROOT / "configs" / CONFIG_FILE_NAME)
    assert cfg.service_design_settings.train_type_category


def test_location_preserves_casing_but_matches_case_insensitively():
    loc = Config().network_settings.hub_locations[0]
    assert loc.raw_city == "Portland"
    assert loc.state_code == "OR"
    assert loc.matches("PORTLAND", "or")


def test_config_default_writes_schedule_config(tmp_path: Path):
    path = tmp_path / "schedule.yaml"
    assert _run("config", "default", "schedule", "-o", str(path)).exit_code == 0
    assert load_config(path) == (True, Config())


def test_config_schema_writes_both_kinds_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _run("config", "schema").exit_code == 0
    for schema_path in (SCHEMA_PATH, SCENARIO_SCHEMA_PATH):
        assert json.loads((tmp_path / schema_path).read_text())["properties"]


def test_config_schema_output_without_a_kind_is_rejected(tmp_path: Path):
    assert _run("config", "schema", "-o", str(tmp_path / "out.json")).exit_code != 0
    assert not (tmp_path / "out.json").exists()


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def load_yaml(text: str) -> Config:
    return Config.model_validate(yaml.safe_load(text))
