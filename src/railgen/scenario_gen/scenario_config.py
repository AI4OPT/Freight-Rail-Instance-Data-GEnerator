"""The `scenario_config.yaml` object model defining all user-tweakable scenario generation knobs."""

from pathlib import Path
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from railgen.models import Coord, Location, TrainCategory, YgKey
from railgen.scenario_gen.distribution_groupings import TrainDistributions
from railgen.scenario_gen.distributions import NormalDist, ScenarioDistribution
from railgen.schedule_gen.schedule_config import CONFIGS_DIR

SCENARIO_CONFIG_FILE_PATH = CONFIGS_DIR / "scenario_config.yaml"
SCENARIO_SCHEMA_PATH = Path("schema") / "scenario_config.schema.json"

_CONFIG_MODEL = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")
_CONFIG_MODEL_MUTABLE = ConfigDict(frozen=False, populate_by_name=True, extra="forbid")


class ScenarioGeneralConfig(BaseModel):
    seed: int | None = Field(
        None,
        description="Value used to seed random scenario generation. Null means a fresh random seed on every run.",
    )
    scenario_count: int = Field(10, description="Number of scenarios to generate.")
    minimum_dwell_time_minutes: int = Field(
        30,
        description="Minimum dwell time at a stop in minutes. If a train's simulated dwell time is less than this, it will be clamped to this value.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL_MUTABLE


class Distributions(BaseModel):
    """Each dist is either a fully defined distribution or the name of a predefined distribution."""

    initial_departure: ScenarioDistribution | str = Field(
        default_factory=lambda: NormalDist(distr_type="normal", mean=0, stdev=5),
        description="Minutes between a train's scheduled and actual first departure.",
    )
    trip_time_percent_diff: ScenarioDistribution | str = Field(
        default_factory=lambda: NormalDist(distr_type="normal", mean=0, stdev=5),
        description="Percentage difference between actual run time of a trip leg and its scheduled run time.",
    )
    dwell_time: ScenarioDistribution | str = Field(
        default_factory=lambda: NormalDist(distr_type="normal", mean=0, stdev=5),
        description="Minutes between a stop's scheduled and actual dwell time.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


class DistributionsOverrides(BaseModel):
    """Same entries as `Distributions` but None fields fallback to default."""

    initial_departure: ScenarioDistribution | str | None = None
    trip_time_percent_diff: ScenarioDistribution | str | None = None
    dwell_time: ScenarioDistribution | str | None = None

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


class RegionDistributions(BaseModel):
    center: Coord | Location = Field(
        description="Center of the region. May be a 'lat,lon' coordinate or a city and state.",
    )

    # If the region's center is a city, state, this is the geocoded coordinate determined at runtime.
    geocoded_center_coordinate: SkipJsonSchema[Coord | None] = Field(
        None,
        exclude_if=lambda v: v is None,
    )

    radius_km: float = Field(description="Radius of the region in kilometers.", gt=0.0)
    dwell_time: ScenarioDistribution | str | None = Field(
        None,
        description="Dwell minutes distribution for every yard group in the region.",
    )
    trip_time_percent_diff: ScenarioDistribution | str | None = Field(
        None,
        description="Trip time percent difference for every leg between two yard groups in the region.",
    )

    # We make this mutable so we can set the geocoded center at runtime
    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL_MUTABLE


class ScenarioGenerationConfig(BaseModel):
    """Root of `scenario_config.yaml`."""

    general_settings: ScenarioGeneralConfig = Field(
        default_factory=ScenarioGeneralConfig
    )
    distributions_pool: dict[str, ScenarioDistribution] = Field(
        default_factory=dict, description="Collection of named reusable distributions."
    )
    default_distributions: Distributions = Field(
        default_factory=Distributions,
        description="Default distributions.",
    )
    train_category_distributions_overrides: dict[
        TrainCategory, DistributionsOverrides
    ] = Field(
        default_factory=dict,
        description="Per-category overrides of `default_distributions`. Categories and entries left out fall back to the defaults.",
    )
    yard_group_dwell_distributions: dict[YgKey, ScenarioDistribution | str] = Field(
        default_factory=dict,
        description="Per-yard-group dwell minutes distributions. Also applies to first departures from the yard group.",
    )
    leg_trip_time_percent_diff_distributions: dict[
        YgKey, dict[YgKey, ScenarioDistribution | str]
    ] = Field(
        default_factory=dict,
        description="Per-one-way-leg (origin -> destination) trip time percent difference.",
    )
    region_distributions: list[RegionDistributions] = Field(
        default_factory=list,
        description="Distributions applied to all yard groups within a radius of a location. A yard group or leg may only get a distribution from one source.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL

    @model_validator(mode="after")
    def _check_distribution_names(self) -> Self:
        refs = [
            (f"default_distributions.{k}", v) for k, v in self.default_distributions
        ]
        refs += [
            (f"train_category_distributions_overrides.{cat}.{k}", v)
            for cat, overrides in self.train_category_distributions_overrides.items()
            for k, v in overrides
        ]
        refs += [
            (f"yard_group_dwell_distributions.{yg}", v)
            for yg, v in self.yard_group_dwell_distributions.items()
        ]
        refs += [
            (f"leg_trip_time_percent_diff_distributions.{orig}.{dest}", v)
            for orig, dests in self.leg_trip_time_percent_diff_distributions.items()
            for dest, v in dests.items()
        ]
        refs += [
            (f"region_distributions[{i}].{k}", getattr(region, k))
            for i, region in enumerate(self.region_distributions)
            for k in ("dwell_time", "trip_time_percent_diff")
        ]
        if unknown := [
            f"{path} -> '{value}'"
            for path, value in refs
            if isinstance(value, str) and value not in self.distributions_pool
        ]:
            raise ValueError(
                f"undefined distributions referenced: {', '.join(unknown)}. "
                f"distributions_pool defines {sorted(self.distributions_pool)}"
            )
        return self

    def train_distributions_for(self, category: TrainCategory) -> TrainDistributions:
        """Get the distributions to be used for a train of a given category."""
        overrides = self.train_category_distributions_overrides.get(category)
        resolved = {}
        for name, default in self.default_distributions:
            value = getattr(overrides, name, None)
            if value is None:
                value = default
            resolved[name] = (
                self.distributions_pool[value] if isinstance(value, str) else value
            )
        return TrainDistributions(**resolved)

    def to_yaml(self) -> str:
        """Serialize to YAML with schema line."""
        body = yaml.safe_dump(
            self.model_dump(mode="json", by_alias=True), sort_keys=False, width=100
        )
        return f"# yaml-language-server: $schema=../{SCENARIO_SCHEMA_PATH.as_posix()}\n\n{body}"


def load_scenario_config(
    path: Path | None = None,
) -> tuple[bool, ScenarioGenerationConfig]:
    """
    Load scenario config from `path`, else `configs/scenario_config.yaml`. Falls back to defaults if not found.

    Returns (config_file_found, config).
    """
    if path is None:
        path = SCENARIO_CONFIG_FILE_PATH
        if not path.is_file():
            return False, ScenarioGenerationConfig()
    elif not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return True, ScenarioGenerationConfig.model_validate(
        yaml.safe_load(path.read_text()) or {}
    )
