from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from railgen.models import Coord, Location, TrainCategory, Yard, YdKey
from railgen.scenario_gen.distribution_groupings import (
    TrainDistributions,
    YardDistributions,
)
from railgen.scenario_gen.distributions import (
    AsymmetricLaplace,
    NormalDist,
    WeibullDist,
)
from railgen.scenario_gen.scenario_config import (
    SCENARIO_CONFIG_FILE_PATH,
    ScenarioGenerationConfig,
    load_scenario_config,
)
from railgen.scenario_gen.yard_resolution import (
    resolve_yard_distributions,
)

REPO_ROOT = Path(__file__).parent.parent.parent


SCENARIO_OVERRIDES = """
distributions_pool:
  jittery: {distr_type: normal, mean: 0, stdev: 45}
  late_tail: {distr_type: asymmetric_laplace, location: 0, scale: 20, skew: 2}
default_distributions:
  initial_departure: jittery
train_category_distributions_overrides:
  B:
    initial_departure: {distr_type: normal, mean: 0, stdev: 2}
  M:
    dwell_time: late_tail
"""


def test_repo_scenario_config_file_is_valid():
    assert load_scenario_config(REPO_ROOT / SCENARIO_CONFIG_FILE_PATH)


def test_default_scenario_config_resolves_for_every_category():
    cfg = load_scenario_yaml("")
    assert cfg == ScenarioGenerationConfig()
    for category in TrainCategory:
        assert cfg.train_distributions_for(category) == TrainDistributions(
            initial_departure=WeibullDist(distr_type="weibull", shape=1, scale=60),
            trip_time_percent_diff=WeibullDist(distr_type="weibull", shape=1, scale=5),
            dwell_time=WeibullDist(distr_type="weibull", shape=1, scale=15),
        )


def test_overrides_layer_over_defaults_per_entry():
    cfg = load_scenario_yaml(SCENARIO_OVERRIDES)
    jittery = NormalDist(distr_type="normal", mean=0, stdev=45)
    default_dwell = WeibullDist(distr_type="weibull", shape=1, scale=15)

    assert cfg.train_distributions_for(TrainCategory.A).initial_departure == jittery
    assert cfg.train_distributions_for(TrainCategory.A).dwell_time == default_dwell
    assert cfg.train_distributions_for(TrainCategory.B).initial_departure == NormalDist(
        distr_type="normal", mean=0, stdev=2
    )
    assert cfg.train_distributions_for(TrainCategory.M).initial_departure == jittery
    assert cfg.train_distributions_for(TrainCategory.M).dwell_time == AsymmetricLaplace(
        distr_type="asymmetric_laplace", location=0, scale=20, skew=2
    )


def test_distribution_names_must_resolve():
    with pytest.raises(
        ValidationError, match=r"default_distributions.dwell_time -> 'typo'"
    ):
        _ = load_scenario_yaml("default_distributions:\n  dwell_time: typo\n")
    with pytest.raises(
        ValidationError,
        match=r"train_category_distributions_overrides.I.trip_time_percent_diff -> 'typo'",
    ):
        _ = load_scenario_yaml(
            "train_category_distributions_overrides:\n  I:\n    trip_time_percent_diff: typo\n"
        )


def test_yard_distributions_resolve():
    cfg = load_scenario_yaml(
        "distributions_pool:\n  slow: {distr_type: normal, mean: 10, stdev: 1}\n"
        "yard_dwell_distributions:\n  NORRIS-1750: slow\n"
        "leg_trip_time_percent_diff_distributions:\n"
        "  NORRIS-1750:\n    NEW BOYLES-1049: {distr_type: normal, mean: 5, stdev: 2}\n"
    )
    assert resolve_yard_distributions(cfg, {}) == YardDistributions(
        yd_dwell_distributions={
            YdKey("NORRIS-1750"): NormalDist(distr_type="normal", mean=10, stdev=1)
        },
        leg_distributions={
            (YdKey("NORRIS-1750"), YdKey("NEW BOYLES-1049")): NormalDist(
                distr_type="normal", mean=5, stdev=2
            )
        },
    )
    with pytest.raises(ValidationError, match="Invalid YdKey format: 'NORRIS'"):
        _ = load_scenario_yaml(
            "yard_dwell_distributions:\n  NORRIS: {distr_type: normal, mean: 0, stdev: 1}\n"
        )


def test_region_distributions_resolve():
    yds = {
        YdKey(f"{name}-{i}"): Yard(
            id=str(i),
            name=name,
            owner="",
            coord=Coord(lat=lat, lon=-71.06),
            elevation=0,
        )
        for i, (name, lat) in enumerate([("A", 42.36), ("B", 42.40), ("FAR", 45.0)])
    }
    region = (
        "region_distributions:\n  - center: {lat: 42.36, lon: -71.06}\n    radius_km: 10\n"
        "    dwell_time: slow\n    trip_time_percent_diff: slow\n"
    )
    pool = "distributions_pool:\n  slow: {distr_type: normal, mean: 10, stdev: 1}\n"
    slow = NormalDist(distr_type="normal", mean=10, stdev=1)
    a, b = YdKey("A-0"), YdKey("B-1")
    assert resolve_yard_distributions(
        load_scenario_yaml(pool + region), yds
    ) == YardDistributions(
        yd_dwell_distributions={a: slow, b: slow},
        leg_distributions={(a, b): slow, (b, a): slow},
    )
    with pytest.raises(ValueError, match=r"region_distributions\[0\] conflicts"):
        _ = resolve_yard_distributions(
            load_scenario_yaml(
                pool + region + "yard_dwell_distributions:\n  A-0: slow\n"
            ),
            yds,
        )
    assert isinstance(
        load_scenario_yaml(
            "region_distributions:\n  - {center: {city: Boston, state: MA}, radius_km: 5}\n"
        )
        .region_distributions[0]
        .center,
        Location,
    )


def test_scenario_config_round_trips(tmp_path: Path):
    cfg = load_scenario_yaml(SCENARIO_OVERRIDES)
    path = tmp_path / "scenario_config.yaml"
    _ = path.write_text(cfg.to_yaml())
    assert load_scenario_config(path) == cfg


def load_scenario_yaml(text: str) -> ScenarioGenerationConfig:
    return ScenarioGenerationConfig.model_validate(yaml.safe_load(text) or {})
