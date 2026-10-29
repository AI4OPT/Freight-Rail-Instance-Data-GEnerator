import json
from pathlib import Path

from test_schedule_input import TRAINS, write_schedule
from typer.testing import CliRunner

from cli.main import app
from railgen.models import TrainCategory, Week, YdKey
from railgen.scenario_gen.distribution_groupings import YardDistributions
from railgen.scenario_gen.scenario_config import ScenarioGenerationConfig
from railgen.scenario_gen.scenario_generation import (
    Sample,
    generate_scenario,
    simulate_day,
)
from railgen.scenario_gen.schedule_input import ScheduledTrain, ScheduledTripLeg
from railgen.scenario_gen.yard_resolution import (
    resolve_yard_distributions,
)
from railgen.schedule_gen.attributes import TonnageByDay, TrainSymbol

ZEROS = Week(mon=0, tue=0, wed=0, thu=0, fri=0, sat=0, sun=0)
WEEKDAYS = Week(mon=True, tue=True, wed=True, thu=True, fri=True, sat=False, sun=False)


def _cfg(dep: float = 0, count: int = 1, dwell: float = 0) -> ScenarioGenerationConfig:
    """stdev 0 everywhere, so every sample is exactly its mean."""
    dist = lambda mean: {"distr_type": "normal", "mean": mean, "stdev": 0}
    return ScenarioGenerationConfig.model_validate(
        {
            "general_settings": {"scenario_count": count},
            "default_distributions": {
                "initial_departure": dist(dep),
                "trip_time_percent_diff": dist(0),
                "dwell_time": dist(dwell),
            },
        }
    )


def _train(*legs: tuple[str, int, int], days: Week[bool] = WEEKDAYS) -> ScheduledTrain:
    yd = YdKey.from_raw_str("NORRIS-1750")
    return ScheduledTrain(
        category=TrainCategory.A,
        symbol=TrainSymbol("ATRN_1"),
        days_active=days,
        trips=[
            ScheduledTripLeg(
                row_id=row_id,
                bgn_yd=yd,
                end_yd=yd,
                tonnage_by_day=TonnageByDay(
                    tonnages=Week(
                        mon=0.0, tue=0.0, wed=0.0, thu=0.0, fri=0.0, sat=0.0, sun=0.0
                    )
                ),
                req_counts_by_day=ZEROS,
                dep_minute=dep,
                arvl_minute=arvl,
            )
            for row_id, dep, arvl in legs
        ],
    )


def _simulate(train: ScheduledTrain, cfg: ScenarioGenerationConfig):
    return simulate_day(
        train,
        train_dists=cfg.train_distributions_for(TrainCategory.A),
        yd_dists=YardDistributions(leg_distributions={}, yd_dwell_distributions={}),
        min_dwell_share=0.75,
    )


def test_late_train_recovers_with_the_buffer_of_the_dwell():
    # 60 min planned dwell at R2: 45 min minimum + 15 min buffer.
    train = _train(("R1", 600, 900), ("R2", 960, 1200))
    # 10 min late: the buffer absorbs all of it.
    assert _simulate(train, _cfg(dep=10))["R2"] == Sample(
        dep_min_delta=0, arvl_min_delta=0
    )
    # 40 min late: only the 15 min buffer is recovered.
    assert _simulate(train, _cfg(dep=40))["R2"] == Sample(
        dep_min_delta=25, arvl_min_delta=25
    )


def test_early_train_is_not_held_and_dwell_never_below_minimum():
    train = _train(("R1", 600, 900), ("R2", 960, 1200))
    # 10 min early: dwells the planned 60 min and stays early.
    assert _simulate(train, _cfg(dep=-10))["R2"] == Sample(
        dep_min_delta=-10, arvl_min_delta=-10
    )
    # A negative dwell change shortens the stop, down to the 45 min minimum dwell;
    # a positive one adds delay.
    assert _simulate(train, _cfg(dwell=-10))["R2"] == Sample(
        dep_min_delta=-10, arvl_min_delta=-10
    )
    assert _simulate(train, _cfg(dwell=-20))["R2"] == Sample(
        dep_min_delta=-15, arvl_min_delta=-15
    )
    assert _simulate(train, _cfg(dwell=7))["R2"] == Sample(
        dep_min_delta=7, arvl_min_delta=7
    )


def test_generate_scenario_only_fills_active_days():
    cfg = _cfg(count=3)
    result = generate_scenario(
        cfg, [_train(("R1", 600, 900))], resolve_yard_distributions(cfg, {})
    )
    week = result["R1"]
    assert [len(s) for _, s in week if s is not None] == [3] * 5
    assert week.sat is None and week.sun is None


def test_scenario_cli_smoke(tmp_path: Path):
    write_schedule(tmp_path, TRAINS)
    config = tmp_path / "scenario_config.yaml"
    config.write_text("")
    res = CliRunner().invoke(
        app,
        ["scenario", "-d", str(tmp_path), "-c", str(config), "-s", "1", "-n", "2"],
    )
    assert res.exit_code == 0, res.output

    out_dir = next((tmp_path / "scenarios").iterdir())
    assert (out_dir / "scenario-generation-config.yaml").is_file()
    scenarios = json.loads((out_dir / "scenario.json").read_text())
    assert len(scenarios) == 3
    # MTRN_7 only runs on weekdays.
    assert set(scenarios["MTRN_7_JESSUP-987_MILLHAM-748"]) == {
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
    }
    for row in scenarios.values():
        for samples in row.values():
            assert len(samples) == 2
            assert samples[0].keys() == {"dep_min_delta", "arvl_min_delta"}


def test_scenario_cli_requires_config(tmp_path: Path):
    write_schedule(tmp_path, TRAINS)
    res = CliRunner().invoke(
        app, ["scenario", "-d", str(tmp_path), "-c", str(tmp_path / "nope.yaml")]
    )
    assert res.exit_code == 1
    assert "No config file found" in res.output
    assert not (tmp_path / "scenarios").exists()
