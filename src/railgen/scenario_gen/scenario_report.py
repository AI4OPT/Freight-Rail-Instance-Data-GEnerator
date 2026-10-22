"""Plain-text summary of generated scenarios, written next to `scenario.json`."""

from collections import defaultdict
from pathlib import Path

import numpy

from railgen.scenario_gen.scenario_config import ScenarioGenerationConfig
from railgen.scenario_gen.scenario_generation import ScenarioResult
from railgen.scenario_gen.schedule_input import ScheduledTrain

REPORT_FILE_NAME = "scenario-generation-report.txt"

# Final arrivals later than each of these many minutes are counted
LATE_THRESHOLDS_MIN = (0, 60, 120, 240)


def write_report(
    scenario_dir: Path,
    config: ScenarioGenerationConfig,
    trains: list[ScheduledTrain],
    results: ScenarioResult,
) -> Path:
    """Write the report into `scenario_dir` and return its path."""
    report_path = scenario_dir / REPORT_FILE_NAME
    report_path.write_text("\n".join(report_lines(config, trains, results)) + "\n")
    return report_path


def report_lines(
    config: ScenarioGenerationConfig,
    trains: list[ScheduledTrain],
    results: ScenarioResult,
) -> list[str]:
    """Delays at the first departure and the final arrival of every train run, by train category.

    A train run is one sample of one train on one day it runs.
    """
    settings = config.general_settings
    stops = sum(len(train.trips) - 1 for train in trains)

    # Per category: first departure delays, final arrival delays, number of legs, one entry per train run
    runs: dict[str, tuple[list[int], list[int], list[int]]] = defaultdict(
        lambda: ([], [], [])
    )
    for train in trains:
        first, last = results[train.trips[0].row_id], results[train.trips[-1].row_id]
        for day, first_samples in first:
            last_samples = last.get_day(day)
            if first_samples is None or last_samples is None:
                continue
            for dep, arr in zip(first_samples, last_samples, strict=True):
                for key in (f"Category {train.category}", "All categories"):
                    runs[key][0].append(dep.dep_min_delta)
                    runs[key][1].append(arr.arvl_min_delta)
                    runs[key][2].append(len(train.trips))

    all_runs = runs.get("All categories", ([], [], []))
    lines = [
        "Delays are actual minus scheduled, in minutes; negative is early.",
        "A train run is one sample of one train on one day it runs.",
        "",
        "Run",
        f"seed: {settings.seed}",
        f"samples per train and running day: {settings.scenario_count}",
        f"train symbols: {len(trains):,}, legs: {sum(len(t.trips) for t in trains):,}",
        f"train runs: {len(all_runs[0]):,}",
        (
            f"intermediate stops: {stops:,}; minimum dwell {settings.minimum_dwell_share:.0%} "
            "of the planned dwell, the rest is a buffer late trains use to recover"
        ),
    ]
    for key in sorted(runs, key=lambda k: (k == "All categories", k)):
        deps, arrs, _ = runs[key]
        lines += [
            "",
            key,
            f"train runs: {len(deps):,}",
            f"first departure delay: {_summary(deps)}",
            f"first departures early: {_share(deps, lambda d: d < 0)}",
            f"final arrival delay: {_summary(arrs)}",
            f"final arrivals early: {_share(arrs, lambda d: d < 0)}",
            *(
                f"final arrivals more than {t} min late: {_share(arrs, lambda d, t=t: d > t)}"
                for t in LATE_THRESHOLDS_MIN
            ),
        ]

    # Delay grows with every leg, so longer trains arrive later
    arrs_by_legs: dict[int, list[int]] = defaultdict(list)
    for arr, legs in zip(all_runs[1], all_runs[2], strict=True):
        arrs_by_legs[legs].append(arr)
    lines += ["", "Final arrival delay by number of legs in the train"]
    lines += [
        f"{legs} leg{'s' if legs > 1 else ''}: train runs {len(arrs):,}, {_summary(arrs)}"
        for legs, arrs in sorted(arrs_by_legs.items())
    ]
    return lines


def _summary(values: list[int]) -> str:
    v = numpy.array(values)
    p10, p50, p90, p99 = numpy.percentile(v, [10, 50, 90, 99])
    return f"mean {v.mean():.0f}, p10 {p10:.0f}, median {p50:.0f}, p90 {p90:.0f}, p99 {p99:.0f}, min {v.min()}, max {v.max()}"


def _share(values: list[int], condition) -> str:
    return f"{100 * sum(map(condition, values)) / len(values):.1f}%"
