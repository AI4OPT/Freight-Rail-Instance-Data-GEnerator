import secrets
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy
from numpy.random import Generator
from rich import print
from rich.progress import track

from railgen.models import DayOfWeek, Week
from railgen.scenario_gen.distribution_groupings import (
    TrainDistributions,
    YardGroupDistributions,
)
from railgen.scenario_gen.distributions import (
    ScenarioDistribution,
    sample_distribution,
)
from railgen.scenario_gen.scenario_config import ScenarioGenerationConfig
from railgen.scenario_gen.schedule_input import ScheduledTrain
from railgen.schedule_gen.attributes import TrainSymbol
from railgen.utils import MINUTES_PER_DAY, get_progress_desc_with_padding

type TrainRowID = str


@dataclass(frozen=True, kw_only=True, slots=True)
class Sample:
    # difference between scheduled and actual departure time in minutes
    dep_min_delta: int
    # difference between scheduled and actual arrival time in minutes
    arvl_min_delta: int


# For every row, per-day-of-week list of actual delta or None if the train does not run on that day
type ScenarioResult = dict[TrainRowID, Week[list[Sample] | None]]


def generate_scenario(
    config: ScenarioGenerationConfig,
    scheduled_trains: list[ScheduledTrain],
    yg_dists: YardGroupDistributions,
) -> ScenarioResult:

    accumulated_results: dict[TrainRowID, Week[list[Sample]]] = {}

    # The CLI always sets the seed; a direct call without one gets a fresh seed
    seed = config.general_settings.seed
    if seed is None:
        seed = secrets.randbits(32)

    for train in track(
        scheduled_trains,
        description=get_progress_desc_with_padding("Generating Scenarios..."),
    ):
        results = process_train(config, train, yg_dists, seed)
        accumulated_results.update(results)

    # Days with no samples become None
    return {
        row: Week(
            mon=week.mon if week.mon else None,
            tue=week.tue if week.tue else None,
            wed=week.wed if week.wed else None,
            thu=week.thu if week.thu else None,
            fri=week.fri if week.fri else None,
            sat=week.sat if week.sat else None,
            sun=week.sun if week.sun else None,
        )
        for row, week in accumulated_results.items()
    }


def process_train(
    config: ScenarioGenerationConfig,
    train: ScheduledTrain,
    yg_dists: YardGroupDistributions,
    seed: int,
) -> dict[TrainRowID, Week[list[Sample]]]:

    dists = config.train_distributions_for(train.category)

    # All the days that this train runs
    active_days = [day for day, active in train.days_active if active]

    # Every row starts with an empty list per day, so we can just append
    output = defaultdict[TrainRowID, Week[list[Sample]]](
        lambda: Week(mon=[], tue=[], wed=[], thu=[], fri=[], sat=[], sun=[])
    )

    # For each day, simulate the train's trip N times
    for day in active_days:
        rng = train_day_rng(seed, train.symbol, day)
        for _ in range(config.general_settings.scenario_count):
            simulation = simulate_day(
                train,
                rng=rng,
                train_dists=dists,
                yg_dists=yg_dists,
                min_dwell_time_mins=config.general_settings.minimum_dwell_time_minutes,
            )
            for row, sample in simulation.items():
                output[row].get_day(day).append(sample)

    return output


def train_day_rng(seed: int, symbol: TrainSymbol, day: DayOfWeek) -> Generator:
    """Random number generator for one train on one day, seeded from (seed, train symbol, day).

    Each train and day has its own sequence, so a train's samples do not depend on the other
    trains in the schedule or on how many numbers they draw.
    """
    return numpy.random.default_rng([seed, int.from_bytes(symbol.encode()), day.value])


def simulate_day(
    train: ScheduledTrain,
    *,
    rng: Generator | None = None,
    train_dists: TrainDistributions,
    yg_dists: YardGroupDistributions,
    min_dwell_time_mins: int,
) -> dict[TrainRowID, Sample]:
    """Simulates a train's full trip on one day. Without `rng`, draws from an unseeded generator."""

    if rng is None:
        rng = numpy.random.default_rng()
    actuals = {}

    assert len(train.trips) > 0, "Train must have at least one trip to simulate."

    # Handle the very first trip separately since it has no dwell time before it
    first_trip = train.trips[0]

    # Where we currently actually are along the trip
    running_time_actual_minutes = (
        first_trip.dep_minute  # starts at the schedule departure
    )

    dep_delta = add_samples(
        train_dists.initial_departure,
        yg_dists.yg_dwell_distributions.get(first_trip.bgn_yg),
        rng,
    )

    running_time_actual_minutes += dep_delta
    planned_trip_time = first_trip.arvl_minute - first_trip.dep_minute
    trip_time_percent = 1 + (
        add_samples(
            train_dists.trip_time_percent_diff,
            yg_dists.leg_distributions.get((first_trip.bgn_yg, first_trip.end_yg)),
            rng,
        )
        # This comes back as a percentage so divide by 100 to get a scalar
        / 100
    )
    actual_trip_time = planned_trip_time * trip_time_percent
    running_time_actual_minutes += actual_trip_time
    actuals[first_trip.row_id] = Sample(
        dep_min_delta=int(dep_delta),
        # actual - planned is the formula
        arvl_min_delta=int(running_time_actual_minutes - first_trip.arvl_minute),
    )

    for _i, (prev_trip, cur_trip) in enumerate(pairwise(train.trips)):
        dwell_time_planned = cur_trip.dep_minute - prev_trip.arvl_minute
        # Sanity check - prev.end and cur.bgn should be the same
        assert prev_trip.end_yg == cur_trip.bgn_yg, (
            "Previous trip's end yard group does not match current trip's begin yard group."
        )
        dwell_delta_mins = add_samples(
            train_dists.dwell_time,
            yg_dists.yg_dwell_distributions.get(prev_trip.end_yg),
            rng,
        )
        actual_dwell_mins = max(
            min_dwell_time_mins, dwell_time_planned + dwell_delta_mins
        )

        running_time_actual_minutes += actual_dwell_mins
        dep_delta = running_time_actual_minutes - cur_trip.dep_minute

        planned_trip_time = cur_trip.arvl_minute - cur_trip.dep_minute
        trip_time_percent = 1 + (
            add_samples(
                train_dists.trip_time_percent_diff,
                yg_dists.leg_distributions.get((cur_trip.bgn_yg, cur_trip.end_yg)),
                rng,
            )
            / 100
        )
        actual_trip_time = planned_trip_time * trip_time_percent
        running_time_actual_minutes += actual_trip_time
        arvl_delta = running_time_actual_minutes - cur_trip.arvl_minute
        actuals[cur_trip.row_id] = Sample(
            dep_min_delta=int(dep_delta),
            arvl_min_delta=int(arvl_delta),
        )

    # Sanity check for when the "actual" is longer than a week
    if running_time_actual_minutes - first_trip.dep_minute > MINUTES_PER_DAY * 7:
        print(
            f"[yellow]Warning[/yellow]: generated an actual running time for train {train.symbol} that is longer than a week: {running_time_actual_minutes} minutes"
        )

    return actuals


def add_samples(
    train_dist: ScenarioDistribution,
    yg_dist: ScenarioDistribution | None,
    rng: Generator,
) -> float:
    """Combine train distribution sample and yard group distribution sample."""
    return sample_distribution(train_dist, rng) + (
        sample_distribution(yg_dist, rng) if yg_dist else 0
    )
