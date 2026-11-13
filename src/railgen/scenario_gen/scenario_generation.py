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
    YardDistributions,
)
from railgen.scenario_gen.distributions import ScenarioDistribution
from railgen.scenario_gen.scenario_config import ScenarioGenerationConfig
from railgen.scenario_gen.schedule_input import ScheduledTrain, ScheduledTripLeg
from railgen.scenario_gen.time import SimulationMinute
from railgen.scenario_gen.track_outage import TrackOutage, determine_outage_delay
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


# logical entry point
def generate_scenario(
    config: ScenarioGenerationConfig,
    scheduled_trains: list[ScheduledTrain],
    yd_dists: YardDistributions,
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
        results = process_train(config, train, yd_dists, seed)
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
    yd_dists: YardDistributions,
    seed: int,
) -> dict[TrainRowID, Week[list[Sample]]]:

    train_dists = config.train_distributions_for(train.category)

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
                day=day,
                rng=rng,
                train_dists=train_dists,
                yd_dists=yd_dists,
                min_dwell_share=config.general_settings.minimum_dwell_share,
                outages=config.track_outages,
            )
            for row, sample in simulation.items():
                output[row].get_day(day).append(sample)

    return output


def train_day_rng(seed: int, symbol: TrainSymbol, day: DayOfWeek) -> Generator:
    """Random number generator for one train on one day, seeded from (seed, train symbol, day).

    Each train and day has its own sequence, so a train's samples do not depend on the other
    trains in the schedule or on how many numbers they draw.
    """
    return numpy.random.default_rng(
        [seed, int.from_bytes(symbol.encode()), day.to_day_int()]
    )


def simulate_day(
    train: ScheduledTrain,
    *,
    day: DayOfWeek,
    rng: Generator,
    train_dists: TrainDistributions,
    yd_dists: YardDistributions,
    outages: list[TrackOutage],
    min_dwell_share: float,
) -> dict[TrainRowID, Sample]:
    """Simulates a train's full trip on one running day."""

    actuals = {}

    assert len(train.trips) > 0, "Train must have at least one trip to simulate."

    # Handle the very first trip separately since it has no dwell time before it
    first_trip = train.trips[0]

    # Where we currently actually are along the trip relative to the start of the running day
    sim_minute = SimulationMinute(
        first_trip.dep_minute  # starts at the schedule departure
    )

    dep_delta = add_samples(
        train_dists.initial_departure,
        yd_dists.yd_dwell_distributions.get(first_trip.bgn_yd),
        sim_minute,
        day,
        rng,
    )

    sim_minute.add_minute_delta(dep_delta)

    # Simulate the trip to next yard
    actual_trip_time = travel_trip(
        first_trip,
        sim_minute,
        day,
        rng,
        train_dists,
        yd_dists,
        outages,
    )

    sim_minute.add_minute_delta(actual_trip_time)
    actuals[first_trip.row_id] = Sample(
        dep_min_delta=int(dep_delta),
        # actual - planned is the formula
        arvl_min_delta=int(sim_minute.min_float() - first_trip.arvl_minute),
    )

    for _i, (prev_trip, cur_trip) in enumerate(pairwise(train.trips)):
        dwell_time_planned = cur_trip.dep_minute - prev_trip.arvl_minute
        # Sanity check - prev.end and cur.bgn should be the same
        assert prev_trip.end_yd == cur_trip.bgn_yd, (
            "Previous trip's end yard does not match current trip's begin yard."
        )
        # Dwell change at the stop: positive is new delay, negative a quicker stop.
        extra_dwell_mins = add_samples(
            train_dists.dwell_time,
            yd_dists.yd_dwell_distributions.get(prev_trip.end_yd),
            sim_minute,
            day,
            rng,
        )
        arrival_delay = sim_minute.min_float() - prev_trip.arvl_minute
        min_dwell = min_dwell_share * dwell_time_planned
        actual_dwell_mins = max(
            min_dwell,
            max(min_dwell, dwell_time_planned - max(arrival_delay, 0.0))
            + extra_dwell_mins,
        )

        sim_minute.add_minute_delta(actual_dwell_mins)
        dep_delta = sim_minute.min_float() - cur_trip.dep_minute

        actual_trip_time = travel_trip(
            cur_trip, sim_minute, day, rng, train_dists, yd_dists, outages
        )
        sim_minute.add_minute_delta(actual_trip_time)
        arvl_delta = sim_minute.min_float() - cur_trip.arvl_minute
        actuals[cur_trip.row_id] = Sample(
            dep_min_delta=int(dep_delta),
            arvl_min_delta=int(arvl_delta),
        )

    # Sanity check for when the "actual" is longer than a week
    if sim_minute.min_int() - first_trip.dep_minute > MINUTES_PER_DAY * 7:
        print(
            f"[yellow]Warning[/yellow]: generated an actual running time for train {train.symbol} that is longer than a week: {sim_minute.min_int()} minutes"
        )

    return actuals


def travel_trip(
    trip: ScheduledTripLeg,
    sim_minute: SimulationMinute,
    day: DayOfWeek,
    rng: Generator,
    train_dists: TrainDistributions,
    yd_dists: YardDistributions,
    outages: list[TrackOutage],
) -> float:
    """Travel across a yard->yard segment and return time taken to travel in minutes. Handles percentage delay and track outages."""

    planned_trip_time = trip.arvl_minute - trip.dep_minute
    trip_time_percent = 1 + (
        add_samples(
            train_dists.trip_time_percent_diff,
            yd_dists.leg_distributions.get((trip.bgn_yd, trip.end_yd)),
            sim_minute,
            day,
            rng,
        )
        # This comes back as a percentage so divide by 100 to get a scalar
        / 100
    )

    travel_time = planned_trip_time * trip_time_percent

    # Handle any track outages and add the resultant delay
    if outage_delay := determine_outage_delay(
        trip.bgn_yd,
        trip.end_yd,
        outages,
        sim_minute=sim_minute,
        day=day,
        travel_minutes=travel_time,
    ):
        return travel_time + outage_delay
    else:
        return travel_time


# -------- Utils ---------


def add_samples(
    dist_a: ScenarioDistribution,
    dist_b: ScenarioDistribution | None,
    running_minute: SimulationMinute,
    day: DayOfWeek,
    rng: Generator,
) -> float:
    """Combine train distribution sample and yard distribution sample."""

    # Trip minutes count from midnight of the running day; time bounds count from Monday 00:00
    week_start = (day.to_day_int() - 1) * MINUTES_PER_DAY

    week_minute = week_start + running_minute.min_float()

    return (dist_a.sample(rng, week_minute) or 0) + (
        (dist_b.sample(rng, week_minute) or 0) if dist_b else 0
    )
