import random
from collections import defaultdict
from typing import Any

from railgen.models import DayOfWeek, YgKey, ZoneCode
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.attributes import RouteSegment, Train, TrainSymbol
from railgen.schedule_gen.candidates import CandidateService
from railgen.schedule_gen.demand import KM_PER_MILE
from railgen.schedule_gen.schedule_config import ServiceDesignSettings
from railgen.schedule_gen.solution_to_trains import return_trip

NEAR_BALANCE_SHARES = (
    0.10,
    0.25,
)  # reported: yard groups within these shares of their traffic


def leg_all_days_loc_count(leg: RouteSegment) -> int:
    """Locomotives on a leg over the week: its required count summed over the days."""
    return sum(leg.req_counts_by_day.get_day(day) for day in DayOfWeek)


def yard_traffic(trains: list[Train]) -> tuple[dict[YgKey, int], dict[YgKey, int]]:
    """Locomotives arriving at and departing from each yard group in a week."""
    arrivals: dict[YgKey, int] = defaultdict(int)
    departures: dict[YgKey, int] = defaultdict(int)
    for train in trains:
        for leg in train.route:
            n = leg_all_days_loc_count(leg)
            arrivals[leg.end_yg] += n
            departures[leg.bgn_yg] += n
    return dict(arrivals), dict(departures)


def net_balance(trains: list[Train]) -> dict[YgKey, int]:
    """Arrivals minus departures of locomotives per yard group in a week."""
    arrivals, departures = yard_traffic(trains)
    return {
        y: arrivals.get(y, 0) - departures.get(y, 0)
        for y in set(arrivals) | set(departures)
    }


def to_move_back(net: dict[Any, int]) -> int:
    """Locomotives that would have to be moved to balance every yard group (or zone)."""
    return sum(v for v in net.values() if v > 0)


def terminals(trains: list[Train]) -> set[YgKey]:
    """Yard groups where some train starts or ends."""
    return {t.route[0].bgn_yg for t in trains} | {t.route[-1].end_yg for t in trains}


def balance_lines(
    trains: list[Train], zone_of: dict[YgKey, ZoneCode | None]
) -> list[str]:
    """Summary of the locomotive balance, for the run log and the report."""
    arrivals, departures = yard_traffic(trains)
    net = net_balance(trains)
    ends = terminals(trains)
    share = {y: abs(net[y]) / (arrivals.get(y, 0) + departures.get(y, 0)) for y in ends}
    near = ", ".join(
        f"within {s:.0%} of their own traffic {sum(v <= s for v in share.values())}"
        for s in NEAR_BALANCE_SHARES
    )
    zone_net: dict[ZoneCode, int] = defaultdict(int)
    for y, v in net.items():
        zone = zone_of.get(y)
        if zone is not None:
            zone_net[zone] += v
    return [
        (
            f"locomotive movements on legs: {sum(arrivals.values()):,} a week; "
            f"to move back to balance every yard group: {to_move_back(net):,}"
        ),
        (
            f"yard groups where trains start or end: {len(ends)}; balanced "
            f"{sum(net[y] == 0 for y in ends)}, {near}, one-way "
            f"{sum(v >= 1.0 for v in share.values())}"
        ),
        (
            f"zones (their yard groups added): balanced {sum(v == 0 for v in zone_net.values())} "
            f"of {len(zone_net)}; to move back between zones: {to_move_back(zone_net):,}"
        ),
    ]


def leg_balance(legs: list[RouteSegment]) -> dict[YgKey, int]:
    """Arrivals minus departures of locomotives that these legs cause, per yard group."""
    net: dict[YgKey, int] = defaultdict(int)
    for leg in legs:
        n = leg_all_days_loc_count(leg)
        net[leg.end_yg] += n
        net[leg.bgn_yg] -= n
    return dict(net)


def pick_helpful(
    net: dict[YgKey, int],
    candidates: dict[TrainSymbol, dict[YgKey, int]],
    category_of: dict[TrainSymbol, str],
    budget: dict[str, int],
) -> list[TrainSymbol]:
    """Candidates chosen one at a time, each the one that most reduces the locomotives to
    move back (`net` updated as they are chosen), within each category's budget. Stops
    when no candidate left would reduce them."""

    def gain(delta: dict[YgKey, int]) -> int:
        return sum(
            max(net.get(y, 0), 0) - max(net.get(y, 0) + d, 0) for y, d in delta.items()
        )

    net = dict(net)
    left = dict(budget)
    chosen: list[TrainSymbol] = []
    pool = dict(candidates)
    while pool:
        open_pool = {s: d for s, d in pool.items() if left.get(category_of[s], 0) > 0}
        if not open_pool:
            break
        best = max(sorted(open_pool), key=lambda s: gain(open_pool[s]))
        if gain(open_pool[best]) <= 0:
            break
        for y, d in pool.pop(best).items():
            net[y] = net.get(y, 0) + d
        left[category_of[best]] -= 1
        chosen.append(best)
    return chosen


def add_return_trips(
    trains: list[Train],
    service_of: dict[TrainSymbol, CandidateService],
    network: RailNetwork,
    settings: ServiceDesignSettings,
    seed: int,
) -> tuple[list[Train], list[str]]:
    """The trains with the return trips of `settings.return_trips`, and report lines.

    Circuitous trains first (a random share of each category's short trains), then empty
    returns (chosen where they reduce the imbalance most) among the trains left.
    """
    levers = settings.return_trips
    rng = random.Random(seed)
    extended: dict[TrainSymbol, Train] = {}
    lines: list[str] = []

    max_km = levers.circuitous_max_miles * KM_PER_MILE
    for category, share in sorted(levers.circuitous_share.items()):
        short = sorted(
            (t for t in trains if t.category == category),
            key=lambda t: t.symbol,
        )
        short = [t for t in short if service_of[t.symbol].total_km <= max_km]
        picked = rng.sample(short, round(share * len(short)))
        added = 0
        for train in sorted(picked, key=lambda t: t.symbol):
            longer = return_trip(train, service_of[train.symbol], network, settings)
            if longer is not None:
                extended[train.symbol] = longer
                added += 1
        lines.append(
            f"circuitous trains, category {category}: {added} of {len(short)} trains of at "
            f"most {levers.circuitous_max_miles:g} mi come back ({share:.0%} asked)"
        )

    current = [extended.get(t.symbol, t) for t in trains]
    candidates: dict[TrainSymbol, Train] = {}
    for train in current:
        if (
            train.symbol in extended
            or levers.empty_return_share.get(train.category, 0) <= 0
        ):
            continue
        longer = return_trip(train, service_of[train.symbol], network, settings)
        if longer is not None:
            candidates[train.symbol] = longer
    budget = {
        c: int(share * sum(t.category == c for t in trains))
        for c, share in levers.empty_return_share.items()
    }
    outbound_legs = {t.symbol: len(t.route) for t in current}
    chosen = pick_helpful(
        net_balance(current),
        {s: leg_balance(t.route[outbound_legs[s] :]) for s, t in candidates.items()},
        {s: t.category for s, t in candidates.items()},
        budget,
    )
    for symbol in chosen:
        extended[symbol] = candidates[symbol]
    for category in sorted(levers.empty_return_share):
        n = sum(candidates[s].category == category for s in chosen)
        lines.append(
            f"empty returns, category {category}: {n} trains come back with their empty "
            f"cars (at most {budget[category]} allowed; {sum(t.category == category for t in candidates.values())} could fit in the week)"
        )
    return [extended.get(t.symbol, t) for t in trains], lines
