from collections import defaultdict
from collections.abc import Hashable, Mapping

from railgen.models import DayOfWeek, YgKey
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


def to_reposition(net: dict[YgKey, int]) -> int:
    """Locomotives to reposition (move without a train that needs them) to balance every yard group."""
    return sum(v for v in net.values() if v > 0)


def terminals(trains: list[Train]) -> set[YgKey]:
    """Yard groups where some train starts or ends."""
    return {t.route[0].bgn_yg for t in trains} | {t.route[-1].end_yg for t in trains}


def balance_lines(trains: list[Train], ends: set[YgKey]) -> list[str]:
    """Summary of the locomotive balance, for the run log and the report.

    `ends` are the yard groups where trains start or end, taken before any circuitous train
    is added, so that the lines before and after count the same yard groups.
    """
    arrivals, departures = yard_traffic(trains)
    net = net_balance(trains)
    share = {y: abs(net[y]) / (arrivals.get(y, 0) + departures.get(y, 0)) for y in ends}
    near = ", ".join(
        f"within {s:.0%} of their own traffic {sum(v <= s for v in share.values())}"
        for s in NEAR_BALANCE_SHARES
    )
    return [
        (
            f"locomotive movements on legs: {sum(arrivals.values()):,} a week; "
            f"to reposition to balance every yard group: {to_reposition(net):,}"
        ),
        (
            f"yard groups where trains start or end: {len(ends)}; balanced "
            f"{sum(net[y] == 0 for y in ends)}, {near}, one-way "
            f"{sum(v >= 1.0 for v in share.values())}"
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


def pick_helpful[G: Hashable](
    net: dict[YgKey, int],
    candidates: dict[TrainSymbol, dict[YgKey, int]],
    group_of: Mapping[TrainSymbol, G],
    budget: Mapping[G, int],
) -> list[TrainSymbol]:
    """Candidates chosen one at a time, each the one that most reduces the locomotives to
    reposition (`net` updated as they are chosen), within the budget of each candidate's group
    (e.g. its category). Stops when no candidate left would reduce them."""

    def gain(delta: dict[YgKey, int]) -> int:
        return sum(
            max(net.get(y, 0), 0) - max(net.get(y, 0) + d, 0) for y, d in delta.items()
        )

    net = dict(net)
    left = dict(budget)
    chosen: list[TrainSymbol] = []
    pool = dict(candidates)
    while pool:
        open_pool = {s: d for s, d in pool.items() if left.get(group_of[s], 0) > 0}
        if not open_pool:
            break
        best = max(sorted(open_pool), key=lambda s: gain(open_pool[s]))
        if gain(open_pool[best]) <= 0:
            break
        for y, d in pool.pop(best).items():
            net[y] = net.get(y, 0) + d
        left[group_of[best]] -= 1
        chosen.append(best)
    return chosen


def add_circuitous_trains(
    trains: list[Train],
    service_of: dict[TrainSymbol, CandidateService],
    network: RailNetwork,
    settings: ServiceDesignSettings,
) -> tuple[list[Train], list[str]]:
    """The trains with the circuitous trains of `settings.circuitous_trains`, and report lines.

    Each category has two budgets, `round(share x its trains)` for its short and for its long
    trains. Returns are then chosen one at a time over all eligible trains, each the one that
    most reduces the locomotives to reposition, while its category and length still have budget.
    """
    levers = settings.circuitous_trains
    max_km = levers.short_max_miles * KM_PER_MILE
    share_of = {"short": levers.short_empty_returns, "long": levers.long_empty_returns}

    def length(train: Train) -> str:
        return "short" if service_of[train.symbol].total_km <= max_km else "long"

    trains_in: dict[tuple[str, str], int] = defaultdict(int)
    for train in trains:
        trains_in[(train.category, length(train))] += 1
    budget = {
        (category, kind): round(share * trains_in[(category, kind)])
        for kind, shares in share_of.items()
        for category, share in shares.items()
    }

    candidates: dict[TrainSymbol, Train] = {}
    group_of: dict[TrainSymbol, tuple[str, str]] = {}
    for train in trains:
        group = (train.category, length(train))
        if budget.get(group, 0) <= 0:
            continue
        longer = return_trip(train, service_of[train.symbol], network, settings)
        if longer is not None:
            candidates[train.symbol] = longer
            group_of[train.symbol] = group
    outbound_legs = {t.symbol: len(t.route) for t in trains}
    chosen = pick_helpful(
        net_balance(trains),
        {s: leg_balance(t.route[outbound_legs[s] :]) for s, t in candidates.items()},
        group_of,
        budget,
    )

    lines = []
    for category, kind in sorted(budget):
        n = sum(group_of[s] == (category, kind) for s in chosen)
        fit = sum(g == (category, kind) for g in group_of.values())
        lines.append(
            f"{kind} empty returns, category {category}: {n} trains come back with their "
            f"empty cars (at most {budget[(category, kind)]} of its {trains_in[(category, kind)]} "
            f"{kind} trains; {fit} could fit in the week)"
        )
    returned = {s: candidates[s] for s in chosen}
    return [returned.get(t.symbol, t) for t in trains], lines
