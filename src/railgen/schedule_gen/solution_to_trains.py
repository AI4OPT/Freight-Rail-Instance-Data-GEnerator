import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from statistics import median

from railgen.models import DayOfWeek, TrainCategory, Week, YgKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.attributes import (
    RouteSegment,
    TonnageByDay,
    Train,
    TrainSymbol,
    YardGroupReason,
)
from railgen.schedule_gen.candidates import CandidateService
from railgen.schedule_gen.demand import Commodity
from railgen.schedule_gen.model import OdModelIndex, Solution
from railgen.schedule_gen.schedule_config import ServiceDesignSettings
from railgen.utils import (
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    MINUTES_PER_WEEK,
    HourMinute,
    get_abs_HHMM,
    get_elapsed_day,
)

_DAYS = list(DayOfWeek)
_EPS = 1e-6
_MAX_START_TRIES = 200


# Balance check: every yard's weekly set-outs minus pick-ups must equal its target.
def solution_yard_net(solution: Solution, index: OdModelIndex) -> dict[YgKey, float]:
    """Weekly set-outs minus pick-ups per yard, from a model solution."""
    net: dict[YgKey, float] = defaultdict(float)
    for (sid, i), tons in solution.set_out.items():
        net[index.services[sid].stops[i]] += tons
    for (sid, i), tons in solution.pick_up.items():
        net[index.services[sid].stops[i]] -= tons
    return dict(net)


def deviations(
    net: dict[YgKey, float], targets: dict[YgKey, float], tol: float = 1e-3
) -> dict[YgKey, float]:
    """Yards whose net differs from the target (targets default to 0 for transit-only yards)."""
    out: dict[YgKey, float] = {}
    for yard in set(net) | set(targets):
        diff = net.get(yard, 0.0) - targets.get(yard, 0.0)
        if abs(diff) > tol:
            out[yard] = diff
    return out


def split_runs(runs: int, max_per_symbol: int = 7) -> list[int]:
    """Weekly runs of one service -> runs per symbol (full symbols first, remainder last).

    E.g. 10 runs -> [7, 3]: a daily symbol plus a 3-day symbol.
    """
    full, rest = divmod(max(runs, 0), max_per_symbol)
    return [max_per_symbol] * full + ([rest] if rest else [])


def pick_days(runs: int, rng: random.Random) -> Week[bool]:
    """Draw `runs` distinct days of the week at random for one symbol."""
    active = set(rng.sample(_DAYS, max(0, min(runs, 7))))
    return Week(
        mon=DayOfWeek.MONDAY in active,
        tue=DayOfWeek.TUESDAY in active,
        wed=DayOfWeek.WEDNESDAY in active,
        thu=DayOfWeek.THURSDAY in active,
        fri=DayOfWeek.FRIDAY in active,
        sat=DayOfWeek.SATURDAY in active,
        sun=DayOfWeek.SUNDAY in active,
    )


@dataclass(frozen=True, kw_only=True)
class CrewPlan:
    change_positions: list[int]  # indices into the train's path where crews change
    over_limit: list[int]  # path indices whose dwell + next hop alone exceed the limit


def crew_change_positions(
    hop_hours: list[float],
    work_dwell_hours: dict[int, float],
    usable_hours: float,
) -> CrewPlan:
    """Greedy crew changes along a path. Any yard on the path may host a crew change.

    Walk the path adding running time plus dwell at work stops. When the next stretch
    would push duty time past the usable limit (duty hours minus terminal hours), change
    crews at the yard just reached and restart the clock.

    `hop_hours[j]` is the running time from path node j to node j+1. `work_dwell_hours`
    maps path indices of work stops to their dwell, which counts against the crew that
    departs that node.
    """
    changes: list[int] = []
    over: list[int] = []
    duty = 0.0
    for j, hop in enumerate(hop_hours):
        need = work_dwell_hours.get(j, 0.0) + hop
        if duty > 0 and duty + need > usable_hours:
            changes.append(j)
            duty = 0.0
        if need > usable_hours:
            over.append(j)
        duty += need
    return CrewPlan(change_positions=changes, over_limit=over)


@dataclass(frozen=True, kw_only=True)
class OutputStop:
    position: int  # index into the service path
    reason: YardGroupReason


def output_stops(
    service: CandidateService,
    solution: Solution,
    runs: int,
    network: RailNetwork,
    settings: ServiceDesignSettings,
) -> tuple[list[OutputStop], dict[int, float]]:
    """Output stops of one service, and per-run freight tons on board after each path position."""
    sid = service.id
    n = len(service.stop_positions) - 1
    pu = {i: solution.pick_up.get((sid, i), 0.0) / runs for i in range(n + 1)}
    so = {i: solution.set_out.get((sid, i), 0.0) / runs for i in range(n + 1)}
    stop_index = {pos: i for i, pos in enumerate(service.stop_positions)}

    # Intermediate work stops and their dwell, which counts against the crew.
    work: dict[int, tuple[bool, bool]] = {}
    for i in range(1, n):
        has_pu, has_so = pu[i] > _EPS, so[i] > _EPS
        if has_pu or has_so:
            work[service.stop_positions[i]] = (has_pu, has_so)
    hop_hours = [
        network.get_travel_time([a, b]) / 60 for a, b in pairwise(service.path)
    ]
    dwell = settings.dwell_minutes
    work_dwell_hours = {
        pos: dwell.minutes(p, s, False) / 60 for pos, (p, s) in work.items()
    }
    crew = set(
        crew_change_positions(
            hop_hours, work_dwell_hours, settings.crew.usable_hours
        ).change_positions
    )

    last = len(service.path) - 1
    stops: list[OutputStop] = []
    for pos in sorted({0, last} | set(work) | crew):
        i = stop_index.get(pos)
        pick = pu[i] if i is not None else 0.0
        drop = so[i] if i is not None else 0.0
        stops.append(
            OutputStop(
                position=pos,
                reason=YardGroupReason(
                    pick_up=pos == 0 or pick > _EPS,
                    set_out=pos == last or drop > _EPS,
                    crew_change=pos in crew,
                    pu_amount=pick,
                    so_amount=drop,
                ),
            )
        )

    # Freight tons on board after departing each path position (constant between model stops).
    on_board: dict[int, float] = {}
    for k in range(n):
        tons = solution.on_board.get((sid, k + 1), 0.0) / runs
        for pos in range(service.stop_positions[k], service.stop_positions[k + 1]):
            on_board[pos] = tons
    return stops, on_board


def wrap_week(cur_min: int, is_wrapped: bool) -> tuple[int, bool]:
    if cur_min >= MINUTES_PER_WEEK:
        # wrap around the week
        cur_min -= MINUTES_PER_WEEK
        is_wrapped = True
    return cur_min, is_wrapped


def build_segments(
    service: CandidateService,
    stops: list[OutputStop],
    on_board: dict[int, float],
    days: Week[bool],
    network: RailNetwork,
    settings: ServiceDesignSettings,
    rng: random.Random,
) -> list[RouteSegment] | None:
    """Timed legs between consecutive output stops.

    Returns None when the trip wraps past a week from this start time (caller retries).
    """
    power = settings.locomotive_power
    category = settings.train_type_category[service.train_type]
    starting_min = rng.randint(0, MINUTES_PER_WEEK - 1)
    cur_min = starting_min
    is_wrapped = False
    segments: list[RouteSegment] = []
    for idx, (bgn, end) in enumerate(pairwise(stops)):
        if idx > 0:
            r = bgn.reason
            cur_min += round(
                settings.dwell_minutes.minutes(r.pick_up, r.set_out, r.crew_change)
            )
        cur_min, is_wrapped = wrap_week(cur_min, is_wrapped)
        dep_time = get_abs_HHMM(cur_min)
        elapsed = cur_min - starting_min
        while elapsed < 0:
            elapsed += MINUTES_PER_WEEK
        dep_day = get_elapsed_day(elapsed, starting_min)

        leg_path = list(service.path[bgn.position : end.position + 1])
        cur_min += round(network.get_travel_time(leg_path))
        if not is_wrapped:
            cur_min, is_wrapped = wrap_week(cur_min, is_wrapped)
        if cur_min > starting_min and is_wrapped:
            return None
        arvl_time = get_abs_HHMM(cur_min)
        elapsed = cur_min - starting_min
        while elapsed < 0:
            elapsed += MINUTES_PER_WEEK
        arvl_day = get_elapsed_day(elapsed, starting_min)

        freight = on_board.get(bgn.position, 0.0)
        trailing = settings.car_weight.trailing_tons(category, freight)
        grade = network.get_ruling_grade(leg_path, power.min_grade_hop_km)
        hp_per_ton = power.hp_per_ton(category, grade)
        segments.append(
            RouteSegment(
                bgn_yg=leg_path[0],
                end_yg=leg_path[-1],
                tonnage_by_day=TonnageByDay.new(freight, days),
                bgn_yg_reason=bgn.reason,
                end_yg_reason=end.reason,
                req_counts_by_day=TonnageByDay.new(trailing, days).get_req_loc_counts(
                    hp_per_ton, power.locomotive_hp
                ),
                dep_day=dep_day,
                arvl_day=arvl_day,
                dep_time=dep_time,
                arvl_time=arvl_time,
            )
        )
    return segments


def to_trains(
    solution: Solution,
    services: dict[str, CandidateService],
    network: RailNetwork,
    settings: ServiceDesignSettings,
    rng: random.Random,
    service_of: dict[TrainSymbol, CandidateService] | None = None,
) -> list[Train]:
    """Trains of a solution. `service_of`, when given, is filled with each symbol's service."""
    trains: list[Train] = []
    counters: dict[TrainCategory, int] = {}
    for sid, runs in sorted(solution.runs_per_week.items()):
        service = services[sid]
        stops, on_board = output_stops(service, solution, runs, network, settings)
        category = TrainCategory(settings.train_type_category[service.train_type])
        for symbol_runs in split_runs(runs, settings.max_runs_per_week):
            days = pick_days(symbol_runs, rng)
            segments = None
            for _ in range(_MAX_START_TRIES):
                segments = build_segments(
                    service, stops, on_board, days, network, settings, rng
                )
                if segments is not None:
                    break
            if segments is None:
                raise ValueError(
                    f"Service {sid} ({service.total_km:,.0f} km) does not fit in one week."
                )
            counters[category] = counters.get(category, 0) + 1
            symbol = TrainSymbol.new(category, counters[category])
            trains.append(
                Train(
                    category=category,
                    symbol=symbol,
                    route=segments,
                    days_active=days,
                )
            )
            if service_of is not None:
                service_of[symbol] = service
    return trains


def _minute_of_day(hhmm: HourMinute) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * MINUTES_PER_HOUR + int(minutes)


def _elapsed(day: int, hhmm: HourMinute, start_of_day: int) -> int:
    """Minutes since the train's first departure, from a trip day and a time of day."""
    return (day - 1) * MINUTES_PER_DAY + _minute_of_day(hhmm) - start_of_day


def _week_of(values: dict[DayOfWeek, int]) -> Week[int]:
    return Week(
        mon=values[DayOfWeek.MONDAY],
        tue=values[DayOfWeek.TUESDAY],
        wed=values[DayOfWeek.WEDNESDAY],
        thu=values[DayOfWeek.THURSDAY],
        fri=values[DayOfWeek.FRIDAY],
        sat=values[DayOfWeek.SATURDAY],
        sun=values[DayOfWeek.SUNDAY],
    )


def return_trip(
    train: Train,
    service: CandidateService,
    network: RailNetwork,
    settings: ServiceDesignSettings,
) -> Train | None:
    """The train extended by a trip straight back to its origin, as the same symbol.

    The return runs the service's path in reverse with the empty cars of the heaviest leg
    (freight / cargo tons per car, times the empty-car weight), shown as the legs'
    tonnage, and the locomotives of the heaviest leg, so they come back with the train.
    The loads are set out where the train turns. Crews change where the duty limit
    requires, counting the duty that the crew bringing the train in has already worked.
    Returns None when the round trip does not fit in a week.
    """
    outbound = list(train.route)
    first, last = outbound[0], outbound[-1]
    start = _minute_of_day(first.dep_time)
    arrival = _elapsed(last.arvl_day, last.arvl_time, start)
    crew_start = max(
        (
            _elapsed(leg.dep_day, leg.dep_time, start)
            for leg in outbound[1:]
            if leg.bgn_yg_reason.crew_change
        ),
        default=0,
    )
    dwell = settings.dwell_minutes
    duty_hours = (arrival - crew_start + dwell.minutes(False, True, False)) / 60

    path = list(reversed(service.path))
    hop_hours = [network.get_travel_time([a, b]) / 60 for a, b in pairwise(path)]
    plan = crew_change_positions(
        [duty_hours, *hop_hours], {}, settings.crew.usable_hours
    )
    crew = {j - 1 for j in plan.change_positions}  # the first hop is the duty worked
    turn_crew = 0 in crew
    end = len(path) - 1
    reason = {
        pos: YardGroupReason(
            pick_up=False, set_out=pos == end, crew_change=pos in crew and pos != end
        )
        for pos in sorted({0, end} | crew)
    }
    stops = sorted(reason)

    category = train.category
    cars = settings.car_weight
    heaviest = max(leg.tonnage_by_day.get_day(d) for leg in outbound for d in _DAYS)
    empty_tons = (
        heaviest
        / cars.cargo_tons_per_car_by_category[category]
        * cars.empty_car_tons_by_category[category]
    )
    locomotives = _week_of(
        {d: max(leg.req_counts_by_day.get_day(d) for leg in outbound) for d in _DAYS}
    )

    if turn_crew:
        outbound[-1] = replace(
            last, end_yg_reason=replace(last.end_yg_reason, crew_change=True)
        )
    cur = arrival + round(dwell.minutes(False, True, turn_crew))
    legs: list[RouteSegment] = []
    for idx, (a, b) in enumerate(pairwise(stops)):
        if idx > 0:
            cur += round(dwell.minutes(False, False, True))
        dep = cur
        leg_path = path[a : b + 1]
        cur += round(network.get_travel_time(leg_path))
        if cur >= MINUTES_PER_WEEK:
            return None
        legs.append(
            RouteSegment(
                bgn_yg=leg_path[0],
                end_yg=leg_path[-1],
                tonnage_by_day=TonnageByDay.new(empty_tons, train.days_active),
                bgn_yg_reason=reason[a],
                end_yg_reason=reason[b],
                req_counts_by_day=locomotives,
                dep_day=get_elapsed_day(dep, start),
                arvl_day=get_elapsed_day(cur, start),
                dep_time=get_abs_HHMM(start + dep),
                arvl_time=get_abs_HHMM(start + cur),
            )
        )
    return replace(train, route=[*outbound, *legs])


# Reports on a solution. They need only the solution and its services, not the network.
type StopEvent = dict[str, str | int | float]


def stop_events(
    category: str, solution: Solution, services: dict[str, CandidateService]
) -> list[StopEvent]:
    """One row per stop of every selected service, in tons per week and per run."""
    rows: list[StopEvent] = []
    for sid, runs in sorted(solution.runs_per_week.items()):
        s = services[sid]
        n = len(s.stop_positions) - 1
        for j, yard in enumerate(s.stops):
            pu = solution.pick_up.get((sid, j), 0.0)
            so = solution.set_out.get((sid, j), 0.0)
            leaving = solution.on_board.get((sid, j + 1), 0.0)
            rows.append(
                {
                    "category": category,
                    "service": sid,
                    "train_type": s.train_type,
                    "runs_per_week": runs,
                    "stop": j,
                    "last_stop": n,
                    "role": "origin"
                    if j == 0
                    else "destination"
                    if j == n
                    else "intermediate",
                    "yard": yard,
                    "pu_tons_week": pu,
                    "so_tons_week": so,
                    "leaving_tons_week": leaving,
                    "pu_tons_run": pu / runs,
                    "so_tons_run": so / runs,
                    "leaving_tons_run": leaving / runs,
                    "leg_capacity_tons": s.leg_capacity_tons[j] if j < n else "",
                }
            )
    return rows


def write_stop_events(path: Path, rows: list[StopEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_lines(rows: list[StopEvent]) -> list[str]:
    def tons(col: str, role: str) -> float:
        return sum(float(r[col]) for r in rows if r["role"] == role)

    inter = [r for r in rows if r["role"] == "intermediate"]
    kinds = Counter(
        (float(r["pu_tons_week"]) > _EPS, float(r["so_tons_week"]) > _EPS)
        for r in inter
    )
    lines = [
        (
            f"picked up t/week: at origin {tons('pu_tons_week', 'origin'):,.0f}, "
            f"at intermediate stops {tons('pu_tons_week', 'intermediate'):,.0f}"
        ),
        (
            f"set out t/week: at destination {tons('so_tons_week', 'destination'):,.0f}, "
            f"at intermediate stops {tons('so_tons_week', 'intermediate'):,.0f}"
        ),
        (
            f"intermediate stops of selected services: {len(inter):,} "
            f"(pick-up only {kinds[(True, False)]:,}, set-out only {kinds[(False, True)]:,}, "
            f"both {kinds[(True, True)]:,}, no work {kinds[(False, False)]:,})"
        ),
    ]
    for label, col in (("pick-up", "pu_tons_run"), ("set-out", "so_tons_run")):
        for role in ("origin", "intermediate", "destination"):
            vals = sorted(
                v for r in rows if r["role"] == role and (v := float(r[col])) > _EPS
            )
            if vals:
                lines.append(
                    f"{label} tons per run, {role}: n {len(vals):,}, min {vals[0]:,.0f}, "
                    f"median {median(vals):,.0f}, mean {sum(vals) / len(vals):,.0f}, "
                    f"max {vals[-1]:,.0f}"
                )
    return lines


def transfers(
    solution: Solution,
    services: dict[str, CandidateService],
    commodities: list[Commodity],
) -> dict[tuple[int, YgKey], float]:
    """(commodity index, yard) -> tons/week of that commodity that change trains there.

    A commodity changes trains where it leaves a train at a yard that is not its
    destination (the z variables of `model`, counted from the flows).
    """
    legs: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for (sid, i, k), tons in solution.flows.items():
        legs[(sid, k)][i] = tons
    out: dict[tuple[int, YgKey], float] = defaultdict(float)
    for (sid, k), flow in legs.items():
        stops = services[sid].stops
        for j in range(1, len(stops)):
            drop = flow.get(j, 0.0) - flow.get(j + 1, 0.0)
            if drop > _EPS and stops[j] != commodities[k].destination:
                out[(k, stops[j])] += drop
    return dict(out)


def transfer_lines(
    solution: Solution,
    services: dict[str, CandidateService],
    commodities: list[Commodity],
) -> list[str]:
    moved = transfers(solution, services, commodities)
    per_od = Counter(k for k, _yard in moved)
    counts = Counter(per_od.get(k, 0) for k in range(len(commodities)))
    demand = sum(c.tons_per_week for c in commodities)
    return [
        (
            f"transfers (a commodity changes trains at a yard): {len(moved):,}; "
            f"commodities with at least one: {len(per_od):,} of {len(commodities):,}"
        ),
        (
            f"transfers per commodity: mean {len(moved) / len(commodities):.2f}, "
            f"max {max(counts)}; distribution {dict(sorted(counts.items()))}"
        ),
        (
            f"tons transferred: {sum(moved.values()):,.0f} t/week "
            f"({sum(moved.values()) / demand:.2f} x demand)"
        ),
    ]


def write_stop_report(
    category: str,
    solution: Solution,
    services: dict[str, CandidateService],
    commodities: list[Commodity],
    output_dir: Path,
) -> list[str]:
    """Write `stop_events_<category>.csv` to `output_dir` and return the report lines.

    The lines summarise the pick-ups and set-outs at the stops, and the transfers.
    """
    rows = stop_events(category, solution, services)
    write_stop_events(output_dir / f"stop_events_{category}.csv", rows)
    return summary_lines(rows) + transfer_lines(solution, services, commodities)
