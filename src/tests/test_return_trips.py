"""Return trips and the locomotive balance (no solver, no graph files)."""

import random
from types import SimpleNamespace

import networkx as nx
import pytest

from railgen.models import DayOfWeek, YgKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.attributes import TrainSymbol
from railgen.schedule_gen.candidates import CandidateService
from railgen.schedule_gen.model import Solution
from railgen.schedule_gen.return_trips import (
    leg_balance,
    net_balance,
    pick_helpful,
)
from railgen.schedule_gen.schedule_config import Config, ServiceDesignSettings
from railgen.schedule_gen.solution_to_trains import return_trip, to_trains

A, B, C, D = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")


def _coal_train(speed_km_h: float):
    """One coal train A -> D (three 60 km hops on level track), 10,000 t, one run a week."""
    network = object.__new__(RailNetwork)
    network.cfg = Config()
    network.train_speed_km_h = speed_km_h
    network.ygs_graph = nx.Graph()
    for u, v in [(A, B), (B, C), (C, D)]:
        network.ygs_graph.add_edge(u, v, weight=60.0)
    network.ygs_dict = {y: SimpleNamespace(elevation=0.0) for y in (A, B, C, D)}  # ty: ignore[invalid-assignment]
    service = CandidateService(
        id="S0",
        train_type="coal",
        path=(A, B, C, D),
        stop_positions=(0, 3),
        leg_km=(180.0,),
        leg_hours=(180.0 / speed_km_h,),
        leg_capacity_tons=(45_000.0,),
    )
    solution = Solution(
        runs_per_week={"S0": 1},
        on_board={("S0", 1): 10_000.0},
        pick_up={("S0", 0): 10_000.0},
        set_out={("S0", 1): 10_000.0},
        objective=0.0,
        bound=0.0,
        gap=0.0,
        status="hand-written",
    )
    settings = ServiceDesignSettings()
    (train,) = to_trains(solution, {"S0": service}, network, settings, random.Random(3))
    return train, service, network, settings


def _minutes(day: int, hhmm: str) -> int:
    h, m = hhmm.split(":")
    return (day - 1) * 1440 + int(h) * 60 + int(m)


def test_return_trip_brings_the_empty_cars_and_locomotives_back():
    train, service, network, settings = _coal_train(40.0)  # 4.5 h each way
    longer = return_trip(train, service, network, settings)
    assert longer is not None and longer.symbol == train.symbol
    out, back = longer.route
    assert (out.bgn_yg, out.end_yg, back.bgn_yg, back.end_yg) == (A, D, D, A)
    day = next(d for d in DayOfWeek if train.days_active.get_day(d))
    # 10,000 t of coal in cars carrying 100 t -> 100 empty cars of 29 t.
    assert back.tonnage_by_day.get_day(day) == pytest.approx(2_900.0)
    assert back.req_counts_by_day == out.req_counts_by_day  # same locomotives come back
    assert not back.bgn_yg_reason.pick_up and back.end_yg_reason.set_out
    # Leaves after the set-out dwell (30 min); one crew for 4.5 + 0.5 + 4.5 h.
    start = _minutes(1, out.dep_time)
    arrive = _minutes(out.arvl_day, out.arvl_time) - start
    leave = _minutes(back.dep_day, back.dep_time) - start
    assert leave - arrive == 30
    assert not back.bgn_yg_reason.crew_change and not out.end_yg_reason.crew_change


def test_return_trip_changes_crew_at_the_turn_when_the_shift_is_used_up():
    train, service, network, settings = _coal_train(18.0)  # 10 h each way
    longer = return_trip(train, service, network, settings)
    assert longer is not None
    out, *back = longer.route
    assert out.end_yg_reason.crew_change and back[0].bgn_yg_reason.crew_change


def test_return_trip_that_does_not_fit_in_a_week_is_not_made():
    train, service, network, settings = _coal_train(1.2)  # 150 h each way
    assert return_trip(train, service, network, settings) is None


def test_return_trip_cancels_the_train_s_imbalance():
    train, service, network, settings = _coal_train(40.0)
    assert net_balance([train])[A] < 0 and net_balance([train])[D] > 0
    longer = return_trip(train, service, network, settings)
    assert longer is not None
    assert all(v == 0 for v in net_balance([longer]).values())
    assert leg_balance(longer.route[1:]) == {
        y: -v for y, v in net_balance([train]).items()
    }


def test_returns_are_picked_where_they_help_most_and_never_when_they_hurt():
    x, y = YgKey("X-1"), YgKey("Y-2")
    net = {x: 5, y: -5}
    s1, s2, s3 = TrainSymbol("BTRN_1"), TrainSymbol("BTRN_2"), TrainSymbol("BTRN_3")
    candidates = {
        s1: {x: -3, y: 3},  # helps by 3
        s2: {x: 2, y: -2},  # makes it worse
        s3: {x: -5, y: 5},  # helps by 5
    }
    category = {s1: "B", s2: "B", s3: "B"}
    assert pick_helpful(net, candidates, category, {"B": 1}) == [s3]
    # After s3 the two yards are balanced: s1 would undo that, so nothing more is added.
    assert pick_helpful(net, candidates, category, {"B": 3}) == [s3]
    assert pick_helpful(net, candidates, category, {"M": 3}) == []
