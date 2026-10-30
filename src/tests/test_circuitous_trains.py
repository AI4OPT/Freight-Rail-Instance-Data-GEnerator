"""Circuitous trains and the locomotive balance (no solver, no graph files)."""

import random
from types import SimpleNamespace

import networkx as nx
import pytest
from pydantic import ValidationError

from railgen.models import DayOfWeek, YdKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.attributes import TrainSymbol
from railgen.schedule_gen.candidates import CandidateService
from railgen.schedule_gen.circuitous_trains import (
    add_circuitous_trains,
    leg_balance,
    net_balance,
    pick_helpful,
)
from railgen.schedule_gen.model import Solution
from railgen.schedule_gen.schedule_config import (
    CircuitousTrains,
    Config,
    ServiceDesignSettings,
)
from railgen.schedule_gen.solution_to_trains import return_trip, to_trains

A, B, C, D = YdKey("A-1"), YdKey("B-2"), YdKey("C-3"), YdKey("D-4")


def _coal_train(
    speed_km_h: float, hops_km: tuple[float, float, float] = (60.0, 60.0, 60.0)
):
    """One coal train A -> D (three hops on level track, 60 km each by default), 10,000 t,
    one run a week."""
    network = object.__new__(RailNetwork)
    network.cfg = Config.model_validate(
        {
            "train_settings": {
                "speed_km_h_by_category": dict.fromkeys("AMBI", speed_km_h)
            }
        }
    )
    network.train_speed_km_h = speed_km_h
    network.yds_graph = nx.Graph()
    for (u, v), km in zip([(A, B), (B, C), (C, D)], hops_km, strict=True):
        network.yds_graph.add_edge(u, v, weight=km)
    network.ruling_grades = dict.fromkeys(network.yds_graph.to_directed().edges, 0.0)
    network.yds_dict = {y: SimpleNamespace(elevation=0.0) for y in (A, B, C, D)}  # ty: ignore[invalid-assignment]
    service = CandidateService(
        id="S0",
        train_type="coal",
        path=(A, B, C, D),
        stop_positions=(0, 3),
        leg_km=(sum(hops_km),),
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
    assert (out.bgn_yd, out.end_yd, back.bgn_yd, back.end_yd) == (A, D, D, A)
    day = next(d for d in DayOfWeek if train.days_active.get_day(d))
    # 10,000 t of coal in cars carrying 100 t -> 100 empty cars of 29 t.
    assert back.tonnage_by_day.get_day(day) == pytest.approx(2_900.0)
    assert back.req_counts_by_day == out.req_counts_by_day  # same locomotives come back
    assert out.bgn_yd_reason.pu_amount == pytest.approx(
        10_000.0
    )  # PU_TON of the first leg
    assert back.end_yd_reason.so_amount == pytest.approx(
        2_900.0
    )  # the empty cars, set out at A
    assert not back.bgn_yd_reason.pick_up and back.end_yd_reason.set_out
    # Leaves after the set-out dwell (30 min); one crew for 4.5 + 0.5 + 4.5 h.
    start = _minutes(1, out.dep_time)
    arrive = _minutes(out.arvl_day, out.arvl_time) - start
    leave = _minutes(back.dep_day, back.dep_time) - start
    assert leave - arrive == 30
    assert not back.bgn_yd_reason.crew_change and not out.end_yd_reason.crew_change


def test_return_trip_changes_crew_at_the_turn_when_the_shift_is_used_up():
    train, service, network, settings = _coal_train(18.0)  # 10 h each way
    longer = return_trip(train, service, network, settings)
    assert longer is not None
    out, *back = longer.route
    assert out.end_yd_reason.crew_change and back[0].bgn_yd_reason.crew_change


def test_return_trip_crew_change_just_past_the_turn_moves_back_to_the_turn():
    # 10 h out at 18 km/h; C is 200 m past the turn at D, and C -> B (6.7 h) is beyond the
    # crew's hours, so the change would fall at C; it merges into the set-out at D.
    train, service, network, settings = _coal_train(18.0, (60.0, 120.0, 0.2))
    longer = return_trip(train, service, network, settings)
    assert longer is not None
    out, *back = longer.route
    assert out.end_yd_reason.crew_change and back[0].bgn_yd_reason.crew_change
    assert not any(leg.end_yd_reason.crew_change for leg in back)


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
    x, y = YdKey("X-1"), YdKey("Y-2")
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


def _with_returns(settings: ServiceDesignSettings, **shares) -> ServiceDesignSettings:
    return settings.model_copy(
        update={
            "circuitous_trains": CircuitousTrains.model_validate(
                {"short_empty_returns": {}, "long_empty_returns": {}, **shares}
            )
        }
    )


def test_short_train_comes_back_only_under_the_short_share():
    train, service, network, settings = _coal_train(40.0)  # 180 km: short (200 mi)
    service_of = {train.symbol: service}
    short = _with_returns(settings, short_empty_returns={"B": 1.0})
    (back,), lines = add_circuitous_trains([train], service_of, network, short)
    assert len(back.route) > len(train.route)
    assert lines == [
        (
            "short empty returns, category B: 1 trains come back with their empty cars "
            "(at most 1 of its 1 short trains; 1 could fit in the week)"
        )
    ]
    long = _with_returns(settings, long_empty_returns={"B": 1.0})
    (same,), lines = add_circuitous_trains([train], service_of, network, long)
    assert same is train
    assert lines == [
        (
            "long empty returns, category B: 0 trains come back with their empty cars "
            "(at most 0 of its 0 long trains; 0 could fit in the week)"
        )
    ]


def test_circuitous_train_shares_accept_only_configured_categories():
    with pytest.raises(ValidationError, match="Manifest"):
        _ = ServiceDesignSettings.model_validate(
            {"circuitous_trains": {"short_empty_returns": {"Manifest": 0.1}}}
        )
    assert ServiceDesignSettings.model_validate(
        {"circuitous_trains": {"long_empty_returns": {"M": 0.1}}}
    ).circuitous_trains.long_empty_returns == {"M": 0.1}
