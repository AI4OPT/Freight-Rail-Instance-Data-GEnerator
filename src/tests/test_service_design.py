"""Unit tests for the pure pieces of the service-design generator (no network, no FAF file)."""

import random

import pytest

from railgen.models import DayOfWeek, TrainCategory, Week, YgKey, ZoneCode
from railgen.schedule_gen.attributes import TonnageByDay, TrainSymbol
from railgen.schedule_gen.demand import (
    MULTIMODAL_MODE,
    RAIL_MODE,
    CandidateYardPair,
    Commodity,
    DemandReport,
    kept_lines,
    select_yard_pairs,
    split_by_size,
    yard_targets,
)
from railgen.schedule_gen.schedule_config import (
    CarWeight,
    LocomotivePower,
    ServiceDesignSettings,
    SolverSettings,
)
from railgen.schedule_gen.solution_to_trains import (
    LongStretch,
    crew_change_positions,
    long_stretch_lines,
    pick_days,
    split_runs,
)

A1, A2, A3 = YgKey("A1-1"), YgKey("A2-2"), YgKey("A3-3")
B1, B2, B3 = YgKey("B1-4"), YgKey("B2-5"), YgKey("B3-6")


def _cands() -> list[CandidateYardPair]:
    # 3 x 3 inter-zone candidates; A1-B1 shortest, A3-B3 far out of the way.
    return [
        CandidateYardPair(origin=o, dest=d, value=km)
        for o, d, km in [
            (A1, B1, 100.0),
            (A1, B2, 105.0),
            (A1, B3, 140.0),
            (A2, B1, 108.0),
            (A2, B2, 110.0),
            (A2, B3, 150.0),
            (A3, B1, 200.0),
            (A3, B2, 210.0),
            (A3, B3, 300.0),
        ]
    ]


def test_small_flow_keeps_one_pair():
    kept = select_yard_pairs(
        _cands(), 5_000, tonnage_step=11_000, max_pairs=3, distance_tolerance=0.15
    )
    assert kept == [(A1, B1)]


def test_large_flow_spreads_within_tolerance():
    kept = select_yard_pairs(
        _cands(), 30_000, tonnage_step=11_000, max_pairs=3, distance_tolerance=0.15
    )
    # n = ceil(30,000 / 11,000) = 3. Within 115 km: A1-B1, A1-B2, A2-B1, A2-B2 (by length).
    # The first three are kept; A3-* and *-B3 are out of tolerance.
    assert kept == [(A1, B1), (A1, B2), (A2, B1)]
    assert all(c.value <= 115 for c in _cands() if (c.origin, c.dest) in kept)


def test_intra_zone_ranks_by_target_distance():
    cands = [
        CandidateYardPair(origin=A1, dest=A2, value=30.0),
        CandidateYardPair(origin=A1, dest=A3, value=250.0),
        CandidateYardPair(origin=A2, dest=A3, value=240.0),
    ]
    kept = select_yard_pairs(
        cands,
        5_000,
        tonnage_step=11_000,
        max_pairs=3,
        distance_tolerance=None,
        target_km=245.0,
    )
    assert kept == [(A1, A3)] or kept == [(A2, A3)]


def test_split_by_size_sums_to_one():
    shares = split_by_size([(A1, B1), (A2, B2)], {A1: 3.0, A2: 1.0, B1: 2.0, B2: 2.0})
    assert shares == pytest.approx([0.75, 0.25])


def test_yard_targets_sum_to_zero():
    zp = (ZoneCode("001"), ZoneCode("002"))
    cs = [
        Commodity(
            origin=A1,
            destination=B1,
            train_type="bulk",
            tons_per_week=100.0,
            zone_pair=zp,
        ),
        Commodity(
            origin=A2,
            destination=B1,
            train_type="manifest",
            tons_per_week=40.0,
            zone_pair=zp,
        ),
        Commodity(
            origin=B1,
            destination=A1,
            train_type="manifest",
            tons_per_week=10.0,
            zone_pair=zp[::-1],
        ),
    ]
    t = yard_targets(cs)
    assert t[B1] == pytest.approx(130.0)
    assert t[A1] == pytest.approx(-90.0)
    assert t[A2] == pytest.approx(-40.0)
    assert sum(t.values()) == pytest.approx(0.0)


def test_crew_changes_include_work_dwell():
    # Five 3-hour hops, work at node 2 with 2 h dwell, 10.5 usable hours.
    plan = crew_change_positions([3, 3, 3, 3, 3], {2: 2.0}, usable_hours=10.5)
    # Duty 3, 6; node 2 needs 2 + 3 -> 11 > 10.5, change at 2; then 5, 8, 11 -> change at 4.
    assert plan.change_positions == [2, 4]
    assert plan.over_limit == []


def test_crew_change_merges_with_a_work_stop_next_door():
    # Work at node 1 (1.5 h dwell), node 2 is 0.1 h further, then a 3 h hop: duty 8 + 1.5 + 0.1
    # at node 2, which needs 3 more -> change. Node 1 is 0.1 h back, so the crew changes there.
    hops, work = [8, 0.1, 3], {1: 1.5}

    def changes(merge_hours):
        return crew_change_positions(hops, work, 12, merge_hours).change_positions

    assert changes(0.0) == [2]
    assert changes(1 / 3) == [1]
    assert changes(0.05) == [2]  # node 1 is 0.1 h back, too far


def test_a_hop_longer_than_a_crew_shift_is_reported():
    # The hop from node 1 takes 14 h: a crew changes at node 1, but no change can split it.
    plan = crew_change_positions([2, 14, 3], {}, usable_hours=12)
    assert plan.change_positions == [1, 2]
    assert plan.over_limit == [1]
    stretch = LongStretch(
        symbol=TrainSymbol("MTRN_1"),
        bgn_yg=YgKey("A-1"),
        end_yg=YgKey("B-2"),
        hours=14.0,
    )
    (line,) = long_stretch_lines([stretch], 12.0)
    assert line.startswith(
        "1 train symbols have 1 stretches longer than a crew's 12 usable"
    )
    assert line.endswith("MTRN_1 A-1 -> B-2 14.0 h")
    assert long_stretch_lines([], 12.0) == []


def test_dwell_table():
    d = ServiceDesignSettings().dwell_minutes
    assert d.minutes(False, False, True) == 20
    assert d.minutes(True, True, False) == 90
    assert d.minutes(True, True, True) == 140
    assert d.minutes(False, False, False) == 0


def test_pick_days_counts():
    rng = random.Random(0)
    for runs in range(8):
        week = pick_days(runs, rng)
        assert sum(week.get_day(d) for d in DayOfWeek) == runs


def test_split_runs_into_symbols():
    assert split_runs(0) == []
    assert split_runs(3) == [3]
    assert split_runs(7) == [7]
    assert split_runs(10) == [7, 3]
    assert split_runs(21) == [7, 7, 7]


def test_vehicles_ride_only_automotive():
    s = ServiceDesignSettings()
    assert s.group_of("36") == "automotive"
    assert s.train_type_of("automotive") == "automotive"
    assert s.group_of("43") == "manifest"


def test_hp_per_ton_is_the_larger_of_category_and_grade():
    power = ServiceDesignSettings().locomotive_power
    # 1.5% grade: (2 + 20 x 1.5) lb/ton x 10 mph / (375 x 0.82) = 1.0407 hp/ton.
    assert power.grade_hp_per_ton(1.5) == pytest.approx(1.0407, abs=1e-4)
    assert power.hp_per_ton("B", 1.5) == pytest.approx(1.0407, abs=1e-4)
    assert power.hp_per_ton("B", 0.0) == 0.78
    assert power.hp_per_ton("M", 1.5) == 1.5


def test_every_output_category_needs_an_hp_per_ton():
    with pytest.raises(ValueError, match="missing categories"):
        ServiceDesignSettings(
            locomotive_power=LocomotivePower(hp_per_ton_by_category={"A": 1.4})
        )


def test_trailing_tons_add_the_cars_to_the_freight():
    cars = ServiceDesignSettings().car_weight
    # Manifest: 8,250 t of freight / 82.5 t per car = 100 cars x 33 t = 3,300 t of cars.
    assert cars.trailing_tons("M", 8_250.0) == pytest.approx(11_550.0)
    # One loaded auto rack: 18 t of vehicles + 53 t of car.
    assert cars.trailing_tons("A", 18.0) == pytest.approx(71.0)
    assert cars.trailing_tons("B", 0.0) == 0.0


def test_no_car_weight_leaves_freight_tons():
    cars = CarWeight(
        empty_car_tons_by_category={"M": 0.0},
        cargo_tons_per_car_by_category={"M": 82.5},
    )
    assert cars.trailing_tons("M", 8_000.0) == 8_000.0


def test_car_weight_values_are_checked():
    with pytest.raises(ValueError, match="cargo_tons_per_car_by_category must be > 0"):
        CarWeight(cargo_tons_per_car_by_category={"M": 0.0})
    with pytest.raises(ValueError, match="empty_car_tons_by_category must be >= 0"):
        CarWeight(empty_car_tons_by_category={"M": -1.0})


def test_every_output_category_needs_car_weights():
    with pytest.raises(ValueError, match="car_weight.empty_car_tons_by_category"):
        ServiceDesignSettings(
            car_weight=CarWeight(empty_car_tons_by_category={"A": 53.0})
        )


def test_locomotive_counts_from_hp_per_ton():
    tons = TonnageByDay(
        tonnages=Week(
            mon=10_000.0, tue=0.0, wed=0.0, thu=0.0, fri=0.0, sat=0.0, sun=0.0
        )
    )
    # 10,000 t x 0.78 hp/t = 7,800 hp -> 2 locomotives of 4,400 hp; x 1.04 -> 3.
    assert tons.get_req_loc_counts(0.78, 4400.0).mon == 2
    assert tons.get_req_loc_counts(1.04, 4400.0).mon == 3
    assert tons.get_req_loc_counts(1.04, 4400.0).tue == 0


def _climbing_line():
    """Four yards in a line, each higher than the last; no graph files are read."""
    from types import SimpleNamespace

    import networkx as nx

    from railgen.rail_network import RailNetwork
    from railgen.schedule_gen.schedule_config import Config

    a, b, c, d = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")
    network = object.__new__(RailNetwork)
    network.cfg = Config()
    network.train_speed_km_h = 40.0
    network.ygs_graph = nx.Graph()
    network.ygs_graph.add_edge(a, b, weight=20.0)  # +100 m over 20 km: 0.5%
    network.ygs_graph.add_edge(b, c, weight=10.0)  # +150 m over 10 km: 1.5%
    network.ygs_graph.add_edge(c, d, weight=1.0)  # +50 m over 1 km: 5%, a short hop
    network.ruling_grades = {
        (a, b): 0.5,
        (b, a): 0.0,
        (b, c): 1.5,
        (c, b): 0.4,
        (c, d): 1.2,
        (d, c): 0.0,
    }
    network.ygs_dict = {  # ty: ignore[invalid-assignment]
        y: SimpleNamespace(elevation=z)
        for y, z in [(a, 0.0), (b, 100.0), (c, 250.0), (d, 300.0)]
    }
    return network, [a, b, c, d]


def test_ruling_grade_is_steepest_edge_grade_in_direction_of_travel():
    network, path = _climbing_line()
    assert network.get_ruling_grade(path) == 1.5
    assert network.get_ruling_grade(path[::-1]) == 0.4


def test_travel_time_is_slower_uphill():
    network, (a, b, c, d) = _climbing_line()
    # Downhill: 20 km at 40 km/h. Uphill at 0.5%: speed cut by 0.5 x 10% = 5%.
    assert network.get_travel_time([b, a]) == pytest.approx(30.0)
    assert network.get_travel_time([a, b]) == pytest.approx(20 / 38 * 60)
    # At a category's own speed: 20 km at 20 km/h downhill.
    assert network.get_travel_time([b, a], 20.0) == pytest.approx(60.0)
    # A hop shorter than grade_window_km has no grade.
    assert network.get_travel_time([c, d]) == pytest.approx(1.5)


def test_extraction_gives_category_and_locomotives_per_leg():
    """A hand-written solution (no solver): one intermodal and one coal train, A to D."""
    from railgen.schedule_gen.candidates import CandidateService
    from railgen.schedule_gen.model import Solution
    from railgen.schedule_gen.solution_to_trains import to_trains

    network, path = _climbing_line()

    def service(sid: str, train_type: str) -> CandidateService:
        return CandidateService(
            id=sid,
            train_type=train_type,
            path=tuple(path),
            stop_positions=(0, 3),
            leg_km=(31.0,),
            leg_capacity_tons=(12_000.0,),
        )

    tons = {"S0": 6_000.0, "S1": 10_000.0}  # one run a week each
    solution = Solution(
        runs_per_week={"S0": 1, "S1": 1},
        on_board={(sid, 1): t for sid, t in tons.items()},
        pick_up={(sid, 0): t for sid, t in tons.items()},
        set_out={(sid, 1): t for sid, t in tons.items()},
        objective=0.0,
        bound=0.0,
        gap=0.0,
        status="hand-written",
    )
    trains = to_trains(
        solution,
        {"S0": service("S0", "intermodal"), "S1": service("S1", "coal")},
        network,
        ServiceDesignSettings(),
        random.Random(1),
    )

    intermodal, coal = trains
    assert (intermodal.category, intermodal.symbol) == (TrainCategory.I, "ITRN_1")
    assert (coal.category, coal.symbol) == (TrainCategory.B, "BTRN_1")
    for train, freight, locomotives in [(intermodal, 6_000.0, 5), (coal, 10_000.0, 4)]:
        (leg,) = train.route  # no work or crew stop on the way
        day = next(x for x in DayOfWeek if train.days_active.get_day(x))
        # The tonnage of the leg is its freight; the locomotives also pull the cars.
        # Intermodal: 6,000 t of freight in cars of 31 t carrying 50 t -> 9,720 t;
        # x 1.9 hp/t = 18,468 hp -> 5 locomotives of 4,400 hp.
        # Coal: 10,000 t in cars of 29 t carrying 100 t -> 12,900 t; the 1.5% grade needs
        # 1.04 hp/t, above its 0.78: 13,424 hp -> 4 (not 3).
        assert leg.tonnage_by_day.get_day(day) == pytest.approx(freight)
        assert leg.req_counts_by_day.get_day(day) == locomotives
        assert sum(leg.req_counts_by_day.get_day(x) for x in DayOfWeek) == locomotives


def _line_service(sid: str, train_type: str):
    from railgen.schedule_gen.candidates import CandidateService

    yards = (YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4"))
    return CandidateService(
        id=sid,
        train_type=train_type,
        path=yards,
        stop_positions=(0, 1, 2, 3),
        leg_km=(100.0, 100.0, 100.0),
        leg_capacity_tons=(1000.0, 1000.0, 1000.0),
    )


class _StubNetwork:
    """Just what extraction needs: config, elevations, and 5 hours per hop."""

    def __init__(self, yards):
        from types import SimpleNamespace

        from railgen.schedule_gen.schedule_config import Config

        self.cfg = Config()
        self.ygs_dict = {y: SimpleNamespace(elevation=0.0) for y in yards}

    def get_travel_time(self, path, speed_km_h=None):
        return 300.0 * (len(path) - 1)

    def get_ruling_grade(self, path):
        return 0.0


def test_extraction_rows_match_solution():
    from railgen.schedule_gen.model import Solution
    from railgen.schedule_gen.solution_to_trains import to_trains

    a, b, c, d = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")
    # A hand-written solution: the A-B-C-D train runs twice a week and carries, per week,
    # 1,500 t A to D, 200 t A to C, 300 t B to D and 100 t C to D.
    solution = Solution(
        runs_per_week={"S0": 2},
        on_board={("S0", 1): 1700.0, ("S0", 2): 2000.0, ("S0", 3): 1900.0},
        pick_up={("S0", 0): 1700.0, ("S0", 1): 300.0, ("S0", 2): 100.0},
        set_out={("S0", 2): 200.0, ("S0", 3): 1900.0},
        objective=0.0,
        bound=0.0,
        gap=0.0,
        status="hand-written",
    )
    trains = to_trains(
        solution,
        {"S0": _line_service("S0", "manifest")},
        _StubNetwork([a, b, c, d]),  # ty: ignore[invalid-argument-type]
        ServiceDesignSettings(),
        random.Random(1),
    )

    assert len(trains) == 1  # 2 runs -> one symbol running 2 days
    train = trains[0]
    assert sum(train.days_active.get_day(x) for x in DayOfWeek) == 2
    segs = train.route
    assert segs[0].bgn_yg == a and segs[-1].end_yg == d
    assert segs[0].bgn_yg_reason.pick_up and segs[-1].end_yg_reason.set_out
    # Per-run tons on the first leg: (1500 + 200) / 2 runs; B and C are work stops.
    active = next(x for x in DayOfWeek if train.days_active.get_day(x))
    assert segs[0].tonnage_by_day.get_day(active) == pytest.approx(850.0)
    # With its cars (33 t carrying 82.5 t each): 1,190 t x 1.5 hp/t -> 1 locomotive.
    assert segs[0].req_counts_by_day.get_day(active) == 1
    assert [s.end_yg for s in segs] == [b, c, d]
    # 5 h hops: duty 5 at B, then B's 60 min pick-up + 5 h -> 11 h, then C would exceed 12 h.
    assert segs[1].end_yg_reason.crew_change


# --- Model (needs gurobipy and a license; skipped otherwise) ---------------------------


def _gurobi_available() -> bool:
    try:
        import gurobipy as gp

        gp.Model().dispose()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_gurobi = pytest.mark.skipif(
    not _gurobi_available(), reason="gurobipy or license unavailable"
)


def _line_candidate(sid: str, path: tuple[YgKey, ...]):
    from railgen.schedule_gen.candidates import CandidateService

    n = len(path) - 1
    return CandidateService(
        id=sid,
        train_type="manifest",
        path=path,
        stop_positions=tuple(range(n + 1)),
        leg_km=(100.0,) * n,
        leg_capacity_tons=(1000.0,) * n,
    )


def test_od_index_keeps_each_commodity_on_its_route():
    from railgen.schedule_gen.model import build_od_index
    from railgen.schedule_gen.routes import Route

    # k0 runs A-B-C, k1 runs D-B-E; the A-B-E train shares A-B with k0 and B-E with k1.
    a, b, c, d, e = (YgKey(f"{x}-{i}") for i, x in enumerate("ABCDE", start=1))
    zp = (ZoneCode("001"), ZoneCode("002"))
    cs = [
        Commodity(
            origin=a,
            destination=c,
            train_type="manifest",
            tons_per_week=100.0,
            zone_pair=zp,
        ),
        Commodity(
            origin=d,
            destination=e,
            train_type="manifest",
            tons_per_week=50.0,
            zone_pair=zp,
        ),
    ]
    routes = {
        (a, c): Route(path=(a, b, c), cost=1.0, km=200.0),
        (d, e): Route(path=(d, b, e), cost=1.0, km=200.0),
    }
    cands = [_line_candidate("S0", (a, b, e))]
    index = build_od_index(cs, routes, cands)
    assert index.riders == {("S0", 1): [0], ("S0", 2): [1]}


def test_stop_events_derived_from_flows():
    from railgen.schedule_gen.model import derive_stop_events

    a, b, c, d = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")
    services = {"S0": _line_candidate("S0", (a, b, c, d))}
    # k0 rides legs 1-3 (A to D); k1 rides leg 2 only (B to C).
    flows = {
        ("S0", 1, 0): 500.0,
        ("S0", 2, 0): 500.0,
        ("S0", 3, 0): 500.0,
        ("S0", 2, 1): 200.0,
    }
    on_board, pick_up, set_out = derive_stop_events(flows, services)
    assert on_board == {("S0", 1): 500.0, ("S0", 2): 700.0, ("S0", 3): 500.0}
    assert pick_up == {("S0", 0): 500.0, ("S0", 1): 200.0}
    assert set_out == {("S0", 2): 200.0, ("S0", 3): 500.0}


@needs_gurobi
def test_od_model_puts_short_flow_on_long_train():
    from railgen.schedule_gen.model import (
        build_od_index,
        solve_od_model,
    )
    from railgen.schedule_gen.routes import Route
    from railgen.schedule_gen.solution_to_trains import deviations, solution_yard_net

    a, b, c, d = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")
    zp = (ZoneCode("001"), ZoneCode("002"))
    cs = [
        Commodity(
            origin=a,
            destination=d,
            train_type="manifest",
            tons_per_week=1500.0,
            zone_pair=zp,
        ),
        Commodity(
            origin=b,
            destination=c,
            train_type="manifest",
            tons_per_week=300.0,
            zone_pair=zp,
        ),
    ]
    routes = {
        (a, d): Route(path=(a, b, c, d), cost=1.0, km=300.0),
        (b, c): Route(path=(b, c), cost=1.0, km=100.0),
    }
    cands = [_line_candidate("S0", (a, b, c, d)), _line_candidate("S1", (b, c))]
    settings = ServiceDesignSettings(
        solver=SolverSettings(log_file=None)
    )  # no log file
    index = build_od_index(cs, routes, cands)
    sol = solve_od_model(index, settings, verbose=False)
    # Two A-D runs (capacity 2,000 per leg) carry 1,500 t A-D plus 300 t B-C; no B-C train.
    assert sol.runs_per_week == {"S0": 2}
    assert sol.on_board[("S0", 2)] == pytest.approx(1800.0)
    assert sol.pick_up[("S0", 1)] == pytest.approx(300.0)
    assert sol.set_out[("S0", 2)] == pytest.approx(300.0)
    assert deviations(solution_yard_net(sol, index), yard_targets(cs)) == {}


def test_small_zone_pair_flows_counted_then_dropped():
    from railgen.schedule_gen.demand import drop_small_zone_pair_flows

    z1, z2, z3 = ZoneCode("001"), ZoneCode("002"), ZoneCode("003")
    # With `shared_bulk` the bulk groups arrive as one `bulk` train type.
    by_pair = {
        (z1, z2): {"automotive": 100.0, "manifest": 100.0},  # A 100 < 150: dropped
        (z1, z3): {"automotive": 200.0},  # A kept
        (z2, z3): {"bulk": 110.0},  # B 110 >= 100: kept
        (z3, z1): {"bulk": 40.0},  # B 40 < 100: dropped, zone pair disappears
    }
    settings = ServiceDesignSettings(min_od_tons_per_week={"A": 150.0, "B": 100.0})
    kept, report = drop_small_zone_pair_flows(by_pair, settings)
    assert kept == {
        (z1, z2): {"manifest": 100.0},  # M has no minimum
        (z1, z3): {"automotive": 200.0},
        (z2, z3): {"bulk": 110.0},
    }
    assert report["A"].ods_below == 1
    assert report["A"].ods_total == 2
    assert report["A"].tons_dropped == pytest.approx(100.0)
    assert report["A"].tons_total == pytest.approx(300.0)
    assert report["A"].share_dropped == pytest.approx(1 / 3)
    assert report["B"].ods_below == 1
    assert report["B"].ods_total == 2
    assert report["B"].tons_dropped == pytest.approx(40.0)


def test_without_shared_bulk_each_group_must_reach_the_minimum():
    from railgen.schedule_gen.demand import drop_small_zone_pair_flows

    z1, z2 = ZoneCode("001"), ZoneCode("002")
    # Each B group rides its own trains, so each needs the B minimum on its own.
    by_pair = {(z1, z2): {"coal": 20_000.0, "grain": 15_000.0, "petroleum_energy": 9.2}}
    settings = ServiceDesignSettings(
        shared_bulk=False, min_od_tons_per_week={"B": 20_000.0}
    )
    kept, report = drop_small_zone_pair_flows(by_pair, settings)
    assert kept == {(z1, z2): {"coal": 20_000.0}}
    assert report["B"].ods_below == 2
    assert report["B"].ods_total == 3
    assert report["B"].tons_dropped == pytest.approx(15_009.2)


def test_shared_bulk_groups_are_one_train_type_and_category_b():
    settings = ServiceDesignSettings()
    assert {settings.train_type_of(g) for g in ("coal", "grain", "aggregates")} == {
        "bulk"
    }
    assert settings.category_of("bulk") == "B"
    assert settings.train_type_of("manifest") == "manifest"
    unshared = ServiceDesignSettings(shared_bulk=False)
    assert unshared.train_type_of("coal") == "coal"
    assert unshared.category_of("coal") == "B"


def test_empty_end_legs_are_cut_and_flows_keep_their_tons():
    from railgen.schedule_gen.model import Solution
    from railgen.schedule_gen.solution_to_trains import cut_empty_legs

    yards = tuple(YgKey(f"Y{i}-{i}") for i in range(6))
    service = _line_candidate("S0", yards)  # 5 legs of 100 km
    # Freight only on legs 2-4: commodity 0 rides 2-4, commodity 1 rides 3.
    flows = {
        ("S0", 2, 0): 300.0,
        ("S0", 3, 0): 300.0,
        ("S0", 4, 0): 300.0,
        ("S0", 3, 1): 50.0,
    }
    full = Solution(
        runs_per_week={"S0": 2, "S1": 1},  # S1 carries nothing
        on_board={},
        pick_up={},
        set_out={},
        objective=0.0,
        bound=0.0,
        gap=0.0,
        status="hand-written",
        flows=flows,
    )
    services = {"S0": service, "S1": _line_candidate("S1", yards)}
    solution, pieces, report = cut_empty_legs(full, services)
    assert set(pieces) == {"S0"}
    cut = pieces["S0"]
    assert cut.path == yards[1:5]
    assert cut.stop_positions == (0, 1, 2, 3)
    assert cut.leg_km == (100.0, 100.0, 100.0)
    assert solution.runs_per_week == {"S0": 2}
    assert solution.flows == {
        ("S0", 1, 0): 300.0,
        ("S0", 2, 0): 300.0,
        ("S0", 3, 0): 300.0,
        ("S0", 2, 1): 50.0,
    }
    assert solution.pick_up == {("S0", 0): 300.0, ("S0", 1): 50.0}
    assert solution.set_out == {("S0", 2): 50.0, ("S0", 3): 300.0}
    assert (report.end_legs, report.middle_legs, report.services) == (2, 0, 1)
    assert (report.added_services, report.empty_services) == (0, 1)
    assert report.train_km_per_week == pytest.approx(2 * 200.0)


def test_empty_middle_legs_split_the_train():
    from railgen.schedule_gen.model import Solution
    from railgen.schedule_gen.solution_to_trains import cut_empty_legs

    yards = tuple(YgKey(f"Y{i}-{i}") for i in range(7))
    service = _line_candidate("S0", yards)  # 6 legs of 100 km
    # Freight on legs 1-2 (commodity 0) and 5 (commodity 1); 3, 4 and 6 are empty.
    flows = {("S0", 1, 0): 300.0, ("S0", 2, 0): 300.0, ("S0", 5, 1): 80.0}
    full = Solution(
        runs_per_week={"S0": 3},
        on_board={},
        pick_up={},
        set_out={},
        objective=0.0,
        bound=0.0,
        gap=0.0,
        status="hand-written",
        flows=flows,
    )
    solution, pieces, report = cut_empty_legs(full, {"S0": service})
    assert {p: s.path for p, s in pieces.items()} == {
        "S0-1": yards[0:3],
        "S0-2": yards[4:6],
    }
    assert solution.runs_per_week == {"S0-1": 3, "S0-2": 3}
    assert solution.flows == {
        ("S0-1", 1, 0): 300.0,
        ("S0-1", 2, 0): 300.0,
        ("S0-2", 1, 1): 80.0,
    }
    assert solution.pick_up == {("S0-1", 0): 300.0, ("S0-2", 0): 80.0}
    assert solution.set_out == {("S0-1", 2): 300.0, ("S0-2", 1): 80.0}
    assert (report.end_legs, report.middle_legs, report.added_services) == (1, 2, 1)
    assert report.train_km_per_week == pytest.approx(3 * 300.0)


def test_capacity_follows_train_category():
    settings = ServiceDesignSettings(
        max_trailing_tons_per_run_by_category={
            "A": 11_000.0,
            "M": 30_000.0,
            "B": 45_000.0,
            "I": 25_000.0,
        },
    )
    cars = settings.car_weight
    # Freight per run = max trailing tons / (1 + empty car / cargo per car).
    assert settings.capacity_of("coal") == pytest.approx(45_000.0 / (1 + 29 / 100))
    assert settings.capacity_of("manifest") == pytest.approx(
        30_000.0 / cars.trailing_tons("M", 1.0)
    )
    assert settings.capacity_of("bulk") == settings.capacity_of("coal")
    with pytest.raises(ValueError, match="missing categories"):
        ServiceDesignSettings(max_trailing_tons_per_run_by_category={"M": 30_000.0})
    # A full run, freight plus cars, weighs exactly the limit.
    assert cars.trailing_tons("M", settings.capacity_of("manifest")) == pytest.approx(
        30_000.0
    )


# --- Saved solutions and per-category models (no solver) -------------------------------


def _saved_solution_case():
    """A-B-C-D train with k0 on legs 1-3 and k1 on leg 2, as a model index and a solution."""
    from railgen.schedule_gen.model import OdModelIndex, Solution, derive_stop_events

    a, b, c, d = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")
    zp = (ZoneCode("001"), ZoneCode("002"))
    commodities = [
        Commodity(
            origin=a,
            destination=d,
            train_type="manifest",
            tons_per_week=500.0,
            zone_pair=zp,
        ),
        Commodity(
            origin=b,
            destination=c,
            train_type="manifest",
            tons_per_week=200.0,
            zone_pair=zp,
        ),
    ]
    index = OdModelIndex(
        services={
            "S0": _line_candidate("S0", (a, b, c, d)),
            "S1": _line_candidate("S1", (b, c)),  # not selected
        },
        commodities=commodities,
        riders={("S0", 1): [0], ("S0", 2): [0, 1], ("S0", 3): [0], ("S1", 1): [1]},
        own_tons={"S0": 500.0, "S1": 200.0},
    )
    flows = {
        ("S0", 1, 0): 500.0,
        ("S0", 2, 0): 500.0,
        ("S0", 3, 0): 500.0,
        ("S0", 2, 1): 200.0,
    }
    on_board, pick_up, set_out = derive_stop_events(flows, index.services)
    solution = Solution(
        runs_per_week={"S0": 2},
        on_board=on_board,
        pick_up=pick_up,
        set_out=set_out,
        objective=1234.5,
        bound=1200.0,
        gap=0.028,
        status="9",
        flows=flows,
    )
    return index, solution


def test_saved_solution_reads_back_unchanged(tmp_path):
    from railgen.schedule_gen.solve_by_category import (
        instance_key,
        read_solution,
        save_solution,
    )

    index, solution = _saved_solution_case()
    settings = ServiceDesignSettings()
    path = tmp_path / "solution_M.json"
    save_solution(path, "M", solution, index, settings)
    saved = read_solution(path)

    assert saved.solution == solution  # runs, flows, tons on board, pick-ups, set-outs
    assert saved.category == "M"
    assert saved.instance_key == instance_key(index, settings)
    assert saved.services == {"S0": index.services["S0"]}  # only services in use
    assert saved.commodities == index.commodities


def test_instance_key_changes_with_the_model():
    from dataclasses import replace

    from railgen.schedule_gen.solve_by_category import instance_key

    index, _ = _saved_solution_case()
    settings = ServiceDesignSettings()
    key = instance_key(index, settings)
    assert instance_key(index, settings) == key

    heavier = replace(index.commodities[0], tons_per_week=501.0)
    changed = replace(index, commodities=[heavier, index.commodities[1]])
    assert instance_key(changed, settings) != key
    # Solver limits do not change the model.
    relaxed = settings.model_copy(
        update={"solver": settings.solver.model_copy(update={"time_limit_s": 5.0})}
    )
    assert instance_key(index, relaxed) == key


def test_every_run_solves_and_saves_in_its_own_folder(tmp_path, monkeypatch):
    from railgen.schedule_gen import solve_by_category
    from railgen.schedule_gen.solve_by_category import CategoryModel, read_solution

    index, solution = _saved_solution_case()
    model = CategoryModel(
        name="M",
        category="M",
        settings=ServiceDesignSettings(),
        commodities=index.commodities,
        candidates=list(index.services.values()),
        index=index,
    )
    log_files = []

    def fake_solve(idx, settings):
        log_files.append(settings.solver.log_file)
        return solution

    monkeypatch.setattr(solve_by_category, "solve_od_model", fake_solve)

    for run in ("run1", "run2"):
        result, path = solve_by_category.solve_and_save(model, tmp_path / run)
        assert result == solution
        assert path == tmp_path / run / "solution_M.json"
        assert read_solution(path).solution.runs_per_week == solution.runs_per_week
    # Solved again on the second run, each with its log in its own folder.
    assert log_files == [
        str(tmp_path / run / "gurobi_service_design.log") for run in ("run1", "run2")
    ]


def test_each_group_rides_only_its_own_train_type():
    from railgen.schedule_gen.solve_by_category import category_settings

    settings = ServiceDesignSettings()
    assert settings.train_type_of("manifest") == "manifest"
    assert settings.train_type_of("coal") == settings.train_type_of("grain") == "bulk"
    assert settings.category_of("bulk") == "B"
    assert settings.capacity_of("bulk") == settings.capacity_of("coal")

    separate = settings.model_copy(update={"shared_bulk": False})
    assert separate.train_type_of("coal") == "coal"
    assert category_settings(settings, "M").solver.log_file == "gurobi_M.log"
    no_log = settings.model_copy(
        update={"solver": settings.solver.model_copy(update={"log_file": None})}
    )
    assert category_settings(no_log, "M").solver.log_file is None


def test_multimodal_rows_count_on_long_zone_pairs_only():
    from railgen.schedule_gen.demand import multimodal_counts

    assert multimodal_counts(500.0, 500.0)
    assert not multimodal_counts(499.9, 500.0)
    assert not multimodal_counts(None, 500.0)  # no FAF5 distance for the zone pair
    assert multimodal_counts(12.0, 0.0) and multimodal_counts(None, 0.0)  # rule off
    assert multimodal_counts(2500.0, 500.0, 2500.0)
    assert not multimodal_counts(2500.1, 500.0, 2500.0)
    assert not multimodal_counts(None, 0.0, 2500.0)
    assert multimodal_counts(9000.0, 500.0, 0.0)  # no upper limit


def test_stop_report_writes_the_csv_and_returns_the_lines(tmp_path):
    from railgen.schedule_gen.solution_to_trains import write_stop_report

    index, solution = _saved_solution_case()
    lines = write_stop_report(
        "M", solution, index.services, index.commodities, tmp_path
    )
    written = (tmp_path / "stop_events_M.csv").read_text().splitlines()
    assert len(written) == 1 + 4  # header + the four stops of the selected train
    assert lines[0].startswith("picked up t/week: at origin 500")
    assert any(line.startswith("transfers") for line in lines)


def test_stop_event_rows():
    from railgen.schedule_gen.solution_to_trains import stop_events

    index, solution = _saved_solution_case()
    rows = stop_events("M", solution, index.services)
    assert [r["role"] for r in rows] == [
        "origin",
        "intermediate",
        "intermediate",
        "destination",
    ]
    # Weekly tons over 2 runs: 500 on at A, 200 on at B, 200 off at C, 500 off at D.
    assert [r["pu_tons_run"] for r in rows] == [250.0, 100.0, 0.0, 0.0]
    assert [r["so_tons_run"] for r in rows] == [0.0, 0.0, 100.0, 250.0]
    assert [r["leaving_tons_run"] for r in rows] == [250.0, 350.0, 250.0, 0.0]


@needs_gurobi
def test_transfer_cost_stops_a_relay():
    from railgen.schedule_gen.model import (
        build_od_index,
        solve_od_model,
    )
    from railgen.schedule_gen.routes import Route
    from railgen.schedule_gen.schedule_config import Costs
    from railgen.schedule_gen.solution_to_trains import transfers

    a, b, c, e, f = (YgKey(f"{x}-{i}") for i, x in enumerate("ABCEF", start=1))
    zp = (ZoneCode("001"), ZoneCode("002"))
    # k0 runs A-B-C. The E-A-B and B-C-F trains must run for k1 and k2 anyway, and
    # together they cover A-B and B-C, so k0 can relay over them with a change at B.
    paths = [(a, b, c), (e, a, b), (b, c, f)]
    cs = [
        Commodity(
            origin=p[0],
            destination=p[-1],
            train_type="manifest",
            tons_per_week=500.0,
            zone_pair=zp,
        )
        for p in paths
    ]
    routes = {(p[0], p[-1]): Route(path=p, cost=1.0, km=200.0) for p in paths}
    cands = [_line_candidate(name, p) for name, p in zip(("AC", "EB", "BF"), paths)]
    train = 400 + 10 * 200 / 1.609344  # one run of any of the three, about 1,643

    def solve(fixed: float, per_ton: float, per_distinct_train: float = 0.0):
        settings = ServiceDesignSettings(
            costs=Costs(
                transfer_fixed=fixed,
                transfer_per_ton=per_ton,
                per_distinct_train=per_distinct_train,
            ),
            solver=SolverSettings(log_file=None),  # no log file
        )
        index = build_od_index(cs, routes, cands)
        sol = solve_od_model(index, settings, verbose=False)
        return sol, transfers(sol, index.services, cs)

    # Free transfers: k0 relays, the A-C train does not run.
    sol, moved = solve(0.0, 0.0)
    assert sol.runs_per_week == {"EB": 1, "BF": 1}
    assert moved == {(0, b): pytest.approx(500.0)}

    # A cheap transfer (200 + 1 x 500 = 700) is still below the A-C train.
    sol, moved = solve(200.0, 1.0)
    assert sol.runs_per_week == {"EB": 1, "BF": 1}
    assert sol.objective == pytest.approx(2 * train + 700)

    # A dear one (200 + 10 x 500 = 5,200) is not: the A-C train runs, nothing transfers.
    sol, moved = solve(200.0, 10.0)
    assert sol.runs_per_week == {"AC": 1, "EB": 1, "BF": 1}
    assert moved == {}
    assert sol.objective == pytest.approx(3 * train)

    # A weekly charge of 10,000 per distinct train makes the dear transfer worth it
    # again: dropping the A-C train saves 10,000 + 1,643 against 5,200.
    sol, moved = solve(200.0, 10.0, per_distinct_train=10_000.0)
    assert sol.runs_per_week == {"EB": 1, "BF": 1}
    assert moved == {(0, b): pytest.approx(500.0)}
    assert sol.objective == pytest.approx(2 * train + 2 * 10_000 + 5_200)


def test_intra_zone_miles_fall_back_to_the_zone_average():
    from railgen.schedule_gen.demand import IntraZoneMiles

    z, other = ZoneCode("111"), ZoneCode("222")
    miles = IntraZoneMiles(
        by_commodity={(z, "15", "2"): 12.0}, by_zone={(z, "2"): 40.0}
    )
    assert miles.miles(z, "15", "2") == 12.0  # FAF5 has this zone, commodity and mode
    assert miles.miles(z, "02", "2") == 40.0  # zone average for the mode
    assert miles.miles(z, "15", "5") is None  # nothing for this mode
    assert miles.miles(other, "15", "2") is None  # zone not in FAF5


def test_multimodal_is_its_own_group_and_category():
    from railgen.schedule_gen.schedule_config import INTERMODAL_GROUP

    settings = ServiceDesignSettings()
    assert settings.train_type_of(INTERMODAL_GROUP) == INTERMODAL_GROUP
    assert settings.train_type_category[INTERMODAL_GROUP] == "I"
    assert settings.multimodal_rail_share == 0.5


def test_one_yard_pair_per_category_largest_pair_to_heaviest_category():
    from railgen.schedule_gen.demand import assign_pair_per_category

    size = {A1: 30.0, A2: 20.0, A3: 10.0, B1: 5.0, B2: 4.0, B3: 1.0}
    # route costs: A1-B1 100 (cheapest) ... A3-B3 300
    cands = _cands()
    tons = {"B": 9000.0, "M": 5000.0, "I": 2000.0}

    # No tolerance: pairs ranked by size product: A1-B1 150, A1-B2 120, A2-B1 100.
    pair_of = assign_pair_per_category(cands, tons, size, distance_tolerance=None)
    assert pair_of == {"B": (A1, B1), "M": (A1, B2), "I": (A2, B1)}

    # 6% tolerance keeps only A1-B1 (100) and A1-B2 (105): the third category reuses
    # the largest pair.
    pair_of = assign_pair_per_category(cands, tons, size, distance_tolerance=0.06)
    assert pair_of == {"B": (A1, B1), "M": (A1, B2), "I": (A1, B1)}


def test_kept_lines_split_kept_freight_by_mode_and_trade_type():
    report = DemandReport(
        kept_by_mode_trade={
            (RAIL_MODE, "1"): 600.0,
            (RAIL_MODE, "3"): 150.0,
            (MULTIMODAL_MODE, "2"): 250.0,
        }
    )
    assert kept_lines(report, ServiceDesignSettings()) == [
        "kept after every drop: 1,000 t/week",
        "  rail: 750 t/week (75.0%; domestic 600, import 0, export 150)",
        "  multimodal at 50%: 250 t/week (25.0%; domestic 0, import 250, export 0)",
    ]
