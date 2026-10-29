"""Unit tests for the pure pieces of graph generation (no geojson, no FAF file)."""

import networkx as nx
import pytest

from railgen.faf_data import LazyLoad
from railgen.models import (
    Coord,
    Yard,
    YdKey,
    ZoneCode,
    rail_network_adapter,
    yd_adapter,
)
from railgen.rail_network_gen import graph_generation
from railgen.rail_network_gen.graph_generation import (
    _assign_od_yards,
    _spaced,
    _track_closeness,
    _with_min_track,
    drop_bypassed_edges,
    drop_long_edges,
    drop_redundant_edges,
    merge_graph_nodes,
    merge_yard_records,
    traverse_graph,
    warn_unmatched_changes,
)


def _graph(edges: list[tuple[str, str, float]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph


def _edges(graph: nx.Graph) -> set[frozenset[str]]:
    return {frozenset(edge) for edge in graph.edges()}


def test_drops_edge_when_joint_is_shorter():
    # A-C is a detour: going through B is shorter.
    graph = _graph([("A", "B", 100.0), ("B", "C", 100.0), ("A", "C", 250.0)])
    kept = _edges(drop_redundant_edges(graph, tolerance=0.03))
    assert kept == {frozenset("AB"), frozenset("BC")}


def test_drops_edge_when_joint_is_within_tolerance():
    # Same track passing B: the joint is 2% longer than A-C.
    graph = _graph([("A", "B", 260.0), ("B", "C", 250.0), ("A", "C", 500.0)])
    kept = _edges(drop_redundant_edges(graph, tolerance=0.03))
    assert kept == {frozenset("AB"), frozenset("BC")}


def test_keeps_edge_when_joint_is_longer_than_tolerance():
    # The joint is 4% longer than A-C, so A-C is a different, shorter connection.
    graph = _graph([("A", "B", 270.0), ("B", "C", 250.0), ("A", "C", 500.0)])
    kept = _edges(drop_redundant_edges(graph, tolerance=0.03))
    assert kept == {frozenset("AB"), frozenset("BC"), frozenset("AC")}


def test_near_duplicate_edges_do_not_drop_each_other():
    # B and C are next to each other. A-B is covered by A-C + C-B, but A-C must stay:
    # its only joint uses A-B, which is not shorter than A-C.
    graph = _graph([("A", "C", 500.0), ("A", "B", 501.0), ("B", "C", 2.0)])
    kept = _edges(drop_redundant_edges(graph, tolerance=0.03))
    assert kept == {frozenset("AC"), frozenset("BC")}


def test_chain_of_yards_keeps_only_consecutive_edges():
    # A, B, C, D in order on one track, with every pass-through edge present.
    graph = _graph(
        [
            ("A", "B", 100.0),
            ("B", "C", 100.0),
            ("C", "D", 100.0),
            ("A", "C", 199.0),
            ("B", "D", 199.0),
            ("A", "D", 298.0),
        ]
    )
    kept = _edges(drop_redundant_edges(graph, tolerance=0.03))
    assert kept == {frozenset("AB"), frozenset("BC"), frozenset("CD")}


def _yard(name: str, track_km: float) -> Yard:
    return Yard(
        id="1",
        name=name,
        owner="",
        coord=Coord(lat=0.0, lon=0.0),
        elevation=0.0,
        shape_length_km=track_km,
    )


def _merges(pairs: dict[str, str]) -> dict[YdKey, YdKey]:
    return {YdKey(a): YdKey(b) for a, b in pairs.items()}


def test_merging_nodes_moves_edges_to_the_kept_yard_with_the_shorter_distance():
    # B2 is a second record of yard B. Both reach C; only B2 reaches D.
    graph = _graph(
        [
            ("A", "B", 100.0),
            ("B", "C", 50.0),
            ("B", "B2", 0.01),
            ("B2", "C", 40.0),
            ("B2", "D", 30.0),
        ]
    )
    graph = merge_graph_nodes(graph, _merges({"B2": "B"}))
    assert set(graph.nodes) == {"A", "B", "C", "D"}
    assert {frozenset((a, b)): w for a, b, w in graph.edges(data="weight")} == {
        frozenset("AB"): 100.0,
        frozenset("BC"): 40.0,
        frozenset("BD"): 30.0,
    }


def test_merging_nodes_follows_a_chain_of_three_records():
    graph = _graph([("A", "B3", 10.0), ("B3", "B2", 0.01), ("B2", "B", 0.01)])
    graph = merge_graph_nodes(graph, _merges({"B3": "B2", "B2": "B"}))
    assert set(graph.nodes) == {"A", "B"}
    assert graph["A"]["B"]["weight"] == 10.0


def test_merging_nodes_leaves_a_graph_generated_with_merged_records_alone():
    # The graph was generated with B2 already part of B, so B2 is not a node.
    graph = merge_graph_nodes(_graph([("A", "B", 10.0)]), _merges({"B2": "B"}))
    assert set(graph.nodes) == {"A", "B"}
    assert graph["A"]["B"]["weight"] == 10.0


def test_merging_records_adds_the_track_length_to_the_kept_yard():
    yards = {
        YdKey(n): _yard(n, km) for n, km in (("A", 1.0), ("B", 80.0), ("B2", 45.0))
    }
    yds = merge_yard_records(yards, _merges({"B2": "B"}))
    assert set(yds) == {"A", "B"}
    assert yds[YdKey("B")].shape_length_km == 125.0


def test_merging_records_follows_a_chain_and_skips_missing_yards():
    yards = {YdKey(n): _yard(n, 1.0) for n in ("A", "B", "B2", "B3")}
    yds = merge_yard_records(
        yards, _merges({"B3": "B2", "B2": "B", "X": "B", "A": "Y"})
    )
    assert set(yds) == {"A", "B"}
    assert yds[YdKey("B")].shape_length_km == 3.0


# --- Walk with dropped and merged records ----------------------------------------------


def _point(i: int) -> list[float]:
    return [0.01 * i, 0.0]  # [lon, lat]: points 1.1 km apart on the equator


def _records(yards: dict[str, list[int]], kept: list[str]):
    """One straight track of 11 points; `yards` gives each record's points.

    The track is two lines: points 0 to 5 are main line (`NET` M) and points 5 to 10 are abandoned (`NET` A).
    """
    track_line = rail_network_adapter.validate_python(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [_point(i) for i in points],
                },
                "properties": {"NET": net},
            }
            for net, points in (("M", range(6)), ("A", range(5, 11)))
        ]
    )
    raw_yds = yd_adapter.validate_python(
        [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [_point(i) for i in points],
                },
                "properties": {"YARDNAME": name, "OBJECTID": 1},
            }
            for name, points in yards.items()
        ]
    )
    yds = {
        YdKey(f"{name}-1"): Yard(
            id="1",
            name=name,
            owner="",
            coord=Coord(lat=0.0, lon=0.01 * yards[name][0]),
            elevation=0.0,
        )
        for name in kept
    }
    return yds, raw_yds, track_line


def _walk(
    yards: dict[str, list[int]],
    kept: list[str],
    merges: dict[str, str],
    excluded_track_net_codes: tuple[str, ...] = (),
) -> dict[frozenset[str], float]:
    yds, raw_yds, track_line = _records(yards, kept)
    edges = traverse_graph(
        yds,
        raw_yds,
        track_line,
        {YdKey(f"{a}-1"): YdKey(f"{b}-1") for a, b in merges.items()},
        excluded_track_net_codes,
    )
    return {frozenset((a, b)): km for a, b, km in edges}


_RECORDS = {"A": [0, 1], "B": [3], "B2": [4], "P": [6], "C": [8, 9]}


def test_walk_stops_at_every_record_by_default():
    edges = _walk(_RECORDS, kept=list(_RECORDS), merges={})
    assert set(edges) == {
        frozenset(("A-1", "B-1")),
        frozenset(("B-1", "B2-1")),
        frozenset(("B2-1", "P-1")),
        frozenset(("P-1", "C-1")),
    }


def test_walk_passes_a_dropped_yard_and_treats_a_merged_record_as_its_yard():
    # P is dropped (not in the yard list) and B2 is a second record of B.
    edges = _walk(_RECORDS, kept=["A", "B", "C"], merges={"B2": "B"})
    assert set(edges) == {frozenset(("A-1", "B-1")), frozenset(("B-1", "C-1"))}
    # An edge is the shorter of the two directions. A to B is two steps (from B back to
    # A's nearest point). B to C is four steps: from C back to B2, which is B's track now.
    assert (
        round(edges[frozenset(("B-1", "C-1"))] / edges[frozenset(("A-1", "B-1"))], 2)
        == 2.0
    )


def test_walk_does_not_use_excluded_track():
    # The track from point 5 on is abandoned, so P and C cannot be reached from B2.
    edges = _walk(
        _RECORDS, kept=list(_RECORDS), merges={}, excluded_track_net_codes=("A",)
    )
    assert set(edges) == {frozenset(("A-1", "B-1")), frozenset(("B-1", "B2-1"))}


def _grade_hops(monkeypatch, yards, hops, elevation):
    yds, raw_yds, track_line = _records(yards, list(yards))
    monkeypatch.setattr(graph_generation, "load_geojson", lambda: (raw_yds, track_line))
    monkeypatch.setattr(
        graph_generation, "sample_elevations", lambda cs: [elevation(c) for c in cs]
    )
    pairs = [(YdKey(f"{a}-1"), YdKey(f"{b}-1")) for a, b in hops]
    km = {(a, b): 1.12 * abs(yards[a][0] - yards[b][0]) for a, b in hops}
    graph = nx.Graph([(*pair, {"weight": km[hop]}) for pair, hop in zip(pairs, hops)])
    return graph_generation.grade_hops(yds, graph, pairs, 2.0)


def test_hop_grade_follows_its_track_in_each_direction(monkeypatch):
    # Flat to point 3, then 22 m per point, 1.11 km apart: 1.98%
    grades = _grade_hops(
        monkeypatch,
        {"A": [0], "B": [3], "C": [5]},
        [("A", "B"), ("C", "B")],
        lambda c: 2200.0 * max(c.lon - 0.03, 0.0),
    )
    assert grades[("A-1", "B-1")] == grades[("B-1", "A-1")] == 0.0
    assert grades[("B-1", "C-1")] == pytest.approx(1.98, abs=0.01)
    assert grades[("C-1", "B-1")] == 0.0


def test_hop_grade_ignores_a_ridge_steeper_than_track(monkeypatch):
    grades = _grade_hops(
        monkeypatch,
        {"A": [0], "C": [10]},
        [("A", "C")],
        lambda c: 300.0 if c.lon == 0.05 else 0.0,
    )
    assert grades == {("A-1", "C-1"): 0.0, ("C-1", "A-1"): 0.0}


def test_hop_grade_finds_track_after_merges_added_to_a_built_graph(monkeypatch):
    # The K-N edge is the walk from X, and M's merge into a missing Q is skipped.
    yds, raw_yds, track_line = _records(
        {"N": [0, 1, 2], "X": [4], "K": [6], "M": [8]}, ["N", "X", "K", "M"]
    )
    monkeypatch.setattr(graph_generation, "load_geojson", lambda: (raw_yds, track_line))
    monkeypatch.setattr(
        graph_generation, "sample_elevations", lambda cs: [0.0] * len(cs)
    )
    edges = traverse_graph(yds, raw_yds, track_line)
    merges = {YdKey("X-1"): YdKey("K-1"), YdKey("M-1"): YdKey("Q-1")}
    graph = merge_graph_nodes(
        nx.Graph([(a, b, {"weight": km}) for a, b, km in edges]), merges
    )
    del yds[YdKey("X-1")]
    grades = graph_generation.grade_hops(yds, graph, list(graph.edges), 2.0, merges)
    assert set(grades) == {
        ("K-1", "N-1"),
        ("N-1", "K-1"),
        ("K-1", "M-1"),
        ("M-1", "K-1"),
    }


def test_walk_handles_two_points_at_the_same_distance():
    # From S the track runs equally far both ways, so two points are at the same distance.
    edges = _walk({"A": [0], "S": [5], "C": [10]}, kept=["A", "S", "C"], merges={})
    assert set(edges) == {frozenset(("A-1", "S-1")), frozenset(("S-1", "C-1"))}


def test_long_edges_are_dropped():
    graph = _graph([("A", "B", 900.0), ("B", "C", 300.0), ("C", "D", 500.0)])
    assert _edges(drop_long_edges(graph, 500.0)) == {
        frozenset(("B", "C")),
        frozenset(("C", "D")),
    }


def test_edge_bypassed_through_several_yards_is_dropped():
    # A-D is 100 km; A-B-C-D is 102 km over two other yards (within 3%); D-E has no bypass.
    graph = _graph(
        [
            ("A", "D", 100.0),
            ("A", "B", 40.0),
            ("B", "C", 30.0),
            ("C", "D", 32.0),
            ("D", "E", 50.0),
        ]
    )
    assert _edges(drop_bypassed_edges(graph)) == {
        frozenset(("A", "B")),
        frozenset(("B", "C")),
        frozenset(("C", "D")),
        frozenset(("D", "E")),
    }
    # At 98 km A-D is more than 3% shorter than the 102-km path, so it stays.
    assert frozenset(("A", "D")) in _edges(
        drop_bypassed_edges(
            _graph(
                [("A", "D", 98.0), ("A", "B", 40.0), ("B", "C", 30.0), ("C", "D", 32.0)]
            )
        )
    )


# --- Choice of zone yards --------------------------------------------------------


def test_spaced_skips_candidates_too_close_to_one_already_taken():
    graph = _graph([("A", "B", 2.0), ("B", "C", 4.0), ("C", "D", 20.0)])
    too_close = _track_closeness(graph, 5.0)
    order = [YdKey(name) for name in ("A", "B", "C", "D")]
    # B is 2 km from A; C is 6 km from A by track; D is far from both.
    assert _spaced(order, 3, too_close) == ["A", "C", "D"]
    assert _spaced(order, 2, too_close) == ["A", "C"]


def test_track_distance_exactly_at_the_minimum_is_not_too_close():
    too_close = _track_closeness(_graph([("A", "B", 5.0)]), 5.0)
    assert not too_close(YdKey("A"), YdKey("B"))


def test_min_track_filters_small_yards_and_is_waived_when_none_is_large_enough():
    yards = {
        YdKey("SMALL"): _yard("SMALL", 0.4),
        YdKey("BIG"): _yard("BIG", 22.0),
        YdKey("TINY"): _yard("TINY", 1.8),
    }
    in_order = [YdKey("SMALL"), YdKey("BIG"), YdKey("TINY")]
    assert _with_min_track(in_order, yards, 2.0) == ["BIG"]
    assert _with_min_track([YdKey("SMALL"), YdKey("TINY")], yards, 2.0) == [
        "SMALL",
        "TINY",
    ]


def test_each_zone_takes_its_largest_yards_inside_it(monkeypatch):
    monkeypatch.setattr(
        graph_generation,
        "FAF_OD_ZONE_CODES",
        LazyLoad(
            lambda: frozenset({ZoneCode("171"), ZoneCode("179"), ZoneCode("109")})
        ),
    )
    tracks = {
        "BIG": 60.0,
        "MID": 30.0,
        "BARRED": 90.0,
        "SMALL": 1.0,
        "RURAL": 40.0,
        "NOFREIGHT": 50.0,
        "OUTSIDE": 70.0,
    }
    yards = {YdKey(n): _yard(n, km) for n, km in tracks.items()}
    zones: dict[YdKey, ZoneCode | None] = {
        YdKey("BIG"): ZoneCode("171"),
        YdKey("MID"): ZoneCode("171"),
        YdKey("BARRED"): ZoneCode("171"),
        YdKey("SMALL"): ZoneCode("171"),
        YdKey("RURAL"): ZoneCode("179"),
        YdKey("NOFREIGHT"): ZoneCode("999"),
        YdKey("OUTSIDE"): None,
    }
    stats = _assign_od_yards(
        yards,
        zones,
        yds_per_zone=3,
        min_track_km=2.0,
        barred={YdKey("BARRED")},
    )
    picked = {n: yd.od_zone_code for n, yd in yards.items() if yd.od_zone_code}
    assert picked == {"BIG": "171", "MID": "171", "RURAL": "179"}
    assert stats.zones_total == 3
    assert stats.zones_represented == 2
    assert stats.zones_unrepresented == ["109"]
    assert stats.zones_under_target == 2
    assert stats.yards_designated == 3


def test_unmatched_changes_are_reported(capsys):
    records = {YdKey("A-1"): _yard("A", 1.0), YdKey("B-2"): _yard("B", 1.0)}
    merges = {YdKey("A-1"): YdKey("B-2"), YdKey("C-3"): YdKey("B-2")}
    missing = warn_unmatched_changes(records, merges, {YdKey("D-4")}, {YdKey("B-2")})
    assert missing == ["C-3", "D-4"]
    assert "2 yard(s)" in capsys.readouterr().out
    assert warn_unmatched_changes(records, {}, set(), set()) == []
