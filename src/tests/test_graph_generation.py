"""Unit tests for the pure pieces of graph generation (no geojson, no FAF file)."""

import networkx as nx

from railgen.models import (
    Coord,
    YardGroup,
    YgKey,
    rail_network_adapter,
    yg_adapter,
)
from railgen.rail_network_gen.graph_generation import (
    _spaced,
    _track_closeness,
    _with_min_track,
    _zone_yard_groups_table,
    drop_redundant_edges,
    merge_graph_nodes,
    merge_yard_group_records,
    traverse_graph,
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


def _yard(name: str, track_km: float) -> YardGroup:
    return YardGroup(
        id="1",
        name=name,
        owner="",
        coord=Coord(lat=0.0, lon=0.0),
        elevation=0.0,
        shape_length_km=track_km,
    )


def _merges(pairs: dict[str, str]) -> dict[YgKey, YgKey]:
    return {YgKey(a): YgKey(b) for a, b in pairs.items()}


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
        YgKey(n): _yard(n, km) for n, km in (("A", 1.0), ("B", 80.0), ("B2", 45.0))
    }
    ygs = merge_yard_group_records(yards, _merges({"B2": "B"}))
    assert set(ygs) == {"A", "B"}
    assert ygs[YgKey("B")].shape_length_km == 125.0


def test_merging_records_follows_a_chain_and_skips_missing_yards():
    yards = {YgKey(n): _yard(n, 1.0) for n in ("A", "B", "B2", "B3")}
    ygs = merge_yard_group_records(
        yards, _merges({"B3": "B2", "B2": "B", "X": "B", "A": "Y"})
    )
    assert set(ygs) == {"A", "B"}
    assert ygs[YgKey("B")].shape_length_km == 3.0


# --- Walk with dropped and merged records ----------------------------------------------


def _point(i: int) -> list[float]:
    return [0.01 * i, 0.0]  # [lon, lat]: points 1.1 km apart on the equator


def _walk(
    yards: dict[str, list[int]],
    kept: list[str],
    merges: dict[str, str],
    excluded_track_net_codes: tuple[str, ...] = (),
) -> dict[frozenset[str], float]:
    """Edges of the walk over one straight track of 11 points; `yards` gives each record's points.

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
    raw_ygs = yg_adapter.validate_python(
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
    ygs = {
        YgKey(f"{name}-1"): YardGroup(
            id="1",
            name=name,
            owner="",
            coord=Coord(lat=0.0, lon=0.01 * yards[name][0]),
            elevation=0.0,
        )
        for name in kept
    }
    edges = traverse_graph(
        ygs,
        raw_ygs,
        track_line,
        {YgKey(f"{a}-1"): YgKey(f"{b}-1") for a, b in merges.items()},
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


def test_walk_handles_two_points_at_the_same_distance():
    # From S the track runs equally far both ways, so two points are at the same distance.
    edges = _walk({"A": [0], "S": [5], "C": [10]}, kept=["A", "S", "C"], merges={})
    assert set(edges) == {frozenset(("A-1", "S-1")), frozenset(("S-1", "C-1"))}


# --- Choice of zone yard groups (pure pieces) ------------------------------------------


def test_spaced_skips_candidates_too_close_to_one_already_taken():
    graph = _graph([("A", "B", 2.0), ("B", "C", 4.0), ("C", "D", 20.0)])
    too_close = _track_closeness(graph, 5.0)
    order = [YgKey(name) for name in ("A", "B", "C", "D")]
    # B is 2 km from A; C is 6 km from A by track; D is far from both.
    assert _spaced(order, 3, too_close) == ["A", "C", "D"]
    assert _spaced(order, 2, too_close) == ["A", "C"]


def test_track_distance_exactly_at_the_minimum_is_not_too_close():
    too_close = _track_closeness(_graph([("A", "B", 5.0)]), 5.0)
    assert not too_close(YgKey("A"), YgKey("B"))


def test_min_track_filters_small_yards_and_is_waived_when_none_is_large_enough():
    yards = {
        YgKey("SMALL"): _yard("SMALL", 0.4),
        YgKey("BIG"): _yard("BIG", 22.0),
        YgKey("TINY"): _yard("TINY", 1.8),
    }
    nearest_first = [YgKey("SMALL"), YgKey("BIG"), YgKey("TINY")]
    assert _with_min_track(nearest_first, yards, 2.0) == ["BIG"]
    assert _with_min_track([YgKey("SMALL"), YgKey("TINY")], yards, 2.0) == [
        "SMALL",
        "TINY",
    ]


def test_zone_table_escapes_latex_characters_and_marks_empty_zones():
    table = _zone_yard_groups_table(
        [
            (
                "172",
                "East St. Louis, IL",
                1,
                [("WIGGINS #2-2005", 6.17), ("A&B-1", 2.0)],
            ),
            ("151", "Honolulu, HI", 2, []),
        ]
    )
    assert (
        r"172 & East St. Louis, IL & 1 & WIGGINS \#2-2005 (6.2); A\&B-1 (2.0) \\"
        in table
    )
    assert r"151 & Honolulu, HI & 2 & none \\" in table
    assert table.startswith("%") and table.rstrip().endswith(r"\end{longtable}")
