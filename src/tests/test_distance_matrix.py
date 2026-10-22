"""The yard group distance cache grows with the yard groups that runs need."""

import json
from types import SimpleNamespace

import networkx as nx
import pytest

from railgen.models import YgKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen import build_distance_matrix
from railgen.schedule_gen.build_distance_matrix import (
    REFERENCE_SPEED_KM_H,
    generate_output_yg_distances,
)
from railgen.schedule_gen.schedule_config import Config

A, B, C, D = YgKey("A-1"), YgKey("B-2"), YgKey("C-3"), YgKey("D-4")


def _climbing_line(train_speed_km_h: float) -> RailNetwork:
    """A-B-C-D in a line, 60 km apart and each 600 m higher (a 1% grade going up).

    A real `RailNetwork`, so travel times come from its own grade rule; no graph files
    are read.
    """
    network = object.__new__(RailNetwork)
    network.cfg = Config()
    network.train_speed_km_h = train_speed_km_h
    network.ygs_graph = nx.Graph()
    for u, v in [(A, B), (B, C), (C, D)]:
        network.ygs_graph.add_edge(u, v, weight=60.0)
    network.ygs_dict = {  # ty: ignore[invalid-assignment]
        y: SimpleNamespace(elevation=z)
        for y, z in [(A, 0.0), (B, 600.0), (C, 1200.0), (D, 1800.0)]
    }
    return network


def _rows(network: RailNetwork, yards: set[YgKey]) -> dict[tuple[str, str], float]:
    rows = generate_output_yg_distances(network, yards)
    assert all(r.distance_km > 0 for r in rows)
    return {(r.begin_yard_group, r.end_yard_group): r.distance_minutes for r in rows}


def test_distance_cache_only_adds_missing_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(build_distance_matrix, "GRAPH_DIR", tmp_path)
    network = _climbing_line(REFERENCE_SPEED_KM_H)
    cache_path = tmp_path / build_distance_matrix.distance_cache_file(network)

    assert set(_rows(network, {A, C})) == {(A, C), (C, A)}
    first = json.loads(cache_path.read_text())
    assert set(first) == {A, C} and set(first[A]) == {C}  # only what was needed
    assert first[A][C][1] == pytest.approx(120.0)  # km

    # A stored pair is reused as it is; only the pairs with the new yard are added.
    first[A][C] = [999.0, 120.0]
    cache_path.write_text(json.dumps(first))
    minutes = _rows(network, {A, C, D})
    assert len(minutes) == 6
    assert minutes[(A, C)] == pytest.approx(999.0)
    second = json.loads(cache_path.read_text())
    assert second[A][D][1] == pytest.approx(180.0)
    assert B not in second and all(B not in inner for inner in second.values())


def test_minutes_keep_the_grade_slowdown_at_any_train_speed(tmp_path, monkeypatch):
    monkeypatch.setattr(build_distance_matrix, "GRAPH_DIR", tmp_path)
    fast, slow = _climbing_line(60.0), _climbing_line(30.0)

    built_at_60 = _rows(fast, {A, C})  # creates the cache
    read_at_30 = _rows(slow, {A, C})  # same cache file, other train speed

    # Uphill A -> C is slower than downhill C -> A, and both equal the network's own times.
    assert built_at_60[(A, C)] > built_at_60[(C, A)]
    for network, minutes in [(fast, built_at_60), (slow, read_at_30)]:
        assert minutes[(A, C)] == pytest.approx(network.get_travel_time([A, B, C]))
        assert minutes[(C, A)] == pytest.approx(network.get_travel_time([C, B, A]))
