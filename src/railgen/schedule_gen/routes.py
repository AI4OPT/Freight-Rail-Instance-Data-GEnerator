from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx

from railgen.models import YdKey
from railgen.rail_network import RailNetwork


@dataclass(frozen=True, kw_only=True)
class Route:
    path: tuple[YdKey, ...]
    cost: float  # routing-graph cost (discounted travel time)
    km: float  # physical length


def count_phase1_segments(
    phase1_graph: nx.DiGraph,
    hub_nodes: set[YdKey],
) -> dict[tuple[YdKey, YdKey], int]:
    """Count segment usage across all hub-to-hub shortest paths on the Phase 1 graph."""
    seg_counts: dict[tuple[YdKey, YdKey], int] = {}
    hubs = list(hub_nodes)
    for origin in hubs:
        for dest in hubs:
            if origin == dest:
                continue
            try:
                raw_path = nx.shortest_path(phase1_graph, origin, dest, weight="weight")
                assert isinstance(raw_path, list)
                path = YdKey.parse_raw_path(raw_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                raise ValueError(
                    f"No path exists between {origin} and {dest} in phase 1 graph."
                )
            for a, b in pairwise(path):
                seg_counts[(a, b)] = seg_counts.get((a, b), 0) + 1
    return seg_counts


def routing_graph(network: RailNetwork) -> nx.DiGraph:
    """Graph every yard pair is routed on: Phase 1 corridor counts turned into Phase 2 discounts."""
    phase1 = network.build_phase1_routing_graph()
    counts = count_phase1_segments(phase1, network.get_hub_yards())
    return network.build_phase2_routing_graph(counts)


def routes_between(
    network: RailNetwork, graph: nx.DiGraph, pairs: set[tuple[YdKey, YdKey]]
) -> dict[tuple[YdKey, YdKey], Route]:
    """Cheapest route for each (origin, destination) pair; unreachable pairs are omitted."""
    targets: dict[YdKey, set[YdKey]] = defaultdict(set)
    for o, d in pairs:
        targets[o].add(d)
    out: dict[tuple[YdKey, YdKey], Route] = {}
    for o, dests in sorted(targets.items()):
        cost, paths = nx.single_source_dijkstra(graph, o, weight="weight")
        for d in dests:
            if d not in paths:
                continue
            path = tuple(YdKey(n) for n in paths[d])
            km = nx.path_weight(network.yds_graph, list(path), weight="weight")
            out[(o, d)] = Route(path=path, cost=float(cost[d]), km=float(km))
    return out
