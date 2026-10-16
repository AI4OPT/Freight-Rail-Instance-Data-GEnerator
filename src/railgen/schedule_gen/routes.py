from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx

from railgen.models import YgKey
from railgen.rail_network import RailNetwork


@dataclass(frozen=True, kw_only=True)
class Route:
    path: tuple[YgKey, ...]
    cost: float  # routing-graph cost (discounted travel time)
    km: float  # physical length


def count_phase1_segments(
    phase1_graph: nx.DiGraph,
    hub_nodes: set[YgKey],
) -> dict[tuple[YgKey, YgKey], int]:
    """Count segment usage across all hub-to-hub shortest paths on the Phase 1 graph."""
    seg_counts: dict[tuple[YgKey, YgKey], int] = {}
    hubs = list(hub_nodes)
    for origin in hubs:
        for dest in hubs:
            if origin == dest:
                continue
            try:
                raw_path = nx.shortest_path(phase1_graph, origin, dest, weight="weight")
                assert isinstance(raw_path, list)
                path = YgKey.parse_raw_path(raw_path)
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
    counts = count_phase1_segments(phase1, network.get_hub_yard_groups())
    return network.build_phase2_routing_graph(counts)


def routes_between(
    network: RailNetwork, graph: nx.DiGraph, pairs: set[tuple[YgKey, YgKey]]
) -> dict[tuple[YgKey, YgKey], Route]:
    """Cheapest route for each (origin, destination) pair; unreachable pairs are omitted."""
    targets: dict[YgKey, set[YgKey]] = defaultdict(set)
    for o, d in pairs:
        targets[o].add(d)
    out: dict[tuple[YgKey, YgKey], Route] = {}
    for o, dests in sorted(targets.items()):
        cost, paths = nx.single_source_dijkstra(graph, o, weight="weight")
        for d in dests:
            if d not in paths:
                continue
            path = tuple(YgKey(n) for n in paths[d])
            km = nx.path_weight(network.ygs_graph, list(path), weight="weight")
            out[(o, d)] = Route(path=path, cost=float(cost[d]), km=float(km))
    return out
