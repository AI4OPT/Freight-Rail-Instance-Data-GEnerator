import csv
import json
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable
from collections.abc import Set as AbstractSet
from itertools import pairwise
from pathlib import Path
from typing import cast

import httpx
import networkx as nx
import numpy as np
from haversine import Unit, haversine
from networkx.classes.graph import Graph
from pydantic import ValidationError
from rich.progress import track

from railgen.constants import (
    EDGE_GRADES_FILE,
    FORWARD_GEOCODE_MIN_INTERVAL_S,
    GRAPH_FILE,
    HUB_COORDS_FILE,
    HUB_DOWNLOAD_URL,
    HUB_EXPORT_POLL_S,
    HUB_EXPORT_TIMEOUT_S,
    MAX_TRACK_GRADE_PERCENT,
    NA_RAIL_NETWORK_FILE,
    NA_RAIL_NETWORK_ITEM_ID,
    SAME_TRACK_TOLERANCE,
    TRACK_OFF_GROUND_M,
    YARD_CHANGES_FILE,
    YARD_RECORDS_FILE,
    YARDS_DICT_FILE,
    YARDS_FILE,
    YARDS_ITEM_ID,
)
from railgen.constants import INPUT_DATA_DIR as PUBLIC_DIR
from railgen.constants import RAIL_NETWORK_DIR as GRAPH_DIR
from railgen.faf_data import (
    FAF_OD_ZONE_CODES,
    FAF_ZONE_NAMES,
    FAF_ZONES,
    faf_zone_of,
)
from railgen.models import (
    Coord,
    LineStringGeometry,
    Location,
    MultiLineStringGeometry,
    RailNetworkFeature,
    Yard,
    YardFeature,
    YdKey,
    ZoneCode,
    rail_network_adapter,
    yd_adapter,
    yds_dict_adapter,
)
from railgen.rail_network_gen.graph_utils import (
    concurrent_geocode,
    custom_dijkstra,
    forward_geocode,
    get_yd_coord,
    sample_elevations,
)
from railgen.schedule_gen.schedule_config import Config, load_config
from railgen.stats import OdAssignmentStats
from railgen.utils import get_progress_desc_with_padding, kilometers_to_meters

YD_CHANGES_PATH = Path(__file__).parent / YARD_CHANGES_FILE


def _calc_shape_length_km(geom: LineStringGeometry | MultiLineStringGeometry) -> float:
    lines = (
        [geom.coordinates] if isinstance(geom, LineStringGeometry) else geom.coordinates
    )
    total = 0.0
    for line in lines:
        for i in range(len(line) - 1):
            p1, p2 = line[i], line[i + 1]
            total += float(
                haversine((p1[1], p1[0]), (p2[1], p2[0]), unit=Unit.KILOMETERS)
            )
    return total


def _never_too_close(_yd_1: YdKey, _yd_2: YdKey) -> bool:
    return False


def _track_closeness(
    graph: Graph, min_spacing_km: float
) -> Callable[[YdKey, YdKey], bool]:
    """Makes a `too_close(a, b)` check: are they within `min_spacing_km` of each other by track?"""
    near: dict[YdKey, set[str]] = {}

    def too_close(yd_1: YdKey, yd_2: YdKey) -> bool:
        if yd_1 not in near:
            lengths = nx.single_source_dijkstra_path_length(
                graph, yd_1, cutoff=min_spacing_km, weight="weight"
            )
            near[yd_1] = {name for name, km in lengths.items() if km < min_spacing_km}
        return yd_2 in near[yd_1]

    return too_close


def _spaced(
    candidates: Iterable[YdKey],
    count: int,
    too_close: Callable[[YdKey, YdKey], bool],
) -> list[YdKey]:
    """First `count` candidates in order, skipping any too close to one already picked."""
    chosen: list[YdKey] = []
    for candidate in candidates:
        if len(chosen) == count:
            break
        if not any(too_close(yd, candidate) for yd in chosen):
            chosen.append(candidate)
    return chosen


def _with_min_track(
    candidates: list[YdKey], yds_dict: dict[YdKey, Yard], min_track_km: float
) -> list[YdKey]:
    """Candidates with at least `min_track_km` of track. If none qualify, return them all."""
    large = [n for n in candidates if yds_dict[n].shape_length_km >= min_track_km]
    return large or candidates


def hub_label(hub: Location) -> str:
    return f"{hub.raw_city}, {hub.state_code}"


def hub_city_coords(hubs: list[Location]) -> dict[str, Coord]:
    """Geocoded coordinates of the hub cities by `hub_label`, from `HUB_COORDS_FILE`."""
    path = GRAPH_DIR / HUB_COORDS_FILE
    saved: dict[str, dict[str, float]] = (
        json.loads(path.read_text()) if path.exists() else {}
    )
    missing = [hub for hub in hubs if hub_label(hub) not in saved]
    for hub in missing:
        coord = forward_geocode(hub.raw_city, hub.state_code)
        time.sleep(FORWARD_GEOCODE_MIN_INTERVAL_S)  # respect Nominatim rate limit
        if coord is not None:
            saved[hub_label(hub)] = {"lat": coord.lat, "lon": coord.lon}
    if missing:
        GRAPH_DIR.mkdir(exist_ok=True)
        _ = path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    return {
        label: Coord(lat=saved[label]["lat"], lon=saved[label]["lon"])
        for label in map(hub_label, hubs)
        if label in saved
    }


def _yd_zone_codes(yds: dict[YdKey, Yard]) -> dict[YdKey, ZoneCode | None]:
    """FAF zone each Yard lies in, by the zone boundaries (`faf_zone_of`), or None."""
    zones = FAF_ZONES.load()
    return {name: faf_zone_of(yd.coord, zones) for name, yd in yds.items()}


def _zone_members(
    yd_zone_codes: dict[YdKey, ZoneCode | None], barred: AbstractSet[YdKey]
) -> dict[ZoneCode, list[YdKey]]:
    """Yards inside each FAF zone that sends or receives rail or multimodal freight, barred ones left out."""
    members: dict[ZoneCode, list[YdKey]] = {}
    for yd_key, code in yd_zone_codes.items():
        if code is None or code not in FAF_OD_ZONE_CODES.load() or yd_key in barred:
            continue
        members.setdefault(code, []).append(yd_key)
    return members


def _od_assignment_summary(
    assigned: dict[ZoneCode, list[YdKey]], *, yds_per_zone: int
) -> OdAssignmentStats:
    """Stats of the Yards `assigned` to each zone, over the zones with rail or multimodal freight."""
    zones = FAF_OD_ZONE_CODES.load()
    represented = {code: yds for code, yds in assigned.items() if yds}
    return OdAssignmentStats(
        yds_per_zone=yds_per_zone,
        zones_total=len(zones),
        zones_represented=len(represented),
        zones_unrepresented=sorted(c for c in zones if c not in represented),
        zones_under_target=sum(
            1 for yds in represented.values() if len(yds) < yds_per_zone
        ),
        yards_designated=sum(len(yds) for yds in represented.values()),
    )


def _assign_od_yards(
    yds_dict: dict[YdKey, Yard],
    yd_zone_codes: dict[YdKey, ZoneCode | None],
    *,
    yds_per_zone: int,
    min_spacing_km: float = 0.0,
    min_track_km: float = 0.0,
    graph: Graph | None = None,
    barred: AbstractSet[YdKey] = frozenset(),
) -> OdAssignmentStats:
    """Pick up to `yds_per_zone` yards to represent each FAF zone: the largest yards inside it.

    Barred yards are never picked, yards with at least `min_track_km` of track
    come first (a zone with none that large takes smaller ones), and picks for the same
    zone have to be at least `min_spacing_km` apart by track. A zone with no yard
    inside stays unrepresented.

    Sets `od_zone_code` on the chosen Yards in place. Returns the stats.
    """
    too_close = (
        _track_closeness(graph, min_spacing_km)
        if graph is not None and min_spacing_km > 0
        else _never_too_close
    )
    for yd in yds_dict.values():
        yd.od_zone_code = None

    assigned: dict[ZoneCode, list[YdKey]] = {}
    for code, yd_names in _zone_members(yd_zone_codes, barred).items():
        largest_first = sorted(
            yd_names, key=lambda n: yds_dict[n].shape_length_km, reverse=True
        )
        assigned[code] = _spaced(
            _with_min_track(largest_first, yds_dict, min_track_km),
            yds_per_zone,
            too_close,
        )
        for yd_key in assigned[code]:
            yds_dict[yd_key].od_zone_code = code

    summary = _od_assignment_summary(assigned, yds_per_zone=yds_per_zone)
    summary.print_summary()
    return summary


def write_zone_yards(
    yds: dict[YdKey, Yard], hubs: dict[str, YdKey | None], path: Path
) -> None:
    """
    Write the Yards that represent each FAF zone with rail or multimodal freight to a CSV, largest first; a zone with none gets one row with no yard. `hub_city` names the hub cities whose hub the Yard is.

    ## Args:
    - yds -- Yards by name, with their zones assigned
    - hubs -- hub Yard of each hub city (`RailNetwork.hub_yards_by_city`)
    - path -- the CSV to write
    """
    assigned: dict[ZoneCode, list[YdKey]] = {}
    for name, yd in yds.items():
        if yd.od_zone_code is not None:
            assigned.setdefault(yd.od_zone_code, []).append(name)
    hub_cities: dict[YdKey, list[str]] = {}
    for city, key in hubs.items():
        if key is not None:
            hub_cities.setdefault(key, []).append(city)
    zone_names = FAF_ZONE_NAMES.load()
    rows: list[tuple[ZoneCode, str, str, str, str]] = []
    for code in sorted(FAF_OD_ZONE_CODES.load()):
        names = sorted(
            assigned.get(code, []), key=lambda n: yds[n].shape_length_km, reverse=True
        )
        rows += [
            (
                code,
                zone_names[code],
                n,
                f"{yds[n].shape_length_km:.1f}",
                "; ".join(hub_cities.get(n, [])),
            )
            for n in names
        ] or [(code, zone_names[code], "", "", "")]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zone_code", "zone_name", "yard", "track_km", "hub_city"])
        writer.writerows(rows)


def assign_od_yards(
    yds: dict[YdKey, Yard], graph: Graph, cfg: Config, barred: AbstractSet[YdKey]
) -> OdAssignmentStats:
    """
    Pick the Yards that represent each FAF zone (rules in `_assign_od_yards`).

    ## Args:
    - yds -- Yards by name; updated in place
    - graph -- Yard graph, for track distances
    - cfg -- Config (`network_settings`)
    - barred -- Yards that can't represent a zone

    ## Returns:
    - The stats.
    """
    net = cfg.network_settings
    return _assign_od_yards(
        yds,
        _yd_zone_codes(yds),
        yds_per_zone=net.yds_per_zone,
        min_spacing_km=net.min_od_yd_spacing_km,
        min_track_km=net.min_od_yd_track_km,
        graph=graph,
        barred=barred,
    )


def od_assignment_stats(yds: dict[YdKey, Yard], cfg: Config) -> OdAssignmentStats:
    """
    Stats of a saved Yard dict, recounted from its `od_zone_code`s.

    ## Args:
    - yds -- Yards by name, with their zones assigned
    - cfg -- Config (`network_settings`)

    ## Returns:
    - The stats, as `assign_od_yards` returns them.
    """
    assigned: dict[ZoneCode, list[YdKey]] = {}
    for name, yd in yds.items():
        if yd.od_zone_code is not None:
            assigned.setdefault(yd.od_zone_code, []).append(name)
    return _od_assignment_summary(
        assigned, yds_per_zone=cfg.network_settings.yds_per_zone
    )


def _kept_name(name: YdKey, merges: dict[YdKey, YdKey]) -> YdKey:
    """Follow merges to the Yard that `name` ends up in (merges can chain)."""
    seen = {name}
    while name in merges:
        name = merges[name]
        if name in seen:
            raise ValueError(f"Yards merged into each other: {sorted(seen)}")
        seen.add(name)
    return name


def _merged_name(
    name: YdKey, yds: Collection[YdKey], merges: dict[YdKey, YdKey]
) -> YdKey:
    """`_kept_name`, or `name` itself when that Yard is missing and the merge is skipped (`merge_yard_records`)."""
    kept = _kept_name(name, merges)
    return kept if kept in yds else name


def track_graph(
    yds: dict[YdKey, Yard],
    raw_yds: list[YardFeature],
    raw_network: list[RailNetworkFeature],
    merges: dict[YdKey, YdKey] | None = None,
    excluded_track_net_codes: Collection[str] = (),
) -> tuple[dict[tuple[float, float], int], list[dict[int, float]], list[YdKey | None]]:
    """
    The track the walk of `traverse_graph` uses, and the Yard of every track node on a Yard's geometry.

    ## Args:
    - yds -- Yards by name
    - raw_yds -- yard features. A merged record's track counts as track of the Yard it's merged into.
    - raw_network -- rail line features
    - merges -- merged Yard name -> the one it's merged into
    - excluded_track_net_codes -- `NET` codes the walk skips (abandoned lines, ferries, ...)

    ## Returns:
    - Track node of each (lat, lon), numbered in order
    - Neighbor track node -> km, for each track node
    - Yard name whose geometry includes each track node, or None
    """
    # Track nodes are numbered: (lat, lon) -> node, and adj[node] is neighbor node -> km.
    nodes: dict[tuple[float, float], int] = {}
    adj: list[dict[int, float]] = []
    for feat in track(
        raw_network, get_progress_desc_with_padding("Building traverse graph...")
    ):
        if feat.properties.get("NET") in excluded_track_net_codes:
            continue
        coords = [(x[1], x[0]) for x in feat.geometry.coordinates]
        line = [nodes.setdefault(c, len(nodes)) for c in coords]
        adj += [{} for _ in range(len(nodes) - len(adj))]
        for c1, c2, n1, n2 in zip(coords, coords[1:], line, line[1:]):
            adj[n1][n2] = adj[n2][n1] = float(haversine(c1, c2, unit=Unit.KILOMETERS))
    # The Yard whose geometry includes each track node. custom_dijkstra stops when it hits one.
    yd_at: list[YdKey | None] = [None] * len(adj)
    for yd_feat in track(
        raw_yds, get_progress_desc_with_padding("Building YD geometry map...")
    ):
        yd_id = str(yd_feat.properties.get("OBJECTID"))
        yd_name_str = str(yd_feat.properties.get("YARDNAME"))
        yd_key = _merged_name(YdKey.new(yd_name_str, yd_id), yds, merges or {})
        if yd_key not in yds:
            continue
        geom = yd_feat.geometry
        lines = (
            [geom.coordinates]
            if isinstance(geom, LineStringGeometry)
            else geom.coordinates
        )
        for line in lines:
            for cp in line:
                node = nodes.get((cp[1], cp[0]))
                if node is not None:
                    yd_at[node] = yd_key
    return nodes, adj, yd_at


def traverse_graph(
    yds: dict[YdKey, Yard],
    raw_yds: list[YardFeature],
    raw_network: list[RailNetworkFeature],
    merges: dict[YdKey, YdKey] | None = None,
    excluded_track_net_codes: Collection[str] = (),
) -> set[tuple[YdKey, YdKey, float]]:
    """
    Walk the rail network out from every Yard (`custom_dijkstra`) and collect Yard-to-Yard edges. Arguments as for `track_graph`.

    ## Returns:
    - Set of (Yard 1, Yard 2, km), keeping the shortest distance per pair.
    """
    nodes, adj, yd_at = track_graph(
        yds, raw_yds, raw_network, merges, excluded_track_net_codes
    )
    srcs = {
        name: nodes[coord]
        for name, yd in yds.items()
        if (coord := (yd.coord.lat, yd.coord.lon)) in nodes
    }
    # Fold each run of plain track (no yard, no junction) into one edge, about 10x fewer nodes to walk.
    # The edge keeps its segment lengths: summing them up front rounds differently and flips exact ties in drop_redundant_edges.
    # The nodes next to a start stay: a start that's also on another Yard is reached by turning back there.
    keep = [len(a) != 2 or y is not None for a, y in zip(adj, yd_at)]
    for src in srcs.values():
        keep[src] = True
        for n in adj[src]:
            keep[n] = True
    trav_graph: dict[int, dict[int, tuple[float, ...]]] = {
        n: {} for n in range(len(adj)) if keep[n]
    }
    for start, out in trav_graph.items():
        for end, km in adj[start].items():
            prev, kms = start, [km]
            while not keep[end]:
                prev, end = end, next(n for n in adj[end] if n != prev)
                kms.append(adj[prev][end])
            if end != start and (end not in out or sum(kms) < sum(out[end])):
                out[end] = tuple(kms)

    edges: list[tuple[YdKey, YdKey, float]] = []
    for src_yd_name, src in track(
        list(srcs.items()),
        description=get_progress_desc_with_padding("Building YD graph..."),
    ):
        found_connected_yds = custom_dijkstra(trav_graph, yd_at, src_yd_name, src)
        for neigh_yd_name, dist_to_neigh_yd in found_connected_yds.items():
            edges.append((src_yd_name, neigh_yd_name, dist_to_neigh_yd))
    best: dict[tuple[YdKey, YdKey], float] = {}
    for yd_1, yd_2, dist in edges:
        yd_pair = (yd_1, yd_2) if yd_1 <= yd_2 else (yd_2, yd_1)
        if yd_pair not in best or dist < best[yd_pair]:
            best[yd_pair] = dist
    norm_edges = {
        (yd_name_1, yd_name_2, distance)
        for (yd_name_1, yd_name_2), distance in best.items()
    }
    return norm_edges


def grade_hops(
    yds: dict[YdKey, Yard],
    yds_graph: Graph,
    hops: Collection[tuple[YdKey, YdKey]],
    window_km: float,
    merges: dict[YdKey, YdKey] | None = None,
    excluded_track_net_codes: Collection[str] = (),
) -> dict[tuple[YdKey, YdKey], float]:
    """
    Steepest ascending grade, in percent, along the track of each hop of the Yard graph. The grade at a track coordinate is its rise over the last `window_km` of track; a hop shorter than that has no grade. Elevations over tunnels and under bridges are interpolated first (`MAX_TRACK_GRADE_PERCENT`).

    The track of a hop is the walk of `traverse_graph` between its Yards: the shorter of the walks from either one or a record merged into it, searched back from the other's geometry no farther than the length of the edge.

    ## Args:
    - yds -- dictionary of Yard names to Yard objects
    - yds_graph -- Yard graph holding every hop as an edge
    - hops -- pairs of Yards to grade, in either order
    - window_km -- track length a grade is measured over
    - merges, excluded_track_net_codes -- as for `track_graph`

    ## Returns:
    - Dictionary of (Yard 1, Yard 2) to the grade travelling from 1 to 2, for both orders of every hop.
    """
    raw_yds, raw_network = load_geojson()
    nodes, adj, yd_at = track_graph(
        yds, raw_yds, raw_network, merges, excluded_track_net_codes
    )
    trav_graph = nx.Graph(
        (u, v, {"weight": km}) for u, out in enumerate(adj) for v, km in out.items()
    )
    yd_track: dict[YdKey, list[int]] = defaultdict(list)
    for node, yd_key in enumerate(yd_at):
        if yd_key is not None:
            yd_track[yd_key].append(node)
    # A graph built before a merge has edges walked from the merged record (`merge_graph_nodes`).
    starts: dict[YdKey, set[int]] = defaultdict(set)
    for feat in raw_yds:
        name = YdKey.new(
            str(feat.properties.get("YARDNAME")), str(feat.properties.get("OBJECTID"))
        )
        coord = get_yd_coord(feat.geometry)
        if (node := nodes.get((coord.lat, coord.lon))) is not None:
            starts[_merged_name(name, yds, merges or {})].add(node)

    paths: dict[tuple[YdKey, YdKey], list[int]] = {}
    for a, b in track(hops, get_progress_desc_with_padding("Finding hop track...")):
        found: list[tuple[float, list[int]]] = []
        for src, dst in ((a, b), (b, a)):
            for start in starts[src]:
                if not yd_track[dst]:
                    continue
                try:
                    dist, path = nx.multi_source_dijkstra(
                        trav_graph,
                        yd_track[dst],
                        target=start,
                        cutoff=float(yds_graph[a][b]["weight"]) * 1.001,
                        # Like the walk, do not pass through another Yard. The walk starts at `start` even when it is on another Yard's track too.
                        weight=lambda _u, v, d, start=start, ends=(None, src, dst): (
                            d["weight"] if v == start or yd_at[v] in ends else None
                        ),
                    )
                except nx.NetworkXNoPath:
                    continue
                found.append((dist, path[::-1] if src == a else path))
        if not found:
            raise ValueError(f"No track found between {a} and {b}")
        paths[(a, b)] = min(found, key=lambda f: f[0])[1]

    coord_of = list(nodes)
    path_nodes = list({node for path in paths.values() for node in path})
    coords = [Coord(lat=coord_of[n][0], lon=coord_of[n][1]) for n in path_nodes]
    elevations = dict(zip(path_nodes, sample_elevations(coords)))
    rise_per_km = kilometers_to_meters(MAX_TRACK_GRADE_PERCENT / 100)
    grades: dict[tuple[YdKey, YdKey], float] = {}
    for (a, b), path in paths.items():
        km = np.cumsum([0.0] + [adj[u][v] for u, v in pairwise(path)])
        elev = np.array([elevations[node] for node in path])
        # A point is on the ground when every earlier point on the ground reaches it within the max grade, give or take TRACK_OFF_GROUND_M: between max(e + s * x) - s * km and min(e - s * x) + s * km over those points.
        on_ground: list[int] = []
        floor, ceiling = -np.inf, np.inf
        for i, (x, e) in enumerate(zip(km.tolist(), elev.tolist())):
            if (
                floor - rise_per_km * x - TRACK_OFF_GROUND_M
                <= e
                <= ceiling + rise_per_km * x + TRACK_OFF_GROUND_M
            ):
                on_ground.append(i)
                floor = max(floor, e + rise_per_km * x)
                ceiling = min(ceiling, e - rise_per_km * x)
        elev = np.interp(km, km[on_ground], elev[on_ground])
        start = np.searchsorted(km, km - window_km, side="right") - 1
        end = start >= 0
        grade = (
            (elev[end] - elev[start[end]])
            / kilometers_to_meters(km[end] - km[start[end]])
            * 100
        )
        grades[(a, b)] = float(grade.max(initial=0.0))
        grades[(b, a)] = float((-grade).max(initial=0.0))
    return grades


def _copy_yard_attributes(graph: Graph, yds: dict[YdKey, Yard]) -> None:
    """Copy each Yard's city, state and track length onto its node."""
    for yd_name, yd in yds.items():
        if yd_name in graph.nodes:
            graph.nodes[yd_name]["city"] = yd.city
            graph.nodes[yd_name]["state"] = yd.state
            graph.nodes[yd_name]["shape_length_km"] = yd.shape_length_km


def build_or_load_graph(
    yds: dict[YdKey, Yard],
    merges: dict[YdKey, YdKey] | None = None,
    excluded_track_net_codes: Collection[str] = (),
) -> Graph:
    """
    Load the Yard graph from `GRAPH_FILE`, or, only when that file is missing, build it with `traverse_graph` (the walk) and save it.

    ## Args:
    - yds -- Yards by name
    - merges -- merged Yard name -> the one it's merged into. Only used when building.
    - excluded_track_net_codes -- `NET` codes the walk skips. Only used when building.

    ## Returns:
    - Graph with a node per Yard and an edge (km) per connection.
    """
    try:
        file_exists = (GRAPH_DIR / GRAPH_FILE).exists()
        if file_exists:
            graph = cast(Graph, nx.read_gexf(path=GRAPH_DIR / GRAPH_FILE))
            print(f"{GRAPH_FILE} exists, skipping Yard Graph generation.")
            _copy_yard_attributes(graph, yds)
            return graph
    except (ValidationError, FileNotFoundError) as e:
        print(
            f"File missing or invalid {GRAPH_FILE} structure, regenerating graph. Error: {e}"
        )
    raw_yds, raw_network = load_geojson()
    yds = {name: yd for name, yd in yds.items() if name not in (merges or {})}
    graph = nx.Graph()
    for yd_name, yd in yds.items():
        graph.add_node(
            yd_name,
            label=yd_name,
            lat=yd.coord.lat,
            lon=yd.coord.lon,
            elevation=yd.elevation,
            city=yd.city,
            state=yd.state,
            shape_length_km=yd.shape_length_km,
        )
    edges = traverse_graph(yds, raw_yds, raw_network, merges, excluded_track_net_codes)
    for a, b, d in edges:
        _ = graph.add_edge(a, b, weight=d)
    (GRAPH_DIR / EDGE_GRADES_FILE).unlink(missing_ok=True)
    nx.write_gexf(G=graph, path=GRAPH_DIR / GRAPH_FILE)
    return graph


def load_yard_changes() -> tuple[dict[YdKey, YdKey], set[YdKey], set[YdKey]]:
    """
    Read the `apply = yes` rows of the yard changes CSV.

    ## Returns:
    - merges: merged Yard name -> the one it's merged into
    - drops: Yards to drop
    - bars: Yards that stay but can't represent a FAF zone
    """
    with open(YD_CHANGES_PATH, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["apply"] == "yes"]
    merges = {
        YdKey(row["yard"]): YdKey(row["merge_into"])
        for row in rows
        if row["action"] == "merge"
    }
    drops = {YdKey(row["yard"]) for row in rows if row["action"] == "drop"}
    bars = {YdKey(row["yard"]) for row in rows if row["action"] == "bar"}
    return merges, drops, bars


def warn_unmatched_changes(
    records: dict[YdKey, Yard],
    merges: dict[YdKey, YdKey],
    drops: set[YdKey],
    bars: set[YdKey],
) -> list[YdKey]:
    """
    Warn about yards named in the changes CSV that the yard records don't have.

    ## Returns:
    - The unmatched names, sorted.
    """
    named = set(merges) | set(merges.values()) | drops | bars
    missing = sorted(name for name in named if name not in records)
    if missing:
        print(
            f"Warning: {len(missing):,} yard(s) in {YARD_CHANGES_FILE} are not in the NTAD yard records, "
            f"so their rows are skipped (the record IDs may have changed): {', '.join(missing)}"
        )
    return missing


def merge_yard_records(
    yds: dict[YdKey, Yard], merges: dict[YdKey, YdKey]
) -> dict[YdKey, Yard]:
    """
    Fold duplicate records of the same yard together. NTAD lists some yards more than once (crossing a county line, spelled two ways, ...).

    The kept Yard gets the merged one's track length. Pairs where either side isn't in `yds` are skipped.

    ## Args:
    - yds -- Yards by name
    - merges -- merged Yard name -> the one it's merged into

    ## Returns:
    - `yds` without the merged Yards.
    """
    merged: set[YdKey] = set()
    for merged_name in merges:
        kept = _kept_name(merged_name, merges)
        if merged_name not in yds or kept not in yds:
            continue
        yds[kept].shape_length_km += yds[merged_name].shape_length_km
        merged.add(merged_name)
    print(
        f"Merged {len(merged):,} of {len(yds):,} Yards into another record of the same yard; {len(yds) - len(merged):,} left."
    )
    return {name: yd for name, yd in yds.items() if name not in merged}


def merge_graph_nodes(graph: Graph, merges: dict[YdKey, YdKey]) -> Graph:
    """
    Like `merge_yard_records`, but for the nodes of a graph built with every record as its own node. A graph built with the merges already applied comes out unchanged.

    The kept node takes over the merged node's edges, keeping the shorter one when both connect to the same Yard.

    ## Args:
    - graph -- Yard graph with `weight` (km) on every edge
    - merges -- merged Yard name -> the one it's merged into

    ## Returns:
    - The same graph without the merged nodes.
    """
    for merged_name in merges:
        kept = _kept_name(merged_name, merges)
        if merged_name not in graph or kept not in graph:
            continue
        for neighbour, data in list(graph[merged_name].items()):
            if neighbour == kept:
                continue
            dist = float(data["weight"])
            if (
                not graph.has_edge(kept, neighbour)
                or float(graph[kept][neighbour]["weight"]) > dist
            ):
                _ = graph.add_edge(kept, neighbour, weight=dist)
        graph.remove_node(merged_name)
    return graph


def drop_long_edges(graph: Graph, max_km: float) -> Graph:
    """
    Drop connections longer than `max_km`. The yards file has US yards only, so a route through Canada or Mexico becomes one long connection with no yard, and so no crew change, on the way.

    ## Args:
    - graph -- Yard graph with `weight` (km) on every edge
    - max_km -- longest connection kept

    ## Returns:
    - The same graph without those edges.
    """
    long = [(a, b) for a, b, d in graph.edges(data=True) if float(d["weight"]) > max_km]
    graph.remove_edges_from(long)
    print(f"Dropped {len(long):,} edges longer than {max_km:g} km.")
    return graph


def drop_redundant_edges(
    graph: Graph, tolerance: float = SAME_TRACK_TOLERANCE
) -> Graph:
    """
    Drop edges that skip over a Yard. The walk can give an edge A-C even when the track runs through B, and then a train on A-C passes B without B being on its path.

    A-C goes when some B (connected to both) has:
    1. A-B + B-C < A-C, or
    2. A-B + B-C within `tolerance` of A-C, with A-B and B-C both shorter than A-C (probably the same track).

    Every edge is checked against the original graph before anything is removed. The `.gexf` file isn't touched.

    ## Args:
    - graph -- Yard graph with `weight` (km) on every edge
    - tolerance -- allowed extra length, as a fraction of A-C

    ## Returns:
    - The same graph without the skipped-over edges.
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    dist = np.full((len(nodes), len(nodes)), np.inf)
    for yd_1, yd_2, weight in graph.edges(data="weight"):
        dist[index[yd_1], index[yd_2]] = dist[index[yd_2], index[yd_1]] = float(weight)

    shorter_joint: list[tuple[str, str]] = []
    same_track: list[tuple[str, str]] = []
    for a in range(len(nodes)):
        # Check the edge once from its lower-indexed end
        ends = np.flatnonzero(np.isfinite(dist[a, a + 1 :])) + a + 1
        if ends.size == 0:
            continue
        direct = dist[a, ends][:, None]
        a_to_b = dist[a][None, :]
        b_to_c = dist[ends]
        joint = a_to_b + b_to_c
        is_shorter = (joint < direct).any(axis=1)
        is_same_track = (
            (joint < (1 + tolerance) * direct) & (a_to_b < direct) & (b_to_c < direct)
        ).any(axis=1)
        for c, shorter, same in zip(ends, is_shorter, is_same_track):
            if shorter:
                shorter_joint.append((nodes[a], nodes[c]))
            elif same:
                same_track.append((nodes[a], nodes[c]))

    edges_before = graph.number_of_edges()
    graph.remove_edges_from(shorter_joint + same_track)
    print(
        f"Dropped {len(shorter_joint):,} of {edges_before:,} edges with a shorter joint length through another Yard and {len(same_track):,} with a joint length less than {tolerance:.0%} longer; {graph.number_of_edges():,} edges left."
    )
    return graph


def drop_bypassed_edges(graph: Graph, tolerance: float = SAME_TRACK_TOLERANCE) -> Graph:
    """
    Drop edges that a path through other yards makes redundant: `drop_redundant_edges` catches the paths through 1 Yard; this catches the rest.

    Edges are checked longest first, each against the graph as it stands, and the graph stays as connected as before. Each check searches only up to the edge's own length times (1 + `tolerance`).

    ## Args:
    - graph -- Yard graph with `weight` (km) on every edge
    - tolerance -- allowed extra length, as a fraction of A-C

    ## Returns:
    - The same graph without the bypassed edges.
    """
    edges = sorted(
        ((a, b, float(w)) for a, b, w in graph.edges(data="weight")),
        key=lambda e: -e[2],
    )
    dropped = 0
    for a, b, km in edges:
        data = graph[a][b]
        graph.remove_edge(a, b)
        try:
            _ = nx.single_source_dijkstra(
                graph, a, target=b, cutoff=km * (1 + tolerance), weight="weight"
            )
            dropped += 1
        except nx.NetworkXNoPath:
            _ = graph.add_edge(a, b, **data)
    print(
        f"Dropped {dropped:,} edges with a path through several Yards at most {tolerance:.0%} longer; {graph.number_of_edges():,} edges left."
    )
    return graph


def _save_yds(yds: dict[YdKey, Yard], path: Path) -> None:
    GRAPH_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {key: value.model_dump(mode="json") for key, value in yds.items()},
            f,
            indent=4,
        )


def _load_yds_dict() -> dict[YdKey, Yard] | None:
    """Saved Yard dict, or None if the file is missing or invalid."""
    try:
        with open(GRAPH_DIR / YARDS_DICT_FILE, "r") as f:
            return yds_dict_adapter.validate_python(json.load(f))
    except (ValidationError, FileNotFoundError) as e:
        print(
            f"File missing or invalid {YARDS_DICT_FILE} structure, regenerating dictionary. Error: {e}"
        )
        return None


def load_yard_records() -> dict[YdKey, Yard]:
    """
    Every NTAD yard record as a Yard with elevation and city. Read from `YARD_RECORDS_FILE`, or built from the geojson and saved.

    ## Returns:
    - Yards by name, one per record. No FAF zones assigned yet.
    """
    try:
        with open(GRAPH_DIR / YARD_RECORDS_FILE, "r") as f:
            records = yds_dict_adapter.validate_python(json.load(f))
        print(f"{YARD_RECORDS_FILE} exists, skipping the elevation and city look-ups.")
        return records
    except (ValidationError, FileNotFoundError) as e:
        print(
            f"File missing or invalid {YARD_RECORDS_FILE} structure, building the yard records. Error: {e}"
        )
    raw_yds = load_yards_geojson()
    yd_coords = [get_yd_coord(yd.geometry) for yd in raw_yds]
    elevations = sample_elevations(yd_coords)
    city_states = concurrent_geocode(yd_coords)
    records: dict[YdKey, Yard] = {}
    for yd, elevation, (city, state) in track(
        list(zip(raw_yds, elevations, city_states)),
        description=get_progress_desc_with_padding("Creating Yard Dictionary..."),
    ):
        id = str(yd.properties.get("OBJECTID"))
        name = str(yd.properties.get("YARDNAME"))
        owner = str(yd.properties.get("RROWNER1_NAME") or "")
        geom = yd.geometry
        coord = get_yd_coord(geom)
        shape_length_km = _calc_shape_length_km(geom)
        yd = Yard(
            id=id,
            name=name,
            owner=owner,
            coord=coord,
            elevation=elevation,
            city=city,
            state=state,
            shape_length_km=shape_length_km,
        )
        records[yd.key()] = yd
    _save_yds(records, GRAPH_DIR / YARD_RECORDS_FILE)
    return records


class DownloadFailedError(Exception):
    """Raised when a download fails."""


def _download_geojson(item_id: str, name: str, dest: Path) -> None:
    """
    Requests a fresh ArcGIS Hub export of an NTAD layer and streams it to `dest`.

    ## Args:
    - item_id -- the layer's ArcGIS Hub item id
    - name -- file name, used only for progress and error messages
    - dest -- path to write the GeoJSON to
    """
    print(f"{name} not found, downloading...")
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        deadline = time.monotonic() + HUB_EXPORT_TIMEOUT_S
        while True:
            response = client.get(
                HUB_DOWNLOAD_URL.format(item_id),
                params={"redirect": "false", "spatialRefId": 4326, "layers": 0},
            )
            # 200 means an export is ready; 202 means one is still being generated.
            if response.status_code not in (200, 202):
                raise DownloadFailedError(
                    f"Failed to request {name} export: {response.status_code}"
                )
            body = response.json()
            status = body.get("status")
            if status == "Completed":
                break
            if status in ("Failed", "ExportError"):
                raise DownloadFailedError(f"Export of {name} failed: {body}")
            if time.monotonic() > deadline:
                raise DownloadFailedError(
                    f"Export of {name} stuck at status {status!r}"
                )
            print(f"  export {status}, waiting...")
            time.sleep(HUB_EXPORT_POLL_S)

        # resultUrl 302s to a blob URL signed for ~an hour, hence follow_redirects.
        with client.stream("GET", cast(str, body["resultUrl"])) as stream:
            if stream.status_code != 200:
                raise DownloadFailedError(
                    f"Failed to download {name}: {stream.status_code}"
                )
            # Temp file first, so an interrupted download doesn't leave a truncated file that looks cached.
            partial = dest.with_name(dest.name + ".part")
            with open(partial, "wb") as f:
                for chunk in stream.iter_bytes():
                    _ = f.write(chunk)
            partial.replace(dest)
    print(f"Download complete: {name}")


def load_yards_geojson() -> list[YardFeature]:
    """
    Opens and reads the raw geojson file of the Yards, downloading it when missing.

    ## Returns:
    - The ["features"] field of the raw geojson US Yards.
    """
    gj_yds = None
    try:
        PUBLIC_DIR.mkdir(exist_ok=True)
        with open(PUBLIC_DIR / YARDS_FILE, "r") as f:
            gj_yds = json.load(f)
    except FileNotFoundError:
        _download_geojson(YARDS_ITEM_ID, YARDS_FILE, PUBLIC_DIR / YARDS_FILE)
        with open(PUBLIC_DIR / YARDS_FILE, "r") as f:
            gj_yds = json.load(f)

    try:
        if not gj_yds:
            raise ValueError("Invalid rail yard GeoJSON features")
        return yd_adapter.validate_python(gj_yds["features"])
    except ValidationError as e:
        raise ValueError("Invalid rail yard GeoJSON features") from e


def load_geojson() -> tuple[list[YardFeature], list[RailNetworkFeature]]:
    """
    Read the raw yard and rail network geojson, downloading them if missing.

    ## Returns:
    - The yard features and the rail network features.
    """
    raw_yds = load_yards_geojson()

    gj_network = None
    try:
        with open(PUBLIC_DIR / NA_RAIL_NETWORK_FILE, "r") as f:
            gj_network = json.load(f)
    except FileNotFoundError:
        _download_geojson(
            NA_RAIL_NETWORK_ITEM_ID,
            NA_RAIL_NETWORK_FILE,
            PUBLIC_DIR / NA_RAIL_NETWORK_FILE,
        )
        with open(PUBLIC_DIR / NA_RAIL_NETWORK_FILE, "r") as f:
            gj_network = json.load(f)

    try:
        if not gj_network:
            raise ValueError("Invalid rail network GeoJSON features")
        raw_network = rail_network_adapter.validate_python(gj_network["features"])
    except ValidationError as e:
        raise ValueError("Invalid rail network GeoJSON features") from e

    return raw_yds, raw_network


def generate_or_load_network(
    cfg: Config | None = None,
) -> tuple[dict[YdKey, Yard], Graph, OdAssignmentStats]:
    """
    Entry point for the Yard dict (`.json`) and the Yard graph (`.gexf`). If both files exist they're loaded; otherwise the missing pieces get built:
    1. Yards: the yard records (`load_yard_records`) with the changes CSV applied. Duplicates get merged, and drops are left out if the graph is being built too.
    2. Graph: loaded, or walked over the track in use.
    3. FAF zone yards (`assign_od_yards`), which need the graph distances.
    4. Save the dict.

    ## Args:
    - cfg -- Config for the track in use and the zone settings. Defaults to the config file (`configs/schedule_config.yaml`).

    ## Returns:
    - The Yard dict, the graph, and the FAF zone stats (recounted from the dict when it was loaded from file).
    """
    cfg = cfg if cfg is not None else load_config()
    max_km = cfg.network_settings.max_connection_km
    merges, drops, bars = load_yard_changes()
    graph_exists = (GRAPH_DIR / GRAPH_FILE).exists()

    yds_dict = _load_yds_dict() if graph_exists else None
    yds_graph: Graph | None = None
    if yds_dict is not None:
        yds_graph = merge_graph_nodes(build_or_load_graph(yds_dict), merges)
        if set(yds_dict) == set(yds_graph.nodes):
            return (
                yds_dict,
                drop_bypassed_edges(
                    drop_redundant_edges(drop_long_edges(yds_graph, max_km))
                ),
                od_assignment_stats(yds_dict, cfg),
            )
        print(
            f"{YARDS_DICT_FILE} does not hold the Yards of {GRAPH_FILE}, regenerating dictionary."
        )

    records = load_yard_records()
    warn_unmatched_changes(records, merges, drops, bars)
    if not graph_exists:
        records = {name: yd for name, yd in records.items() if name not in drops}
    yds_dict = merge_yard_records(records, merges)
    if yds_graph is None:
        yds_graph = build_or_load_graph(
            yds_dict, merges, cfg.network_settings.excluded_track_net_codes
        )
    else:  # graph.gexf was already read above; only the Yard attributes change
        _copy_yard_attributes(yds_graph, yds_dict)
    if not graph_exists:
        parts = sorted(map(len, nx.connected_components(yds_graph)), reverse=True)
        print(
            f"Generated {GRAPH_FILE}: {yds_graph.number_of_nodes():,} Yards and {yds_graph.number_of_edges():,} edges in {len(parts):,} connected part(s); Yards per part: {parts[:10]}."
        )

    yds_graph = merge_graph_nodes(yds_graph, merges)
    yds_dict = {name: yd for name, yd in yds_dict.items() if name in yds_graph}
    yds_graph = drop_bypassed_edges(
        drop_redundant_edges(drop_long_edges(yds_graph, max_km))
    )
    od_assignment = assign_od_yards(yds_dict, yds_graph, cfg, barred=drops | bars)
    _save_yds(yds_dict, GRAPH_DIR / YARDS_DICT_FILE)
    return yds_dict, yds_graph, od_assignment
