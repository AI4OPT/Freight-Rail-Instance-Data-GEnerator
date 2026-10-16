import asyncio
import csv
import json
import time
from collections.abc import Callable, Collection, Iterable
from collections.abc import Set as AbstractSet
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
    FAF_ZONE_COORDS_FILE,
    FORWARD_GEOCODE_MIN_INTERVAL_S,
    GRAPH_FILE,
    HUB_DOWNLOAD_URL,
    HUB_EXPORT_POLL_S,
    HUB_EXPORT_TIMEOUT_S,
    NA_RAIL_NETWORK_FILE,
    NA_RAIL_NETWORK_ITEM_ID,
    SAME_TRACK_TOLERANCE,
    YARD_GROUP_CHANGES_FILE,
    YARD_GROUPS_DICT_FILE,
    YARD_GROUPS_FILE,
    YARD_GROUPS_ITEM_ID,
    YARD_RECORDS_FILE,
    ZONE_YARD_GROUPS_FILE,
)
from railgen.constants import INPUT_DATA_DIR as PUBLIC_DIR
from railgen.constants import RAIL_NETWORK_DIR as GRAPH_DIR
from railgen.faf_data import (
    FAF_LOCATION_LOOKUP,
    FAF_OD_ZONE_CODES,
    FAF_ZONE_LOOKUP,
)
from railgen.models import (
    Coord,
    LineStringGeometry,
    Location,
    MultiLineStringGeometry,
    RailNetworkFeature,
    YardGroup,
    YardGroupFeature,
    YgKey,
    ZoneCode,
    rail_network_adapter,
    yg_adapter,
    ygs_dict_adapter,
)
from railgen.rail_network_gen.graph_utils import (
    concurrent_elevation_fetch,
    concurrent_geocode,
    custom_dijkstra,
    forward_geocode,
    get_yg_coord,
)
from railgen.schedule_gen.schedule_config import Config, load_config
from railgen.stats import OdAssignmentStats, OdPassStats
from railgen.utils import get_progress_desc_with_padding

YG_CHANGES_PATH = Path(__file__).parent / YARD_GROUP_CHANGES_FILE


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


def _find_zone_code(city: str, state: str) -> ZoneCode | None:
    """Return the FAF numeric zone code for a given city/state, or None if no match."""
    for zone, code in FAF_ZONE_LOOKUP.load().items():
        if zone.matches(city, state):
            return code
    return None


def _haversine_km(a: "Coord", b: "Coord") -> float:
    return float(haversine((a.lat, a.lon), (b.lat, b.lon), unit=Unit.KILOMETERS))


def _never_too_close(_yg_1: YgKey, _yg_2: YgKey) -> bool:
    return False


def _track_closeness(
    graph: Graph, min_spacing_km: float
) -> Callable[[YgKey, YgKey], bool]:
    """Makes a `too_close(a, b)` check: are they within `min_spacing_km` of each other by track?"""
    near: dict[YgKey, set[str]] = {}

    def too_close(yg_1: YgKey, yg_2: YgKey) -> bool:
        if yg_1 not in near:
            lengths = nx.single_source_dijkstra_path_length(
                graph, yg_1, cutoff=min_spacing_km, weight="weight"
            )
            near[yg_1] = {name for name, km in lengths.items() if km < min_spacing_km}
        return yg_2 in near[yg_1]

    return too_close


def _spaced(
    candidates: Iterable[YgKey],
    count: int,
    too_close: Callable[[YgKey, YgKey], bool],
) -> list[YgKey]:
    """First `count` candidates in order, skipping any too close to one already picked."""
    chosen: list[YgKey] = []
    for candidate in candidates:
        if len(chosen) == count:
            break
        if not any(too_close(yg, candidate) for yg in chosen):
            chosen.append(candidate)
    return chosen


def _with_min_track(
    candidates: list[YgKey], ygs_dict: dict[YgKey, YardGroup], min_track_km: float
) -> list[YgKey]:
    """Candidates with at least `min_track_km` of track. If none qualify, return them all."""
    large = [n for n in candidates if ygs_dict[n].shape_length_km >= min_track_km]
    return large or candidates


def _zone_city_coords(
    codes: list[ZoneCode], code_to_faf_zone: dict[ZoneCode, Location]
) -> dict[ZoneCode, Coord]:
    """Geocoded city of each zone, from `FAF_ZONE_COORDS_FILE`. Zones missing from the file get looked up and added."""
    path = GRAPH_DIR / FAF_ZONE_COORDS_FILE
    saved: dict[str, dict[str, float]] = (
        json.loads(path.read_text()) if path.exists() else {}
    )
    missing = [code for code in codes if code not in saved]
    for code in missing:
        faf_zone = code_to_faf_zone[code]
        coord = forward_geocode(faf_zone.raw_city, faf_zone.state_code)
        time.sleep(FORWARD_GEOCODE_MIN_INTERVAL_S)  # respect Nominatim rate limit
        if coord is not None:
            saved[code] = {"lat": coord.lat, "lon": coord.lon}
    if missing:
        _ = path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    return {
        code: Coord(lat=saved[code]["lat"], lon=saved[code]["lon"])
        for code in codes
        if code in saved
    }


def _assign_nearest_ygs_for_real_city_zones(
    ygs_dict: dict[YgKey, YardGroup],
    codes: list[ZoneCode],
    code_to_faf_zone: dict[ZoneCode, Location],
    used_ygs: set[YgKey],
    assign: Callable[[ZoneCode, list[YgKey]], None],
    *,
    ygs_per_zone: int,
    max_fallback_km: float,
    min_track_km: float = 0.0,
    too_close: Callable[[YgKey, YgKey], bool] = _never_too_close,
    barred: AbstractSet[YgKey] = frozenset(),
) -> None:
    """Fallback for city zones that matched no yard group by city/state.

    Geocode the zone's city and take up to `ygs_per_zone` of the nearest free,
    unbarred yard groups within `max_fallback_km`, preferring ones with at least
    `min_track_km` of track and skipping any `too_close` to one already picked.

    Without the distance cap this would happily pick yards hundreds of km away, so a
    zone with nothing in range just stays unrepresented.
    """
    zone_coords = _zone_city_coords(codes, code_to_faf_zone)
    for code in codes:
        zone_coord = zone_coords.get(code)
        if zone_coord is None:  # the zone city could not be geocoded
            continue
        in_range = sorted(
            (
                (name, dist)
                for name, yg in ygs_dict.items()
                if name not in used_ygs
                and name not in barred
                and (dist := _haversine_km(zone_coord, yg.coord)) <= max_fallback_km
            ),
            key=lambda pair: pair[1],
        )
        chosen = _spaced(
            _with_min_track([name for name, _ in in_range], ygs_dict, min_track_km),
            ygs_per_zone,
            too_close,
        )
        assign(code, chosen)


def _assign_largest_ygs_for_rest_of_zones(
    ygs_dict: dict[YgKey, YardGroup],
    codes: list[ZoneCode],
    code_to_city_state: dict[ZoneCode, Location],
    used_ygs: set[YgKey],
    assign: Callable[[ZoneCode, list[YgKey]], None],
    *,
    ygs_per_zone: int,
    named_zone_ygs: set[YgKey],
    too_close: Callable[[YgKey, YgKey], bool] = _never_too_close,
    barred: AbstractSet[YgKey] = frozenset(),
) -> None:
    """Fallback for "Rest of <state>" zones: the `ygs_per_zone` largest free yard groups in the state.

    Skips `named_zone_ygs` (yards in a city that has its own zone, like Detroit).
    Otherwise the biggest yards, which sit in the big metros, would end up representing
    the rest of the state. No distance cap since these zones have no center.
    """
    ygs_by_state: dict[str, list[YgKey]] = {}
    for name, yg in ygs_dict.items():
        if yg.state:
            ygs_by_state.setdefault(yg.state.upper(), []).append(name)

    for code in codes:
        faf_zone = code_to_city_state[code]
        candidates = [
            n
            for n in ygs_by_state.get(faf_zone.state_code, [])
            if n not in used_ygs and n not in named_zone_ygs and n not in barred
        ]
        best = _spaced(
            sorted(candidates, key=lambda n: ygs_dict[n].shape_length_km, reverse=True),
            ygs_per_zone,
            too_close,
        )
        assign(code, best)


def _assign_od_yard_groups(
    ygs_dict: dict[YgKey, YardGroup],
    yg_zone_codes: dict[YgKey, ZoneCode | None],
    *,
    ygs_per_zone: int,
    max_fallback_km: float,
    min_spacing_km: float = 0.0,
    min_track_km: float = 0.0,
    graph: Graph | None = None,
    barred: AbstractSet[YgKey] = frozenset(),
) -> tuple[OdAssignmentStats, dict[ZoneCode, tuple[int, list[YgKey]]]]:
    """Pick up to `ygs_per_zone` yard groups to represent each FAF zone, in three passes:

    1. City/state match: the largest matching yard groups.
    2. City zones with no match: the nearest yard groups to the geocoded city
       (`_assign_nearest_ygs_for_real_city_zones`).
    3. "Rest of <state>" zones: the largest leftover yard groups in the state
       (`_assign_largest_ygs_for_rest_of_zones`).

    Barred yard groups are never picked, and picks for the same zone have to be at
    least `min_spacing_km` apart by track (skipped without a graph). A zone can end up
    with fewer yard groups, or none. Better than a yard that doesn't really represent it.

    Sets `od_zone_code` on the chosen YardGroups in place. Returns per-pass stats and,
    per zone, the pass that handled it and its yard groups.
    """
    zone_to_ygs: dict[ZoneCode, list[YgKey]] = {}
    for yg_key, code in yg_zone_codes.items():
        if code is None or code not in FAF_OD_ZONE_CODES.load() or yg_key in barred:
            continue
        zone_to_ygs.setdefault(code, []).append(yg_key)
    too_close = (
        _track_closeness(graph, min_spacing_km)
        if graph is not None and min_spacing_km > 0
        else _never_too_close
    )

    # Reset everyone first; only the chosen YG per zone is set as OD below.
    for yg in ygs_dict.values():
        yg.od_zone_code = None

    used_ygs: set[YgKey] = set()
    assigned: dict[ZoneCode, list[YgKey]] = {}

    def assign(code: ZoneCode, yg_keys: list[YgKey]) -> None:
        """Hand a zone to its yard groups."""
        if not yg_keys:
            return
        for yg_key in yg_keys:
            yg = ygs_dict[yg_key]
            # A yard group only represents one zone. Downstream relies on this to
            # keep inter-zone flows from ending up with origin == destination.
            assert yg.od_zone_code is None, (
                f"{yg_key} already represents zone {yg.od_zone_code}, "
                f"cannot also represent zone {code}."
            )
            yg.od_zone_code = code
            used_ygs.add(yg_key)
        assigned.setdefault(code, []).extend(yg_keys)

    # Pass 1: the largest city/state-matched YGs per zone.
    for code, yg_names in zone_to_ygs.items():
        best = _spaced(
            sorted(yg_names, key=lambda n: ygs_dict[n].shape_length_km, reverse=True),
            ygs_per_zone,
            too_close,
        )
        assign(code, best)

    # Passes 2 and 3: fall back for OD zones that matched no yard group above.
    unmatched = [
        code
        for code in sorted(FAF_OD_ZONE_CODES.load())
        if code not in zone_to_ygs and code in FAF_LOCATION_LOOKUP.load()
    ]
    rest_of_codes = [
        c
        for c in unmatched
        if FAF_LOCATION_LOOKUP.load()[c].normalized_city.startswith("rest of")
    ]
    real_city_codes = [
        c
        for c in unmatched
        if not FAF_LOCATION_LOOKUP.load()[c].normalized_city.startswith("rest of")
    ]

    _assign_nearest_ygs_for_real_city_zones(
        ygs_dict,
        real_city_codes,
        FAF_LOCATION_LOOKUP.load(),
        used_ygs,
        assign,
        ygs_per_zone=ygs_per_zone,
        max_fallback_km=max_fallback_km,
        min_track_km=min_track_km,
        too_close=too_close,
        barred=barred,
    )
    _assign_largest_ygs_for_rest_of_zones(
        ygs_dict,
        rest_of_codes,
        FAF_LOCATION_LOOKUP.load(),
        used_ygs,
        assign,
        ygs_per_zone=ygs_per_zone,
        # Yards in a named zone's city belong to that zone, not the rest of the state.
        named_zone_ygs={n for n, c in yg_zone_codes.items() if c is not None},
        too_close=too_close,
        barred=barred,
    )

    def tally(codes: Iterable[ZoneCode]) -> OdPassStats:
        handled = [c for c in codes if c in assigned]
        return OdPassStats(
            zones=len(handled),
            yard_groups=sum(len(assigned[c]) for c in handled),
            zones_under_target=sum(
                1 for c in handled if len(assigned[c]) < ygs_per_zone
            ),
        )

    targeted = [*zone_to_ygs, *real_city_codes, *rest_of_codes]
    summary = OdAssignmentStats(
        ygs_per_zone=ygs_per_zone,
        max_fallback_km=max_fallback_km,
        zones_total=len(targeted),
        zones_represented=len(assigned),
        zones_unrepresented=sorted(c for c in targeted if c not in assigned),
        yard_groups_designated=sum(len(v) for v in assigned.values()),
        city_match=tally(zone_to_ygs),
        nearest_within_radius=tally(real_city_codes),
        rest_of_state=tally(rest_of_codes),
    )
    summary.print_summary()
    chosen = {
        code: (zone_pass, assigned.get(code, []))
        for zone_pass, codes in enumerate(
            (zone_to_ygs, real_city_codes, rest_of_codes), start=1
        )
        for code in codes
    }
    return summary, chosen


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _zone_yard_groups_table(
    rows: list[tuple[str, str, int, list[tuple[str, float]]]],
) -> str:
    """LaTeX table of each zone's yard groups, from rows of (zone code, zone name, pass, [(yard group, track km)])."""

    def tex(text: str) -> str:
        return "".join(_TEX_ESCAPES.get(char, char) for char in text)

    header = "Zone & City & Stage & Yard groups (km of track) \\\\"
    lines = [
        "% Yard groups that represent each FAF zone. Written when the yard groups are generated (assign_od_yard_groups); do not edit.",
        "% Requires: longtable, booktabs.",
        "\\begin{longtable}{@{}l l c p{9cm}@{}}",
        "\\caption{Yard groups representing each FAF zone.}\\label{tab:zone-yard-groups}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for code, name, zone_pass, yards in rows:
        chosen = "; ".join(f"{tex(yg)} ({km:.1f})" for yg, km in yards) or "none"
        lines.append(f"{tex(code)} & {tex(name)} & {zone_pass} & {chosen} \\\\")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _save_zone_yard_groups(
    ygs: dict[YgKey, YardGroup], chosen: dict[ZoneCode, tuple[int, list[YgKey]]]
) -> None:
    """Write the zone table to `ZONE_YARD_GROUPS_FILE`, only if it changed."""
    zones = FAF_LOCATION_LOOKUP.load()
    rows: list[tuple[str, str, int, list[tuple[str, float]]]] = []
    for code in sorted(chosen):
        zone_pass, names = chosen[code]
        zone = zones[code]
        zone_name = (
            f"Rest of {zone.state_code}"
            if zone.normalized_city.startswith("rest of")
            else f"{zone.raw_city.title()}, {zone.state_code}"
        )
        rows.append(
            (code, zone_name, zone_pass, [(n, ygs[n].shape_length_km) for n in names])
        )
    table = _zone_yard_groups_table(rows)
    path = GRAPH_DIR / ZONE_YARD_GROUPS_FILE
    if not path.exists() or path.read_text() != table:
        _ = path.write_text(table)


def assign_od_yard_groups(
    ygs: dict[YgKey, YardGroup], graph: Graph, cfg: Config, barred: AbstractSet[YgKey]
) -> OdAssignmentStats:
    """
    Pick the YardGroups that represent each FAF zone (rules in `_assign_od_yard_groups`) and write the zone table.

    ## Args:
    - ygs -- YardGroups by name; updated in place
    - graph -- YardGroup graph, for track distances
    - cfg -- Config (`network_settings`)
    - barred -- YardGroups that can't represent a zone

    ## Returns:
    - Per-pass stats.
    """
    net = cfg.network_settings
    stats, chosen = _assign_od_yard_groups(
        ygs,
        {
            name: _find_zone_code(yg.city, yg.state) if yg.city and yg.state else None
            for name, yg in ygs.items()
        },
        ygs_per_zone=net.ygs_per_zone,
        max_fallback_km=net.max_od_fallback_km,
        min_spacing_km=net.min_od_yg_spacing_km,
        min_track_km=net.min_od_yg_track_km,
        graph=graph,
        barred=barred,
    )
    _save_zone_yard_groups(ygs, chosen)
    return stats


def _kept_name(name: YgKey, merges: dict[YgKey, YgKey]) -> YgKey:
    """Follow merges to the YardGroup that `name` ends up in (merges can chain)."""
    seen = {name}
    while name in merges:
        name = merges[name]
        if name in seen:
            raise ValueError(f"YardGroups merged into each other: {sorted(seen)}")
        seen.add(name)
    return name


def traverse_graph(
    ygs: dict[YgKey, YardGroup],
    raw_ygs: list[YardGroupFeature],
    raw_network: list[RailNetworkFeature],
    merges: dict[YgKey, YgKey] | None = None,
    excluded_track_net_codes: Collection[str] = (),
) -> set[tuple[YgKey, YgKey, float]]:
    """
    Walk the rail network out from every YardGroup (`custom_dijkstra`) and collect YardGroup-to-YardGroup edges.

    ## Args:
    - ygs -- YardGroups by name
    - raw_ygs -- yard features. A merged record's track counts as track of the YardGroup it's merged into; track of records not in `ygs` is just track.
    - raw_network -- rail line features
    - merges -- merged YardGroup name -> the one it's merged into
    - excluded_track_net_codes -- `NET` codes the walk skips (abandoned lines, ferries, ...)

    ## Returns:
    - Set of (YardGroup 1, YardGroup 2, km), keeping the shortest distance per pair.
    """
    trav_graph = nx.Graph()
    for feat in track(
        raw_network, get_progress_desc_with_padding("Building traverse graph...")
    ):
        if feat.properties.get("NET") in excluded_track_net_codes:
            continue
        geom = feat.geometry

        coords: list[Coord] = [Coord(lat=x[1], lon=x[0]) for x in geom.coordinates]
        for i in range(len(coords) - 1):
            node1, node2 = coords[i], coords[i + 1]
            dist_from_src_yg = float(
                haversine(
                    (node1.lat, node1.lon), (node2.lat, node2.lon), unit=Unit.KILOMETERS
                )
            )
            _ = trav_graph.add_edge(node1, node2, weight=dist_from_src_yg)
    # Every coordinate of every YardGroup's geometry -> its key. custom_dijkstra stops when it hits one.
    all_yg_geom_coords: dict[Coord, YgKey] = {}
    for yg_feat in track(
        raw_ygs, get_progress_desc_with_padding("Building YG geometry map...")
    ):
        yg_id = str(yg_feat.properties.get("OBJECTID"))
        yg_name_str = str(yg_feat.properties.get("YARDNAME"))
        yg_key = _kept_name(YgKey.new(yg_name_str, yg_id), merges or {})
        if yg_key not in ygs:
            continue
        geom = yg_feat.geometry
        if isinstance(geom, LineStringGeometry):
            for cp in geom.coordinates:
                all_yg_geom_coords[Coord(lat=cp[1], lon=cp[0])] = yg_key
        else:
            for line in geom.coordinates:
                for cp in line:
                    all_yg_geom_coords[Coord(lat=cp[1], lon=cp[0])] = yg_key

    edges: list[tuple[YgKey, YgKey, float]] = []
    for yg in track(
        list(ygs.items()),
        description=get_progress_desc_with_padding("Building YG graph..."),
    ):
        src_yg_name = yg[0]
        found_connected_ygs = custom_dijkstra(
            trav_graph,
            all_yg_geom_coords,
            yg,
        )
        for neigh_yg_name, dist_to_neigh_yg in found_connected_ygs.items():
            edges.append((src_yg_name, neigh_yg_name, dist_to_neigh_yg))
    best: dict[tuple[YgKey, YgKey], float] = {}
    for yg_1, yg_2, dist in edges:
        yg_pair = (yg_1, yg_2) if yg_1 <= yg_2 else (yg_2, yg_1)
        if yg_pair not in best or dist < best[yg_pair]:
            best[yg_pair] = dist
    norm_edges = {
        (yg_name_1, yg_name_2, distance)
        for (yg_name_1, yg_name_2), distance in best.items()
    }
    return norm_edges


def build_graph(
    ygs: dict[YgKey, YardGroup],
    merges: dict[YgKey, YgKey] | None = None,
    excluded_track_net_codes: Collection[str] = (),
) -> Graph:
    """
    Load the YardGroup graph from `GRAPH_FILE`, or build it with `traverse_graph` and save it.

    ## Args:
    - ygs -- YardGroups by name
    - merges -- merged YardGroup name -> the one it's merged into. Only used when building.
    - excluded_track_net_codes -- `NET` codes the walk skips. Only used when building.

    ## Returns:
    - Graph with a node per YardGroup and an edge (km) per connection.
    """
    try:
        file_exists = (GRAPH_DIR / GRAPH_FILE).exists()
        if file_exists:
            graph = cast(Graph, nx.read_gexf(path=GRAPH_DIR / GRAPH_FILE))
            print(f"{GRAPH_FILE} exists, skipping YardGroup Graph generation.")
            for yg_name, yg in ygs.items():
                if yg_name in graph.nodes:
                    graph.nodes[yg_name]["city"] = yg.city
                    graph.nodes[yg_name]["state"] = yg.state
                    graph.nodes[yg_name]["shape_length_km"] = yg.shape_length_km
            return graph
    except (ValidationError, FileNotFoundError) as e:
        print(
            f"File missing or invalid {GRAPH_FILE} structure, regenerating graph. Error: {e}"
        )
    raw_ygs, raw_network = load_geojson()
    ygs = {name: yg for name, yg in ygs.items() if name not in (merges or {})}
    graph = nx.Graph()
    for yg_name, yg in ygs.items():
        graph.add_node(
            yg_name,
            label=yg_name,
            lat=yg.coord.lat,
            lon=yg.coord.lon,
            elevation=yg.elevation,
            city=yg.city,
            state=yg.state,
            shape_length_km=yg.shape_length_km,
        )
    edges = traverse_graph(ygs, raw_ygs, raw_network, merges, excluded_track_net_codes)
    for a, b, d in edges:
        _ = graph.add_edge(a, b, weight=d)
    nx.write_gexf(G=graph, path=GRAPH_DIR / GRAPH_FILE)
    return graph


def load_yard_group_changes() -> tuple[dict[YgKey, YgKey], set[YgKey], set[YgKey]]:
    """
    Read the `apply = yes` rows of the yard group changes CSV.

    ## Returns:
    - merges: merged YardGroup name -> the one it's merged into
    - drops: YardGroups to drop
    - bars: YardGroups that stay but can't represent a FAF zone
    """
    with open(YG_CHANGES_PATH, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["apply"] == "yes"]
    merges = {
        YgKey(row["yard_group"]): YgKey(row["merge_into"])
        for row in rows
        if row["action"] == "merge"
    }
    drops = {YgKey(row["yard_group"]) for row in rows if row["action"] == "drop"}
    bars = {YgKey(row["yard_group"]) for row in rows if row["action"] == "bar"}
    return merges, drops, bars


def merge_yard_group_records(
    ygs: dict[YgKey, YardGroup], merges: dict[YgKey, YgKey]
) -> dict[YgKey, YardGroup]:
    """
    Fold duplicate records of the same yard together. NTAD lists some yards more than once (crossing a county line, spelled two ways, ...).

    The kept YardGroup gets the merged one's track length. Pairs where either side isn't in `ygs` are skipped.

    ## Args:
    - ygs -- YardGroups by name
    - merges -- merged YardGroup name -> the one it's merged into

    ## Returns:
    - `ygs` without the merged YardGroups.
    """
    merged: set[YgKey] = set()
    for merged_name in merges:
        kept = _kept_name(merged_name, merges)
        if merged_name not in ygs or kept not in ygs:
            continue
        ygs[kept].shape_length_km += ygs[merged_name].shape_length_km
        merged.add(merged_name)
    print(
        f"Merged {len(merged):,} of {len(ygs):,} YardGroups into another record of the same yard; {len(ygs) - len(merged):,} left."
    )
    return {name: yg for name, yg in ygs.items() if name not in merged}


def merge_graph_nodes(graph: Graph, merges: dict[YgKey, YgKey]) -> Graph:
    """
    Like `merge_yard_group_records`, but for the nodes of a graph built with every record as its own node. A graph built with the merges already applied comes out unchanged.

    The kept node takes over the merged node's edges, keeping the shorter one when both connect to the same YardGroup.

    ## Args:
    - graph -- YardGroup graph with `weight` (km) on every edge
    - merges -- merged YardGroup name -> the one it's merged into

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


def drop_redundant_edges(
    graph: Graph, tolerance: float = SAME_TRACK_TOLERANCE
) -> Graph:
    """
    Drop edges that skip over a YardGroup. The walk can give an edge A-C even when the track runs through B, and then a train on A-C passes B without B being on its path.

    A-C goes when some B (connected to both) has:
    1. A-B + B-C < A-C, or
    2. A-B + B-C within `tolerance` of A-C, with A-B and B-C both shorter than A-C (probably the same track).

    Every edge is checked against the original graph before anything is removed. The `.gexf` file isn't touched.

    ## Args:
    - graph -- YardGroup graph with `weight` (km) on every edge
    - tolerance -- allowed extra length, as a fraction of A-C

    ## Returns:
    - The same graph without the skipped-over edges.
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    dist = np.full((len(nodes), len(nodes)), np.inf)
    for yg_1, yg_2, weight in graph.edges(data="weight"):
        dist[index[yg_1], index[yg_2]] = dist[index[yg_2], index[yg_1]] = float(weight)

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
        f"Dropped {len(shorter_joint):,} of {edges_before:,} edges with a shorter joint length through another YardGroup and {len(same_track):,} with a joint length less than {tolerance:.0%} longer; {graph.number_of_edges():,} edges left."
    )
    return graph


def _save_ygs(ygs: dict[YgKey, YardGroup], path: Path) -> None:
    GRAPH_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {key: value.model_dump(mode="json") for key, value in ygs.items()},
            f,
            indent=4,
        )


def _load_ygs_dict() -> dict[YgKey, YardGroup] | None:
    """Saved YardGroup dict, or None if the file is missing or invalid."""
    try:
        with open(GRAPH_DIR / YARD_GROUPS_DICT_FILE, "r") as f:
            return ygs_dict_adapter.validate_python(json.load(f))
    except (ValidationError, FileNotFoundError) as e:
        print(
            f"File missing or invalid {YARD_GROUPS_DICT_FILE} structure, regenerating dictionary. Error: {e}"
        )
        return None


def load_yard_records() -> dict[YgKey, YardGroup]:
    """
    Every NTAD yard record as a YardGroup with elevation and city. Read from `YARD_RECORDS_FILE`, or built from the geojson (slow, lots of elevation and geocoding lookups) and saved.

    ## Returns:
    - YardGroups by name, one per record. No FAF zones assigned yet.
    """
    try:
        with open(GRAPH_DIR / YARD_RECORDS_FILE, "r") as f:
            records = ygs_dict_adapter.validate_python(json.load(f))
        print(f"{YARD_RECORDS_FILE} exists, skipping the elevation and city look-ups.")
        return records
    except (ValidationError, FileNotFoundError) as e:
        print(
            f"File missing or invalid {YARD_RECORDS_FILE} structure, building the yard records. Error: {e}"
        )
    raw_ygs = load_yards_geojson()
    yg_coords = [get_yg_coord(yg.geometry) for yg in raw_ygs]
    elevations = asyncio.run(concurrent_elevation_fetch(yg_coords))
    city_states = concurrent_geocode(yg_coords)
    records: dict[YgKey, YardGroup] = {}
    for yg, elevation, (city, state) in track(
        list(zip(raw_ygs, elevations, city_states)),
        description=get_progress_desc_with_padding("Creating YardGroup Dictionary..."),
    ):
        id = str(yg.properties.get("OBJECTID"))
        name = str(yg.properties.get("YARDNAME"))
        owner = str(yg.properties.get("RROWNER1_NAME") or "")
        geom = yg.geometry
        coord = get_yg_coord(geom)
        shape_length_km = _calc_shape_length_km(geom)
        yg = YardGroup(
            id=id,
            name=name,
            owner=owner,
            coord=coord,
            elevation=elevation,
            city=city,
            state=state,
            shape_length_km=shape_length_km,
        )
        records[yg.key()] = yg
    _save_ygs(records, GRAPH_DIR / YARD_RECORDS_FILE)
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


def load_yards_geojson() -> list[YardGroupFeature]:
    """
    Opens and reads the raw geojson file of the YardGroups, downloading it when missing.

    ## Returns:
    - The ["features"] field of the raw geojson US YardGroups.
    """
    gj_ygs = None
    try:
        PUBLIC_DIR.mkdir(exist_ok=True)
        with open(PUBLIC_DIR / YARD_GROUPS_FILE, "r") as f:
            gj_ygs = json.load(f)
    except FileNotFoundError:
        _download_geojson(
            YARD_GROUPS_ITEM_ID, YARD_GROUPS_FILE, PUBLIC_DIR / YARD_GROUPS_FILE
        )
        with open(PUBLIC_DIR / YARD_GROUPS_FILE, "r") as f:
            gj_ygs = json.load(f)

    try:
        if not gj_ygs:
            raise ValueError("Invalid rail yard GeoJSON features")
        return yg_adapter.validate_python(gj_ygs["features"])
    except ValidationError as e:
        raise ValueError("Invalid rail network GeoJSON features") from e


def load_geojson() -> tuple[list[YardGroupFeature], list[RailNetworkFeature]]:
    """
    Read the raw yard and rail network geojson, downloading them if missing.

    ## Returns:
    - The yard features and the rail network features.
    """
    raw_ygs = load_yards_geojson()

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

    return raw_ygs, raw_network


def generate_or_load_network(
    cfg: Config | None = None,
) -> tuple[dict[YgKey, YardGroup], Graph, OdAssignmentStats]:
    """
    Entry point for the YardGroup dict (`.json`) and the YardGroup graph (`.gexf`). If both files exist they're just loaded. Otherwise the missing pieces get built:
    1. YardGroups: the yard records (`load_yard_records`) with the changes CSV applied. Duplicates get merged, and drops are left out if the graph is being built too.
    2. Graph: loaded, or walked over the track in use.
    3. FAF zone yard groups (`assign_od_yard_groups`), which need the graph distances.
    4. Save the dict.

    Changed the zone settings or merges? Delete the `.json`. Changed the drops or the track in use? Delete the `.gexf` too. Redundant edges get dropped on every load (`drop_redundant_edges`); the `.gexf` keeps them all.

    ## Args:
    - cfg -- Config for the track in use and the zone settings. Defaults to the config file, or `Config()` if there isn't one.

    ## Returns:
    - The YardGroup dict, the graph, and the FAF zone stats (empty when the dict was loaded from file).
    """
    cfg = cfg if cfg is not None else load_config()[1]
    merges, drops, bars = load_yard_group_changes()
    graph_exists = (GRAPH_DIR / GRAPH_FILE).exists()

    ygs_dict = _load_ygs_dict() if graph_exists else None
    if ygs_dict is not None:
        ygs_graph = merge_graph_nodes(build_graph(ygs_dict), merges)
        if set(ygs_dict) == set(ygs_graph.nodes):
            return ygs_dict, drop_redundant_edges(ygs_graph), OdAssignmentStats()
        print(
            f"{YARD_GROUPS_DICT_FILE} does not hold the YardGroups of {GRAPH_FILE}, regenerating dictionary."
        )

    records = load_yard_records()
    if not graph_exists:
        records = {name: yg for name, yg in records.items() if name not in drops}
    ygs_dict = merge_yard_group_records(records, merges)
    ygs_graph = build_graph(
        ygs_dict, merges, cfg.network_settings.excluded_track_net_codes
    )
    if not graph_exists:
        parts = sorted(map(len, nx.connected_components(ygs_graph)), reverse=True)
        print(
            f"Generated {GRAPH_FILE}: {ygs_graph.number_of_nodes():,} YardGroups and {ygs_graph.number_of_edges():,} edges in {len(parts):,} connected part(s); YardGroups per part: {parts[:10]}."
        )
    # An older graph can still have merged records as nodes (merged here) or dropped yards (kept, but barred from FAF zones).
    ygs_graph = merge_graph_nodes(ygs_graph, merges)
    ygs_dict = {name: yg for name, yg in ygs_dict.items() if name in ygs_graph}
    ygs_graph = drop_redundant_edges(ygs_graph)
    od_assignment = assign_od_yard_groups(ygs_dict, ygs_graph, cfg, barred=drops | bars)
    _save_ygs(ygs_dict, GRAPH_DIR / YARD_GROUPS_DICT_FILE)
    return ygs_dict, ygs_graph, od_assignment
