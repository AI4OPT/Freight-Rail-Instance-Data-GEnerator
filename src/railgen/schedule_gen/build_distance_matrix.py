from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple, cast

import networkx as nx
from pydantic import TypeAdapter
from rich.progress import track

from railgen.constants import RAIL_NETWORK_DIR as GRAPH_DIR
from railgen.models import YdKey
from railgen.rail_network import RailNetwork
from railgen.utils import get_progress_desc_with_padding


@dataclass(frozen=True, kw_only=True)
class DistanceOutputRow:
    begin_yard: str
    end_yard: str
    distance_minutes: float
    distance_km: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def get_headers() -> list[str]:
        return [field.name for field in fields(DistanceOutputRow)]


# Cache minutes are stored at this speed and rescaled to the run's (travel time ∝ 1 / speed, grades included).
REFERENCE_SPEED_KM_H = 100.0


# Stored as 2 element array [minutes, km] - smaller than an object with named fields
class YdDistance(NamedTuple):
    minutes: float  # at REFERENCE_SPEED_KM_H
    km: float


# Contents of the JSON cache file: {yd1: {yd2: [minutes, km]}}
type DistanceCache = dict[YdKey, dict[YdKey, YdDistance]]
distance_cache_adapter = TypeAdapter[DistanceCache](DistanceCache)


def add_missing_yd_distances(
    network: RailNetwork, yards: list[YdKey], distances: DistanceCache
) -> int:
    """Compute the distances between `yards` that `distances` lacks and add them to it.

    One shortest-path search per yard that has a missing pair. Returns the number
    of pairs added.
    """
    added = 0
    for yd_1 in track(
        yards,
        description=get_progress_desc_with_padding(
            "Calculating missing yard distances..."
        ),
    ):
        known = distances.get(yd_1, {})
        missing = [yd_2 for yd_2 in yards if yd_2 != yd_1 and yd_2 not in known]
        if not missing:
            continue
        lengths, paths = nx.single_source_dijkstra(
            G=network.yds_graph, source=yd_1, weight="weight"
        )
        assert isinstance(lengths, dict) and isinstance(paths, dict)
        lengths = cast(dict[str, float | int], lengths)
        paths = cast(dict[str, list[str]], paths)

        for yd_2 in missing:
            if yd_2 not in paths:
                continue  # no route; reported when the output rows are built
            yd_path = [YdKey.from_raw_str(node) for node in paths[yd_2]]
            minutes = network.get_travel_time(yd_path)  # at the speed of this run
            distances.setdefault(yd_1, {})[yd_2] = YdDistance(
                minutes=minutes * network.train_speed_km_h / REFERENCE_SPEED_KM_H,
                km=lengths[yd_2],
            )
            added += 1
    return added


def distance_cache_file(network: RailNetwork) -> str:
    """Name of the distance cache. It names the grade settings, since its minutes depend on them."""
    net = network.cfg.network_settings
    window_km = network.cfg.service_design_settings.locomotive_power.grade_window_km
    return (
        f"all_distances_grade{net.per_grade_speed_reduction:g}"
        f"_max{net.max_speed_reduction:g}_window{window_km:g}.json"
    )


def generate_output_yd_distances(
    network: RailNetwork,
    output_yds: set[YdKey],
) -> list[DistanceOutputRow]:
    """Return distances between all pairs within output_yds, using the JSON cache.

    The cache grows with use: it holds the pairs that earlier runs needed, and the pairs
    of this run that it lacks are computed and added to it. Its minutes include the
    slowdown on grades and are stored at `REFERENCE_SPEED_KM_H`; the rows returned here
    are at the train speed of this run, so the cache stays valid when the speed changes.
    Each pair of grade settings has its own cache file (`distance_cache_file`). Delete the
    caches after the network changes.
    """
    cache_file = distance_cache_file(network)
    all_distances_path: Path = GRAPH_DIR / cache_file
    distances: DistanceCache = (
        distance_cache_adapter.validate_json(all_distances_path.read_bytes())
        if all_distances_path.exists()
        else {}
    )

    yards = sorted(yd for yd in network.get_all_yard_names() if yd in output_yds)
    added = add_missing_yd_distances(network, yards, distances)
    pair_count = len(yards) * (len(yards) - 1)
    print(
        f"{cache_file}: {added:,} of the {pair_count:,} output yard pairs "
        f"were missing and have been calculated."
    )
    if added:
        tmp_path = all_distances_path.with_suffix(".tmp")
        tmp_path.write_bytes(distance_cache_adapter.dump_json(distances, indent=2))
        tmp_path.replace(all_distances_path)

    output_rows: list[DistanceOutputRow] = []
    for yd_1_name in track(
        yards,
        description=get_progress_desc_with_padding(
            "Consolidating output yard distances..."
        ),
    ):
        for yd_2_name in yards:
            if yd_1_name == yd_2_name:
                continue
            entry = distances.get(yd_1_name, {}).get(yd_2_name)
            if entry is None:
                raise ValueError(f"Missing distance for {yd_1_name} -> {yd_2_name}")
            output_rows.append(
                DistanceOutputRow(
                    begin_yard=yd_1_name,
                    end_yard=yd_2_name,
                    distance_minutes=entry.minutes
                    * REFERENCE_SPEED_KM_H
                    / network.train_speed_km_h,
                    distance_km=entry.km,
                )
            )
    return output_rows
