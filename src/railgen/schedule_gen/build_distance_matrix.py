from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple, cast

import networkx as nx
from pydantic import TypeAdapter
from rich.progress import track

from railgen.constants import ALL_DISTANCES_FILE
from railgen.constants import RAIL_NETWORK_DIR as GRAPH_DIR
from railgen.models import YgKey
from railgen.rail_network import RailNetwork
from railgen.utils import get_progress_desc_with_padding


@dataclass(frozen=True, kw_only=True)
class DistanceOutputRow:
    begin_yard_group: str
    end_yard_group: str
    distance_minutes: float
    distance_km: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def get_headers() -> list[str]:
        return [field.name for field in fields(DistanceOutputRow)]


# The minutes in the JSON cache file are always those at this train speed. Travel time is
# proportional to 1 / train speed (the slowdown on a grade is a share of the speed), so the
# minutes at any other speed follow from them exactly.
REFERENCE_SPEED_KM_H = 100.0


# Stored as 2 element array [minutes, km] - smaller than an object with named fields
class YgDistance(NamedTuple):
    minutes: float  # at REFERENCE_SPEED_KM_H
    km: float


# Contents of the JSON cache file: {yg1: {yg2: [minutes, km]}}
type DistanceCache = dict[YgKey, dict[YgKey, YgDistance]]
distance_cache_adapter = TypeAdapter[DistanceCache](DistanceCache)


def add_missing_yg_distances(
    network: RailNetwork, yard_groups: list[YgKey], distances: DistanceCache
) -> int:
    """Compute the distances between `yard_groups` that `distances` lacks and add them to it.

    One shortest-path search per yard group that has a missing pair. Returns the number
    of pairs added.
    """
    added = 0
    for yg_1 in track(
        yard_groups,
        description=get_progress_desc_with_padding(
            "Calculating missing yard group distances..."
        ),
    ):
        known = distances.get(yg_1, {})
        missing = [yg_2 for yg_2 in yard_groups if yg_2 != yg_1 and yg_2 not in known]
        if not missing:
            continue
        lengths, paths = nx.single_source_dijkstra(
            G=network.ygs_graph, source=yg_1, weight="weight"
        )
        assert isinstance(lengths, dict) and isinstance(paths, dict)
        lengths = cast(dict[str, float | int], lengths)
        paths = cast(dict[str, list[str]], paths)

        for yg_2 in missing:
            if yg_2 not in paths:
                continue  # no route; reported when the output rows are built
            yg_path = [YgKey.from_raw_str(node) for node in paths[yg_2]]
            minutes = network.get_travel_time(yg_path)  # at the speed of this run
            distances.setdefault(yg_1, {})[yg_2] = YgDistance(
                minutes=minutes * network.train_speed_km_h / REFERENCE_SPEED_KM_H,
                km=lengths[yg_2],
            )
            added += 1
    return added


def generate_output_yg_distances(
    network: RailNetwork,
    output_ygs: set[YgKey],
) -> list[DistanceOutputRow]:
    """Return distances between all pairs within output_ygs, using the JSON cache.

    The cache grows with use: it holds the pairs that earlier runs needed, and the pairs
    of this run that it lacks are computed and added to it. Its minutes include the
    slowdown on grades and are stored at `REFERENCE_SPEED_KM_H`; the rows returned here
    are at the train speed of this run, so the cache stays valid when the speed changes.
    Delete it after the network or the grade settings (`per_grade_speed_reduction`,
    `max_speed_reduction`) change.
    """
    all_distances_path: Path = GRAPH_DIR / ALL_DISTANCES_FILE
    distances: DistanceCache = (
        distance_cache_adapter.validate_json(all_distances_path.read_bytes())
        if all_distances_path.exists()
        else {}
    )

    yard_groups = sorted(
        yg for yg in network.get_all_yard_group_names() if yg in output_ygs
    )
    added = add_missing_yg_distances(network, yard_groups, distances)
    pair_count = len(yard_groups) * (len(yard_groups) - 1)
    print(
        f"{ALL_DISTANCES_FILE}: {added:,} of the {pair_count:,} output yard group pairs "
        f"were missing and have been calculated."
    )
    if added:
        tmp_path = all_distances_path.with_suffix(".tmp")
        tmp_path.write_bytes(distance_cache_adapter.dump_json(distances, indent=2))
        tmp_path.replace(all_distances_path)

    output_rows: list[DistanceOutputRow] = []
    for yg_1_name in track(
        yard_groups,
        description=get_progress_desc_with_padding(
            "Consolidating output yard group distances..."
        ),
    ):
        for yg_2_name in yard_groups:
            if yg_1_name == yg_2_name:
                continue
            entry = distances.get(yg_1_name, {}).get(yg_2_name)
            if entry is None:
                raise ValueError(f"Missing distance for {yg_1_name} -> {yg_2_name}")
            output_rows.append(
                DistanceOutputRow(
                    begin_yard_group=yg_1_name,
                    end_yard_group=yg_2_name,
                    distance_minutes=entry.minutes
                    * REFERENCE_SPEED_KM_H
                    / network.train_speed_km_h,
                    distance_km=entry.km,
                )
            )
    return output_rows
