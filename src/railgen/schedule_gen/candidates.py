from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from railgen.models import Yard, YdKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.demand import Commodity
from railgen.schedule_gen.routes import Route
from railgen.schedule_gen.schedule_config import ServiceDesignSettings


@dataclass(frozen=True, kw_only=True)
class CandidateService:
    id: str
    train_type: str
    path: tuple[YdKey, ...]  # every yard on the physical route
    stop_positions: tuple[
        int, ...
    ]  # indices into `path`: origin, potential stops, destination
    leg_km: tuple[float, ...]  # one per leg between consecutive stops
    leg_capacity_tons: tuple[float, ...]

    @property
    def stops(self) -> tuple[YdKey, ...]:
        return tuple(self.path[i] for i in self.stop_positions)

    @property
    def total_km(self) -> float:
        return sum(self.leg_km)


def largest_yards_per_state(yds: dict[YdKey, Yard], per_state: int) -> set[YdKey]:
    """The `per_state` largest yards (by track length) in each state.

    Yards with no recorded state are skipped rather than grouped together.
    """
    by_state: dict[str, list[tuple[float, YdKey]]] = defaultdict(list)
    for key, yd in yds.items():
        if yd.state:
            by_state[yd.state].append((yd.shape_length_km, key))
    return {
        key
        for yards in by_state.values()
        for _km, key in sorted(yards, reverse=True)[:per_state]
    }


def allowed_yards(network: RailNetwork, settings: ServiceDesignSettings) -> set[YdKey]:
    """Yards where freight can be picked up or set out: the FAF zone yards plus the
    `work_yards_per_state` largest per state. Non-zone yards only see transfers, so their
    weekly net is 0.
    """
    zone = {key for key, yd in network.yds_dict.items() if yd.is_od_yd}
    return zone | largest_yards_per_state(
        network.yds_dict, settings.work_yards_per_state
    )


def build_candidate(
    network: RailNetwork,
    path: list[YdKey],
    train_type: str,
    allowed: set[YdKey],
    capacity: float,
    cid: str,
) -> CandidateService:
    last = len(path) - 1
    positions = [i for i, node in enumerate(path) if i in (0, last) or node in allowed]
    leg_km: list[float] = []
    for a, b in pairwise(positions):
        seg = path[a : b + 1]
        leg_km.append(
            sum(float(network.yds_graph[u][v]["weight"]) for u, v in pairwise(seg))
        )
    return CandidateService(
        id=cid,
        train_type=train_type,
        path=tuple(path),
        stop_positions=tuple(positions),
        leg_km=tuple(leg_km),
        leg_capacity_tons=tuple(capacity for _ in leg_km),
    )


def generate_candidates(
    network: RailNetwork,
    commodities: list[Commodity],
    routes: dict[tuple[YdKey, YdKey], Route],
    allowed: set[YdKey],
    settings: ServiceDesignSettings,
) -> list[CandidateService]:
    """Candidates for every kept yard pair (see `demand.build_commodities`)."""
    pair_tons: dict[tuple[YdKey, YdKey], float] = defaultdict(float)
    pair_types: dict[tuple[YdKey, YdKey], set[str]] = defaultdict(set)
    for c in commodities:
        pair_tons[(c.origin, c.destination)] += c.tons_per_week
        pair_types[(c.origin, c.destination)].add(c.train_type)
    pairs = sorted(pair_tons, key=lambda p: (-pair_tons[p], p))

    out: list[CandidateService] = []
    for pair in pairs:
        route = routes[pair]
        for t in sorted(pair_types[pair]):
            out.append(
                build_candidate(
                    network,
                    list(route.path),
                    t,
                    allowed,
                    settings.capacity_of(t),
                    f"S{len(out):06d}",
                )
            )
    return out
