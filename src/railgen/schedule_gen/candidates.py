from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

from railgen.models import YardGroup, YgKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.demand import Commodity
from railgen.schedule_gen.routes import Route
from railgen.schedule_gen.schedule_config import ServiceDesignSettings


@dataclass(frozen=True, kw_only=True)
class CandidateService:
    id: str
    train_type: str
    path: tuple[YgKey, ...]  # every yard group on the physical route
    stop_positions: tuple[
        int, ...
    ]  # indices into `path`: origin, potential stops, destination
    leg_km: tuple[float, ...]  # one per leg between consecutive stops
    leg_hours: tuple[float, ...]
    leg_capacity_tons: tuple[float, ...]

    @property
    def stops(self) -> tuple[YgKey, ...]:
        return tuple(self.path[i] for i in self.stop_positions)

    @property
    def total_km(self) -> float:
        return sum(self.leg_km)


def largest_yards_per_state(ygs: dict[YgKey, YardGroup], per_state: int) -> set[YgKey]:
    """The `per_state` largest yard groups (by track length) in each state.

    Yard groups with no recorded state are skipped rather than grouped together.
    """
    by_state: dict[str, list[tuple[float, YgKey]]] = defaultdict(list)
    for key, yg in ygs.items():
        if yg.state:
            by_state[yg.state].append((yg.shape_length_km, key))
    return {
        key
        for yards in by_state.values()
        for _km, key in sorted(yards, reverse=True)[:per_state]
    }


def allowed_yards(network: RailNetwork, settings: ServiceDesignSettings) -> set[YgKey]:
    """Yards where freight can be picked up or set out: the FAF zone yard groups plus the
    `work_yards_per_state` largest per state. Non-zone yards only see transfers, so their
    weekly net is 0.
    """
    zone = {key for key, yg in network.ygs_dict.items() if yg.is_od_yg}
    return zone | largest_yards_per_state(
        network.ygs_dict, settings.work_yards_per_state
    )


def build_candidate(
    network: RailNetwork,
    path: list[YgKey],
    train_type: str,
    allowed: set[YgKey],
    capacity: float,
    cid: str,
) -> CandidateService:
    last = len(path) - 1
    positions = [i for i, node in enumerate(path) if i in (0, last) or node in allowed]
    leg_km: list[float] = []
    leg_hours: list[float] = []
    for a, b in pairwise(positions):
        seg = path[a : b + 1]
        leg_km.append(
            sum(float(network.ygs_graph[u][v]["weight"]) for u, v in pairwise(seg))
        )
        leg_hours.append(network.get_travel_time(seg) / 60)
    return CandidateService(
        id=cid,
        train_type=train_type,
        path=tuple(path),
        stop_positions=tuple(positions),
        leg_km=tuple(leg_km),
        leg_hours=tuple(leg_hours),
        leg_capacity_tons=tuple(capacity for _ in leg_km),
    )


def generate_candidates(
    network: RailNetwork,
    commodities: list[Commodity],
    routes: dict[tuple[YgKey, YgKey], Route],
    allowed: set[YgKey],
    settings: ServiceDesignSettings,
) -> list[CandidateService]:
    """Candidates for every kept yard pair (see `demand.build_commodities`)."""
    pair_tons: dict[tuple[YgKey, YgKey], float] = defaultdict(float)
    pair_groups: dict[tuple[YgKey, YgKey], set[str]] = defaultdict(set)
    for c in commodities:
        pair_tons[(c.origin, c.destination)] += c.tons_per_week
        pair_groups[(c.origin, c.destination)].add(c.group)
    pairs = sorted(pair_tons, key=lambda p: (-pair_tons[p], p))

    out: list[CandidateService] = []
    for pair in pairs:
        route = routes[pair]
        types = sorted(
            {t for g in pair_groups[pair] for t in settings.group_train_types[g]}
        )
        for t in types:
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
