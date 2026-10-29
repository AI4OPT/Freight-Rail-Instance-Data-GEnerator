from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING

import networkx as nx
from haversine import Unit, haversine

from railgen.models import YdKey
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.routes import Route
from railgen.schedule_gen.schedule_config import ServiceDesignSettings

if TYPE_CHECKING:
    from railgen.schedule_gen.demand import Commodity

EAST, WEST = "east", "west"
GATEWAY = "gateway"  # side of a yard in both regions
_WEST_OF_LONGITUDE = -90.0  # a yard with no state is West if west of 90 W

# What a region does with a flow, in run-log order.
WHOLE = "kept whole: both ends on this side, or a gateway end and the other end here"
GATEWAY_PAIR = "kept whole: gateway to gateway, most of the route on this side"
SINGLE_YARD = "kept whole: crossing flow with a single yard on the other side"
CUT = "cut at the boundary"
OTHER_SIDE = "not in this region (the other region keeps it)"
NEITHER = "in neither region (gateway area only, tie, or one leg across the line)"


@dataclass(kw_only=True)
class RegionReport:
    """What the region rules did: commodities and tons per week per outcome."""

    region: str
    count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tons: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cut_at_gateway: int = 0  # cut flows whose part ends (or starts) at a gateway yard
    gateway_yards: list[YdKey] = field(default_factory=list)

    def add(self, outcome: str, c: "Commodity") -> None:
        self.count[outcome] += 1
        self.tons[outcome] += c.tons_per_week


def yard_regions(
    network: RailNetwork, settings: ServiceDesignSettings
) -> dict[YdKey, set[str]]:
    """The region(s) of every yard: {"east"}, {"west"} or both at a gateway."""
    rule = settings.region
    west_states = set(rule.west_states)
    gateway_zones = set(rule.gateway_zones)
    anchors = [
        yd.coord for yd in network.yds_dict.values() if yd.od_zone_code in gateway_zones
    ]
    out: dict[YdKey, set[str]] = {}
    for key, yd in network.yds_dict.items():
        if yd.state:
            side = WEST if yd.state in west_states else EAST
        else:  # no state recorded: go by longitude
            side = WEST if yd.coord.lon < _WEST_OF_LONGITUDE else EAST
        gateway = any(
            haversine(
                (yd.coord.lat, yd.coord.lon), (a.lat, a.lon), unit=Unit.KILOMETERS
            )
            <= rule.gateway_radius_km
            for a in anchors
        )
        out[key] = {EAST, WEST} if gateway else {side}
    return out


def _side(regions: dict[YdKey, set[str]], yard: YdKey) -> str:
    r = regions[yard]
    return GATEWAY if len(r) == 2 else next(iter(r))


def _majority_side(path: list[YdKey], sides: list[str], graph: nx.Graph) -> str | None:
    """Gateway-to-gateway flows: the side with most of the route km, else None."""
    km = {EAST: 0.0, WEST: 0.0}
    for (a, b), pair in zip(pairwise(path), pairwise(sides)):
        # A leg counts for the side(s) whose yard it touches: an East-West leg
        # half for each, a leg between two gateway yards for neither.
        touched = set(pair) - {GATEWAY}
        for s in touched:
            km[s] += float(graph[a][b]["weight"]) / len(touched)
    if km[EAST] == km[WEST]:  # a tie, or no km outside the gateway areas
        return None
    return EAST if km[EAST] > km[WEST] else WEST


def _split(path: list[YdKey], sides: list[str]) -> tuple[list[YdKey], list[YdKey]]:
    """The origin's part and the destination's part of a route between the sides."""
    o_side, d_side = sides[0], sides[-1]
    k = next(i for i, s in enumerate(sides) if s == d_side)
    j = k - 1
    while sides[j] == GATEWAY and sides[j - 1] == GATEWAY:
        j -= 1  # back to the origin's edge of the gateway area
    head = path[: j + 1]
    m = max(i for i, s in enumerate(sides) if s == o_side)
    j = m + 1
    while sides[j] == GATEWAY and sides[j + 1] == GATEWAY:
        j += 1  # on to the destination's edge of the gateway area
    return head, path[j:]


def _part_route(part: list[YdKey], route: Route, graph: nx.Graph) -> Route:
    km = sum(float(graph[a][b]["weight"]) for a, b in pairwise(part))
    return Route(
        path=tuple(part),
        cost=route.cost * km / route.km if route.km else 0.0,
        km=km,
    )


def cut_to_region(
    commodities: list["Commodity"],
    routes: dict[tuple[YdKey, YdKey], Route],
    network: RailNetwork,
    settings: ServiceDesignSettings,
) -> tuple[list["Commodity"], dict[tuple[YdKey, YdKey], Route], RegionReport]:
    """The commodities and routes of `settings.region.instance` (east or west)."""
    region = settings.region.instance
    regions = yard_regions(network, settings)
    graph = network.yds_graph
    report = RegionReport(
        region=region,
        gateway_yards=sorted(y for y, r in regions.items() if len(r) == 2),
    )
    kept: list[Commodity] = []
    kept_routes: dict[tuple[YdKey, YdKey], Route] = {}

    for c in commodities:
        route = routes[(c.origin, c.destination)]
        path = list(route.path)
        sides = [_side(regions, y) for y in path]
        o, d = sides[0], sides[-1]
        if o == d == GATEWAY:
            side, outcome = _majority_side(path, sides, graph), GATEWAY_PAIR
        elif o == d or GATEWAY in (o, d):
            side, outcome = (d if o == GATEWAY else o), WHOLE
        else:  # one end East, the other West
            head, tail = _split(path, sides)
            if len(head) >= 2 and len(tail) >= 2:
                part = head if o == region else tail
                pair = (part[0], part[-1])
                kept_routes[pair] = _part_route(part, route, graph)
                kept.append(replace(c, origin=pair[0], destination=pair[1]))
                report.add(CUT, c)
                if _side(regions, part[-1] if o == region else part[0]) == GATEWAY:
                    report.cut_at_gateway += 1
                continue
            if len(head) < 2 and len(tail) < 2:
                side = None  # one leg across the line
            else:
                side = d if len(head) < 2 else o
            outcome = SINGLE_YARD
        if side is None:
            report.add(NEITHER, c)
        elif side == region:
            kept.append(c)
            kept_routes[(c.origin, c.destination)] = route
            report.add(outcome, c)
        else:
            report.add(OTHER_SIDE, c)
    return kept, kept_routes, report


def region_lines(report: RegionReport) -> list[str]:
    """Run-log lines for a region instance."""
    lines = [f"region {report.region}:"]
    for outcome in (WHOLE, GATEWAY_PAIR, SINGLE_YARD, CUT, OTHER_SIDE, NEITHER):
        lines.append(
            f"  {outcome}: {report.count[outcome]:,} commodities "
            f"({report.tons[outcome]:,.0f} t/week)"
        )
        if outcome == CUT:
            lines.append(
                f"    of which cut at a gateway yard: {report.cut_at_gateway:,}"
            )
    lines.append(f"gateway yards (in both regions): {len(report.gateway_yards):,}")
    return lines
