"""East and West instances: which yard groups are in a region, and which flows each keeps."""

from itertools import pairwise
from types import SimpleNamespace

import networkx as nx
import pytest

from railgen.models import YgKey, ZoneCode
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.demand import Commodity
from railgen.schedule_gen.regions import (
    CUT,
    EAST,
    NEITHER,
    WEST,
    cut_to_region,
    yard_regions,
)
from railgen.schedule_gen.routes import Route
from railgen.schedule_gen.schedule_config import RegionSettings, ServiceDesignSettings

# West: Texas, Kansas, Missouri. East: Illinois (far from Chicago), Ohio, Georgia.
# Gateways: Chicago, and Memphis with a yard on each bank (Arkansas, Tennessee).
TX, KS, MO = YgKey("TX-1"), YgKey("KS-2"), YgKey("MO-3")
IL, OH, GA = YgKey("IL-4"), YgKey("OH-5"), YgKey("GA-6")
CHI, AR_G, MEM, TN_G = YgKey("CHI-7"), YgKey("AR-8"), YgKey("MEM-9"), YgKey("TN-10")
YARDS = {
    TX: ("TX", 32.8, -97.0, None),
    KS: ("KS", 38.0, -98.0, None),
    MO: ("MO", 38.6, -92.2, None),
    IL: ("IL", 40.1, -89.4, None),
    OH: ("OH", 40.0, -83.0, None),
    GA: ("GA", 32.0, -83.0, None),
    CHI: ("IL", 41.8, -87.6, "171"),  # represents the Chicago zone
    AR_G: ("AR", 35.15, -90.18, None),  # 12 km from MEM
    MEM: ("TN", 35.15, -90.05, "471"),  # represents the Memphis zone
    TN_G: ("TN", 35.05, -89.95, None),  # 14 km from MEM
}
PATHS = [
    (TX, AR_G, MEM, TN_G, OH),  # West -> East through Memphis
    (OH, TN_G, MEM, AR_G, TX),  # East -> West through Memphis
    (KS, AR_G, MO, IL, GA),  # passes Memphis, crosses the line elsewhere
    (CHI, IL, MO, TX),  # gateway -> West, through Illinois
    (CHI, OH),  # gateway -> East
    (CHI, IL, MEM),  # gateway -> gateway, all of it on East track
    (AR_G, MEM),  # gateway -> gateway inside one gateway area
    (MO, IL, GA),  # a single yard group on the West side
    (MO, IL),  # one leg across the line
    (TX, MO),  # West -> West
]
ROUTES = {
    (p[0], p[-1]): Route(path=p, cost=10.0 * (len(p) - 1), km=100.0 * (len(p) - 1))
    for p in PATHS
}


def _network() -> RailNetwork:
    network = object.__new__(RailNetwork)
    network.ygs_dict = {  # ty: ignore[invalid-assignment]
        k: SimpleNamespace(
            state=s, coord=SimpleNamespace(lat=lat, lon=lon), od_zone_code=z
        )
        for k, (s, lat, lon, z) in YARDS.items()
    }
    network.ygs_graph = nx.Graph()
    for p in PATHS:
        for a, b in pairwise(p):
            network.ygs_graph.add_edge(a, b, weight=100.0)
    return network


def _settings(instance: str) -> ServiceDesignSettings:
    return ServiceDesignSettings().model_copy(
        update={"region": RegionSettings(instance=instance)}  # ty: ignore[invalid-argument-type]
    )


def _kept(instance: str):
    flows = [
        Commodity(
            origin=o,
            destination=d,
            train_type="manifest",
            tons_per_week=100.0,
            zone_pair=(ZoneCode("481"), ZoneCode("391")),
        )
        for o, d in ROUTES
    ]
    kept, routes, report = cut_to_region(flows, ROUTES, _network(), _settings(instance))
    return {
        (c.origin, c.destination): routes[(c.origin, c.destination)].path for c in kept
    }, report


def test_states_and_gateways_decide_the_region():
    regions = yard_regions(_network(), _settings("west"))
    assert regions[TX] == {WEST} and regions[MO] == {WEST}
    assert regions[OH] == {EAST} and regions[IL] == {EAST}
    for gateway in (CHI, AR_G, MEM, TN_G):
        assert regions[gateway] == {EAST, WEST}


def test_a_gateway_end_goes_to_the_side_of_the_other_end():
    west, _ = _kept("west")
    east, _ = _kept("east")
    assert west[(CHI, TX)] == (CHI, IL, MO, TX)  # whole, Illinois track included
    assert east[(CHI, OH)] == (CHI, OH)
    assert {k for k in west if k[0] == CHI} == {(CHI, TX)}
    assert {k for k in east if k[0] == CHI} == {(CHI, OH), (CHI, MEM)}


def test_gateway_to_gateway_goes_where_most_of_the_route_is():
    west, west_report = _kept("west")
    east, east_report = _kept("east")
    assert east[(CHI, MEM)] == (CHI, IL, MEM) and (CHI, MEM) not in west
    assert (AR_G, MEM) not in east and (AR_G, MEM) not in west
    assert west_report.count[NEITHER] == east_report.count[NEITHER] == 2


def test_a_crossing_flow_stops_at_its_own_edge_of_the_gateway():
    west, report = _kept("west")
    east, _ = _kept("east")
    assert west[(TX, AR_G)] == (TX, AR_G) and west[(AR_G, TX)] == (AR_G, TX)
    assert east[(TN_G, OH)] == (TN_G, OH) and east[(OH, TN_G)] == (OH, TN_G)
    assert report.cut_at_gateway == 2


def test_a_crossing_elsewhere_is_cut_at_the_line():
    west, report = _kept("west")
    east, _ = _kept("east")
    assert west[(KS, MO)] == (KS, AR_G, MO)  # the Memphis gateway passed earlier
    assert east[(IL, GA)] == (IL, GA)
    assert report.count[CUT] == 3


def test_a_single_yard_on_one_side_goes_whole_to_the_other():
    west, _ = _kept("west")
    east, _ = _kept("east")
    assert east[(MO, GA)] == (MO, IL, GA)
    assert not any(o == MO for o, _ in west)
    assert (MO, IL) not in east and (MO, IL) not in west


def test_part_routes_keep_their_share_of_km_and_cost():
    _, routes, _ = cut_to_region(
        [
            Commodity(
                origin=TX,
                destination=OH,
                train_type="manifest",
                tons_per_week=100.0,
                zone_pair=(ZoneCode("481"), ZoneCode("391")),
            )
        ],
        ROUTES,
        _network(),
        _settings("east"),
    )
    assert routes[(TN_G, OH)].km == pytest.approx(100.0)
    assert routes[(TN_G, OH)].cost == pytest.approx(10.0)
