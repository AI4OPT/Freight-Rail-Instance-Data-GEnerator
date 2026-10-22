from itertools import pairwise
from typing import cast

import networkx as nx
from haversine import haversine
from networkx.classes.graph import Graph

from railgen.models import YardGroup, YgKey
from railgen.rail_network_gen import graph_generation
from railgen.schedule_gen.schedule_config import Config
from railgen.stats import OdAssignmentStats
from railgen.utils import MINUTES_PER_HOUR, kilometers_to_meters


class RailNetwork:
    ygs_dict: dict[YgKey, YardGroup]
    ygs_graph: Graph
    od_assignment: OdAssignmentStats
    cfg: Config
    train_speed_km_h: float

    def __init__(self, cfg: Config) -> None:
        self.ygs_dict, self.ygs_graph, self.od_assignment = (
            graph_generation.generate_or_load_network(cfg)
        )
        self.cfg = cfg
        self.train_speed_km_h = cfg.train_settings.speed_km_h

    def get_all_yard_group_names(self) -> list[YgKey]:
        """
        Returns all available YardGroup names in the network graph/dict.
        """
        return list(self.ygs_dict.keys())

    def get_hub_yard_groups(self) -> set[YgKey]:
        """Hub yard groups of every hub city that has one (`hub_yard_groups_by_city`)."""
        return {key for key in self.hub_yard_groups_by_city().values() if key}

    def hub_yard_groups_by_city(self) -> dict[str, YgKey | None]:
        """Hub yard group of each of `hub_locations`, by `hub_label`: the largest FAF zone yard group (by track length) in the city's state within `hub_radius_km` of the geocoded city. A city with none in range gets None."""
        net = self.cfg.network_settings
        coords = graph_generation.hub_city_coords(net.hub_locations)
        zone_ygs = [(key, yg) for key, yg in self.ygs_dict.items() if yg.is_od_yg]
        hubs: dict[str, YgKey | None] = {}
        for hub_loc in net.hub_locations:
            label = graph_generation.hub_label(hub_loc)
            hubs[label] = None
            coord = coords.get(label)
            if coord is None:
                print(f"Hub {label} could not be geocoded; it gets no hub.")
                continue
            near = [
                (key, yg)
                for key, yg in zone_ygs
                if (yg.state or "").upper() == hub_loc.state_code
                and haversine((coord.lat, coord.lon), (yg.coord.lat, yg.coord.lon))
                <= net.hub_radius_km
            ]
            if not near:
                print(
                    f"Hub {label}: no zone yard group of {hub_loc.state_code} within {net.hub_radius_km:g} km; it gets no hub."
                )
                continue
            hubs[label] = max(near, key=lambda x: x[1].shape_length_km)[0]
        return hubs

    def build_phase1_routing_graph(self) -> nx.DiGraph:
        """Phase 1 routing graph: travel time, with edges touching OD yard groups discounted to pull routes through them."""
        od_nodes = {name for name, yg in self.ygs_dict.items() if yg.is_od_yg}
        graph: nx.DiGraph = nx.DiGraph()
        for node, attrs in self.ygs_graph.nodes(data=True):
            graph.add_node(node, **attrs)
        for u, v in self.ygs_graph.edges():
            discount = (
                self.cfg.network_settings.od_discount_factor
                if (u in od_nodes or v in od_nodes)
                else 1.0
            )
            _ = graph.add_edge(u, v, weight=self.get_travel_time([u, v]) * discount)
            _ = graph.add_edge(v, u, weight=self.get_travel_time([v, u]) * discount)
        return graph

    def build_phase2_routing_graph(
        self, segment_frequencies: dict[tuple[YgKey, YgKey], int]
    ) -> nx.DiGraph:
        """Phase 2 routing graph: travel time, discounted more for segments Phase 1 used more."""
        graph: nx.DiGraph = nx.DiGraph()
        for node, attrs in self.ygs_graph.nodes(data=True):
            graph.add_node(node, **attrs)
        for u, v in self.ygs_graph.edges():
            for src, dst in [(u, v), (v, u)]:
                travel_time = self.get_travel_time([src, dst])
                freq = segment_frequencies.get((src, dst), 0)
                if freq > 0 and self.cfg.general_settings.median_f > 0:
                    raw_discount = cast(
                        float,
                        1.0
                        - self.cfg.general_settings.seg_freq_arc_cost_discount_alpha
                        * (freq / self.cfg.general_settings.median_f)
                        ** self.cfg.general_settings.normalized_seg_arc_cost_exponent_lambda,
                    )
                    discount = max(raw_discount, 0.01)
                else:
                    discount = 1.0
                _ = graph.add_edge(src, dst, weight=travel_time * discount)
        return graph

    def get_travel_time(
        self, path: list[YgKey], speed_km_h: float | None = None
    ) -> float:
        """
        Returns the minutes required to travel the given path at given speed.
        """
        total_time = 0
        for sub_origin, sub_dest in pairwise(path):
            distance_km = float(self.ygs_graph[sub_origin][sub_dest]["weight"])
            # If the trip is uphill, reduce speed based on grade
            train_speed = speed_km_h or self.train_speed_km_h
            grade_percent = self.get_grade_percent(sub_origin, sub_dest)
            if grade_percent > 0:
                speed_reduction = (
                    grade_percent * self.cfg.network_settings.per_grade_speed_reduction
                )
                speed_reduction = min(
                    speed_reduction, self.cfg.network_settings.max_speed_reduction
                )
                train_speed *= 1 - speed_reduction

            time = distance_km / train_speed * MINUTES_PER_HOUR
            total_time += time
        return total_time

    def get_grade_percent(self, yg1: YgKey, yg2: YgKey) -> float:
        """
        Returns the ascending grade (%) of the hop from yg1 to yg2 (0 when the hop is level or descending).
        """
        distance_km = float(self.ygs_graph[yg1][yg2]["weight"])
        elevation_change = self.ygs_dict[yg2].elevation - self.ygs_dict[yg1].elevation
        if elevation_change <= 0 or distance_km <= 0:
            return 0.0
        return (elevation_change / kilometers_to_meters(distance_km)) * 100

    def get_ruling_grade(self, path: list[YgKey], min_hop_km: float) -> float:
        """
        Returns the steepest ascending grade (%) over the hops of the given path. A hop shorter than min_hop_km has no grade.
        """
        return max(
            (
                self.get_grade_percent(sub_origin, sub_dest)
                for sub_origin, sub_dest in pairwise(path)
                if float(self.ygs_graph[sub_origin][sub_dest]["weight"]) >= min_hop_km
            ),
            default=0.0,
        )
