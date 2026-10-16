import time
from itertools import permutations

from haversine import haversine
from rich import print

from railgen.constants import FORWARD_GEOCODE_MIN_INTERVAL_S
from railgen.models import (
    Coord,
    Location,
    YardGroup,
    YgKey,
)
from railgen.rail_network_gen.graph_utils import forward_geocode
from railgen.scenario_gen.distribution_groupings import YardGroupDistributions
from railgen.scenario_gen.distributions import ScenarioDistribution
from railgen.scenario_gen.scenario_config import ScenarioGenerationConfig


def geocode_yg_regions(config: ScenarioGenerationConfig) -> None:
    """Geocode city/state region centers and store them on the config, so the saved config shows the coordinates."""
    for i, region in enumerate(config.region_distributions):
        if isinstance(region.center, Location):
            city, state = region.center.raw_city, region.center.state_code
            coord = forward_geocode(city, state)
            if coord is None:
                raise ValueError(f"Failed to geocode region center {city}, {state}")
            print(
                f"[green]Geocoded[/green] {city}, {state} [green]to[/green] {coord.lat}, {coord.lon}"
            )
            # respect Nominatim rate limit if there are remaining regions to geocode
            if i < len(config.region_distributions) - 1:
                time.sleep(FORWARD_GEOCODE_MIN_INTERVAL_S)
            region.geocoded_center_coordinate = coord


def resolve_yard_group_distributions(
    config_geocoded: ScenarioGenerationConfig, ygs_dict: dict[YgKey, YardGroup]
) -> YardGroupDistributions:
    """Per-yard-group and per-leg distributions, with pool names and regions resolved.

    City/state regions need to be geocoded first (`geocode_yg_regions`). Raises if a yard
    group or leg gets a distribution from more than one source.
    """

    def resolve_dist(dist: ScenarioDistribution | str) -> ScenarioDistribution:
        resolved = (
            config_geocoded.distributions_pool.get(dist)
            if isinstance(dist, str)
            else dist
        )
        if resolved is None:
            raise ValueError(f"distribution {dist} not found in distributions_pool")
        return resolved

    leg_dists = {
        (orig, dest): resolve_dist(dist)
        for orig, dests in config_geocoded.leg_trip_time_percent_diff_distributions.items()
        for dest, dist in dests.items()
    }
    dwell_dists = {
        yg: resolve_dist(v)
        for yg, v in config_geocoded.yard_group_dwell_distributions.items()
    }
    for i, region in enumerate(config_geocoded.region_distributions):
        center_coord = (
            region.center
            if isinstance(region.center, Coord)
            else region.geocoded_center_coordinate
        )
        if center_coord is None:
            raise ValueError(
                f"region_distributions[{i}] center location not geocoded: {region.center}"
            )
        ygs_inside_region = [
            yg_key
            for yg_key, yg in ygs_dict.items()
            if haversine(
                (center_coord.lat, center_coord.lon), (yg.coord.lat, yg.coord.lon)
            )
            <= region.radius_km
        ]
        if region.dwell_time is not None:
            for yg_key in ygs_inside_region:
                if yg_key in dwell_dists:
                    raise ValueError(
                        f"region_distributions[{i}] conflicts with another distribution for {yg_key}"
                    )
                dwell_dists[yg_key] = resolve_dist(region.dwell_time)

        if region.trip_time_percent_diff is not None:
            for orig_dest in permutations(ygs_inside_region, 2):
                if orig_dest in leg_dists:
                    raise ValueError(
                        f"region_distributions[{i}] conflicts with another distribution for {orig_dest}"
                    )
                leg_dists[orig_dest] = resolve_dist(region.trip_time_percent_diff)
    return YardGroupDistributions(
        leg_distributions=leg_dists, yg_dwell_distributions=dwell_dists
    )
