import asyncio
import heapq
import math
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import httpx
import numpy as np
import rasterio
from rasterio.transform import rowcol
from rich.progress import track

from railgen.constants import (
    BDC_KEY,
    DEM_DIR,
    DEM_DOWNLOAD_CONCURRENCY,
    DEM_TILE_URL,
    FORWARD_GEOCODE_API_URL,
    FRIDGE_USER_AGENT,
    GEOCODE_API_URL,
    GEOCODE_CONCURRENCY,
)
from railgen.models import (
    Coord,
    LineStringGeometry,
    MultiLineStringGeometry,
    YdKey,
)
from railgen.utils import get_progress_desc_with_padding


async def _semaphore_fetch[T](
    coords: list[Coord],
    fetch_one: Callable[[httpx.AsyncClient, asyncio.Semaphore, Coord], Awaitable[T]],
    concurrency: int,
    description: str,
) -> list[T]:
    """One API call per coord, at most `concurrency` at a time."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[T | None] = [None] * len(coords)

    async def _fetch_indexed(
        client: httpx.AsyncClient, i: int, coord: Coord
    ) -> tuple[int, T]:
        return (i, await fetch_one(client, semaphore, coord))

    async with httpx.AsyncClient() as client:
        tasks = [_fetch_indexed(client, i, coord) for i, coord in enumerate(coords)]
        for fut in track(
            asyncio.as_completed(tasks),
            description=get_progress_desc_with_padding(description),
            total=len(coords),
        ):
            i, result = await fut
            results[i] = result

    return cast(list[T], results)


def sample_elevations(coords: list[Coord]) -> list[float]:
    """Elevation in metres of each coordinate, read from the Copernicus GLO-90 tiles in `DEM_DIR`. Missing tiles are downloaded, `DEM_DOWNLOAD_CONCURRENCY` at a time; a coordinate with no tile (open water) gets 0."""
    by_tile: dict[str, list[int]] = defaultdict(list)
    for i, coord in enumerate(coords):
        lat, lon = math.floor(coord.lat), math.floor(coord.lon)
        by_tile[
            f"Copernicus_DSM_COG_30_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
        ].append(i)
    DEM_DIR.mkdir(parents=True, exist_ok=True)

    def download(name: str) -> bool:
        resp = httpx.get(DEM_TILE_URL.format(name=name), timeout=300.0)
        if resp.status_code == 404:
            return False
        partial = DEM_DIR / f"{name}.tif.{os.getpid()}.part"  # own file per run
        _ = partial.write_bytes(resp.raise_for_status().content)
        partial.replace(DEM_DIR / f"{name}.tif")
        return True

    missing = [name for name in by_tile if not (DEM_DIR / f"{name}.tif").exists()]
    with ThreadPoolExecutor(DEM_DOWNLOAD_CONCURRENCY) as pool:
        found = list(
            track(
                pool.map(download, missing),
                total=len(missing),
                description=get_progress_desc_with_padding("Downloading DEM tiles..."),
            )
        )
    not_found = [name for name, ok in zip(missing, found) if not ok]
    if not_found:
        print(
            f"Warning: {len(not_found):,} DEM tile(s) not found at {DEM_TILE_URL}, so "
            f"{sum(len(by_tile[name]) for name in not_found):,} coordinate(s) get 0 m as if "
            f"on open water. If any are on land, check DEM_TILE_URL: {', '.join(not_found)}"
        )
    elevations = [0.0] * len(coords)
    for name, idx in track(
        by_tile.items(),
        description=get_progress_desc_with_padding("Sampling elevations..."),
    ):
        path = DEM_DIR / f"{name}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            rows, cols = rowcol(
                src.transform,
                [coords[i].lon for i in idx],
                [coords[i].lat for i in idx],
            )
            band = src.read(1)
            # Tiles are offset half a pixel, so a point on the bottom or right edge lands one past the last pixel.
            tile_elevations = band[
                np.clip(rows, 0, band.shape[0] - 1),
                np.clip(cols, 0, band.shape[1] - 1),
            ]
        for i, elevation in zip(idx, tile_elevations):
            elevations[i] = float(elevation)
    return elevations


async def _fetch_one_geocode(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    coord: Coord,
) -> tuple[str, str]:
    async with semaphore:
        if not BDC_KEY:
            raise ValueError("BDC_KEY is not set in the environment")
        resp = await client.get(
            GEOCODE_API_URL,
            params={
                "latitude": coord.lat,
                "longitude": coord.lon,
                "localityLanguage": "en",
                "key": BDC_KEY,
            },
            timeout=30.0,
        )
        if not resp.is_success:
            raise ValueError(
                f"Geocode API returned status {resp.status_code} for {coord.lat}, {coord.lon}"
            )
        data: dict[str, object] = resp.json()
        city = str(data.get("city") or "").strip()
        subdivision_code = str(data.get("principalSubdivisionCode") or "").strip()
        country = str(data.get("countryCode") or "").strip()
        state = ""
        if country == "US" and subdivision_code.startswith("US-"):
            state = subdivision_code.removeprefix("US-")
        return (city, state)


def concurrent_geocode(coords: list[Coord]) -> list[tuple[str, str]]:
    """Reverse-geocode a list of coordinates, returning (city, state_code) per coord."""
    return asyncio.run(
        _semaphore_fetch(
            coords,
            fetch_one=_fetch_one_geocode,
            concurrency=GEOCODE_CONCURRENCY,
            description="Geocoding yards...",
        )
    )


def forward_geocode(city: str, state: str) -> Coord | None:
    """Look up a US city/state on Nominatim. None if the lookup fails or finds nothing.

    Callers handle rate limiting (Nominatim allows about 1 request/second).
    """
    try:
        resp = httpx.get(
            FORWARD_GEOCODE_API_URL,
            params={
                "city": city,
                "state": state,
                "country": "USA",
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": FRIDGE_USER_AGENT},
            timeout=30.0,
        )
        if not resp.is_success:
            return None
        data: list[dict[str, object]] = resp.json()
        if not data:
            return None
        return Coord(
            lat=float(cast(str, data[0]["lat"])), lon=float(cast(str, data[0]["lon"]))
        )
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        return None


def get_yd_coord(geom: LineStringGeometry | MultiLineStringGeometry) -> Coord:
    """A Yard's position: just the first coordinate of its geometry."""
    if isinstance(geom, LineStringGeometry):
        assert geom.coordinates is not None
        lon, lat = geom.coordinates[0][0], geom.coordinates[0][1]
        return Coord(lat=lat, lon=lon)
    else:
        assert geom.coordinates is not None
        first_line = geom.coordinates[0]
        assert first_line is not None
        lon, lat = first_line[0][0], first_line[0][1]
        return Coord(lat=lat, lon=lon)


def custom_dijkstra(
    trav_graph: dict[int, dict[int, tuple[float, ...]]],
    yd_at: list[YdKey | None],
    src_yd_name: YdKey,
    src: int,
) -> dict[YdKey, float]:
    """
    Dijkstra outward from a Yard over the track graph. When a path hits another Yard's geometry, that Yard is a neighbor and the path stops there.

    ## Args:
    - trav_graph -- track node -> neighbor track node -> lengths (km) of the track segments between them
    - yd_at -- Yard name whose geometry includes each track node, or None
    - src_yd_name -- name of the Yard to start from
    - src -- its track node

    ## Returns:
    - Neighbor Yard names -> distance (km)
    """
    dist_from_src_yd = {src: 0.0}
    pq = [(0.0, src)]
    found_connected_yds: dict[YdKey, float] = {}
    while pq:
        dist_to_curr, curr = heapq.heappop(pq)
        if dist_to_curr != dist_from_src_yd[curr]:
            continue
        for next_node, edge_kms in trav_graph[curr].items():
            new_dist = dist_to_curr
            for km in edge_kms:
                new_dist += km
            yd_at_next = yd_at[next_node]
            if yd_at_next is not None and yd_at_next != src_yd_name:
                if (
                    yd_at_next not in found_connected_yds
                    or new_dist < found_connected_yds[yd_at_next]
                ):
                    found_connected_yds[yd_at_next] = new_dist
                continue
            if new_dist < dist_from_src_yd.get(next_node, float("inf")):
                dist_from_src_yd[next_node] = new_dist
                heapq.heappush(pq, (new_dist, next_node))
    return found_connected_yds
