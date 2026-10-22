import asyncio
import heapq
import itertools
import time
from collections.abc import Awaitable, Callable
from typing import cast

import httpx
from networkx.classes.graph import Graph
from rich.progress import track

from railgen.constants import (
    BDC_KEY,
    ELEVATION_API_BATCH_SIZE,
    ELEVATION_API_BATCHES_PER_MINUTE,
    ELEVATION_API_MAX_ATTEMPTS,
    ELEVATION_API_RETRY_WAIT_S,
    ELEVATION_API_URL,
    FORWARD_GEOCODE_API_URL,
    FRIDGE_USER_AGENT,
    GEOCODE_API_URL,
    GEOCODE_CONCURRENCY,
)
from railgen.models import (
    Coord,
    ElevationResponse,
    LineStringGeometry,
    MultiLineStringGeometry,
    YardGroup,
    YgKey,
)
from railgen.utils import get_progress_desc_with_padding


async def _batched_fetch[T](
    coords: list[Coord],
    fetch_batch: Callable[[httpx.AsyncClient, list[Coord]], Awaitable[list[T]]],
    batch_size: int,
    batches_per_minute: int,
    description: str,
) -> list[T]:
    """Batched API calls, sent in waves of `batches_per_minute` to stay under a per-minute rate limit."""
    batches = [coords[i : i + batch_size] for i in range(0, len(coords), batch_size)]
    all_results: list[list[T]] = []
    prev_time_start: float | None = None
    waves = list(range(0, len(batches), batches_per_minute))
    async with httpx.AsyncClient() as client:
        for wave_start in track(
            waves, description=get_progress_desc_with_padding(description)
        ):
            now = time.monotonic()
            if prev_time_start is not None:
                wait = 61.0 - (now - prev_time_start)
                if wait > 0:
                    await asyncio.sleep(wait)
            prev_time_start = time.monotonic()
            wave_batches = batches[wave_start : wave_start + batches_per_minute]
            wave_results = await asyncio.gather(
                *[fetch_batch(client, batch) for batch in wave_batches]
            )
            all_results.extend(wave_results)
    return [item for batch_result in all_results for item in batch_result]


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


async def _fetch_elevation_batch(
    client: httpx.AsyncClient, coords: list[Coord]
) -> list[float]:
    if not coords:
        return []
    attempt = 1
    while True:
        try:
            resp = await client.get(
                ELEVATION_API_URL,
                params={
                    "latitude": ",".join(str(c.lat) for c in coords),
                    "longitude": ",".join(str(c.lon) for c in coords),
                },
                timeout=120.0,
            )
            _ = resp.raise_for_status()
            return ElevationResponse.model_validate_json(resp.text).elevation
        except httpx.HTTPError as e:
            if attempt == ELEVATION_API_MAX_ATTEMPTS:
                raise
            reason = (
                e.response.status_code
                if isinstance(e, httpx.HTTPStatusError)
                else type(e).__name__
            )
            print(
                f"Elevation request failed ({reason}), retrying in {ELEVATION_API_RETRY_WAIT_S}s"
            )
            await asyncio.sleep(ELEVATION_API_RETRY_WAIT_S)
            attempt += 1


async def concurrent_elevation_fetch(coords: list[Coord]) -> list[float]:
    return await _batched_fetch(
        coords,
        fetch_batch=_fetch_elevation_batch,
        batch_size=ELEVATION_API_BATCH_SIZE,
        batches_per_minute=ELEVATION_API_BATCHES_PER_MINUTE,
        description="Fetching elevations...",
    )


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
            description="Geocoding yard groups...",
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


def get_yg_coord(geom: LineStringGeometry | MultiLineStringGeometry) -> Coord:
    """A YardGroup's position: just the first coordinate of its geometry."""
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
    trav_graph: Graph,
    all_yg_geom_coords: dict[Coord, YgKey],
    yg: tuple[str, YardGroup],
) -> dict[YgKey, float]:
    """
    Dijkstra outward from a YardGroup over the track graph. When a path hits another YardGroup's geometry, that YardGroup is a neighbor and the path stops there.

    ## Args:
    - trav_graph -- track graph built from the geojson
    - all_yg_geom_coords -- every YardGroup geometry coordinate -> its YardGroup name
    - yg -- (name, YardGroup) to start from

    ## Returns:
    - Neighbor YardGroup names -> distance (km)
    """
    src_yg_name = yg[0]
    src_coord = yg[1].coord
    dist_from_src_yg = {src_coord: 0.0}
    # Coords aren't orderable, so heap ties break on push order.
    pushed = itertools.count()
    pq = [(0.0, next(pushed), src_coord)]
    found_connected_ygs: dict[YgKey, float] = {}
    while pq:
        dist_to_curr, _, curr_coord = heapq.heappop(pq)
        if dist_to_curr != dist_from_src_yg.get(curr_coord):
            continue
        if curr_coord not in trav_graph:
            continue
        for next_coord, edge_data in trav_graph[curr_coord].items():
            edge_dist = float(edge_data.get("weight", 0.0))
            new_dist = dist_to_curr + edge_dist
            yg_at_next = all_yg_geom_coords.get(next_coord)
            if yg_at_next is not None and yg_at_next != src_yg_name:
                if (
                    yg_at_next not in found_connected_ygs
                    or new_dist < found_connected_ygs[yg_at_next]
                ):
                    found_connected_ygs[yg_at_next] = new_dist
                continue
            if new_dist < dist_from_src_yg.get(next_coord, float("inf")):
                dist_from_src_yg[next_coord] = new_dist
                heapq.heappush(pq, (new_dist, next(pushed), next_coord))
    return found_connected_ygs
