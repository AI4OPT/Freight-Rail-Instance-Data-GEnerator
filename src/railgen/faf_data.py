import csv
import io
import math
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import shapefile
import shapely
from haversine import Unit, haversine
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import nearest_points

from railgen.constants import (
    FAF5_CSV_NAME,
    FAF5_ZIP_URL,
    FAF6_CSV_NAME,
    FAF6_ZIP_URL,
    FAF6_ZONES_BASE_NAME,
    FAF6_ZONES_DIR_NAME,
    FAF6_ZONES_ZIP_URL,
    FAF_YEAR,
    FAF_ZONE_TOLERANCE_KM,
    FRIDGE_USER_AGENT,
    INPUT_DATA_DIR,
)
from railgen.models import Coord, ZoneCode


class LazyLoad[T]:
    """Lazily load a value resulting from an expensive operation."""

    def __init__(self, loader: Callable[[], T]):
        self._loader = loader
        self._value: T | None = None

    def load(self) -> T:
        """Get the value if loaded, or perform the initial load."""
        if self._value is None:
            self._value = self._loader()
        return self._value


def _download_zip(
    url: str, required: set[str]
) -> tuple[zipfile.ZipFile, dict[str, str]]:
    """Download a FAF zip into memory. Returns it with a map of file name to archive path."""
    print(f"FAF files not found, downloading {url}...")

    # FAF data website only offers static-RSA ciphers, which Python's default cipher list excludes.
    # We have to manually create an SSL context to succeed.
    ssl_context = httpx.create_ssl_context()
    ssl_context.set_ciphers("DEFAULT")

    buffer = io.BytesIO()
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=120,
        verify=ssl_context,
        headers={"User-Agent": FRIDGE_USER_AGENT},
    ) as response:
        _ = response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task("Downloading FAF zip...", total=total)
            for chunk in response.iter_bytes():
                _ = buffer.write(chunk)
                progress.update(task, advance=len(chunk))
    zf = zipfile.ZipFile(buffer)
    # In case the files are not at the zip root
    names = {Path(n).name: n for n in zf.namelist()}
    missing = required - names.keys()
    if missing:
        raise FileNotFoundError(
            f"{missing} not found inside downloaded zip. Zip contents: {zf.namelist()}",
        )
    return zf, names


def ensure_faf6_files() -> None:
    """Download the FAF6 zip and extract its CSV if missing."""
    INPUT_DATA_DIR.mkdir(exist_ok=True)
    dest = INPUT_DATA_DIR / FAF6_CSV_NAME
    if dest.exists():
        return
    zf, names = _download_zip(FAF6_ZIP_URL, {FAF6_CSV_NAME})
    with zf:
        dest.write_bytes(zf.read(names[FAF6_CSV_NAME]))
    print("FAF download complete.")


def ensure_faf5_file() -> None:
    """Download the FAF5 zip and extract its CSV if missing."""
    INPUT_DATA_DIR.mkdir(exist_ok=True)
    dest = INPUT_DATA_DIR / FAF5_CSV_NAME
    if dest.exists():
        return
    zf, names = _download_zip(FAF5_ZIP_URL, {FAF5_CSV_NAME})
    with zf:
        dest.write_bytes(zf.read(names[FAF5_CSV_NAME]))
    print("FAF5 download complete.")


def _load_faf_od_zone_codes() -> frozenset[ZoneCode]:
    """FAF zones that send or receive rail or multimodal freight (any trade type, domestic part of the move)."""
    ensure_faf6_files()
    codes: set[ZoneCode] = set()
    with open(INPUT_DATA_DIR / FAF6_CSV_NAME, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dms_mode", "").strip() not in (
                "2",
                "5",
            ):  # rail, multiple modes
                continue
            tons = row.get(f"tons_{FAF_YEAR}", "").strip()
            orig = row.get("dms_orig", "").strip()
            dest = row.get("dms_dest", "").strip()
            if tons and float(tons) > 0 and orig and dest:
                codes.update((ZoneCode(orig), ZoneCode(dest)))
    return frozenset(codes)


FAF_OD_ZONE_CODES = LazyLoad[frozenset[ZoneCode]](_load_faf_od_zone_codes)


def ensure_faf6_zone_files() -> None:
    """Download the FAF6 zone boundaries (2022 CFS geography shapefile) if any file read is missing."""
    zone_dir = INPUT_DATA_DIR / FAF6_ZONES_DIR_NAME
    needed = {FAF6_ZONES_BASE_NAME + ext for ext in (".shp", ".shx", ".dbf")}
    if all((zone_dir / name).exists() for name in needed):
        return
    zone_dir.mkdir(parents=True, exist_ok=True)
    zf, names = _download_zip(FAF6_ZONES_ZIP_URL, needed)
    with zf:
        for name in needed:
            _ = (zone_dir / name).write_bytes(zf.read(names[name]))
    print("FAF6 zone download complete.")


@dataclass(frozen=True, kw_only=True)
class FafZone:
    """A FAF6 zone and its boundary, in lon, lat."""

    code: ZoneCode
    polygon: Polygon | MultiPolygon


def _zone_reader() -> shapefile.Reader:
    ensure_faf6_zone_files()
    return shapefile.Reader(INPUT_DATA_DIR / FAF6_ZONES_DIR_NAME / FAF6_ZONES_BASE_NAME)


def _load_faf_zones() -> list[FafZone]:
    """Every FAF6 zone with its boundary, from the zone shapefile (downloaded if missing)."""
    reader = _zone_reader()
    zones: list[FafZone] = []
    for item, record in zip(reader.shapes(), reader.records(), strict=True):
        if record is None:  # deleted record
            continue
        polygon = shape(item)
        assert isinstance(polygon, Polygon | MultiPolygon)
        shapely.prepare(polygon)  # speeds up the many `covers` calls
        zones.append(FafZone(code=ZoneCode(str(record["FAF6"])), polygon=polygon))
    return zones


def _load_faf_zone_names() -> dict[ZoneCode, str]:
    """Short name of each FAF6 zone, e.g. "Chicago IL-IN-WI (IL Part)", from the zone shapefile's table."""
    return {
        ZoneCode(str(record["FAF6"])): str(record["FAF6_SHORT"])
        for record in _zone_reader().records()
        if record is not None
    }


FAF_ZONES = LazyLoad[list[FafZone]](_load_faf_zones)
FAF_ZONE_NAMES = LazyLoad[dict[ZoneCode, str]](_load_faf_zone_names)


def faf_zone_of(
    coord: Coord,
    zones: list[FafZone],
    tolerance_km: float = FAF_ZONE_TOLERANCE_KM,
) -> ZoneCode | None:
    """The zone containing `coord`; else the nearest zone within `tolerance_km`; else None."""
    point = Point(coord.lon, coord.lat)
    for zone in zones:
        if zone.polygon.covers(point):
            return zone.code

    def km_to(zone: FafZone) -> float:
        nearest, _ = nearest_points(zone.polygon, point)
        return haversine(
            (coord.lat, coord.lon), (nearest.y, nearest.x), unit=Unit.KILOMETERS
        )

    km, code = min(
        ((km_to(zone), zone.code) for zone in zones), default=(math.inf, None)
    )
    return code if km <= tolerance_km else None
