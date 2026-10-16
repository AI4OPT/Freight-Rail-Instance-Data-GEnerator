"""FAF data: downloads FAF6 (and FAF5, for intra-zone distances) if needed and builds the OD tonnage and zone lookups."""

import csv
import io
import math
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import openpyxl
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from railgen.constants import (
    FAF5_CSV_NAME,
    FAF5_ZIP_URL,
    FAF6_CSV_NAME,
    FAF6_METADATA_NAME,
    FAF6_ZIP_URL,
    FAF6_ZONES_CSV_NAME,
    FRIDGE_USER_AGENT,
    INPUT_DATA_DIR,
    MULTIMODAL_INCLUDE_PROPORTION,
)
from railgen.models import FafODTonnage, Location, ZoneCode

WHOLE_STATE_NAMES: dict[str, str] = {
    "Alaska": "AK",
    "Arkansas": "AR",
    "Idaho": "ID",
    "Iowa": "IA",
    "Maine": "ME",
    "Montana": "MT",
    "Mississippi": "MS",
    "New Mexico": "NM",
    "North Dakota": "ND",
    "South Dakota": "SD",
    "Vermont": "VT",
    "West Virginia": "WV",
    "Wyoming": "WY",
}

# FAF zone short descriptions replaced with a single city that has rail yards.
# We do this so yard groups can be matched to zones.
ZONE_LABEL_OVERRIDES: dict[str, str] = {
    "020": "Anchorage AK",
    "050": "Little Rock AR",
    "064": "Oakland CA",
    "081": "Englewood CO",
    "092": "Stamford CT",
    "101": "Wilmington DE",
    "111": "Washington DC",
    "121": "Jacksonville FL",
    "160": "Pocatello ID",
    "171": "Cicero IL",
    "172": "East St. Louis IL",
    "181": "East Chicago IN",
    "201": "Kansas City KS",
    "211": "Taylor Mill KY",
    "212": "Louisville KY",
    "222": "Lake Charles LA",
    "223": "Jefferson Parish LA",
    "230": "South Portland ME",
    "242": "Gaithersburg MD",
    "251": "Boston MA",
    "271": "Saint Paul MN",
    "280": "Jackson MS",
    "291": "Kansas City MO",
    "292": "St. Louis MO",
    "300": "Billings MT",
    "311": "Omaha NE",
    "331": "Nashua NH",
    "341": "Secaucus NJ",
    "342": "Camden NJ",
    "350": "Gallup NM",
    "363": "New York NY",
    "371": "Charlotte NC",
    "373": "Raleigh NC",
    "380": "Minot ND",
    "391": "Cincinnati OH",
    "411": "Portland OR",
    "421": "Philadelphia PA",
    "422": "Monroeville PA",
    "441": "Providence RI",
    "460": "Pierre SD",
    "471": "Memphis TN",
    "484": "Fort Worth TX",
    "485": "El Paso TX",
    "500": "St Albans City VT",
    "512": "Norfolk VA",
    "513": "Arlington VA",
    "532": "Vancouver WA",
    "540": "Huntington WV",
    "560": "Cheyenne WY",
}


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


def parse_faf_zone(zone: str) -> tuple[str, str] | None:
    """Parse FAF zone string to (city, state_code), or None for whole-state zones."""
    m = re.match(r"^\d{3}-(.+)$", zone)
    if not m:
        return None
    rest = m.group(1).strip()

    part_m = re.search(r"\(([A-Z]{2}) Part\)", rest)
    if part_m:
        state = part_m.group(1)
        city = re.sub(
            r"\s+[A-Z]{2}(?:-[A-Z]{2})*\s+\([A-Z]{2} Part\)\s*$", "", rest
        ).strip()
        city = re.sub(r"\s+[A-Z]{2}(?:-[A-Z]{2})*$", "", city).strip()
        return (city, state)

    rest = re.sub(r"\s+CFS\s+Area\s*$", "", rest, flags=re.IGNORECASE).strip()

    m2 = re.match(r"^(.+?)\s+([A-Z]{2})$", rest)
    if m2:
        return (m2.group(1).strip(), m2.group(2))

    for state_name in WHOLE_STATE_NAMES:
        if rest == state_name:
            return None

    return None


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
    """Download the FAF zip if the data CSV or edited zones CSV is missing, and write whichever is missing."""
    INPUT_DATA_DIR.mkdir(exist_ok=True)
    csv_dest = INPUT_DATA_DIR / FAF6_CSV_NAME
    zones_dest = INPUT_DATA_DIR / FAF6_ZONES_CSV_NAME
    if csv_dest.exists() and zones_dest.exists():
        return
    zf, names = _download_zip(FAF6_ZIP_URL, {FAF6_CSV_NAME, FAF6_METADATA_NAME})
    with zf:
        if not csv_dest.exists():
            csv_dest.write_bytes(zf.read(names[FAF6_CSV_NAME]))
        if not zones_dest.exists():
            metadata = io.BytesIO(zf.read(names[FAF6_METADATA_NAME]))
            sheet = openpyxl.load_workbook(metadata, read_only=True)[
                "FAF Zone (Domestic)"
            ]
            header, *rows = [
                [str(c).strip() for c in row]
                for row in sheet.iter_rows(values_only=True)
                if row[0]
            ]
            with open(zones_dest, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(
                    [code, ZONE_LABEL_OVERRIDES.get(code, short), long]
                    for code, short, long in rows
                )
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


def _load_faf_data() -> FafODTonnage:
    """Weekly OD tons from one pass over the FAF6 CSV: rail plus a share of multimodal, all trade types."""
    ensure_faf6_files()

    # stats
    domestic_total_rail_only = 0
    import_total_rail_only = 0
    export_total_rail_only = 0
    total_dms_mode_rail = 0
    total_dms_mode_multimodal_rail = 0

    od_totals: dict[tuple[ZoneCode, ZoneCode], float] = {}
    od_commodity_totals: dict[tuple[ZoneCode, ZoneCode], dict[str, float]] = {}
    with open(INPUT_DATA_DIR / FAF6_CSV_NAME, newline="") as f:
        RAIL_DMS_MODE = "2"
        MULTIMODAL_DMS_MODE = "5"

        TRADE_TYPE_DOMESTIC = "1"
        TRADE_TYPE_IMPORT = "2"
        TRADE_TYPE_EXPORT = "3"
        for row in csv.DictReader(f):
            dms_mode = row.get("dms_mode", "").strip()
            if dms_mode not in [RAIL_DMS_MODE, MULTIMODAL_DMS_MODE]:
                continue

            tons_str = row.get("tons_2022", "").strip()
            tons = float(tons_str) if tons_str else 0.0
            if tons <= 0:
                continue

            # If this is multimodal we only want to count a portion of the tonnage
            if dms_mode == MULTIMODAL_DMS_MODE:
                tons *= MULTIMODAL_INCLUDE_PROPORTION

            # Track stats
            if dms_mode == RAIL_DMS_MODE:
                total_dms_mode_rail += tons
            elif dms_mode == MULTIMODAL_DMS_MODE:
                total_dms_mode_multimodal_rail += tons

            commodity = row.get("sctg2", "").strip().zfill(2)

            # Imports and exports count too, but only their domestic leg (dms_orig/dms_dest)
            trade_type = row.get("trade_type", "").strip()
            # Track stats
            if dms_mode == RAIL_DMS_MODE:
                if trade_type == TRADE_TYPE_DOMESTIC:
                    domestic_total_rail_only += tons
                elif trade_type == TRADE_TYPE_IMPORT:
                    import_total_rail_only += tons
                elif trade_type == TRADE_TYPE_EXPORT:
                    export_total_rail_only += tons

            # Only look at the domestic portion of this row
            raw_orig = row.get("dms_orig", "").strip()
            raw_dest = row.get("dms_dest", "").strip()

            if not raw_orig or not raw_dest:
                continue
            orig, dest = ZoneCode(raw_orig), ZoneCode(raw_dest)
            od_totals[(orig, dest)] = od_totals.get((orig, dest), 0.0) + tons
            if commodity:
                per_commodity = od_commodity_totals.setdefault((orig, dest), {})
                per_commodity[commodity] = per_commodity.get(commodity, 0.0) + tons

    weekly_factor = 1000 / 52
    weekly_od_totals = {pair: tons * weekly_factor for pair, tons in od_totals.items()}
    weekly_od_commodity = {
        pair: {code: tons * weekly_factor for code, tons in per_commodity.items()}
        for pair, per_commodity in od_commodity_totals.items()
    }

    # Print the stats
    assert math.isclose(
        total_dms_mode_rail,
        domestic_total_rail_only + import_total_rail_only + export_total_rail_only,
    )
    print(
        f"[FAF Tonnage] Total Domestic Tons (Rail Only): {domestic_total_rail_only:.1f}"
    )
    print(f"[FAF Tonnage] Total Import Tons (Rail Only): {import_total_rail_only:.1f}")
    print(f"[FAF Tonnage] Total Export Tons (Rail Only): {export_total_rail_only:.1f}")
    print(f"[FAF Tonnage] Total Rail DMS Mode Tons: {total_dms_mode_rail:.1f}")
    print(
        f"[FAF Tonnage] Total Multimodal DMS Mode Tons (with {(MULTIMODAL_INCLUDE_PROPORTION * 100):.0f}% rail multiplier): {total_dms_mode_multimodal_rail:.1f}"
    )
    print(
        f"[FAF Tonnage] Total Rail + Multimodal Tons: {(total_dms_mode_rail + total_dms_mode_multimodal_rail):.1f}"
    )

    return FafODTonnage(
        od_tonnage=weekly_od_totals, od_commodity_tonnage=weekly_od_commodity
    )


FAF_OD_TONNAGE = LazyLoad[FafODTonnage](_load_faf_data)

FAF_OD_ZONE_CODES = LazyLoad[frozenset["ZoneCode"]](
    lambda: frozenset(
        code
        for zone_code_pair in FAF_OD_TONNAGE.load().od_tonnage
        for code in zone_code_pair
    )
)


def _build_faf_zone_lookup() -> dict[Location, ZoneCode]:
    """Zone metadata as Location -> zone code."""
    ensure_faf6_files()
    result: dict[Location, ZoneCode] = {}
    with open(
        INPUT_DATA_DIR / FAF6_ZONES_CSV_NAME,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.DictReader(f):
            code = ZoneCode(row["Numeric Label"].strip())
            short_desc = row["Short Description"].strip()
            label = f"{code}-{short_desc}"
            parsed = parse_faf_zone(label)
            if parsed is not None:
                city, state = parsed
                result[Location(raw_city=city, state=state)] = code
    return result


FAF_ZONE_LOOKUP = LazyLoad[dict[Location, ZoneCode]](_build_faf_zone_lookup)
FAF_LOCATION_LOOKUP = LazyLoad[dict[ZoneCode, Location]](
    lambda: {code: loc for loc, code in FAF_ZONE_LOOKUP.load().items()}
)
