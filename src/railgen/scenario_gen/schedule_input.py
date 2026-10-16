"""Reads and parses a previously generated train schedule."""

import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from railgen.constants import TRAIN_SCHEDULE_FILE_NAME
from railgen.models import TrainCategory, Week, YgKey
from railgen.schedule_gen.attributes import TonnageByDay, TrainSymbol
from railgen.schedule_gen.schedule_output import DataOutputRow
from railgen.utils import get_min_from_day_and_HHMM

_DAY_PREFIXES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


@dataclass(frozen=True, kw_only=True, slots=True)
class ScheduledTripLeg:
    """Represents a single leg of a train's route, i.e. a single row in the CSV."""

    row_id: str
    bgn_yg: YgKey
    end_yg: YgKey
    tonnage_by_day: TonnageByDay
    req_counts_by_day: Week[int]
    dep_minute: int
    arvl_minute: int


@dataclass(frozen=True, kw_only=True)
class ScheduledTrain:
    """A single train."""

    category: TrainCategory
    symbol: TrainSymbol
    days_active: Week[bool]
    trips: list[ScheduledTripLeg]


def parse_schedule(schedule_dir: Path) -> list[ScheduledTrain]:
    """Parse schedule file into list of trains."""
    path = schedule_dir / TRAIN_SCHEDULE_FILE_NAME
    if not path.is_file():
        raise ValueError(f"No {TRAIN_SCHEDULE_FILE_NAME} found in {schedule_dir}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        # Make sure every column we write is there
        if missing := set(DataOutputRow.get_headers()) - set(reader.fieldnames or []):
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows_by_symbol: dict[TrainSymbol, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            rows_by_symbol[TrainSymbol(row["TRN_SYMB"])].append(row)

    return [build_train(symbol, rows) for symbol, rows in rows_by_symbol.items()]


def parse_week[T](
    row: dict[str, str], column: str, and_parse: Callable[[str], T]
) -> Week[T]:
    """`column` is a template with `{}` where the day goes, e.g. `"{}_TON"`."""
    mon, tue, wed, thu, fri, sat, sun = (
        and_parse(row[column.format(day)]) for day in _DAY_PREFIXES
    )
    return Week(mon=mon, tue=tue, wed=wed, thu=thu, fri=fri, sat=sat, sun=sun)


def parse_int_bool(flag: str) -> bool:
    return bool(int(flag))


def build_train(symbol: TrainSymbol, rows: list[dict[str, str]]) -> ScheduledTrain:
    category = TrainCategory(rows[0]["TRN_CATG"])
    days_active = parse_week(rows[0], "RUNS_{}", parse_int_bool)
    for row in rows[1:]:
        if TrainCategory(row["TRN_CATG"]) != category:
            raise ValueError(f"{symbol} has rows with differing TRN_CATG")
        if parse_week(row, "RUNS_{}", parse_int_bool) != days_active:
            raise ValueError(f"{symbol} has rows with differing day-of-week flags")

    trips = [build_leg(row) for row in rows]
    # Rows should be in route order. If not, something upstream went wrong.
    for prev, cur in pairwise(trips):
        if cur.bgn_yg != prev.end_yg:
            raise ValueError(
                f"{symbol} route is not contiguous: leg ends at {prev.end_yg} but next begins at {cur.bgn_yg}"
            )

    return ScheduledTrain(
        category=category, symbol=symbol, days_active=days_active, trips=trips
    )


def build_leg(row: dict[str, str]) -> ScheduledTripLeg:
    dep_minute = get_min_from_day_and_HHMM(row["DEP_DAY"], row["DEP_TIME"])
    arvl_minute = get_min_from_day_and_HHMM(row["ARVL_DAY"], row["ARVL_TIME"])
    assert arvl_minute >= dep_minute, (
        f"Arrival time {arvl_minute} is not after departure time {dep_minute}"
    )
    return ScheduledTripLeg(
        row_id=row["ROW_ID"],
        bgn_yg=YgKey.from_raw_str(row["BGN_YG"]),
        end_yg=YgKey.from_raw_str(row["END_YG"]),
        tonnage_by_day=TonnageByDay(tonnages=parse_week(row, "{}_TON", float)),
        req_counts_by_day=parse_week(row, "{}_REQ_LOC", int),
        dep_minute=dep_minute,
        arvl_minute=arvl_minute,
    )
