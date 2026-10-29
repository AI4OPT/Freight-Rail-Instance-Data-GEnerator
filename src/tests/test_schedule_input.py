import csv
from itertools import pairwise
from pathlib import Path

import pytest

from railgen.constants import TRAIN_SCHEDULE_FILE_NAME
from railgen.models import DayOfWeek, TrainCategory, Week, YdKey
from railgen.scenario_gen.schedule_input import parse_schedule
from railgen.schedule_gen.attributes import (
    RouteSegment,
    TonnageByDay,
    Train,
    TrainSymbol,
    YardReason,
)
from railgen.schedule_gen.schedule_output import DataOutputRow
from railgen.utils import MINUTES_PER_DAY, MINUTES_PER_HOUR

ALL_DAYS = Week(mon=True, tue=True, wed=True, thu=True, fri=True, sat=True, sun=True)
WEEKDAYS = Week(mon=True, tue=True, wed=True, thu=True, fri=True, sat=False, sun=False)


def _segment(
    bgn: str, end: str, tonnage: float, *, first: bool, last: bool
) -> RouteSegment:
    return RouteSegment(
        bgn_yd=YdKey.from_raw_str(bgn),
        end_yd=YdKey.from_raw_str(end),
        tonnage_by_day=TonnageByDay(
            tonnages=Week(
                mon=tonnage, tue=tonnage, wed=tonnage, thu=0, fri=tonnage, sat=0, sun=0
            )
        ),
        bgn_yd_reason=YardReason(pick_up=first, set_out=False, crew_change=not first),
        end_yd_reason=YardReason(pick_up=False, set_out=last, crew_change=False),
        req_counts_by_day=Week(mon=2, tue=2, wed=2, thu=0, fri=3, sat=0, sun=0),
        dep_day=1,
        arvl_day=2,
        dep_time="04:15",
        arvl_time="09:45",
    )


def _train(symbol: str, yds: list[str], days_active: Week[bool]) -> Train:
    legs = list(pairwise(yds))
    return Train(
        category=TrainCategory(symbol[0]),
        symbol=TrainSymbol(symbol),
        days_active=days_active,
        route=[
            _segment(
                bgn,
                end,
                1000.0 + i * 100,
                first=i == 0,
                last=i == len(legs) - 1,
            )
            for i, (bgn, end) in enumerate(legs)
        ],
    )


def write_schedule(dir: Path, trains: list[Train], *, mangle=None) -> Path:
    """Write `trains` through the real output writer, optionally mangling the rows first."""
    rows = [
        DataOutputRow.new(train, trip).to_dict()
        for train in trains
        for trip in train.route
    ]
    if mangle is not None:
        mangle(rows)
    headers = [h for h in DataOutputRow.get_headers() if h in rows[0]]
    path = dir / TRAIN_SCHEDULE_FILE_NAME
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


TRAINS = [
    _train("ATRN_1", ["NORRIS-1750", "DE BUTTS-1591", "ROANOKE-1794"], ALL_DAYS),
    _train("MTRN_7", ["JESSUP-987", "MILLHAM-748"], WEEKDAYS),
]


def test_round_trips_trains_and_trips(tmp_path: Path):
    write_schedule(tmp_path, TRAINS)
    parsed = parse_schedule(tmp_path)

    assert [t.symbol for t in parsed] == ["ATRN_1", "MTRN_7"]
    assert [t.category for t in parsed] == [TrainCategory.A, TrainCategory.M]
    assert [len(t.trips) for t in parsed] == [2, 1]
    assert parsed[1].days_active == WEEKDAYS

    first, second = parsed[0].trips
    assert (first.bgn_yd, first.end_yd) == ("NORRIS-1750", "DE BUTTS-1591")
    assert (second.bgn_yd, second.end_yd) == ("DE BUTTS-1591", "ROANOKE-1794")
    assert first.tonnage_by_day.get_day(DayOfWeek.MONDAY) == 1000.0
    assert second.tonnage_by_day.get_day(DayOfWeek.MONDAY) == 1100.0
    assert first.tonnage_by_day.get_day(DayOfWeek.THURSDAY) == 0
    assert first.req_counts_by_day == Week(
        mon=2, tue=2, wed=2, thu=0, fri=3, sat=0, sun=0
    )
    assert first.dep_minute == 0 * MINUTES_PER_DAY + 4 * MINUTES_PER_HOUR + 15
    assert first.arvl_minute == 1 * MINUTES_PER_DAY + 9 * MINUTES_PER_HOUR + 45


def test_missing_schedule_file_raises(tmp_path: Path):
    with pytest.raises(ValueError, match=TRAIN_SCHEDULE_FILE_NAME):
        _ = parse_schedule(tmp_path)


def test_missing_column_raises(tmp_path: Path):
    def drop_row_id(rows: list[dict[str, object]]) -> None:
        for row in rows:
            del row["ROW_ID"]

    write_schedule(tmp_path, TRAINS, mangle=drop_row_id)
    with pytest.raises(ValueError, match="ROW_ID"):
        _ = parse_schedule(tmp_path)


def test_inconsistent_days_active_raises(tmp_path: Path):
    def unset_monday(rows: list[dict[str, object]]) -> None:
        rows[1]["RUNS_MON"] = 0

    write_schedule(tmp_path, TRAINS, mangle=unset_monday)
    with pytest.raises(ValueError, match=r"ATRN_1 .* day-of-week flags"):
        _ = parse_schedule(tmp_path)


def test_non_contiguous_route_raises(tmp_path: Path):
    def break_chain(rows: list[dict[str, object]]) -> None:
        rows[1]["BGN_YD"] = YdKey.from_raw_str("KINNEY-1703")

    write_schedule(tmp_path, TRAINS, mangle=break_chain)
    with pytest.raises(ValueError, match="ATRN_1 route is not contiguous"):
        _ = parse_schedule(tmp_path)
