from dataclasses import asdict, dataclass, fields
from typing import Literal

from railgen.models import DayOfWeek, TrainCategory, YgKey
from railgen.schedule_gen.attributes import RouteSegment, Train, TrainSymbol
from railgen.utils import HourMinute

IntBool = Literal[1, 0]


@dataclass(frozen=True, kw_only=True)
class DataOutputRow:
    ROW_ID: str

    TRN_CATG: TrainCategory

    TRN_SYMB: TrainSymbol

    BGN_YG: YgKey
    END_YG: YgKey

    MON_TON: float
    TUE_TON: float
    WED_TON: float
    THU_TON: float
    FRI_TON: float
    SAT_TON: float
    SUN_TON: float

    IS_PU: IntBool
    IS_SO: IntBool
    IS_CG: IntBool

    MON_REQ_LOC: int
    TUE_REQ_LOC: int
    WED_REQ_LOC: int
    THU_REQ_LOC: int
    FRI_REQ_LOC: int
    SAT_REQ_LOC: int
    SUN_REQ_LOC: int

    RUNS_MON: IntBool
    RUNS_TUE: IntBool
    RUNS_WED: IntBool
    RUNS_THU: IntBool
    RUNS_FRI: IntBool
    RUNS_SAT: IntBool
    RUNS_SUN: IntBool

    # Operating day counters. Starts from 1 (the first day of operation).
    DEP_DAY: int
    DEP_TIME: HourMinute
    ARVL_DAY: int
    ARVL_TIME: HourMinute

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def get_headers() -> list[str]:
        return [field.name for field in fields(DataOutputRow)]

    @staticmethod
    def new(
        train: Train,
        trip: RouteSegment,
    ) -> "DataOutputRow":
        is_crew_change = (
            trip.bgn_yg_reason.crew_change or trip.end_yg_reason.crew_change
        )
        return DataOutputRow(
            ROW_ID=f"{train.symbol}_{trip.bgn_yg}_{trip.end_yg}",
            TRN_CATG=train.category,
            TRN_SYMB=train.symbol,
            BGN_YG=trip.bgn_yg,
            END_YG=trip.end_yg,
            MON_TON=trip.tonnage_by_day.get_day(DayOfWeek.MONDAY),
            TUE_TON=trip.tonnage_by_day.get_day(DayOfWeek.TUESDAY),
            WED_TON=trip.tonnage_by_day.get_day(DayOfWeek.WEDNESDAY),
            THU_TON=trip.tonnage_by_day.get_day(DayOfWeek.THURSDAY),
            FRI_TON=trip.tonnage_by_day.get_day(DayOfWeek.FRIDAY),
            SAT_TON=trip.tonnage_by_day.get_day(DayOfWeek.SATURDAY),
            SUN_TON=trip.tonnage_by_day.get_day(DayOfWeek.SUNDAY),
            IS_PU=1 if trip.bgn_yg_reason.pick_up else 0,
            IS_SO=1 if trip.end_yg_reason.set_out else 0,
            IS_CG=1 if is_crew_change else 0,
            MON_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.MONDAY),
            TUE_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.TUESDAY),
            WED_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.WEDNESDAY),
            THU_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.THURSDAY),
            FRI_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.FRIDAY),
            SAT_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.SATURDAY),
            SUN_REQ_LOC=trip.req_counts_by_day.get_day(DayOfWeek.SUNDAY),
            RUNS_MON=1 if train.days_active.get_day(DayOfWeek.MONDAY) else 0,
            RUNS_TUE=1 if train.days_active.get_day(DayOfWeek.TUESDAY) else 0,
            RUNS_WED=1 if train.days_active.get_day(DayOfWeek.WEDNESDAY) else 0,
            RUNS_THU=1 if train.days_active.get_day(DayOfWeek.THURSDAY) else 0,
            RUNS_FRI=1 if train.days_active.get_day(DayOfWeek.FRIDAY) else 0,
            RUNS_SAT=1 if train.days_active.get_day(DayOfWeek.SATURDAY) else 0,
            RUNS_SUN=1 if train.days_active.get_day(DayOfWeek.SUNDAY) else 0,
            ARVL_DAY=trip.arvl_day,
            DEP_DAY=trip.dep_day,
            ARVL_TIME=trip.arvl_time,
            DEP_TIME=trip.dep_time,
        )
