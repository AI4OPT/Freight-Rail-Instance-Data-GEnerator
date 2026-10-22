import math
from dataclasses import dataclass

from railgen.models import DayOfWeek, TrainCategory, Week, YgKey
from railgen.utils import HourMinute


@dataclass(frozen=True, kw_only=True)
class YardGroupReason:
    pick_up: bool
    set_out: bool
    crew_change: bool
    pu_amount: float = 0.0  # tons of freight picked up per run
    so_amount: float = 0.0  # tons set out per run


@dataclass(frozen=True, kw_only=True)
class TonnageByDay:
    tonnages: Week[float]

    def get_day(self, day: DayOfWeek) -> float:
        """Get the tonnage value for a given day."""
        return self.tonnages.get_day(day)

    def get_req_loc_counts(self, hp_per_ton: float, locomotive_hp: float) -> Week[int]:
        """Locomotives per day for a leg that needs `hp_per_ton` horsepower per ton."""

        def calc_req_count(tonnage: float) -> int:
            return math.ceil(hp_per_ton * tonnage / locomotive_hp)

        return Week(
            mon=calc_req_count(self.get_day(DayOfWeek.MONDAY)),
            tue=calc_req_count(self.get_day(DayOfWeek.TUESDAY)),
            wed=calc_req_count(self.get_day(DayOfWeek.WEDNESDAY)),
            thu=calc_req_count(self.get_day(DayOfWeek.THURSDAY)),
            fri=calc_req_count(self.get_day(DayOfWeek.FRIDAY)),
            sat=calc_req_count(self.get_day(DayOfWeek.SATURDAY)),
            sun=calc_req_count(self.get_day(DayOfWeek.SUNDAY)),
        )

    @staticmethod
    def new(tons: float, days: Week[bool]) -> "TonnageByDay":
        """`tons` on the days the symbol runs and 0 on the others."""
        return TonnageByDay(
            tonnages=Week(
                mon=tons if days.mon else 0.0,
                tue=tons if days.tue else 0.0,
                wed=tons if days.wed else 0.0,
                thu=tons if days.thu else 0.0,
                fri=tons if days.fri else 0.0,
                sat=tons if days.sat else 0.0,
                sun=tons if days.sun else 0.0,
            )
        )


@dataclass(frozen=True, kw_only=True)
class RouteSegment:
    bgn_yg: YgKey
    end_yg: YgKey
    tonnage_by_day: TonnageByDay
    bgn_yg_reason: YardGroupReason
    end_yg_reason: YardGroupReason
    req_counts_by_day: Week[int]
    dep_day: int  # Day relative to trip start (1 = first day, 2 = second, etc.)
    arvl_day: int  # Day relative to trip start (1 = first day, 2 = second, etc.)
    dep_time: HourMinute
    arvl_time: HourMinute


class TrainSymbol(str):
    """Represents a unique identifier for a train."""

    @staticmethod
    def new(category: TrainCategory, number: int) -> "TrainSymbol":
        """Returns a new unique train symbol."""
        return TrainSymbol(f"{category}TRN_{number}")


@dataclass(frozen=True, kw_only=True)
class Train:
    category: TrainCategory
    symbol: TrainSymbol
    route: list[RouteSegment]
    days_active: Week[bool]
