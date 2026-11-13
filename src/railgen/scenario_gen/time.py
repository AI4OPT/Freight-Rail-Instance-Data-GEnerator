from functools import cached_property
from typing import Self

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from railgen.models import DayOfWeek
from railgen.utils import MINUTES_PER_DAY, MINUTES_PER_HOUR, HourMinute


class WeekMinute(int):
    """Represents a minute in the schedule week relative to Monday at midnight.
    Monday 00:00 = 0
    Tuesday 00:00 = 1440
    and so on...
    """


class SimulationMinute:
    """Represents a minute relative to the start of a train's simulation trip.
    For example, if a train scheduled on Tuesday departs at 10:00, the minute would be 600 (10 hours * 60 minutes).
    """

    def __init__(self, minute: int):
        self._minute = float(minute)

    def min_int(self) -> int:
        """Grab the underlying minute as an integer."""
        return int(self._minute)

    def min_float(self) -> float:
        """Grab the underlying minute as a float."""
        return self._minute

    def add_minute_delta(self, minute_delta: float) -> None:
        """Adds a minute delta to the simulation minute."""
        self._minute += minute_delta


class Time(BaseModel):
    """Represents a specific time in the schedule week."""

    day: DayOfWeek = Field(description="Day of the week. Use 'mon', 'tue', etc.")
    hour_min: HourMinute = Field(
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Time of day in HH:MM format (00:00 - 23:59).",
    )

    @field_validator("day", mode="before", json_schema_input_type=str)
    @classmethod
    def _parse_day(cls, value: object) -> DayOfWeek:
        if isinstance(value, DayOfWeek):
            return value
        if (day := DayOfWeek.from_str(str(value))) is None:
            raise ValueError(f"Invalid day of week: {value}")
        return day

    @field_serializer("day")
    def _serialize_day(self, day: DayOfWeek) -> str:
        return day.name[:3].lower()

    @field_validator("hour_min", mode="before")
    @classmethod
    def _parse_hour_min(cls, value: object) -> object:
        # YAML 1.1 reads unquoted 10:00 to 23:59 as base 60 ints, e.g. 12:30 -> 750
        return (
            f"{value // 60:02d}:{value % 60:02d}" if isinstance(value, int) else value
        )

    def minute_of_week(self) -> WeekMinute:
        hour, minute = map(int, self.hour_min.split(":"))
        return WeekMinute(
            (self.day.to_day_int() - 1) * MINUTES_PER_DAY
            + hour * MINUTES_PER_HOUR
            + minute
        )


class TimeBound(BaseModel):
    """A range of time in the schedule week during which a distribution is applied."""

    start: Time = Field(description="Start of the time bound.")
    end: Time = Field(description="End of the time bound.")

    @model_validator(mode="after")
    def _validate_bound(self) -> Self:
        start_min, end_min = self.as_minutes()
        if start_min >= end_min:
            raise ValueError("start time must be before end time in TimeBound")
        return self

    @cached_property
    def _mins(self) -> tuple[WeekMinute, WeekMinute]:
        return self.start.minute_of_week(), self.end.minute_of_week()

    def as_minutes(self) -> tuple[WeekMinute, WeekMinute]:
        """Returns the start and end times as minutes since Monday 00:00."""
        return self._mins
