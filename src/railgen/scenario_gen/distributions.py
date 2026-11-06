import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Annotated, Literal, Self

from numpy.random import Generator
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.stats import laplace_asymmetric

from railgen.models import DayOfWeek
from railgen.utils import (
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    MINUTES_PER_WEEK,
    HourMinute,
)


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

    def minute_of_week(self) -> int:
        hour, minute = map(int, self.hour_min.split(":"))
        return (
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
    def _mins(self) -> tuple[int, int]:
        return self.start.minute_of_week(), self.end.minute_of_week()

    def as_minutes(self) -> tuple[int, int]:
        """Returns the start and end times as minutes since Monday 00:00."""
        return self._mins


class Distribution(BaseModel, ABC):
    """Base class of all distributions. Derived classes should specify the `distr_type` field."""

    time_bounds: list[TimeBound] = Field(
        default_factory=list,
        description="Time ranges, each with a 'start' and 'end', during which this distribution is applied. Empty applies at all times.",
    )

    @abstractmethod
    def _sample_impl(self, rng: Generator) -> float:
        """Implementation of sample logic for derived classes. Should not be called directly; use `sample` instead."""

    def sample(self, rng: Generator, minute: float) -> float | None:
        """Returns a single sample drawn with `rng`, or None if `minute` is outside every time bound.

        `minute` counts from Monday 00:00 and wraps weekly.
        """
        if self.time_bounds:
            week_minute = minute % MINUTES_PER_WEEK
            bounds = (b.as_minutes() for b in self.time_bounds)
            if not any(start <= week_minute < end for start, end in bounds):
                return None
        return self._sample_impl(rng)


class NormalDist(Distribution):
    distr_type: Literal["normal"]
    mean: float
    stdev: float

    def _sample_impl(self, rng: Generator) -> float:
        return float(rng.normal(loc=self.mean, scale=self.stdev))


class AsymmetricLaplace(Distribution):
    distr_type: Literal["asymmetric_laplace"]
    location: float
    scale: float
    skew: float

    def _sample_impl(self, rng: Generator) -> float:
        # Convert numpy.float64 to regular float
        return float(
            laplace_asymmetric.rvs(
                loc=self.location, scale=self.scale, kappa=self.skew, random_state=rng
            )
        )


class WeibullDist(Distribution):
    """`location` + `scale` x a standard Weibull draw; never below `location`."""

    distr_type: Literal["weibull"]
    shape: float = Field(gt=0)
    scale: float = Field(gt=0)
    location: float = 0.0

    def _sample_impl(self, rng: Generator) -> float:
        return float(self.location + self.scale * rng.weibull(self.shape))


class LogNormalDist(Distribution):
    """`location` + exp(N(`log_mean`, `log_stdev`)); never below `location`."""

    distr_type: Literal["log_normal"]
    log_mean: float
    log_stdev: float = Field(ge=0)
    location: float = 0.0

    def _sample_impl(self, rng: Generator) -> float:
        return float(
            self.location + rng.lognormal(mean=self.log_mean, sigma=self.log_stdev)
        )


# Every type except the mixture, so a mixture's components cannot be mixtures themselves
type _SingleDistribution = Annotated[
    NormalDist | AsymmetricLaplace | WeibullDist | LogNormalDist,
    Field(discriminator="distr_type"),
]


class MixtureComponent(BaseModel):
    weight: float = Field(
        gt=0, description="Probability that a sample comes from this component."
    )
    negate: bool = Field(
        False,
        description="Use minus the drawn value, e.g. an early side given as positive minutes early.",
    )
    distribution: _SingleDistribution


class Mixture(Distribution):
    """Picks one component with probability `weight`, then draws from it."""

    distr_type: Literal["mixture"]
    components: list[MixtureComponent] = Field(min_length=1)

    def _sample_impl(self, rng: Generator) -> float:
        component = self.components[
            rng.choice(len(self.components), p=[c.weight for c in self.components])
        ]
        # we're ignoring the component's time bounds here so it's fine to call the private method
        value = component.distribution._sample_impl(rng)
        return -value if component.negate else value

    @model_validator(mode="after")
    def _validate(self) -> Self:
        total = sum(c.weight for c in self.components)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"mixture weights must add up to 1, got {total}")

        if any(c.distribution.time_bounds for c in self.components):
            raise ValueError(
                "Mixture components cannot have time bounds; use the mixture's time bounds instead."
            )
        return self


type ScenarioDistribution = Annotated[
    NormalDist | AsymmetricLaplace | WeibullDist | LogNormalDist | Mixture,
    Field(discriminator="distr_type"),
]
