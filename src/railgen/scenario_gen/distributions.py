import math
from abc import ABC, abstractmethod
from typing import Annotated, Literal, Self

from numpy.random import Generator
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from scipy.stats import laplace_asymmetric

from railgen.scenario_gen.time import TimeBound
from railgen.utils import (
    MINUTES_PER_WEEK,
)


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
