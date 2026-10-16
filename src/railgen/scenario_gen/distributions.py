import math
from typing import Annotated, Literal, Self

from numpy.random import Generator
from pydantic import BaseModel, Field, model_validator
from scipy.stats import laplace_asymmetric


class NormalDist(BaseModel):
    distr_type: Literal["normal"]
    mean: float
    stdev: float


class AsymmetricLaplace(BaseModel):
    distr_type: Literal["asymmetric_laplace"]
    location: float
    scale: float
    skew: float


class WeibullDist(BaseModel):
    """`location` + `scale` x a standard Weibull draw; never below `location`."""

    distr_type: Literal["weibull"]
    shape: float = Field(gt=0)
    scale: float = Field(gt=0)
    location: float = 0.0


class LogNormalDist(BaseModel):
    """`location` + exp(N(`log_mean`, `log_stdev`)); never below `location`."""

    distr_type: Literal["log_normal"]
    log_mean: float
    log_stdev: float = Field(ge=0)
    location: float = 0.0


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


class Mixture(BaseModel):
    """Picks one component with probability `weight`, then draws from it."""

    distr_type: Literal["mixture"]
    components: list[MixtureComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        total = sum(c.weight for c in self.components)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"mixture weights must add up to 1, got {total}")
        return self


type ScenarioDistribution = Annotated[
    NormalDist | AsymmetricLaplace | WeibullDist | LogNormalDist | Mixture,
    Field(discriminator="distr_type"),
]


def sample_distribution(distribution: ScenarioDistribution, rng: Generator) -> float:
    """Returns a single sample from the given distribution, drawn with `rng`."""

    match distribution:
        case NormalDist(mean=m, stdev=s):
            return float(rng.normal(loc=m, scale=s))
        case AsymmetricLaplace(location=l, scale=s, skew=k):
            # Convert numpy.float64 to regular float
            return float(
                laplace_asymmetric.rvs(loc=l, scale=s, kappa=k, random_state=rng)
            )
        case WeibullDist(shape=k, scale=s, location=l):
            return float(l + s * rng.weibull(k))
        case LogNormalDist(log_mean=m, log_stdev=s, location=l):
            return float(l + rng.lognormal(mean=m, sigma=s))
        case Mixture(components=components):
            component = components[
                rng.choice(len(components), p=[c.weight for c in components])
            ]
            value = sample_distribution(component.distribution, rng)
            return -value if component.negate else value
