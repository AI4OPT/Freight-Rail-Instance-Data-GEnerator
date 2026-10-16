from dataclasses import dataclass

from railgen.models import YgKey
from railgen.scenario_gen.distributions import ScenarioDistribution


@dataclass(frozen=True, kw_only=True, slots=True)
class TrainDistributions:
    """Distributions used for various parts of scenario generation related to a specific train category."""

    initial_departure: ScenarioDistribution
    trip_time_percent_diff: ScenarioDistribution
    dwell_time: ScenarioDistribution


@dataclass(frozen=True, kw_only=True, slots=True)
class YardGroupDistributions:
    """Distributions used for various parts of scenario generation related to a specific yard group."""

    # Trip percentage for a specific one-way Orig->Dest rail leg
    leg_distributions: dict[tuple[YgKey, YgKey], ScenarioDistribution]

    # Dwell delay for a specific yard group
    yg_dwell_distributions: dict[YgKey, ScenarioDistribution]
