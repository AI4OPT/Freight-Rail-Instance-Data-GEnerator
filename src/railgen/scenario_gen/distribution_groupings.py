from dataclasses import dataclass

from railgen.models import YdKey
from railgen.scenario_gen.distributions import ScenarioDistribution


@dataclass(frozen=True, kw_only=True, slots=True)
class TrainDistributions:
    """Distributions used for various parts of scenario generation related to a specific train category."""

    initial_departure: ScenarioDistribution
    trip_time_percent_diff: ScenarioDistribution
    dwell_time: ScenarioDistribution


@dataclass(frozen=True, kw_only=True, slots=True)
class YardDistributions:
    """Distributions used for various parts of scenario generation related to a specific yard."""

    # Trip percentage for a specific one-way Orig->Dest rail leg
    leg_distributions: dict[tuple[YdKey, YdKey], ScenarioDistribution]

    # Dwell delay for a specific yard
    yd_dwell_distributions: dict[YdKey, ScenarioDistribution]
