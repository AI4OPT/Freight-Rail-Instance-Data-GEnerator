from pathlib import Path
from typing import ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from railgen.constants import OUTPUT_DIR_NAME
from railgen.models import Location

CONFIG_FILE_NAME = "schedule_config.yaml"
CONFIGS_DIR = Path("configs")
CONFIG_FILE_PATH = CONFIGS_DIR / CONFIG_FILE_NAME
SCHEMA_DIR = Path("schema")
SCHEDULE_SCHEMA_FILE = "schedule_config.schema.json"

_CONFIG_MODEL = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")


class GeneralConfig(BaseModel):
    """General application settings."""

    train_seed: int | None = Field(
        None,
        description="Seed for the random parts of train generation; null draws a new one each run.",
    )
    data_output_dir: str = Field(
        OUTPUT_DIR_NAME, description="Directory that generated data is written to."
    )
    seg_freq_arc_cost_discount_alpha: float = Field(
        0.2, description="Weight α of the phase-2 discount on busy segments."
    )
    normalized_seg_arc_cost_exponent_lambda: float = Field(
        0.6,
        alias="lambda",
        description="Exponent λ of the phase-2 discount on busy segments.",
    )
    median_f: float = Field(
        12.0,
        description="Reference count of the phase-2 discount; meant to be the median phase-1 path count over the segments used.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


def _default_hub_locations() -> list[Location]:
    return [
        Location(raw_city=city, state=state)
        for city, state in [
            ("Portland", "OR"),
            ("Seattle", "WA"),
            ("Oakland", "CA"),
            ("Los Angeles", "CA"),
            ("Salt Lake City", "UT"),
            ("Kansas City", "MO"),
            ("Fort Worth", "TX"),
            ("Houston", "TX"),
            ("Minneapolis", "MN"),
            ("Chicago", "IL"),
            ("Memphis", "TN"),
            ("New Orleans", "LA"),
            ("San Antonio", "TX"),
            ("Pittsburgh", "PA"),
            ("Secaucus", "NJ"),
            ("Birmingham", "AL"),
            ("Atlanta", "GA"),
            ("Norfolk", "VA"),
            ("Baltimore", "MD"),
            ("Detroit", "MI"),
            ("St. Louis", "MO"),
            ("Omaha", "NE"),
            ("North Platte", "NE"),
            ("Denver", "CO"),
            ("El Paso", "TX"),
            ("Spokane", "WA"),
            ("Jacksonville", "FL"),
            ("Cincinnati", "OH"),
            ("Buffalo", "NY"),
            ("Laredo", "TX"),
        ]
    ]


class NetworkConfig(BaseModel):
    """Settings controlling rail network generation."""

    hub_locations: list[Location] = Field(
        default_factory=_default_hub_locations,
        description="Hub cities that phase-1 routes run between their hub yard groups.",
    )
    hub_radius_km: float = Field(
        50.0,
        gt=0,
        description="A hub city's hub: the largest zone yard group of its state within this many km of the city.",
    )
    ygs_per_zone: int = Field(
        3,
        ge=1,
        description="Most yard groups that represent one FAF zone.",
    )
    min_od_yg_spacing_km: float = Field(
        5.0,
        ge=0,
        description="Least track distance (km) between two yard groups of the same zone.",
    )
    min_od_yg_track_km: float = Field(
        2.0,
        ge=0,
        description="Least track length (km) of a zone's yard group; waived for a zone with no candidate that large.",
    )
    excluded_track_net_codes: list[str] = Field(
        default_factory=lambda: ["A", "R", "X", "T", "F"],
        description="NTAD `NET` codes of lines the network walk skips: A abandoned, R removed, X out of service, T trail, F ferry. Applied when `graph.gexf` is built.",
    )
    max_connection_km: float = Field(
        800.0,
        gt=0,
        description="Longest yard-to-yard connection kept on load; longer ones are mostly routes through Canada or Mexico, which have no yards.",
    )
    od_discount_factor: float = Field(
        0.7,
        description="Phase-1 cost factor on connections at a zone yard group (0.7 = 30% cheaper).",
    )
    per_grade_speed_reduction: float = Field(
        0.1, description="Share of speed lost per 1% of uphill grade."
    )
    max_speed_reduction: float = Field(
        0.50,
        description="Largest share of speed lost on a grade.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


class TrainConfig(BaseModel):
    """Settings controlling train generation."""

    speed_km_h: int = Field(
        40,
        description="Speed (km/h) for the routing costs and the minutes of `distance_matrix.csv`; it does not change the routes.",
    )
    speed_km_h_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 40.0, "M": 36.0, "B": 38.0, "I": 50.0},
        description="Average speed (km/h) of a train of each category between yards, delays on the line included; sets the schedule's times and crew changes.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


# For the `service_design_settings` blocks.
_MODEL = ConfigDict(frozen=True, extra="forbid")

# Constants of the grade formula in `LocomotivePower`, not settings. Pull of a 1% grade: 1% of a 2,000-lb ton.
GRADE_RESISTANCE_LB_PER_TON_PER_PERCENT = 20.0
# 1 hp = 375 lb of pull at 1 mph (33,000 ft-lb per minute).
LB_MPH_PER_HP = 375.0

# Freight group of every FAF "multiple modes" row, whatever its commodity code.
INTERMODAL_GROUP = "intermodal"

# With `shared_bulk`, the freight groups of this category ride one train type of this name.
BULK_CATEGORY = "B"
SHARED_BULK_TYPE = "bulk"


# SCTG2 code -> freight group; unlisted codes are manifest.
_DEFAULT_FREIGHT_GROUPS: dict[str, list[str]] = {
    "coal": ["15"],
    "grain": ["02"],
    "ores_minerals": ["13", "14"],
    "aggregates": ["10", "11", "12"],
    "petroleum_energy": ["16", "17", "18", "19"],
    "automotive": ["36", "37"],
}

# Freight group / train type -> output category (symbol prefix).
_DEFAULT_TRAIN_TYPE_CATEGORY: dict[str, str] = {
    "coal": "B",
    "grain": "B",
    "ores_minerals": "B",
    "aggregates": "B",
    "petroleum_energy": "B",
    "manifest": "M",
    "automotive": "A",
    INTERMODAL_GROUP: "I",
}


class ExcludedFlow(BaseModel):
    """FAF rows removed from demand: origin zone, destination zone, SCTG codes, and why."""

    origin_zone: str
    destination_zone: str
    sctg: list[str]
    reason: str = ""

    model_config: ClassVar[ConfigDict] = _MODEL


class YardPairRule(BaseModel):
    """Which of a zone pair's candidate yard pairs carry its tons."""

    assignment: Literal["split_by_tonnage", "one_per_category"] = Field(
        "one_per_category",
        description="split_by_tonnage: n = min(max_pairs, ceil(tons / tonnage_step)) yard pairs, each category split over them by yard size. one_per_category: one yard pair per category, the largest pair (origin x destination track) to the heaviest category.",
    )
    tonnage_step: float = Field(
        11_000.0,
        description="split_by_tonnage only: tons/week per extra yard pair.",
    )
    distance_tolerance: float = Field(
        0.15,
        description="Between two zones, yard pairs whose route cost is over (1 + this) x the cheapest are left out.",
    )
    max_pairs: int = Field(3, ge=1, description="split_by_tonnage only.")
    min_route_km: float = Field(
        40.0,
        ge=0,
        description="Local-train distance (km). Inter-zone: yard pairs with a shorter route are not candidates; a zone pair with none left is dropped. Intra-zone: tons whose assigned yard pair is shorter are dropped.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


class Costs(BaseModel):
    """Objective weights."""

    train_start: float = Field(400.0, description="Cost of one run of a train.")
    per_train_mile: float = Field(
        10.0, description="Cost per mile of one run, over its whole route."
    )
    per_distinct_train: float = Field(
        0.0,
        description="Weekly charge for each service (yard pair and train type) that runs at all. 0 leaves out its binary variables.",
    )
    transfer_fixed: float = Field(
        0.0,
        description="Cost of each change of train of a commodity at a yard.",
    )
    transfer_per_ton: float = Field(
        0.0,
        description="Added per ton/week of the commodity that changes trains. Both 0 leaves out the transfer variables.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


class CrewRule(BaseModel):
    """Crew changes placed after the model along each selected train."""

    duty_hours: float = Field(12.0, description="On-duty limit of a crew, in hours.")
    terminal_hours: float = Field(
        0.0,
        description="On-duty time not spent running; usable hours = duty_hours - terminal_hours.",
    )
    merge_with_work_stop_minutes: float = Field(
        15.0,
        ge=0,
        description="A crew change that falls at most this many minutes' run after a work stop is moved back to that stop.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL

    @property
    def usable_hours(self) -> float:
        return self.duty_hours - self.terminal_hours


class DwellMinutes(BaseModel):
    """Dwell at an intermediate stop, in minutes, by the events that happen there. A stop
    with no event has no dwell."""

    crew_only: float = Field(20.0, description="Crew change only.")
    so: float = Field(30.0, description="Set-out only.")
    pu: float = Field(60.0, description="Pick-up only.")
    pu_so: float = Field(90.0, description="Pick-up and set-out.")
    so_crew: float = Field(70.0, description="Set-out and crew change.")
    pu_crew: float = Field(80.0, description="Pick-up and crew change.")
    pu_so_crew: float = Field(140.0, description="Pick-up, set-out and crew change.")

    model_config: ClassVar[ConfigDict] = _MODEL

    def minutes(self, pick_up: bool, set_out: bool, crew_change: bool) -> float:
        """Dwell for a stop's combination of events (0 if nothing happens)."""
        table = {
            (False, False, True): self.crew_only,
            (False, True, False): self.so,
            (True, False, False): self.pu,
            (True, True, False): self.pu_so,
            (False, True, True): self.so_crew,
            (True, False, True): self.pu_crew,
            (True, True, True): self.pu_so_crew,
        }
        return table.get((pick_up, set_out, crew_change), 0.0)


class CarWeight(BaseModel):
    """Car weights by category. Trailing tons = freight x (1 + empty car tons / cargo tons
    per car); they give the locomotive count and a run's freight limit. Schedule tonnage is
    freight, except on a circuitous train's way back (its empty cars)."""

    empty_car_tons_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 53.0, "M": 33.0, "B": 29.0, "I": 31.0},
        description="Tons of one empty car. 0 = freight only.",
    )
    cargo_tons_per_car_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 18.0, "M": 82.5, "B": 100.0, "I": 50.0},
        description="Tons of freight in one loaded car.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL

    @model_validator(mode="after")
    def _check_tons(self) -> Self:
        if low := {c: t for c, t in self.empty_car_tons_by_category.items() if t < 0}:
            raise ValueError(f"empty_car_tons_by_category must be >= 0, got {low}")
        if low := {
            c: t for c, t in self.cargo_tons_per_car_by_category.items() if t <= 0
        }:
            raise ValueError(f"cargo_tons_per_car_by_category must be > 0, got {low}")
        return self

    def trailing_tons(self, category: str, freight_tons: float) -> float:
        """Tons behind the locomotives: the freight plus the cars that carry it.

        The freight fills freight / cargo-per-car cars, and each adds its empty weight.
        """
        cargo = self.cargo_tons_per_car_by_category[category]
        return freight_tons * (1.0 + self.empty_car_tons_by_category[category] / cargo)


class LocomotivePower(BaseModel):
    """Locomotives on a leg = hp per ton x trailing tons / `locomotive_hp`, rounded up.
    hp per ton = max(category value, (`rolling_resistance_lb_per_ton` + 20 x ruling grade %)
    x `min_speed_mph` / (375 x `transmission_efficiency`))."""

    locomotive_hp: float = Field(
        4400.0, gt=0, description="Horsepower of one locomotive."
    )
    hp_per_ton_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 1.4, "M": 1.5, "B": 0.78, "I": 1.9},
        description="hp per trailing ton by category.",
    )
    rolling_resistance_lb_per_ton: float = Field(
        2.0,
        ge=0,
        description="Level-track resistance at low speed, lb per ton.",
    )
    min_speed_mph: float = Field(
        10.0, gt=0, description="Slowest speed accepted on a leg's ruling grade."
    )
    transmission_efficiency: float = Field(
        0.82,
        gt=0,
        le=1,
        description="Share of a locomotive's horsepower that reaches the rails.",
    )
    min_grade_hop_km: float = Field(
        5.0,
        ge=0,
        description="Hops shorter than this (km) have no grade: elevations of nearby yards differ without a real climb.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL

    def grade_hp_per_ton(self, grade_percent: float) -> float:
        """Horsepower per ton that holds `min_speed_mph` on an ascending grade."""
        pull_lb_per_ton = (
            self.rolling_resistance_lb_per_ton
            + GRADE_RESISTANCE_LB_PER_TON_PER_PERCENT * grade_percent
        )
        return (
            pull_lb_per_ton
            * self.min_speed_mph
            / (LB_MPH_PER_HP * self.transmission_efficiency)
        )

    def hp_per_ton(self, category: str, grade_percent: float) -> float:
        """The larger of the category's value and what the ruling grade needs."""
        return max(
            self.hp_per_ton_by_category[category],
            self.grade_hp_per_ton(grade_percent),
        )


class SolverSettings(BaseModel):
    """Gurobi settings, for each category's model."""

    time_limit_s: float = Field(
        600.0,
        description="Time limit of each category's solve (s); the best solution found is used.",
    )
    mip_gap: float = Field(
        0.02,
        description="Relative gap at which a category's solve stops.",
    )
    log_file: str | None = Field(
        "gurobi_service_design.log",
        description="null: console only. Otherwise each category logs to gurobi_<category>.log in the run's folder under gurobi_solutions/ (this name is not used).",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


_DEFAULT_WEST_STATES = [
    # West bank of the Mississippi, Wisconsin (no CSX or NS yards), and everything west.
    "MN", "IA", "MO", "AR", "LA", "WI",
    "ND", "SD", "NE", "KS", "OK", "TX",
    "MT", "WY", "CO", "NM", "ID", "UT", "AZ", "WA", "OR", "NV", "CA", "AK", "HI",
]  # fmt: skip
# FAF zones of the gateway cities where western and eastern railroads meet.
_DEFAULT_GATEWAY_ZONES = ["171", "181", "172", "292", "201", "291", "471", "223"]


class RegionSettings(BaseModel):
    """Which part of the network an instance covers (`regions.py`)."""

    instance: Literal["full", "east", "west"] = Field(
        "full",
        description="full: the whole network. east / west: flows within the side, or between a gateway and the side, are kept whole; gateway-to-gateway flows go to the side with most of their route; crossing flows are cut where they reach the other side (`regions.py`).",
    )
    west_states: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_WEST_STATES),
        description="States of the West (west bank of the Mississippi, Wisconsin, and everything west); the rest is the East.",
    )
    gateway_zones: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_GATEWAY_ZONES),
        description="FAF zones of the gateway cities, in both regions: Chicago (171, 181), St. Louis (172, 292), Kansas City (201, 291), Memphis (471), New Orleans (223).",
    )
    gateway_radius_km: float = Field(
        25.0,
        gt=0,
        description="Yard groups within this many km of a gateway zone's yard group are in both regions.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


class CircuitousTrains(BaseModel):
    """Trains sent straight back to their origin as the same symbol, with their empty cars and
    the locomotives of their heaviest leg, to reduce the locomotive imbalance. Chosen one at a
    time, the most helpful first, within each category's share; a return that does not help,
    or does not fit in the week, is not added."""

    short_empty_returns: dict[str, float] = Field(
        default_factory=dict,
        description="Per category, e.g. {M: 0.5}: share of its short trains (at most `short_max_miles` one way) that may come back; unlisted get none.",
    )
    long_empty_returns: dict[str, float] = Field(
        default_factory=dict,
        description="Per category: share of its long trains that may come back; unlisted get none.",
    )
    short_max_miles: float = Field(
        225.0,
        gt=0,
        description="One-way miles up to which a train is short.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL

    @model_validator(mode="after")
    def _check_shares(self) -> Self:
        for name in ("short_empty_returns", "long_empty_returns"):
            if bad := {c: v for c, v in getattr(self, name).items() if not 0 <= v <= 1}:
                raise ValueError(
                    f"circuitous_trains.{name} must be between 0 and 1, got {bad}"
                )
        return self


class ServiceDesignSettings(BaseModel):
    """All knobs of the service-design generator."""

    multimodal_rail_share: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Share of a FAF 'multiple modes' row's tons that moves by rail (the intermodal group).",
    )
    multimodal_min_miles: float = Field(
        500.0,
        ge=0.0,
        description="Multiple-modes rows count only on zone pairs averaging at least this many miles (FAF5); pairs with no FAF5 distance are dropped. 0 counts all.",
    )
    multimodal_max_miles: float = Field(
        2500.0,
        ge=0.0,
        description="Multiple-modes rows do not count on zone pairs averaging more than this many miles (coast to coast is not one train). 0 = no limit.",
    )
    region: RegionSettings = Field(default_factory=RegionSettings)
    excluded_zones: list[str] = Field(
        default_factory=lambda: ["020"],
        description="FAF zones whose rows are dropped, as origin or destination. 020 Alaska: its track is not connected to the rest.",
    )
    freight_groups: dict[str, list[str]] = Field(
        default_factory=lambda: dict(_DEFAULT_FREIGHT_GROUPS),
        description="SCTG2 codes of each freight group (rail rows); unlisted codes are manifest, multiple-modes rows intermodal.",
    )
    shared_bulk: bool = Field(
        True,
        description="true: the B groups of a zone pair are one flow and ride `bulk` trains; false: each group is its own flow, on its own trains (coal on coal trains, ...), and must reach `min_od_tons_per_week` alone.",
    )
    train_type_category: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_TRAIN_TYPE_CATEGORY),
        description="Output category (A/M/B/I, the symbol prefix) of each freight group and its train type. Each group rides only its own train type; each category is solved on its own.",
    )
    excluded_flows: list[ExcludedFlow] = Field(
        default_factory=list,
        description="FAF flows left out of demand (origin zone, destination zone, SCTG codes), with a reason for the report.",
    )
    intra_zone_min_miles: float = Field(
        25.0,
        description="Intra-zone FAF rows averaging fewer miles (FAF5) are local switching and dropped; rows with no FAF5 distance are kept.",
    )
    min_od_tons_per_week: dict[str, float] = Field(
        default_factory=lambda: {"A": 3000.0, "M": 15000.0, "B": 20000.0, "I": 8000.0},
        description="Per category: a train type's tons/week on a zone pair below this are dropped, before the yard-pair split (without `shared_bulk`, each B group on its own); unlisted categories keep all.",
    )
    yard_pairs: YardPairRule = Field(default_factory=YardPairRule)
    work_yards_per_state: int = Field(
        3,
        ge=0,
        description="Besides zone yards, the N largest yard groups per state may host work events.",
    )
    max_trailing_tons_per_run_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 6500.0, "M": 13000.0, "B": 17000.0, "I": 12000.0},
        description="Max trailing tons (freight + cars) of a run on any leg, by category; the freight limit is this / (1 + empty car tons / cargo tons per car).",
    )
    max_runs_per_week: int = Field(
        7,
        description="Most runs a week of one symbol (one a day); a service's runs are split into symbols of this many.",
    )
    max_symbols_per_service: int = Field(
        3,
        description="Most parallel symbols of one service; its runs a week are at most max_runs_per_week x max_symbols_per_service, or more if its own freight needs more.",
    )
    costs: Costs = Field(default_factory=Costs)
    crew: CrewRule = Field(default_factory=CrewRule)
    dwell_minutes: DwellMinutes = Field(default_factory=DwellMinutes)
    car_weight: CarWeight = Field(default_factory=CarWeight)
    locomotive_power: LocomotivePower = Field(default_factory=LocomotivePower)
    circuitous_trains: CircuitousTrains = Field(default_factory=CircuitousTrains)
    solver: SolverSettings = Field(default_factory=SolverSettings)

    model_config: ClassVar[ConfigDict] = _MODEL

    @model_validator(mode="after")
    def _check_hp_per_ton_categories(self) -> Self:
        listed = self.locomotive_power.hp_per_ton_by_category
        if missing := set(self.train_type_category.values()) - set(listed):
            raise ValueError(
                f"locomotive_power.hp_per_ton_by_category is missing categories {sorted(missing)}"
            )
        return self

    @model_validator(mode="after")
    def _check_circuitous_train_categories(self) -> Self:
        categories = set(self.train_type_category.values())
        for name in ("short_empty_returns", "long_empty_returns"):
            if unknown := set(getattr(self.circuitous_trains, name)) - categories:
                raise ValueError(
                    f"circuitous_trains.{name} has categories {sorted(unknown)} that no train type "
                    f"has (train_type_category: {sorted(categories)})"
                )
        return self

    @model_validator(mode="after")
    def _check_trailing_ton_categories(self) -> Self:
        categories = set(self.train_type_category.values())
        listed = self.max_trailing_tons_per_run_by_category
        if missing := categories - set(listed):
            raise ValueError(
                f"max_trailing_tons_per_run_by_category is missing categories {sorted(missing)}"
            )
        if low := {c: v for c, v in listed.items() if v <= 0}:
            raise ValueError(
                f"max_trailing_tons_per_run_by_category must be > 0, got {low}"
            )
        return self

    @model_validator(mode="after")
    def _check_car_weight_categories(self) -> Self:
        categories = set(self.train_type_category.values())
        for name in ("empty_car_tons_by_category", "cargo_tons_per_car_by_category"):
            if missing := categories - set(getattr(self.car_weight, name)):
                raise ValueError(
                    f"car_weight.{name} is missing categories {sorted(missing)}"
                )
        return self

    def train_type_of(self, group: str) -> str:
        """The train type a freight group rides: its own, or the shared `bulk` type for the
        bulk groups with `shared_bulk`. Freight never rides a train of another type."""
        if self.shared_bulk and self.train_type_category.get(group) == BULK_CATEGORY:
            return SHARED_BULK_TYPE
        return group

    def category_of(self, train_type: str) -> str:
        """Output category of a train type."""
        if train_type == SHARED_BULK_TYPE:
            return BULK_CATEGORY
        return self.train_type_category[train_type]

    def capacity_of(self, train_type: str) -> float:
        """Max tons of freight per run of a train type: the freight whose trailing tons
        (freight plus its cars) equal the category's limit."""
        category = self.category_of(train_type)
        trailing = self.max_trailing_tons_per_run_by_category[category]
        return trailing / self.car_weight.trailing_tons(category, 1.0)

    def group_of(self, sctg: str) -> str:
        """Freight group of an sctg2 code; unlisted codes are manifest freight."""
        for group, codes in self.freight_groups.items():
            if sctg in codes:
                return group
        return "manifest"


class Config(BaseModel):
    """Root of `schedule_config.yaml`."""

    general_settings: GeneralConfig = Field(default_factory=GeneralConfig)
    network_settings: NetworkConfig = Field(default_factory=NetworkConfig)
    train_settings: TrainConfig = Field(default_factory=TrainConfig)
    service_design_settings: ServiceDesignSettings = Field(
        default_factory=ServiceDesignSettings,
        description="Settings of the train generator (service network design).",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL

    @model_validator(mode="after")
    def _check_speed_categories(self) -> Self:
        categories = set(self.service_design_settings.train_type_category.values())
        listed = self.train_settings.speed_km_h_by_category
        if missing := categories - set(listed):
            raise ValueError(
                f"train_settings.speed_km_h_by_category is missing categories {sorted(missing)}"
            )
        if low := {c: v for c, v in listed.items() if v <= 0}:
            raise ValueError(
                f"train_settings.speed_km_h_by_category must be > 0, got {low}"
            )
        return self

    def to_yaml(self) -> str:
        """Serialize to YAML: the settings a run used, saved next to its output."""
        return yaml.safe_dump(
            self.model_dump(mode="json", by_alias=True), sort_keys=False, width=100
        )


def load_config(path: Path | None = None) -> Config:
    """Load config from `path`, else `CONFIG_FILE_PATH`. Raises if file not found."""
    path = path or CONFIG_FILE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return Config.model_validate(yaml.safe_load(path.read_text()) or {})
