from pathlib import Path
from typing import ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from railgen.constants import OUTPUT_DIR_NAME
from railgen.models import Location

CONFIG_FILE_NAME = "schedule_config.yaml"
CONFIGS_DIR = Path("configs")
CONFIG_FILE_PATH = CONFIGS_DIR / CONFIG_FILE_NAME
SCHEMA_PATH = Path("schema") / "schedule_config.schema.json"

_CONFIG_MODEL = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")


class GeneralConfig(BaseModel):
    """General application settings."""

    train_seed: int | None = Field(
        None,
        description="Value used to seed random train generation. Null means a fresh random seed on every run.",
    )
    data_output_dir: str = Field(
        OUTPUT_DIR_NAME, description="Directory that generated data is written to."
    )
    seg_freq_arc_cost_discount_alpha: float = Field(
        0.2, description="Weight of the segment-frequency discount on arc cost."
    )
    normalized_seg_arc_cost_exponent_lambda: float = Field(
        0.6,
        alias="lambda",
        description="Exponent applied to the normalized segment frequency in the arc cost discount.",
    )
    median_f: float = Field(
        72.0,
        description="Reference median segment frequency used to normalize the discount.",
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
        ]
    ]


class NetworkConfig(BaseModel):
    """Settings controlling rail network generation."""

    hub_locations: list[Location] = Field(
        default_factory=_default_hub_locations,
        description="Locations for hub terminal yard groups.",
    )
    ygs_per_zone: int = Field(
        3,
        ge=1,
        description="How many yard groups represent each FAF zone. A zone's OD tonnage and its net tonnage delta are split across them, so 1 reproduces the single-yard-per-zone behaviour.",
    )
    max_od_fallback_km: float = Field(
        50.0,
        gt=0,
        description="Fallback radius for FAF zones that match no yard group by city/state: such a zone may only be represented by yard groups within this many km of its geocoded city. A zone with nothing in range is left unrepresented rather than mapped to a distant yard group.",
    )
    min_od_yg_spacing_km: float = Field(
        5.0,
        ge=0,
        description="Minimum track distance (km) between two yard groups that represent the same FAF zone: a candidate closer than this to a yard group already chosen for the zone is skipped.",
    )
    min_od_yg_track_km: float = Field(
        2.0,
        ge=0,
        description="Minimum track length (km) of a yard group chosen by distance for a FAF zone that matches no yard group by city/state. Waived for a zone when no yard group within `max_od_fallback_km` has that much track.",
    )
    excluded_track_net_codes: list[str] = Field(
        default_factory=lambda: ["A", "R", "X", "T", "F"],
        description="`NET` codes of the NTAD rail network lines that the walk between yard groups does not use: A abandoned, R abandoned and physically removed, X out of service, T trail on a former rail right-of-way, F rail ferry connection. Applied when `graph.gexf` is generated.",
    )
    od_discount_factor: float = Field(
        0.7,
        description="OD-based routing discount: arc cost multiplier for edges adjacent to OD yard groups in Phase 1. 0.7 means 30% cheaper to route through OD yard groups.",
    )
    per_grade_speed_reduction: float = Field(
        0.1, description="Fractional speed reduction per 1% grade."
    )
    max_speed_reduction: float = Field(
        0.50,
        description="Speed floor: never penalize below this fraction of full speed.",
    )

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


class TrainConfig(BaseModel):
    """Settings controlling train generation."""

    speed_km_h: int = Field(100, description="Train speed in km/h.")

    model_config: ClassVar[ConfigDict] = _CONFIG_MODEL


# Settings for train generation (`service_design_settings` in `config.yaml`).
_MODEL = ConfigDict(frozen=True, extra="forbid")

# Physics of the grade requirement in `LocomotivePower`; neither is a setting.
# Grade resistance: 1% of a 2,000-lb ton acts along a 1% grade.
GRADE_RESISTANCE_LB_PER_TON_PER_PERCENT = 20.0
# 1 hp = 33,000 ft-lb per minute = 375 lb of pull at 1 mph (33,000 x 60 / 5,280).
LB_MPH_PER_HP = 375.0

# Freight group of every FAF "multiple modes" row, whatever its commodity code.
INTERMODAL_GROUP = "intermodal"


# FAF sctg2 code -> freight group of a rail row. Codes not listed fall into `manifest`.
_DEFAULT_FREIGHT_GROUPS: dict[str, list[str]] = {
    "coal": ["15"],
    "grain": ["02"],
    "ores_minerals": ["13", "14"],
    "aggregates": ["10", "11", "12"],
    "petroleum_energy": ["16", "17", "18", "19"],
    "automotive": ["36"],
}

# Freight group -> train types it may ride (compatibility table). Bulk groups ride their
# own bulk train type or a manifest train; vehicles ride only automotive trains.
_DEFAULT_GROUP_TRAIN_TYPES: dict[str, list[str]] = {
    "coal": ["coal", "manifest"],
    "grain": ["grain", "manifest"],
    "ores_minerals": ["ores_minerals", "manifest"],
    "aggregates": ["aggregates", "manifest"],
    "petroleum_energy": ["petroleum_energy", "manifest"],
    "automotive": ["automotive"],
    "manifest": ["manifest"],
    INTERMODAL_GROUP: [INTERMODAL_GROUP],
}

# Train type -> category label written to the output (symbol prefix).
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
    """FAF rows removed from demand."""

    origin_zone: str
    destination_zone: str
    sctg: list[str]
    reason: str

    model_config: ClassVar[ConfigDict] = _MODEL


def _default_excluded_flows() -> list[ExcludedFlow]:
    return [
        ExcludedFlow(
            origin_zone="261",
            destination_zone="261",
            sctg=["11", "12"],
            reason=(
                "Detroit intra-zone sand and gravel: 1.08M t/week, 28% of FAF's national "
                "rail aggregates, about half of AAR's national total."
            ),
        )
    ]


class YardPairRule(BaseModel):
    """Which of a zone pair's candidate yard pairs carry its tons."""

    assignment: Literal["split_by_tonnage", "one_per_category"] = Field(
        "split_by_tonnage",
        description="split_by_tonnage: n = min(max_pairs, ceil(tons / tonnage_step)) yard pairs, every category split over them by yard size. one_per_category: each category of the zone pair rides one yard pair; the largest yard pair (origin track length x destination track length) goes to the category with the most tons, the next largest to the next category, and so on.",
    )
    tonnage_step: float = Field(
        11_000.0,
        description="split_by_tonnage only. Tons/week per extra yard pair: n = min(max_pairs, ceil(tons / tonnage_step)).",
    )
    distance_tolerance: float = Field(
        0.15,
        description="Inter-zone: keep yard pairs whose route cost is within (1 + tolerance) x the cheapest.",
    )
    max_pairs: int = Field(3, ge=1, description="split_by_tonnage only.")

    model_config: ClassVar[ConfigDict] = _MODEL


class Costs(BaseModel):
    """Objective weights. Defaults follow the INFORMS RAS 2011 train design competition."""

    train_start: float = 400.0
    per_train_mile: float = 10.0
    per_distinct_train: float = Field(
        0.0,
        description="Charged once a week for each candidate train that runs at all, whatever its number of runs (binary w per candidate). At 0 the model has no w variables.",
    )
    transfer_fixed: float = Field(
        0.0,
        description="Cost each time a commodity changes trains at a yard (binary z per commodity and yard).",
    )
    transfer_per_ton: float = Field(
        0.0,
        description="Added to `transfer_fixed`, per ton/week of the commodity that changes trains. With both at 0 the model has no transfer variables.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


class CrewRule(BaseModel):
    """Crew changes placed after the model along each selected train."""

    duty_hours: float = Field(12.0, description="On-duty limit, 49 U.S.C. 21103.")
    terminal_hours: float = Field(
        0.0,
        description="On-duty time not spent running; usable hours = duty_hours - terminal_hours.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL

    @property
    def usable_hours(self) -> float:
        return self.duty_hours - self.terminal_hours


class DwellMinutes(BaseModel):
    """Dwell at an intermediate stop, in minutes, by the events that happen there."""

    crew_only: float = 20.0
    so: float = 30.0
    pu: float = 60.0
    pu_so: float = 90.0
    so_crew: float = 70.0
    pu_crew: float = 80.0
    pu_so_crew: float = 140.0

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
    """Weight of the cars that carry a train's freight, by output category (A/M/B/I).

    Demand and the schedule's tonnage are in tons of freight; locomotives pull the
    freight and its cars, so the locomotive count uses `trailing_tons`.
    """

    empty_car_tons_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 53.0, "M": 33.0, "B": 29.0, "I": 31.0},
        description="Weight of one empty car, in tons. A, M, I: medians of the auto, mixed and double-stack trains in FRA (2009), Comparative Evaluation of Rail and Truck Fuel Efficiency on Competitive Corridors, Exhibit A-1 (I: one well of a double-stack car). B: the ore car of the BNSF trains in Lustbader et al. (2025), ALTRIOS final report, Table 13. At 0 a category's tonnage is its freight alone.",
    )
    cargo_tons_per_car_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 18.0, "M": 82.5, "B": 100.0, "I": 50.0},
        description="Freight in one loaded car, in tons. Same sources as `empty_car_tons_by_category` (I: two stacked containers).",
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
    """Locomotives a train leg needs, from horsepower per trailing ton (hp/ton)."""

    locomotive_hp: float = Field(
        4400.0, gt=0, description="Horsepower of one locomotive."
    )
    hp_per_ton_by_category: dict[str, float] = Field(
        default_factory=lambda: {"A": 1.4, "M": 1.5, "B": 0.78, "I": 1.9},
        description="Horsepower per trailing ton a train of each output category (A/M/B/I) runs with on every leg. A, I: medians of the auto and double-stack trains in FRA (2009), Comparative Evaluation of Rail and Truck Fuel Efficiency on Competitive Corridors, Exhibit A-1. M: the value calibrated to BNSF manifest trains in Lustbader et al. (2025), ALTRIOS final report, Table 15; the eight mixed trains of Exhibit A-1 have a median of 1.35 and a mean of 1.56. B: Dingler, Lai and Barkan (2014), Table 2.",
    )
    rolling_resistance_lb_per_ton: float = Field(
        2.0,
        ge=0,
        description="Level-track train resistance at low speed, in lb of pull per ton.",
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
        description="A hop shorter than this many km has no grade: yards a few hundred metres apart can differ by tens of metres in recorded elevation, which is not a grade a train climbs.",
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
    time_limit_s: float = 1800.0
    mip_gap: float = 0.02
    log_file: str | None = Field(
        "gurobi_service_design.log",
        description="null for console only. Otherwise each category logs to gurobi_<category>.log in the run's folder under gurobi_solutions/; the name set here is not used.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


_DEFAULT_WEST_STATES = [
    # The five states on the west bank of the Mississippi, Wisconsin (no CSX or NS yards),
    # then every state west of them.
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
        description="full: the whole network. east / west: flows with both ends on the side, or a gateway end and the other end on the side, are kept whole; gateway-to-gateway flows go to the side with most of their route; flows between the sides are cut where their route reaches the other side, at the near edge of a gateway area when it crosses through one (`regions.py`).",
    )
    west_states: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_WEST_STATES),
        description="States of the West: the five on the west bank of the Mississippi (MN, IA, MO, AR, LA), Wisconsin, and every state west of them. Every other state is the East.",
    )
    gateway_zones: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_GATEWAY_ZONES),
        description="FAF zones of the gateway cities, whose yard groups belong to both regions: Chicago (171 IL, 181 IN), St. Louis (172 IL, 292 MO), Kansas City (201 KS, 291 MO), Memphis (471), New Orleans (223).",
    )
    gateway_radius_km: float = Field(
        25.0,
        gt=0,
        description="A yard group within this many km of a yard group representing a gateway zone belongs to both regions.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL


class ReturnTrips(BaseModel):
    """Levers that reduce the locomotive imbalance between yard groups, applied after the
    trains of every category are built. Each adds a return trip to a train: it continues as
    the same symbol straight back to its origin, carrying its empty cars, with the
    locomotives of its heaviest leg. A train whose round trip does not fit in a week is
    left as it is."""

    circuitous_share: dict[str, float] = Field(
        default_factory=dict,
        description="Per category (A/M/B/I): share of its trains of at most `circuitous_max_miles` (one way) that get a return trip, chosen at random with the train seed. Categories not listed get none.",
    )
    circuitous_max_miles: float = Field(
        225.0,
        gt=0,
        description="One-way route length, in miles, up to which a train counts as short for `circuitous_share`.",
    )
    empty_return_share: dict[str, float] = Field(
        default_factory=dict,
        description="Per category: at most this share of its trains get a return trip that brings their empty cars back, chosen one by one where it reduces the locomotives to move back the most; a return that would not reduce them is not added. Categories not listed get none.",
    )

    model_config: ClassVar[ConfigDict] = _MODEL

    @model_validator(mode="after")
    def _check_shares(self) -> Self:
        for name in ("circuitous_share", "empty_return_share"):
            if bad := {c: v for c, v in getattr(self, name).items() if not 0 <= v <= 1}:
                raise ValueError(
                    f"return_trips.{name} must be between 0 and 1, got {bad}"
                )
        return self


class ServiceDesignSettings(BaseModel):
    """All knobs of the service-design generator."""

    faf_year: str = Field(
        "2022",
        description="Year of the FAF tons column (FAF6 has 2022 only). The same year's FAF5 ton-miles give the intra-zone distances.",
    )
    multimodal_rail_share: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Share of the tons of a FAF 'multiple modes' row that moves by rail; those tons are the intermodal freight group.",
    )
    multimodal_min_miles: float = Field(
        500.0,
        ge=0.0,
        description="A FAF 'multiple modes' row counts only when the multiple-modes freight of its zone pair travels at least this many miles on average (FAF5 ton-miles / tons of the zone pair, same year): rail carries such freight on long hauls only. Rows of zone pairs FAF5 has no distance for are dropped too. 0 counts every row.",
    )
    multimodal_max_miles: float = Field(
        2500.0,
        ge=0.0,
        description="A FAF 'multiple modes' row does not count when the multiple-modes freight of its zone pair travels more than this many miles on average (FAF5, as for `multimodal_min_miles`): a move from coast to coast changes railroad on the way and is not one train. 0 is no upper limit.",
    )
    region: RegionSettings = Field(default_factory=RegionSettings)
    excluded_zones: list[str] = Field(
        default_factory=lambda: ["020"],
        description="FAF zones whose rows are dropped, whether the zone is the origin or the destination. 020 is Alaska: its track is not connected to the rest of the network, so no distance exists between its yard groups and the others.",
    )
    freight_groups: dict[str, list[str]] = Field(
        default_factory=lambda: dict(_DEFAULT_FREIGHT_GROUPS)
    )
    group_train_types: dict[str, list[str]] = Field(
        default_factory=lambda: dict(_DEFAULT_GROUP_TRAIN_TYPES)
    )
    train_type_category: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_TRAIN_TYPE_CATEGORY)
    )
    excluded_flows: list[ExcludedFlow] = Field(default_factory=_default_excluded_flows)
    intra_zone_min_miles: float = Field(
        25.0,
        description="Intra-zone FAF rows with a shorter average distance are local switching. The distance comes from FAF5 (same zone, commodity and mode; else the zone's average); rows FAF5 has no distance for are kept.",
    )
    min_od_tons_per_week: dict[str, float] = Field(
        default_factory=lambda: {"A": 150.0},
        description="Per output category (A/M/B/I): a zone pair's flow of that category below this many tons/week is dropped, before the zone pair is split over yard pairs. Categories not listed keep every flow.",
    )
    yard_pairs: YardPairRule = Field(default_factory=YardPairRule)
    work_yards_per_state: int = Field(
        3,
        ge=0,
        description="Besides zone yards, the N largest yard groups per state (inclusive of zone yards) may host work events.",
    )
    max_trailing_tons_per_run: float = Field(
        11_000.0,
        description="Max weight of one train run on any leg: freight plus the cars that carry it (trailing tons; RAS 2011 median link limit). Used for a category not listed in `max_trailing_tons_per_run_by_category`.",
    )
    max_trailing_tons_per_run_by_category: dict[str, float] = Field(
        default_factory=dict,
        description="Max weight of one train run on any leg, per output category (A/M/B/I): freight plus the cars that carry it (trailing tons). The freight a run may carry is this divided by (1 + empty car tons / cargo tons per car) of `car_weight`. Categories not listed use `max_trailing_tons_per_run`.",
    )
    max_runs_per_week: int = 7
    max_symbols_per_service: int = Field(
        3,
        description="A route may run as up to this many parallel symbols (y <= 7 x this).",
    )
    costs: Costs = Field(default_factory=Costs)
    crew: CrewRule = Field(default_factory=CrewRule)
    dwell_minutes: DwellMinutes = Field(default_factory=DwellMinutes)
    car_weight: CarWeight = Field(default_factory=CarWeight)
    locomotive_power: LocomotivePower = Field(default_factory=LocomotivePower)
    return_trips: ReturnTrips = Field(default_factory=ReturnTrips)
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
    def _check_car_weight_categories(self) -> Self:
        categories = set(self.train_type_category.values())
        for name in ("empty_car_tons_by_category", "cargo_tons_per_car_by_category"):
            if missing := categories - set(getattr(self.car_weight, name)):
                raise ValueError(
                    f"car_weight.{name} is missing categories {sorted(missing)}"
                )
        return self

    def capacity_of(self, train_type: str) -> float:
        """Max tons of freight per run of a train type: the freight whose trailing tons
        (freight plus its cars) equal the category's limit, else the default's."""
        category = self.train_type_category.get(train_type)
        if category is None:
            return self.max_trailing_tons_per_run
        trailing = self.max_trailing_tons_per_run_by_category.get(
            category, self.max_trailing_tons_per_run
        )
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

    def to_yaml(self) -> str:
        """Serialize to YAML with schema line."""
        body = yaml.safe_dump(
            self.model_dump(mode="json", by_alias=True), sort_keys=False, width=100
        )
        return f"# yaml-language-server: $schema=../{SCHEMA_PATH.as_posix()}\n\n{body}"


def load_config(path: Path | None = None) -> tuple[bool, Config]:
    """
    Load config from `path`, else `CONFIG_FILE_PATH` (defaults if that's missing).

    Returns (config_file_found, config).
    """
    if path is None:
        path = CONFIG_FILE_PATH
        if not path.is_file():
            return False, Config()
    elif not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return True, Config.model_validate(yaml.safe_load(path.read_text()) or {})
