import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import ClassVar, Literal, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    TypeAdapter,
    field_validator,
)
from pydantic_core import core_schema


class TrainCategory(StrEnum):
    """Category a train symbol belongs to."""

    A = "A"  # Automotive
    M = "M"  # Manifest
    B = "B"  # Bulk commodity service (coal, grain, ores & minerals, aggregates, petroleum & energy)
    I = "I"  # Intermodal


class ZoneCode(str):
    """String representing a FAF zone code which represents a single FAF zone."""

    # This makes pydantic happy when ZoneCode is used as a model field type.
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _t: type,
        _h: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )


@dataclass(frozen=True, kw_only=True)
class FafODTonnage:
    """Weekly FAF rail tons per (origin, destination) zone pair, in total and per sctg2 commodity code."""

    od_tonnage: dict[tuple[ZoneCode, ZoneCode], float]
    od_commodity_tonnage: dict[tuple[ZoneCode, ZoneCode], dict[str, float]]

    def get_proportions(self) -> dict[tuple[ZoneCode, ZoneCode], float]:
        total = sum(self.od_tonnage.values())
        if total == 0:
            return {}
        return {pair: t / total for pair, t in self.od_tonnage.items()}


class DayOfWeek(Enum):
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


@dataclass(frozen=True, kw_only=True, slots=True)
class Week[T]:
    """Generic class representing values of type `T` for each day of the week."""

    mon: T
    tue: T
    wed: T
    thu: T
    fri: T
    sat: T
    sun: T

    def get_day(self, day: DayOfWeek) -> T:
        """Get the week's value for the given day."""
        match day:
            case DayOfWeek.MONDAY:
                return self.mon
            case DayOfWeek.TUESDAY:
                return self.tue
            case DayOfWeek.WEDNESDAY:
                return self.wed
            case DayOfWeek.THURSDAY:
                return self.thu
            case DayOfWeek.FRIDAY:
                return self.fri
            case DayOfWeek.SATURDAY:
                return self.sat
            case DayOfWeek.SUNDAY:
                return self.sun

    def __iter__(self) -> Iterator[tuple[DayOfWeek, T]]:
        """Iterate over each day of the week and its corresponding value."""
        for day in DayOfWeek:
            yield day, self.get_day(day)


class Coord(BaseModel):
    """A lat/lon coordinate."""

    lat: float = Field(description="Latitude")
    lon: float = Field(description="Longitude")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @override
    def __hash__(self):
        return hash((self.lat, self.lon))


def _clean_city(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\bst\.\s*", "saint ", name).strip()
    return name


class Location(BaseModel):
    """Normalized city and state code."""

    raw_city: str = Field(
        alias="city", description="City name. Compared case-insensitively."
    )
    state_code: str = Field(
        alias="state", description="Two-letter US state code normalized to uppercase."
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, populate_by_name=True, extra="forbid"
    )

    @field_validator("state_code")
    @classmethod
    def _upper_state(cls, state: str) -> str:
        return state.upper()

    @property
    def normalized_city(self) -> str:
        """Lowercased, `St.`-expanded city name. All matching happens on this form."""
        return _clean_city(self.raw_city)

    # Equality is a fuzzy match on purpose
    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Location) and self.matches(
            other.raw_city, other.state_code
        )

    @override
    def __hash__(self) -> int:
        return hash((self.normalized_city, self.state_code))

    def matches(self, other_city: str, other_state_code: str) -> bool:
        # Check state code matches
        if self.state_code != other_state_code.upper():
            return False

        # Check city matches
        this_city = self.normalized_city
        clean_other = _clean_city(other_city)
        if this_city == other_city:
            return True
        if this_city in clean_other or clean_other in this_city:
            return True
        for part in re.split(r"[-/]", this_city):
            if _clean_city(part.strip()) == clean_other:
                return True
        for part in re.split(r"[-/]", clean_other):
            if _clean_city(part.strip()) == this_city:
                return True
        return False


class YgKey(str):
    """Represents a unique key for a `YardGroup`, formatted as `"YARDNAME-OBJECTID"`."""

    @staticmethod
    def new(yard_name: str, object_id: str) -> "YgKey":
        return YgKey(f"{yard_name}-{object_id}")

    @staticmethod
    def from_raw_str(yg_key_str: str) -> "YgKey":
        """Creates a `YgKey` from a raw string and ensures it follows the expected format."""
        if "-" not in yg_key_str:
            raise ValueError(
                f"Invalid YgKey format: '{yg_key_str}'. Expected format is 'YARDNAME-OBJECTID'."
            )
        return YgKey(yg_key_str)

    @staticmethod
    def parse_raw_path(raw_path: list[str]) -> list["YgKey"]:
        """Parses a list of raw strings into a list of `YgKey` instances."""
        return [YgKey.from_raw_str(raw) for raw in raw_path]

    # This makes pydantic happy for the TypeAdapter
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _t: type,
        _h: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.from_raw_str,
            core_schema.str_schema(),
        )


class YardGroup(BaseModel):
    """
    A yard group: one NTAD yard record, or a few merged ones.

    - city, state -- reverse-geocoded city and 2-letter state
    - od_zone_code -- the FAF zone it represents, if any (`is_od_yg` comes from this)
    """

    # raw properties - maybe can be removed
    id: str  # OBJECTID
    name: str  # YARDNAME
    owner: str

    coord: Coord
    elevation: float

    # It's a pain to serialize this as a Location instance so just using raw city/state instead.
    city: str = ""
    state: str = ""

    od_zone_code: ZoneCode | None = None
    shape_length_km: float = 0.0

    _key: YgKey | None = None

    @property
    def is_od_yg(self) -> bool:
        """Whether this YardGroup is an OD (origin/destination) YardGroup."""
        return self.od_zone_code is not None

    def key(self) -> YgKey:
        """Returns a unique key for the YardGroup (`"YARDNAME-OBJECTID"`)."""
        if self._key is None:
            self._key = YgKey.new(self.name, self.id)
        return self._key


ygs_dict_adapter = TypeAdapter(dict[YgKey, YardGroup])


class LineStringGeometry(BaseModel):
    """GeoJSON LineString geometry."""

    type: Literal["LineString"]
    coordinates: list[list[float]]


class MultiLineStringGeometry(BaseModel):
    """GeoJSON MultiLineString geometry."""

    type: Literal["MultiLineString"]
    coordinates: list[list[list[float]]]


class RailNetworkFeature(BaseModel):
    """A rail line from the NTAD rail network geojson."""

    type: Literal["Feature"]
    id: int | None = None
    geometry: LineStringGeometry
    properties: dict[str, str | int | float | None]


rail_network_adapter: TypeAdapter[list[RailNetworkFeature]] = TypeAdapter(
    list[RailNetworkFeature]
)


class ElevationResponse(BaseModel):
    """Elevation API response: one elevation per input coordinate."""

    elevation: list[float]


class YardGroupFeature(BaseModel):
    """A yard record from the NTAD rail yards geojson."""

    type: Literal["Feature"]
    id: int | None = None
    geometry: LineStringGeometry | MultiLineStringGeometry
    properties: dict[str, str | int | float | None]


yg_adapter: TypeAdapter[list[YardGroupFeature]] = TypeAdapter(list[YardGroupFeature])
