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
    B = "B"  # Bulk (coal, grain, ores & minerals, aggregates, petroleum & energy)
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


class Location(BaseModel):
    """City and state code."""

    raw_city: str = Field(alias="city", description="City name.")
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


class YdKey(str):
    """Represents a unique key for a `Yard`, formatted as `"YARDNAME-OBJECTID"`."""

    @staticmethod
    def new(yard_name: str, object_id: str) -> "YdKey":
        return YdKey(f"{yard_name}-{object_id}")

    @staticmethod
    def from_raw_str(yd_key_str: str) -> "YdKey":
        """Creates a `YdKey` from a raw string and ensures it follows the expected format."""
        if "-" not in yd_key_str:
            raise ValueError(
                f"Invalid YdKey format: '{yd_key_str}'. Expected format is 'YARDNAME-OBJECTID'."
            )
        return YdKey(yd_key_str)

    @staticmethod
    def parse_raw_path(raw_path: list[str]) -> list["YdKey"]:
        """Parses a list of raw strings into a list of `YdKey` instances."""
        return [YdKey.from_raw_str(raw) for raw in raw_path]

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


class Yard(BaseModel):
    """
    A rail yard: one NTAD yard record, or a few merged ones.

    - city, state -- reverse-geocoded city and 2-letter state
    - od_zone_code -- the FAF zone it represents, if any (`is_od_yd` comes from this)
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

    _key: YdKey | None = None

    @property
    def is_od_yd(self) -> bool:
        """Whether this Yard is an OD (origin/destination) Yard."""
        return self.od_zone_code is not None

    def key(self) -> YdKey:
        """Returns a unique key for the Yard (`"YARDNAME-OBJECTID"`)."""
        if self._key is None:
            self._key = YdKey.new(self.name, self.id)
        return self._key


yds_dict_adapter = TypeAdapter(dict[YdKey, Yard])


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


class YardFeature(BaseModel):
    """A yard record from the NTAD rail yards geojson."""

    type: Literal["Feature"]
    id: int | None = None
    geometry: LineStringGeometry | MultiLineStringGeometry
    properties: dict[str, str | int | float | None]


yd_adapter: TypeAdapter[list[YardFeature]] = TypeAdapter(list[YardFeature])
