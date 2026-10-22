import pytest
from shapely.geometry import Polygon, box

from railgen.constants import FAF6_ZONES_BASE_NAME, FAF6_ZONES_DIR_NAME, INPUT_DATA_DIR
from railgen.faf_data import FAF_ZONE_NAMES, FAF_ZONES, FafZone, faf_zone_of
from railgen.models import Coord, ZoneCode

# Zone A: a square with a square hole; zone B: the square next to it.
ZONE_A = FafZone(
    code=ZoneCode("011"),
    polygon=Polygon(box(0, 0, 10, 10).exterior, [box(4, 4, 6, 6).exterior]),
)
ZONE_B = FafZone(code=ZoneCode("012"), polygon=box(10, 0, 20, 10))


def test_point_is_in_the_zone_around_it_and_not_in_a_hole():
    zones = [ZONE_A, ZONE_B]
    assert faf_zone_of(Coord(lat=2, lon=2), zones) == "011"
    assert faf_zone_of(Coord(lat=5, lon=15), zones) == "012"
    assert faf_zone_of(Coord(lat=5, lon=5), zones) is None  # in A's hole


def test_point_just_outside_a_zone_belongs_to_it_within_the_tolerance():
    # 0.005 degrees of longitude at latitude 5 is about 0.55 km; 0.02 degrees about 2.2 km.
    assert faf_zone_of(Coord(lat=5, lon=-0.005), [ZONE_A], tolerance_km=1.0) == "011"
    assert faf_zone_of(Coord(lat=5, lon=-0.02), [ZONE_A], tolerance_km=1.0) is None


@pytest.mark.skipif(
    not (INPUT_DATA_DIR / FAF6_ZONES_DIR_NAME / f"{FAF6_ZONES_BASE_NAME}.shp").exists(),
    reason="FAF6 zone shapefile not downloaded",
)
def test_real_zone_file_places_chicago_and_el_paso_yards():
    zones = FAF_ZONES.load()
    assert len(zones) == 134
    assert FAF_ZONE_NAMES.load()[ZoneCode("171")] == "Chicago IL-IN-WI (IL Part)"
    clearing = Coord(lat=41.7606489764185, lon=-87.7196420852658)  # CLEARING-446
    assert faf_zone_of(clearing, zones) == "171"  # Chicago (IL part), not Rest of IL
    # EL PASO-193's first track point is 7 m south of the zone boundary, on the Mexican border.
    el_paso = Coord(lat=31.7478988845201, lon=-106.488172438858)
    assert faf_zone_of(el_paso, zones) == "485"
