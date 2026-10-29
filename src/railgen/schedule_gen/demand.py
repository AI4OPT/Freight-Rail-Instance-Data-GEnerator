import csv
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from railgen.constants import FAF5_CSV_NAME, FAF6_CSV_NAME, FAF_YEAR
from railgen.constants import INPUT_DATA_DIR as DATA_DIR
from railgen.faf_data import ensure_faf5_file, ensure_faf6_files
from railgen.models import YdKey, ZoneCode
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.regions import RegionReport, cut_to_region, region_lines
from railgen.schedule_gen.routes import Route, routes_between, routing_graph
from railgen.schedule_gen.schedule_config import INTERMODAL_GROUP, ServiceDesignSettings

KM_PER_MILE = 1.609344

RAIL_MODE = "2"  # FAF dms_mode "Rail"
MULTIMODAL_MODE = "5"  # FAF dms_mode "Multiple modes & mail"


@dataclass(frozen=True, kw_only=True)
class Commodity:
    origin: YdKey
    destination: YdKey
    train_type: str  # its freight group, or `bulk` for all B groups with `shared_bulk`
    tons_per_week: float
    zone_pair: tuple[ZoneCode, ZoneCode]


@dataclass(frozen=True, kw_only=True)
class FafRow:
    origin_zone: ZoneCode
    destination_zone: ZoneCode
    sctg: str
    mode: str  # RAIL_MODE or MULTIMODAL_MODE
    trade_type: str  # FAF trade_type: "1" domestic, "2" import, "3" export
    tons_per_week: float


@dataclass(kw_only=True)
class IntraZoneMiles:
    """Average shipment miles of intra-zone flows (1000 * ton-miles / tons), from FAF5."""

    by_commodity: dict[tuple[ZoneCode, str, str], float]  # (zone, sctg, mode)
    by_zone: dict[tuple[ZoneCode, str], float]  # (zone, mode), all commodities

    def miles(self, zone: ZoneCode, sctg: str, mode: str) -> float | None:
        """The flow's own average, else its zone's average for that mode, else None."""
        own = self.by_commodity.get((zone, sctg, mode))
        return own if own is not None else self.by_zone.get((zone, mode))


@dataclass(frozen=True, kw_only=True)
class SmallFlowDrop:
    """Zone-pair flows of one category below its minimum (before the yard-pair split)."""

    threshold: float  # tons/week
    # Zone pairs of the category below the threshold (dropped) and in all; without
    # `shared_bulk` a B zone pair counts once per bulk group.
    ods_below: int
    ods_total: int
    tons_dropped: float  # tons/week
    tons_total: float  # tons/week of the category before dropping

    @property
    def share_dropped(self) -> float:
        """Dropped tons as a fraction of the category's tons before dropping."""
        return self.tons_dropped / self.tons_total if self.tons_total else 0.0


@dataclass(kw_only=True)
class DemandReport:
    """Tons kept and dropped at each step, for the generation stats."""

    faf_tons: float = 0.0
    excluded_tons: dict[int, float] = field(
        default_factory=dict
    )  # by index in `excluded_flows`
    local_switching_tons: float = 0.0
    local_switching_rows: int = 0  # FAF rows dropped as local switching
    intra_zone_tons: float = 0.0  # all intra-zone tons read, before any drop
    intra_no_distance_tons: float = (
        0.0  # intra-zone tons FAF5 has no distance for (kept)
    )
    # Multimodal rows of zone pairs under `multimodal_min_miles` or with no FAF5 distance.
    short_multimodal_tons: float = 0.0
    short_multimodal_rows: int = 0
    # Multimodal rows of zone pairs over `multimodal_max_miles`.
    long_multimodal_tons: float = 0.0
    long_multimodal_rows: int = 0
    excluded_zone_tons: float = 0.0  # rows from or to a zone of `excluded_zones`
    no_second_yard_tons: float = 0.0
    no_yard_pair_tons: float = (
        0.0  # inter-zone tons of zone pairs with no routed yard pair
    )
    short_inter_tons: float = 0.0  # inter-zone: every yard pair under `min_route_km`
    short_intra_tons: float = 0.0  # intra-zone: assigned yard pair under `min_route_km`
    small_od_flows: dict[str, SmallFlowDrop] = field(default_factory=dict)
    kept_tons: float = 0.0
    # Kept tons (after every drop) by FAF mode and trade type
    kept_by_mode_trade: dict[tuple[str, str], float] = field(default_factory=dict)
    zone_pairs: int = 0
    commodities: int = 0
    region: RegionReport | None = None  # east / west instances only


def read_faf_rows(multimodal_rail_share: float) -> Iterator[FafRow]:
    """Rail and multimodal rows of the FAF CSV (every trade type), in tons per week.

    For imports and exports only the domestic part of the move is used (`dms_orig`,
    `dms_dest`). A multimodal row counts `multimodal_rail_share` of its tons.
    """
    ensure_faf6_files()
    with open(DATA_DIR / FAF6_CSV_NAME, newline="") as f:
        for r in csv.DictReader(f):
            mode = r["dms_mode"].strip()
            if mode not in (RAIL_MODE, MULTIMODAL_MODE):
                continue
            tons = float(r[f"tons_{FAF_YEAR}"] or 0)
            if mode == MULTIMODAL_MODE:
                tons *= multimodal_rail_share
            if tons <= 0:
                continue
            yield FafRow(
                origin_zone=ZoneCode(r["dms_orig"].strip()),
                destination_zone=ZoneCode(r["dms_dest"].strip()),
                sctg=r["sctg2"].strip().zfill(2),
                mode=mode,
                trade_type=r["trade_type"].strip(),
                tons_per_week=tons * 1000 / 52,
            )


def read_faf_miles() -> tuple[IntraZoneMiles, dict[tuple[ZoneCode, ZoneCode], float]]:
    """Average shipment miles from FAF5 (1000 * ton-miles / tons), every trade type.

    Returns the miles of the intra-zone rail and multimodal flows.
    """
    ensure_faf5_file()
    tons: dict[tuple[str, ...], float] = defaultdict(float)
    ton_miles: dict[tuple[str, ...], float] = defaultdict(float)
    pair_tons: dict[tuple[str, str], float] = defaultdict(float)
    pair_ton_miles: dict[tuple[str, str], float] = defaultdict(float)
    with open(DATA_DIR / FAF5_CSV_NAME, newline="") as f:
        for r in csv.DictReader(f):
            mode = r["dms_mode"].strip()
            if mode not in (RAIL_MODE, MULTIMODAL_MODE):
                continue
            t = float(r[f"tons_{FAF_YEAR}"] or 0)
            if t <= 0:
                continue
            tm = float(r[f"tmiles_{FAF_YEAR}"] or 0)
            zone, dest = r["dms_orig"].strip(), r["dms_dest"].strip()
            if mode == MULTIMODAL_MODE:
                pair_tons[(zone, dest)] += t
                pair_ton_miles[(zone, dest)] += tm
            if zone != dest:
                continue
            for key in ((zone, r["sctg2"].strip().zfill(2), mode), (zone, mode)):
                tons[key] += t
                ton_miles[key] += tm
    miles = {key: 1000 * ton_miles[key] / t for key, t in tons.items()}
    intra_miles = IntraZoneMiles(
        by_commodity={
            (ZoneCode(k[0]), k[1], k[2]): v for k, v in miles.items() if len(k) == 3
        },
        by_zone={(ZoneCode(k[0]), k[1]): v for k, v in miles.items() if len(k) == 2},
    )
    multimodal_miles = {
        (ZoneCode(o), ZoneCode(d)): 1000 * pair_ton_miles[(o, d)] / t
        for (o, d), t in pair_tons.items()
    }
    return intra_miles, multimodal_miles


def multimodal_counts(
    miles: float | None, min_miles: float, max_miles: float = 0.0
) -> bool:
    """
    Whether the multimodal rows of a zone pair with this FAF5 distance are counted.
    """
    if min_miles <= 0 and max_miles <= 0:
        return True
    if miles is None:
        return False
    return miles >= min_miles and (max_miles <= 0 or miles <= max_miles)


@dataclass(frozen=True, kw_only=True)
class CandidateYardPair:
    origin: YdKey
    dest: YdKey
    value: float  # route cost for inter-zone, route km for intra-zone


def select_yard_pairs(
    candidates: list[CandidateYardPair],
    tons_per_week: float,
    *,
    tonnage_step: float,
    max_pairs: int,
    distance_tolerance: float | None,
    target_km: float | None = None,
) -> list[tuple[YdKey, YdKey]]:
    """Pick which yard pairs carry a zone pair's tons.

    `candidates` are (origin, destination, value).
    Inter-zone: value is route cost, and pairs within `distance_tolerance` of the cheapest are kept.
    Intra-zone: value is route km, ranked by closeness to `target_km` (the FAF average distance).
    Keeps n = min(max_pairs, ceil(tons / tonnage_step)) pairs.
    """
    if not candidates:
        return []
    n = max(1, min(max_pairs, math.ceil(tons_per_week / tonnage_step)))
    if target_km is None:
        ranked = sorted(candidates, key=lambda c: (c.value, c.origin, c.dest))
        limit = (
            ranked[0].value * (1 + distance_tolerance)
            if distance_tolerance is not None
            else math.inf
        )
        ranked = [c for c in ranked if c.value <= limit]
    else:
        ranked = sorted(
            candidates, key=lambda c: (abs(c.value - target_km), c.origin, c.dest)
        )

    kept: list[tuple[YdKey, YdKey]] = []
    used_o: set[YdKey] = set()
    used_d: set[YdKey] = set()
    for cand in ranked:
        if len(kept) == n:
            break
        if cand.origin in used_o and cand.dest in used_d:
            continue
        kept.append((cand.origin, cand.dest))
        used_o.add(cand.origin)
        used_d.add(cand.dest)
    return kept


def assign_pair_per_category(
    candidates: list[CandidateYardPair],
    category_tons: dict[str, float],
    size: dict[YdKey, float],
    *,
    distance_tolerance: float | None,
) -> dict[str, tuple[YdKey, YdKey]]:
    """One yard pair per category of a zone pair.

    `candidates` carry the route cost as `value`. Pairs costing more than
    (1 + `distance_tolerance`) x the cheapest are dropped first. Then the biggest pair
    (origin x destination track length) goes to the heaviest category, etc.
    """
    if distance_tolerance is not None:
        limit = min(c.value for c in candidates) * (1 + distance_tolerance)
        candidates = [c for c in candidates if c.value <= limit]
    pairs = sorted(
        ((c.origin, c.dest) for c in candidates),
        key=lambda p: (-size.get(p[0], 0.0) * size.get(p[1], 0.0), p[0], p[1]),
    )
    ranked = sorted(category_tons, key=lambda c: (-category_tons[c], c))
    return {cat: pairs[i % len(pairs)] for i, cat in enumerate(ranked)}


def split_by_size(
    pairs: list[tuple[YdKey, YdKey]], size: dict[YdKey, float]
) -> list[float]:
    """Share of the zone pair's tons for each kept pair: origin size x destination size."""
    weights = [
        max(size.get(o, 0.0), 1e-9) * max(size.get(d, 0.0), 1e-9) for o, d in pairs
    ]
    total = sum(weights)
    return [w / total for w in weights]


def build_commodities(
    network: RailNetwork, settings: ServiceDesignSettings
) -> tuple[list[Commodity], DemandReport, dict[tuple[YdKey, YdKey], Route]]:
    """Commodities, the demand report, and the route of every kept yard pair."""
    report = DemandReport()
    excluded = {
        (e.origin_zone, e.destination_zone, code): i
        for i, e in enumerate(settings.excluded_flows)
        for code in e.sctg
    }

    tons: dict[tuple[ZoneCode, ZoneCode, str], float] = defaultdict(float)
    # The same tons by FAF trade type, for the report of what is kept
    tons_by_trade: dict[tuple[ZoneCode, ZoneCode, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    # Per (zone, train type), over the intra-zone tons that have a FAF5 distance.
    ton_miles_intra: dict[tuple[ZoneCode, str], float] = defaultdict(float)
    tons_with_miles: dict[tuple[ZoneCode, str], float] = defaultdict(float)
    intra_miles, multimodal_miles = read_faf_miles()
    excluded_zones = set(settings.excluded_zones)
    for row in read_faf_rows(settings.multimodal_rail_share):
        report.faf_tons += row.tons_per_week
        if row.origin_zone in excluded_zones or row.destination_zone in excluded_zones:
            report.excluded_zone_tons += row.tons_per_week
            continue
        key = (row.origin_zone, row.destination_zone, row.sctg)
        if key in excluded:
            i = excluded[key]
            report.excluded_tons[i] = (
                report.excluded_tons.get(i, 0.0) + row.tons_per_week
            )
            continue
        intra = row.origin_zone == row.destination_zone
        avg_miles = (
            intra_miles.miles(row.origin_zone, row.sctg, row.mode) if intra else None
        )
        if intra:
            report.intra_zone_tons += row.tons_per_week
            if avg_miles is None:
                report.intra_no_distance_tons += row.tons_per_week
            elif avg_miles < settings.intra_zone_min_miles:
                report.local_switching_tons += row.tons_per_week
                report.local_switching_rows += 1
                continue
        if row.mode == MULTIMODAL_MODE:
            miles = multimodal_miles.get((row.origin_zone, row.destination_zone))
            if not multimodal_counts(
                miles, settings.multimodal_min_miles, settings.multimodal_max_miles
            ):
                if miles is not None and miles >= settings.multimodal_min_miles:
                    report.long_multimodal_tons += row.tons_per_week
                    report.long_multimodal_rows += 1
                else:
                    report.short_multimodal_tons += row.tons_per_week
                    report.short_multimodal_rows += 1
                continue
        # Freight of one train type is one flow (with `shared_bulk`, all B groups together).
        train_type = settings.train_type_of(
            INTERMODAL_GROUP
            if row.mode == MULTIMODAL_MODE
            else settings.group_of(row.sctg)
        )
        tons[(row.origin_zone, row.destination_zone, train_type)] += row.tons_per_week
        tons_by_trade[(row.origin_zone, row.destination_zone, train_type)][
            row.trade_type
        ] += row.tons_per_week
        if avg_miles is not None:
            ton_miles_intra[(row.origin_zone, train_type)] += (
                row.tons_per_week * avg_miles
            )
            tons_with_miles[(row.origin_zone, train_type)] += row.tons_per_week

    zone_to_yds: dict[ZoneCode, list[YdKey]] = defaultdict(list)
    size: dict[YdKey, float] = {}
    for key, yd in network.yds_dict.items():
        if yd.od_zone_code is not None:
            zone_to_yds[yd.od_zone_code].append(key)
            size[key] = yd.shape_length_km

    by_pair: dict[tuple[ZoneCode, ZoneCode], dict[str, float]] = defaultdict(dict)
    for (oz, dz, tt), t in tons.items():
        by_pair[(oz, dz)][tt] = t

    yard_pairs = {
        (o, d)
        for oz, dz in by_pair
        for o in zone_to_yds.get(oz, [])
        for d in zone_to_yds.get(dz, [])
        if o != d
    }
    routes = routes_between(network, routing_graph(network), yard_pairs)

    routes_of: dict[tuple[ZoneCode, ZoneCode], list[tuple[YdKey, YdKey, Route]]] = {}
    placeable: dict[tuple[ZoneCode, ZoneCode], dict[str, float]] = {}
    for (oz, dz), types in sorted(by_pair.items()):
        pair_routes = [
            (o, d, routes[(o, d)])
            for o in zone_to_yds.get(oz, [])
            for d in zone_to_yds.get(dz, [])
            if (o, d) in routes
        ]
        if not pair_routes:
            if oz == dz:
                report.no_second_yard_tons += sum(types.values())
            else:
                report.no_yard_pair_tons += sum(types.values())
            continue
        if oz != dz:
            pair_routes = [
                p for p in pair_routes if p[2].km >= settings.yard_pairs.min_route_km
            ]
            if not pair_routes:
                report.short_inter_tons += sum(types.values())
                continue
        routes_of[(oz, dz)] = pair_routes
        placeable[(oz, dz)] = types
    kept_by_pair, report.small_od_flows = drop_small_zone_pair_flows(
        placeable, settings
    )
    rule = settings.yard_pairs
    commodities: list[Commodity] = []
    kept_routes: dict[tuple[YdKey, YdKey], Route] = {}
    category = settings.category_of
    for (oz, dz), types in kept_by_pair.items():
        total = sum(types.values())
        pair_routes = routes_of[(oz, dz)]
        if rule.assignment == "one_per_category":
            category_tons: dict[str, float] = defaultdict(float)
            for tt, t in types.items():
                category_tons[category(tt)] += t
            pair_of = assign_pair_per_category(
                [
                    CandidateYardPair(origin=o, dest=d, value=r.cost)
                    for o, d, r in pair_routes
                ],
                category_tons,
                size,
                distance_tolerance=None if oz == dz else rule.distance_tolerance,
            )
            for tt, t in types.items():
                o, d = pair_of[category(tt)]
                if oz == dz and routes[(o, d)].km < rule.min_route_km:
                    report.short_intra_tons += t
                    continue
                kept_routes[(o, d)] = routes[(o, d)]
                commodities.append(
                    Commodity(
                        origin=o,
                        destination=d,
                        train_type=tt,
                        tons_per_week=t,
                        zone_pair=(oz, dz),
                    )
                )
            continue
        with_miles = sum(tons_with_miles[(oz, tt)] for tt in types) if oz == dz else 0.0
        if with_miles > 0:
            avg_miles = sum(ton_miles_intra[(oz, tt)] for tt in types) / with_miles
            pairs = select_yard_pairs(
                [
                    CandidateYardPair(origin=o, dest=d, value=r.km)
                    for o, d, r in pair_routes
                ],
                total,
                tonnage_step=rule.tonnage_step,
                max_pairs=rule.max_pairs,
                distance_tolerance=None,
                target_km=avg_miles * KM_PER_MILE,
            )
        else:
            pairs = select_yard_pairs(
                [
                    CandidateYardPair(origin=o, dest=d, value=r.cost)
                    for o, d, r in pair_routes
                ],
                total,
                tonnage_step=rule.tonnage_step,
                max_pairs=rule.max_pairs,
                distance_tolerance=rule.distance_tolerance,
            )
        shares = split_by_size(pairs, size)
        for (o, d), share in zip(pairs, shares):
            if oz == dz and routes[(o, d)].km < rule.min_route_km:
                report.short_intra_tons += total * share
                continue
            kept_routes[(o, d)] = routes[(o, d)]
            for tt, t in types.items():
                commodities.append(
                    Commodity(
                        origin=o,
                        destination=d,
                        train_type=tt,
                        tons_per_week=t * share,
                        zone_pair=(oz, dz),
                    )
                )
    # Kept tons by FAF mode and trade type, each commodity split like its zone-pair flow.
    kept_by_mode_trade: dict[tuple[str, str], float] = defaultdict(float)
    for c in commodities:
        mode = MULTIMODAL_MODE if c.train_type == INTERMODAL_GROUP else RAIL_MODE
        by_trade = tons_by_trade[(*c.zone_pair, c.train_type)]
        flow = sum(by_trade.values())
        if flow <= 0:
            continue
        for trade, t in by_trade.items():
            kept_by_mode_trade[(mode, trade)] += t * c.tons_per_week / flow
    report.kept_by_mode_trade = dict(kept_by_mode_trade)
    report.kept_tons = sum(c.tons_per_week for c in commodities)
    report.zone_pairs = len({c.zone_pair for c in commodities})

    if settings.region.instance != "full":
        commodities, kept_routes, report.region = cut_to_region(
            commodities, kept_routes, network, settings
        )
    report.commodities = len(commodities)
    return commodities, report, kept_routes


def drop_small_zone_pair_flows(
    by_pair: dict[tuple[ZoneCode, ZoneCode], dict[str, float]],
    settings: ServiceDesignSettings,
) -> tuple[dict[tuple[ZoneCode, ZoneCode], dict[str, float]], dict[str, SmallFlowDrop]]:
    """Drop the flows of a zone pair below their category's `min_od_tons_per_week`.

    A flow is one train type's tons on the zone pair, so the minimum is what justifies a
    train of that type: with `shared_bulk` all bulk of the zone pair together, without it
    each bulk group on its own. Also returns per-category counts from before the drop.
    Zone pairs left empty go away.
    """
    minimum = settings.min_od_tons_per_week
    flows = [
        (zone_pair, tt, t, settings.category_of(tt))
        for zone_pair, types in by_pair.items()
        for tt, t in types.items()
    ]
    report: dict[str, SmallFlowDrop] = {}
    for cat, threshold in minimum.items():
        own = [t for _zp, _g, t, c in flows if c == cat]
        below = [t for t in own if t < threshold]
        report[cat] = SmallFlowDrop(
            threshold=threshold,
            ods_below=len(below),
            ods_total=len(own),
            tons_dropped=sum(below),
            tons_total=sum(own),
        )
    kept: dict[tuple[ZoneCode, ZoneCode], dict[str, float]] = {}
    for zone_pair, tt, t, cat in flows:
        if t >= minimum.get(cat, 0.0):
            kept.setdefault(zone_pair, {})[tt] = t
    return kept, report


def demand_lines(report: DemandReport, settings: ServiceDesignSettings) -> list[str]:
    """What the demand stage read and dropped before the yard-pair split, for the run log."""
    faf, intra = report.faf_tons, report.intra_zone_tons
    switching = report.local_switching_tons
    region = region_lines(report.region) if report.region is not None else []
    return region + [
        (
            f"FAF {FAF_YEAR} rail + {settings.multimodal_rail_share:.0%} of "
            f"multimodal, all trade types: {faf:,.0f} t/week, of which intra-zone {intra:,.0f}"
        ),
        (
            f"excluded zones ({', '.join(settings.excluded_zones) or 'none'}), rows from or "
            f"to them dropped: {report.excluded_zone_tons:,.0f} t/week"
        ),
        f"excluded flows: {sum(report.excluded_tons.values()):,.0f} t/week",
        *(
            f"  {e.origin_zone}->{e.destination_zone} SCTG {', '.join(e.sctg)}: "
            f"{report.excluded_tons.get(i, 0.0):,.0f} t/week"
            + (f" ({e.reason})" if e.reason else "")
            for i, e in enumerate(settings.excluded_flows)
        ),
        (
            f"intra-zone under {settings.intra_zone_min_miles:g} mi (FAF5 distance), dropped: "
            f"{report.local_switching_rows:,} FAF rows, {switching:,.0f} t/week "
            f"({switching / intra:.2%} of intra-zone tons, {switching / faf:.2%} of all tons)"
        ),
        f"intra-zone with no FAF5 distance, kept: {report.intra_no_distance_tons:,.0f} t/week",
        (
            f"multimodal on zone pairs under {settings.multimodal_min_miles:g} mi (FAF5 "
            f"distance) or with no FAF5 distance, dropped: {report.short_multimodal_rows:,} "
            f"FAF rows, {report.short_multimodal_tons:,.0f} t/week "
            f"({report.short_multimodal_tons / faf:.2%} of all tons)"
        ),
        (
            f"multimodal on zone pairs over {settings.multimodal_max_miles:g} mi (FAF5 "
            f"distance), dropped: {report.long_multimodal_rows:,} FAF rows, "
            f"{report.long_multimodal_tons:,.0f} t/week "
            f"({report.long_multimodal_tons / faf:.2%} of all tons)"
        ),
        (
            f"intra-zone in zones with a single yard, dropped: "
            f"{report.no_second_yard_tons:,.0f} t/week"
        ),
        (
            f"zone pairs with no yard pair (a zone has no yard), dropped: "
            f"{report.no_yard_pair_tons:,.0f} t/week"
        ),
        (
            f"inter-zone, every yard pair under {settings.yard_pairs.min_route_km:g} km "
            f"(local trains), dropped: {report.short_inter_tons:,.0f} t/week"
        ),
    ]


def small_flow_lines(report: DemandReport) -> list[str]:
    """One line per category with a minimum, then the total over them, for the run log."""
    drops = report.small_od_flows
    if not drops:
        return []
    lines = [
        f"category {cat}: {d.ods_below:,} of {d.ods_total:,} zone pairs below "
        f"{d.threshold:g} t/week, dropped ({d.tons_dropped:,.0f} of "
        f"{d.tons_total:,.0f} t/week, {d.share_dropped:.2%} of the category's tons)"
        for cat, d in drops.items()
    ]
    dropped = sum(d.tons_dropped for d in drops.values())
    # All categories, before this step (intra-zone local moves are dropped after it).
    before = report.kept_tons + report.short_intra_tons + dropped
    lines.append(
        f"all categories: {sum(d.ods_below for d in drops.values()):,} zone-pair flows dropped, "
        f"{dropped:,.0f} of {before:,.0f} t/week ({dropped / before:.2%} of all tons)"
    )
    return lines


def short_intra_lines(
    report: DemandReport, settings: ServiceDesignSettings
) -> list[str]:
    """Intra-zone tons whose assigned yard pair is too short, for the run log."""
    return [
        (
            f"intra-zone on a yard pair under {settings.yard_pairs.min_route_km:g} km "
            f"(local trains), dropped: {report.short_intra_tons:,.0f} t/week"
        )
    ]


TRADE_TYPE_NAMES = {"1": "domestic", "2": "import", "3": "export"}


def kept_lines(report: DemandReport, settings: ServiceDesignSettings) -> list[str]:
    """Where the kept freight comes from (FAF mode and trade type), after every drop, for the run log."""
    kept = report.kept_by_mode_trade
    total = sum(kept.values())
    if not total:
        return []

    def split(mode: str) -> str:
        parts = ", ".join(
            f"{name} {kept.get((mode, code), 0.0):,.0f}"
            for code, name in TRADE_TYPE_NAMES.items()
        )
        tons = sum(t for (m, _), t in kept.items() if m == mode)
        return f"{tons:,.0f} t/week ({tons / total:.1%}; {parts})"

    return [
        f"kept after every drop: {total:,.0f} t/week",
        f"  rail: {split(RAIL_MODE)}",
        f"  multimodal at {settings.multimodal_rail_share:.0%}: {split(MULTIMODAL_MODE)}",
    ]


def yard_targets(commodities: list[Commodity]) -> dict[YdKey, float]:
    """Weekly inbound minus outbound tons per yard (the balance each yard must show)."""
    target: dict[YdKey, float] = defaultdict(float)
    for c in commodities:
        target[c.destination] += c.tons_per_week
        target[c.origin] -= c.tons_per_week
    return dict(target)
