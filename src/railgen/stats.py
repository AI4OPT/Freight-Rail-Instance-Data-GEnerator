"""Statistics collected during a schedule generation run."""

import json
import statistics
from dataclasses import asdict, dataclass, field

from rich import print as pretty_print

from railgen.models import TrainCategory, ZoneCode


@dataclass(kw_only=True, slots=True)
class TrainCategoryStats:
    """Per-category stats."""

    # aggregated over the output rows
    train_symbol_count: int = 0
    total_pick_ups: int = 0
    total_set_outs: int = 0
    req_loc_count_avg: float = 0.0
    req_loc_count_min: int = 0
    req_loc_count_max: int = 0
    req_loc_count_stdev: float = 0.0


@dataclass(kw_only=True, slots=True)
class OdPassStats:
    """What one pass of the OD yard group assignment achieved."""

    zones: int = 0
    yard_groups: int = 0
    zones_under_target: int = 0


@dataclass(kw_only=True, slots=True)
class OdAssignmentStats:
    """How FAF zones were mapped onto yard groups. Built in `_assign_od_yard_groups` and carried on the RailNetwork into the run stats."""

    ygs_per_zone: int = 0
    max_fallback_km: float = 0.0
    zones_total: int = 0
    zones_represented: int = 0
    zones_unrepresented: list[ZoneCode] = field(default_factory=list)
    yard_groups_designated: int = 0
    # One entry per pass, in the order they run.
    city_match: OdPassStats = field(default_factory=OdPassStats)
    nearest_within_radius: OdPassStats = field(default_factory=OdPassStats)
    rest_of_state: OdPassStats = field(default_factory=OdPassStats)

    def print_summary(self) -> None:
        """Print the per-pass breakdown."""
        labels = (
            ("pass 1 (city match)", self.city_match),
            (
                f"pass 2 (within {self.max_fallback_km:.0f} km)",
                self.nearest_within_radius,
            ),
            ("pass 3 (rest of state)", self.rest_of_state),
        )
        for label, s in labels:
            under = (
                f" ({s.zones_under_target} under target of {self.ygs_per_zone})"
                if s.zones_under_target
                else ""
            )
            print(
                f"[OD] {label:<26}: {s.zones:3d} zones -> {s.yard_groups:3d} yard groups{under}"
            )
        print(
            f"[OD] total: {self.zones_represented} of {self.zones_total} zones represented "
            f"by {self.yard_groups_designated} yard groups; "
            f"{len(self.zones_unrepresented)} unrepresented"
            + (
                f": {sorted(self.zones_unrepresented)}"
                if self.zones_unrepresented
                else "."
            )
        )


@dataclass(kw_only=True, slots=True)
class OutputStats:
    # aggregated over the output rows
    unique_route_od_pairs_count: int = 0
    unique_segment_od_pairs_count: int = 0
    used_yard_groups_count: int = 0
    total_pick_ups: int = 0
    total_set_outs: int = 0
    output_rows_count: int = 0


@dataclass(kw_only=True, slots=True)
class ScheduleGenerationStats:
    generation_rng_seed: int = 0

    od_assignment: OdAssignmentStats = field(default_factory=OdAssignmentStats)

    output_stats: OutputStats = field(default_factory=OutputStats)

    train_category_statistics: dict[TrainCategory, TrainCategoryStats] = field(
        default_factory=dict
    )

    def _get_train_category_stats(self, category: TrainCategory) -> TrainCategoryStats:
        """Get category stats, creating if not set."""
        return self.train_category_statistics.setdefault(category, TrainCategoryStats())

    def report_od_assignment(self, od_assignment: OdAssignmentStats) -> None:
        """Record the OD assignment summary (already printed during graph generation)."""
        self.od_assignment = od_assignment

    def report_totals(
        self,
        *,
        route_od_pairs: int,
        segment_od_pairs: int,
        yg_count: int,
        pu_count: int,
        so_count: int,
        record_count: int,
    ) -> None:
        self.output_stats.unique_route_od_pairs_count = route_od_pairs
        self.output_stats.unique_segment_od_pairs_count = segment_od_pairs
        self.output_stats.used_yard_groups_count = yg_count
        self.output_stats.output_rows_count = record_count
        self.output_stats.total_pick_ups = pu_count
        self.output_stats.total_set_outs = so_count
        pretty_print(
            f"[green]Unique route OD pairs (train origin → destination):[/green] {self.output_stats.unique_route_od_pairs_count}"
        )
        pretty_print(
            f"[green]Unique segment OD pairs (CSV leg BGN_YG → END_YG):[/green] {self.output_stats.unique_segment_od_pairs_count}"
        )
        pretty_print(
            f"[green]Total yard groups used in schedule.csv:[/green] {self.output_stats.used_yard_groups_count}"
        )
        pretty_print(
            f"[green]Total pick-ups (IS_PU=1):[/green] {self.output_stats.total_pick_ups}"
        )
        pretty_print(
            f"[green]Total set-outs (IS_SO=1):[/green] {self.output_stats.total_set_outs}"
        )

    def report_category_trains(
        self,
        category: TrainCategory,
        *,
        symbols: int,
        pick_ups: int,
        set_outs: int,
        locos: list[int],
    ) -> None:
        cs = self._get_train_category_stats(category)
        cs.train_symbol_count, cs.total_pick_ups, cs.total_set_outs = (
            symbols,
            pick_ups,
            set_outs,
        )
        cs.req_loc_count_avg = statistics.mean(locos) if locos else 0.0
        cs.req_loc_count_stdev = statistics.pstdev(locos) if locos else 0.0
        cs.req_loc_count_min = min(locos) if locos else 0
        cs.req_loc_count_max = max(locos) if locos else 0
        pretty_print(
            f"  Category {category}: {cs.train_symbol_count} symbols, {cs.total_pick_ups} PUs, {cs.total_set_outs} SOs; Counts/day: avg {cs.req_loc_count_avg:.2f}, min {cs.req_loc_count_min}, max {cs.req_loc_count_max}, stdev {cs.req_loc_count_stdev:.2f}"
        )

    def to_json(self) -> str:
        """Convert stats to a JSON string."""
        return json.dumps(asdict(self), indent=2) + "\n"
