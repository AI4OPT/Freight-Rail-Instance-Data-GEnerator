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
class OdAssignmentStats:
    """How FAF zones were mapped onto yards."""

    yds_per_zone: int = 0
    zones_total: int = 0  # zones that send or receive rail or multimodal freight
    zones_represented: int = 0
    zones_unrepresented: list[ZoneCode] = field(default_factory=list)  # no yard inside
    zones_under_target: int = 0  # represented by fewer than `yds_per_zone`
    yards_designated: int = 0

    def print_summary(self) -> None:
        """Print the totals."""
        under = (
            f", {self.zones_under_target} by fewer than {self.yds_per_zone}"
            if self.zones_under_target
            else ""
        )
        unrepresented = (
            f": {self.zones_unrepresented}" if self.zones_unrepresented else "."
        )
        print(
            f"[OD] {self.zones_represented} of {self.zones_total} zones represented by "
            f"{self.yards_designated} yards{under}; "
            f"{len(self.zones_unrepresented)} unrepresented{unrepresented}"
        )


@dataclass(kw_only=True, slots=True)
class OutputStats:
    # aggregated over the output rows
    unique_route_od_pairs_count: int = 0
    unique_segment_od_pairs_count: int = 0
    used_yards_count: int = 0
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
        yd_count: int,
        pu_count: int,
        so_count: int,
        record_count: int,
    ) -> None:
        self.output_stats.unique_route_od_pairs_count = route_od_pairs
        self.output_stats.unique_segment_od_pairs_count = segment_od_pairs
        self.output_stats.used_yards_count = yd_count
        self.output_stats.output_rows_count = record_count
        self.output_stats.total_pick_ups = pu_count
        self.output_stats.total_set_outs = so_count
        pretty_print(
            f"[green]Unique route OD pairs (train origin → destination):[/green] {self.output_stats.unique_route_od_pairs_count}"
        )
        pretty_print(
            f"[green]Unique segment OD pairs (CSV leg BGN_YD → END_YD):[/green] {self.output_stats.unique_segment_od_pairs_count}"
        )
        pretty_print(
            f"[green]Total yards used in schedule.csv:[/green] {self.output_stats.used_yards_count}"
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
