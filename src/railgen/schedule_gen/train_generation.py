"""FAF flows -> trains, with one service network design model per category.

Demand gets built once (`demand.py`). Then for each category (A, M, B, I): build the
candidates and model, solve it or load a saved solution (`solve_by_category.py`), check
yard balances, and extract trains (`solution_to_trains.py`). Circuitous trains get added at the
end to even out locomotives (`circuitous_trains.py`).
"""

import random
import shutil
import time
from pathlib import Path

from railgen.rail_network import RailNetwork
from railgen.schedule_gen.attributes import Train, TrainSymbol
from railgen.schedule_gen.candidates import CandidateService, allowed_yards
from railgen.schedule_gen.circuitous_trains import (
    add_circuitous_trains,
    balance_lines,
    terminals,
)
from railgen.schedule_gen.demand import (
    build_commodities,
    demand_lines,
    kept_lines,
    short_intra_lines,
    small_flow_lines,
    yard_targets,
)
from railgen.schedule_gen.model import model_size
from railgen.schedule_gen.solution_to_trains import (
    LongStretch,
    cut_empty_legs,
    deviations,
    long_stretch_lines,
    solution_yard_net,
    to_trains,
    write_stop_report,
)
from railgen.schedule_gen.solve_by_category import (
    build_category_model,
    categories,
    solve_and_save,
)

REPORT_FILE_NAME = "schedule-generation-report.txt"


def generate_trains(
    seed: int, network: RailNetwork, output_dir: Path, solution_dir: Path
) -> list[Train]:
    settings = network.cfg.service_design_settings
    start = time.perf_counter()
    last = start

    def log(msg: str) -> None:
        nonlocal last
        now = time.perf_counter()
        print(
            f"[Service design {now - start:7.1f}s | step {now - last:6.1f}s] {msg}",
            flush=True,
        )
        last = now

    log("demand: reading FAF, choosing yard pairs, routing them (phase 1 + 2)")
    commodities, report, routes = build_commodities(network, settings)
    log(
        f"demand done: {report.kept_tons:,.0f} of {report.faf_tons:,.0f} t/week kept; "
        f"{report.zone_pairs:,} zone pairs, {report.commodities:,} commodities, {len(routes):,} yard pairs"
    )
    demand_report = (
        demand_lines(report, settings)
        + small_flow_lines(report)
        + short_intra_lines(report, settings)
        + kept_lines(report, settings)
    )
    for line in demand_report:
        log(line)
    report_lines = ["Demand", *demand_report]
    allowed = allowed_yards(network, settings)

    trains: list[Train] = []
    service_of: dict[TrainSymbol, CandidateService] = {}
    for cat in categories(commodities, settings):
        log(f"category {cat} 1/4 candidates and riders")
        region = settings.region.instance
        model = build_category_model(
            network,
            commodities,
            routes,
            allowed,
            settings,
            cat,
            name=cat if region == "full" else f"{cat}_{region}",  # own solution file
        )
        size = model_size(model.index)
        log(
            f"category {cat}: {len(model.commodities):,} commodities, "
            f"{len(model.candidates):,} candidate trains over {len(allowed):,} allowed yards; "
            f"{size.variables:,} variables ({size.flow_vars:,} flows, "
            f"{size.integer_vars:,} integer), {size.constraints:,} constraints "
            f"({size.balance_rows:,} balance, {size.capacity_rows:,} capacity); "
            f"busiest leg has {size.max_riders_per_leg:,} commodities"
        )

        log(f"category {cat} 2/4 solve")
        solution, path = solve_and_save(model, solution_dir)
        log(
            f"category {cat}: solution solved and saved to {path}: "
            f"objective {solution.objective:,.0f}, bound {solution.bound:,.0f}, "
            f"gap {solution.gap:.2%}, {len(solution.runs_per_week):,} trains selected, "
            f"{sum(solution.runs_per_week.values()):,} runs/week"
        )
        shutil.copyfile(path, output_dir / path.name)
        log(f"category {cat}: solution copied to {output_dir / path.name}")

        log(f"category {cat} 3/4 yard balance check")
        off = deviations(
            solution_yard_net(solution, model.index), yard_targets(model.commodities)
        )
        if off:
            raise AssertionError(
                f"category {cat}: {len(off)} yards off their FAF target, "
                f"e.g. {next(iter(off.items()))}"
            )
        log(f"category {cat}: every yard matches its target")

        solution, services, cut = cut_empty_legs(solution, model.index.services)
        log(f"category {cat}: {cut.line()}")

        log(f"category {cat} 4/4 extraction: symbols, days, work and crew stops, times")
        long_stretches: list[LongStretch] = []
        cat_trains = to_trains(
            solution,
            services,
            network,
            model.settings,
            # New rng instance for each train category
            random.Random(f"{seed}-{cat}"),
            service_of,
            long_stretches,
        )
        log(f"category {cat}: {len(cat_trains):,} train symbols")
        crew_lines = long_stretch_lines(long_stretches, settings.crew.usable_hours)
        for line in crew_lines:
            log(f"category {cat}: warning: {line}")
        trains.extend(cat_trains)

        stop_report = write_stop_report(
            cat, solution, services, model.index.commodities, output_dir
        )
        for line in stop_report:
            log(f"category {cat}: {line}")
        report_lines += ["", f"Category {cat}", cut.line(), *stop_report, *crew_lines]

    ends = terminals(
        trains
    )  # before circuitous trains, so both summaries count the same yards
    before = balance_lines(trains, ends)
    trains, lever_lines = add_circuitous_trains(trains, service_of, network, settings)
    after = balance_lines(trains, ends)
    for line in before:
        log(f"locomotive balance before circuitous trains: {line}")
    for line in lever_lines:
        log(line)
    for line in after:
        log(f"locomotive balance: {line}")
    report_lines += [
        "",
        "Locomotive balance before circuitous trains",
        *before,
        "",
        "Circuitous trains",
        *lever_lines,
        "",
        "Locomotive balance",
        *after,
    ]

    report_path = output_dir / REPORT_FILE_NAME
    report_path.write_text("\n".join(report_lines) + "\n")
    log(f"done: {len(trains):,} train symbols; report written to {report_path}")
    return trains
