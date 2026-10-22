import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from railgen.models import YgKey, ZoneCode
from railgen.rail_network import RailNetwork
from railgen.schedule_gen.candidates import CandidateService, generate_candidates
from railgen.schedule_gen.demand import Commodity
from railgen.schedule_gen.model import (
    OdModelIndex,
    Solution,
    build_od_index,
    derive_stop_events,
    solve_od_model,
)
from railgen.schedule_gen.routes import Route
from railgen.schedule_gen.schedule_config import ServiceDesignSettings

# Relative to where FRIDGE is run. Each run writes its solutions and Gurobi logs to its own `<timestamp>/` folder in here.
SOLUTION_DIR = Path("gurobi_solutions")
_FORMAT = 2  # version of the saved solution file; commodities have `train_type`


@dataclass(kw_only=True)
class CategoryModel:
    name: str  # names the solution file and the Gurobi log
    category: str
    settings: ServiceDesignSettings
    commodities: list[Commodity]
    candidates: list[CandidateService]
    index: OdModelIndex


def categories(
    commodities: list[Commodity], settings: ServiceDesignSettings
) -> list[str]:
    """Output categories that have freight."""
    return sorted({settings.category_of(c.train_type) for c in commodities})


def category_settings(
    settings: ServiceDesignSettings, name: str
) -> ServiceDesignSettings:
    """The settings of one category's model: its own Gurobi log file, unless logging is off."""
    if settings.solver.log_file is None:
        return settings
    return settings.model_copy(
        update={
            "solver": settings.solver.model_copy(
                update={"log_file": f"gurobi_{name}.log"}
            ),
        }
    )


def build_category_model(
    network: RailNetwork,
    commodities: list[Commodity],
    routes: dict[tuple[YgKey, YgKey], Route],
    allowed: set[YgKey],
    settings: ServiceDesignSettings,
    category: str,
    *,
    name: str | None = None,
) -> CategoryModel:
    name = name or category
    cat_settings = category_settings(settings, name)
    own = [c for c in commodities if cat_settings.category_of(c.train_type) == category]
    candidates = generate_candidates(network, own, routes, allowed, cat_settings)
    return CategoryModel(
        name=name,
        category=category,
        settings=cat_settings,
        commodities=own,
        candidates=candidates,
        index=build_od_index(own, routes, candidates),
    )


def solution_path(name: str, run_dir: Path) -> Path:
    return run_dir / f"solution_{name}.json"


@dataclass(kw_only=True)
class SavedSolution:
    category: str
    instance_key: str
    saved_at: str
    solution: Solution
    services: dict[str, CandidateService]  # services with runs or flows
    commodities: list[Commodity]  # flow index k -> commodity


def instance_key(index: OdModelIndex, settings: ServiceDesignSettings) -> str:
    """Hash of everything that defines the model: services, commodities, riders, costs, bounds."""
    content = {
        "services": [
            [
                s.id,
                s.train_type,
                s.path,
                s.stop_positions,
                [round(v, 6) for v in s.leg_km],
                [round(v, 6) for v in s.leg_capacity_tons],
            ]
            for s in index.services.values()
        ],
        "commodities": [
            [c.origin, c.destination, c.train_type, round(c.tons_per_week, 6)]
            for c in index.commodities
        ],
        "riders": [[sid, i, ks] for (sid, i), ks in sorted(index.riders.items())],
        "costs": [settings.costs.train_start, settings.costs.per_train_mile],
        "run_bounds": [settings.max_runs_per_week, settings.max_symbols_per_service],
    }
    transfer = [settings.costs.transfer_fixed, settings.costs.transfer_per_ton]
    if any(
        transfer
    ):  # left out at 0, so solutions saved before these costs existed stay valid
        content["transfer_costs"] = transfer
    if settings.costs.per_distinct_train:  # left out at 0, for the same reason
        content["distinct_train_cost"] = settings.costs.per_distinct_train
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()


def save_solution(
    path: Path,
    category: str,
    solution: Solution,
    index: OdModelIndex,
    settings: ServiceDesignSettings,
) -> None:
    used = set(solution.runs_per_week) | {sid for sid, _i, _k in solution.flows}
    content = {
        "format": _FORMAT,
        "category": category,
        "instance_key": instance_key(index, settings),
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "solver": {
            "time_limit_s": settings.solver.time_limit_s,
            "mip_gap": settings.solver.mip_gap,
            "status": solution.status,
        },
        "objective": solution.objective,
        "bound": solution.bound,
        "gap": solution.gap,
        "runs_per_week": solution.runs_per_week,
        "flows": [[sid, i, k, tons] for (sid, i, k), tons in solution.flows.items()],
        "services": [asdict(index.services[sid]) for sid in sorted(used)],
        "commodities": [asdict(c) for c in index.commodities],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content))
    tmp.replace(path)


def read_solution(path: Path) -> SavedSolution:
    content = json.loads(path.read_text())
    if content["format"] != _FORMAT:
        raise ValueError(f"{path}: unknown solution file format {content['format']}")
    services = {
        d["id"]: CandidateService(
            id=d["id"],
            train_type=d["train_type"],
            path=tuple(YgKey(y) for y in d["path"]),
            stop_positions=tuple(d["stop_positions"]),
            leg_km=tuple(d["leg_km"]),
            leg_capacity_tons=tuple(d["leg_capacity_tons"]),
        )
        for d in content["services"]
    }
    commodities = [
        Commodity(
            origin=YgKey(d["origin"]),
            destination=YgKey(d["destination"]),
            train_type=d["train_type"],
            tons_per_week=d["tons_per_week"],
            zone_pair=(ZoneCode(d["zone_pair"][0]), ZoneCode(d["zone_pair"][1])),
        )
        for d in content["commodities"]
    ]
    flows = {(sid, i, k): tons for sid, i, k, tons in content["flows"]}
    on_board, pick_up, set_out = derive_stop_events(flows, services)
    return SavedSolution(
        category=content["category"],
        instance_key=content["instance_key"],
        saved_at=content["saved_at"],
        solution=Solution(
            runs_per_week=content["runs_per_week"],
            on_board=on_board,
            pick_up=pick_up,
            set_out=set_out,
            objective=content["objective"],
            bound=content["bound"],
            gap=content["gap"],
            status=content["solver"]["status"],
            flows=flows,
        ),
        services=services,
        commodities=commodities,
    )


def solve_and_save(model: CategoryModel, run_dir: Path) -> tuple[Solution, Path]:
    """Solve the category, with the Gurobi log in `run_dir`, and save the solution there."""
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = model.settings
    if settings.solver.log_file:
        settings = settings.model_copy(
            update={
                "solver": settings.solver.model_copy(
                    update={"log_file": str(run_dir / settings.solver.log_file)}
                )
            }
        )
    solution = solve_od_model(model.index, settings)
    path = solution_path(model.name, run_dir)
    save_solution(path, model.name, solution, model.index, model.settings)
    return solution, path
