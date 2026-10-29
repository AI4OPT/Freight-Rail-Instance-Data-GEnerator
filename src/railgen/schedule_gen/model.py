import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

from railgen.models import YdKey
from railgen.schedule_gen.candidates import CandidateService
from railgen.schedule_gen.demand import Commodity
from railgen.schedule_gen.routes import Route
from railgen.schedule_gen.schedule_config import ServiceDesignSettings

KM_PER_MILE = 1.609344

Arc = tuple[YdKey, YdKey]
Leg = tuple[str, int]  # (service id, leg 1..n)


@dataclass(kw_only=True)
class OdModelIndex:
    services: dict[str, CandidateService]
    commodities: list[Commodity]
    riders: dict[Leg, list[int]]  # leg -> indices of commodities that may ride it
    own_tons: dict[str, float]  # service id -> tons of its yard pair it may carry


@dataclass(kw_only=True)
class Solution:
    runs_per_week: dict[str, int]  # y_s for selected services (y_s >= 1)
    on_board: dict[tuple[str, int], float]  # (service, leg 1..n) -> tons/week
    pick_up: dict[tuple[str, int], float]  # (service, stop 0..n) -> tons/week
    set_out: dict[tuple[str, int], float]
    objective: float
    bound: float
    gap: float
    status: str
    # (service, leg 1..n, commodity index) -> tons/week
    flows: dict[tuple[str, int, int], float] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ModelSize:
    integer_vars: int
    flow_vars: int
    capacity_rows: int
    balance_rows: int
    max_riders_per_leg: int

    @property
    def variables(self) -> int:
        return self.integer_vars + self.flow_vars

    @property
    def constraints(self) -> int:
        return self.capacity_rows + self.balance_rows


def build_od_index(
    commodities: list[Commodity],
    routes: dict[tuple[YdKey, YdKey], Route],
    candidates: list[CandidateService],
) -> OdModelIndex:
    arc_users: dict[Arc, set[int]] = defaultdict(set)
    pair_type_tons: dict[tuple[YdKey, YdKey], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for k, c in enumerate(commodities):
        pair_type_tons[(c.origin, c.destination)][c.train_type] += c.tons_per_week
        for arc in pairwise(routes[(c.origin, c.destination)].path):
            arc_users[arc].add(k)

    riders: dict[Leg, list[int]] = {}
    own_tons: dict[str, float] = {}
    for s in candidates:
        own_tons[s.id] = pair_type_tons[(s.path[0], s.path[-1])].get(s.train_type, 0.0)
        for i, (a, b) in enumerate(pairwise(s.stop_positions), start=1):
            arcs = list(pairwise(s.path[a : b + 1]))
            users = set(arc_users.get(arcs[0], ()))
            for arc in arcs[1:]:
                if not users:
                    break
                users &= arc_users.get(arc, set())
            ks = sorted(k for k in users if commodities[k].train_type == s.train_type)
            if ks:
                riders[(s.id, i)] = ks

    return OdModelIndex(
        services={s.id: s for s in candidates},
        commodities=commodities,
        riders=riders,
        own_tons=own_tons,
    )


def model_size(index: OdModelIndex) -> ModelSize:
    """Size of the model `solve_od_model` (without building it)."""
    balance: set[tuple[int, YdKey]] = set()
    for (sid, i), ks in index.riders.items():
        stops = index.services[sid].stops
        for k in ks:
            balance.add((k, stops[i - 1]))
            balance.add((k, stops[i]))
    for k, c in enumerate(index.commodities):
        balance.add((k, c.origin))
        balance.add((k, c.destination))
    return ModelSize(
        integer_vars=len(index.services),
        flow_vars=sum(len(ks) for ks in index.riders.values()),
        capacity_rows=len(index.riders),
        balance_rows=len(balance),
        max_riders_per_leg=max((len(ks) for ks in index.riders.values()), default=0),
    )


def solve_od_model(
    index: OdModelIndex, settings: ServiceDesignSettings, *, verbose: bool = True
) -> Solution:
    import gurobipy as gp
    from gurobipy import GRB

    costs = settings.costs
    m = gp.Model("service_design_od")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = settings.solver.time_limit_s
    m.Params.MIPGap = settings.solver.mip_gap
    if settings.solver.log_file:
        m.Params.LogFile = settings.solver.log_file

    y: dict[str, gp.Var] = {}
    x: dict[tuple[str, int, int], gp.Var] = {}  # (service, leg, commodity)
    balance: dict[tuple[int, YdKey], gp.LinExpr] = defaultdict(gp.LinExpr)
    obj = gp.LinExpr()

    for sid, s in index.services.items():
        capacity = max(s.leg_capacity_tons)
        ub = max(
            settings.max_runs_per_week * settings.max_symbols_per_service,
            math.ceil(index.own_tons[sid] / capacity),
        )
        y[sid] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name=f"y[{sid}]")
        obj.add(
            y[sid], costs.train_start + costs.per_train_mile * s.total_km / KM_PER_MILE
        )
        if costs.per_distinct_train > 0:
            w = m.addVar(vtype=GRB.BINARY, name=f"w[{sid}]")
            obj.add(w, costs.per_distinct_train)
            m.addConstr(y[sid] <= ub * w, name=f"open[{sid}]")

    for (sid, i), ks in index.riders.items():
        s = index.services[sid]
        load = gp.LinExpr()
        for k in ks:
            x[(sid, i, k)] = v = m.addVar(name=f"x[{sid},{i},{k}]")
            load.add(v)
            balance[(k, s.stops[i - 1])].add(v)
            balance[(k, s.stops[i])].add(v, -1.0)
        m.addConstr(load <= s.leg_capacity_tons[i - 1] * y[sid], name=f"cap[{sid},{i}]")

    # Transfers: z[k, yard] = 1 if k gets off a train at a yard that is not its destination.
    transfer_fixed, transfer_per_ton = costs.transfer_fixed, costs.transfer_per_ton
    z: dict[tuple[int, YdKey], gp.Var] = {}
    if transfer_fixed > 0 or transfer_per_ton > 0:
        for (sid, i), ks in index.riders.items():
            yard = index.services[sid].stops[i]  # leg i ends at stop i
            for k in ks:
                c = index.commodities[k]
                if yard == c.destination:
                    continue
                if (k, yard) not in z:
                    z[(k, yard)] = m.addVar(vtype=GRB.BINARY, name=f"z[{k},{yard}]")
                    obj.add(
                        z[(k, yard)],
                        transfer_fixed + transfer_per_ton * c.tons_per_week,
                    )
                staying = x.get((sid, i + 1, k), 0.0)
                m.addConstr(
                    x[(sid, i, k)] - staying <= c.tons_per_week * z[(k, yard)],
                    name=f"transfer[{sid},{i},{k}]",
                )

    for k, c in enumerate(index.commodities):
        _ = balance[(k, c.origin)]
        _ = balance[(k, c.destination)]
    for (k, yard), expr in balance.items():
        c = index.commodities[k]
        rhs = (
            c.tons_per_week
            if yard == c.origin
            else -c.tons_per_week
            if yard == c.destination
            else 0.0
        )
        m.addConstr(expr == rhs, name=f"bal[{k},{yard}]")

    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    print(
        f"[Service design] model built: {m.NumConstrs:,} constraints, "
        f"{m.NumVars:,} variables ({m.NumIntVars:,} integer); solving (stops at gap <= "
        f"{settings.solver.mip_gap:.0%} or {settings.solver.time_limit_s:,.0f} s)",
        flush=True,
    )
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"No feasible solution found (Gurobi status {m.Status}).")

    flows = {key: v.X for key, v in x.items() if v.X > 1e-6}
    on_board, pick_up, set_out = derive_stop_events(flows, index.services)
    return Solution(
        runs_per_week={sid: round(v.X) for sid, v in y.items() if v.X > 0.5},
        on_board=on_board,
        pick_up=pick_up,
        set_out=set_out,
        objective=m.ObjVal,
        bound=m.ObjBound,
        gap=m.MIPGap,
        status=str(m.Status),
        flows=flows,
    )


def derive_stop_events(
    flows: dict[tuple[str, int, int], float],
    services: dict[str, CandidateService],
) -> tuple[
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
]:
    """Per-leg load and per-stop pick-ups / set-outs of each service, from the flows.

    At stop j, commodity k is picked up by the increase and set out by the decrease of
    its load between leg j and leg j+1.
    """
    per_service: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (sid, i, k), tons in flows.items():
        per_service[sid][k][i] = tons

    on_board: dict[tuple[str, int], float] = defaultdict(float)
    pick_up: dict[tuple[str, int], float] = defaultdict(float)
    set_out: dict[tuple[str, int], float] = defaultdict(float)
    for sid, by_k in per_service.items():
        n = len(services[sid].stop_positions) - 1
        for legs in by_k.values():
            for i, tons in legs.items():
                on_board[(sid, i)] += tons
            for j in range(n + 1):
                change = legs.get(j + 1, 0.0) - legs.get(j, 0.0)
                if change > 1e-6:
                    pick_up[(sid, j)] += change
                elif change < -1e-6:
                    set_out[(sid, j)] -= change
    return dict(on_board), dict(pick_up), dict(set_out)
