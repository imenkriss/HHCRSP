"""Deterministic and budget-robust evaluation of the MTWMO-UnHHCRSP model.

This pure module implements the problem model requirements of AGENTS.md:

* P1  routes are ``depot -> patients -> laboratory`` (the laboratory leg is
      skipped when the instance carries no laboratory node);
* P3  soft time windows: early arrival waits, late arrival is tardiness;
* P5  Bertsimas-Sim budget uncertainty over travel, departure and skill demand;
* P6  three separated objectives Z1 (cost), Z2 (satisfaction), Z3 (balance);
* P7  constraint-violation measures for (5), (10), (12), (13), (14), (15);
* P8  permutation (separator) chromosome decoding.

No LLM/RAG/agent imports: only ``.domain`` is used.

ASSUMPTION (P5): the Bertsimas-Sim budget is applied per route. Each route's
travel margin is the budgeted sum of its own arc deviations, the departure
margin uses the single caregiver's deviation, and the skill-demand margin uses
the single patient's deviation. This keeps the deterministic model exactly the
Gamma=0 case and keeps every margin monotone in its budget.

ASSUMPTION (P5): the worst-case travel deviation is realised on the route arc
with the largest deviation and delays every node after it. Unlisted
``alpha_ij`` values default to 0; missing ``travel_deviation_fraction`` /
``departure_deviation_minutes`` / ``demand_deviation_level`` attributes default
to 0.2 / 30.0 / 1.0.

ASSUMPTION (P6): the service cost uses theta_ik = theta_i * p_i / g_k (P4), so
the extra p_i factor written in the paper's Z1 is already folded into theta_ik;
the patient-satisfaction coverage term o_i is read as "patient i is served".
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .domain import Caregiver, Instance, Patient, evaluate_window_arrival

DEFAULT_TRAVEL_DEVIATION_FRACTION = 0.2
DEFAULT_DEPARTURE_DEVIATION_MINUTES = 30.0
DEFAULT_DEMAND_DEVIATION_LEVEL = 1.0


@dataclass(frozen=True)
class CostParameters:
    """Cost rates for the six Z1 components (P6)."""

    service_rate: float = 1.0
    waiting_rate: float = 1.0
    overtime_rate: float = 1.0
    travel_rate: float = 1.0
    fixed_rate: float = 0.0
    tardiness_rate: float = 1.0


@dataclass(frozen=True)
class CostBreakdown:
    """The six separate cost components of Z1 (P6)."""

    service: float = 0.0
    waiting: float = 0.0
    overtime: float = 0.0
    travel: float = 0.0
    fixed: float = 0.0
    tardiness: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.service
            + self.waiting
            + self.overtime
            + self.travel
            + self.fixed
            + self.tardiness
        )


@dataclass(frozen=True)
class ConstraintViolation:
    """Scalar violation measures for the hard constraints (P7)."""

    single_visit: float = 0.0
    incompatibility: float = 0.0
    skill: float = 0.0
    workload: float = 0.0
    lab_closing: float = 0.0
    window: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.single_visit
            + self.incompatibility
            + self.skill
            + self.workload
            + self.lab_closing
            + self.window
        )


@dataclass
class Route:
    """One caregiver's ordered patient list from a decoded chromosome."""

    caregiver: Caregiver
    patients: list[Patient] = field(default_factory=list)


@dataclass
class Solution:
    """Decoded chromosome: per-caregiver routes plus unassigned patients."""

    routes: list[Route] = field(default_factory=list)
    unassigned: list[Patient] = field(default_factory=list)


@dataclass
class RouteDetail:
    """Scheduled route with every per-patient timing and the route totals."""

    caregiver: Caregiver
    patient_ids: list[str | int] = field(default_factory=list)
    arrival: list[float] = field(default_factory=list)
    service_start: list[float] = field(default_factory=list)
    finish: list[float] = field(default_factory=list)
    waiting: list[float] = field(default_factory=list)
    tardiness: list[float] = field(default_factory=list)
    windows: list[tuple[float, float] | None] = field(default_factory=list)
    service_minutes: list[float] = field(default_factory=list)
    departure_time: float = 0.0
    return_time: float = 0.0
    working_minutes: float = 0.0
    travel_minutes: float = 0.0
    travel_distance: float = 0.0
    service_total: float = 0.0
    waiting_total: float = 0.0
    tardiness_total: float = 0.0
    window_missing: int = 0


@dataclass
class Evaluation:
    """Full evaluation result. ``objectives`` is the minimisation vector."""

    costs: CostBreakdown
    z2: float
    z3: float
    violations: ConstraintViolation
    routes: list[RouteDetail]
    unassigned: list[Patient]
    gamma: tuple[float, float, float]

    @property
    def z1(self) -> float:
        return self.costs.total

    @property
    def objectives(self) -> list[float]:
        """Minimisation vector (Z1, -Z2, Z3) for a Pareto optimiser."""
        return [self.z1, -self.z2, self.z3]

    @property
    def feasible(self) -> bool:
        return self.violations.total <= 1e-12


def budget_sum(deviations: Sequence[float], gamma: float) -> float:
    """Bertsimas-Sim budgeted deviation sum (P5).

    Sort the positive deviations descending, take the ``floor(gamma)`` largest
    in full, then the fractional remainder on the next largest. ``gamma <= 0``
    returns 0, so the deterministic model is the Gamma=0 special case.
    """
    ordered = sorted((float(value) for value in deviations if value > 0.0), reverse=True)
    if gamma <= 0.0 or not ordered:
        return 0.0
    whole = int(math.floor(gamma))
    total = sum(ordered[:whole])
    if whole < len(ordered):
        total += (gamma - whole) * ordered[whole]
    return total


def _resolve_gamma(gamma: float | Sequence[float] | Mapping[str, float]) -> tuple[float, float, float]:
    """Return (Gamma_t, Gamma_s, Gamma_p) from a scalar, triple, or mapping."""
    if isinstance(gamma, Mapping):
        return (
            float(gamma.get("t", 0.0)),
            float(gamma.get("s", 0.0)),
            float(gamma.get("p", 0.0)),
        )
    if isinstance(gamma, (tuple, list)):
        values = [float(value) for value in gamma] + [0.0, 0.0, 0.0]
        return values[0], values[1], values[2]
    value = float(gamma)
    return value, value, value


def _speed(instance: Instance) -> float:
    speed = instance.travel_speed if instance.travel_speed else 1.0
    return float(speed) if speed and speed > 0 else 1.0


def _arc_alpha(instance: Instance, a: str | int, b: str | int) -> float:
    alpha = instance.travel_uncertainty.get((a, b))
    if alpha is None:
        alpha = instance.travel_uncertainty.get((b, a), 0.0)
    return float(alpha)


def choose_window(
    arrival_time: float, windows: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float] | None, float, float]:
    """Return (window, waiting, tardiness) minimising tardiness then waiting.

    ``window`` is None when the patient carries no usable window (constraint 12
    cannot be verified). Ties break on the earliest window start.
    """
    best: tuple[tuple[float, float, float], tuple[float, float], float, float] | None = None
    for window in windows:
        timing = evaluate_window_arrival(arrival_time, window)
        key = (timing.tardiness, timing.waiting_time, window[0])
        if best is None or key < best[0]:
            best = (key, window, timing.waiting_time, timing.tardiness)
    if best is None:
        return None, 0.0, 0.0
    return best[1], best[2], best[3]


def service_minutes(patient: Patient, caregiver: Caregiver, demand_extra: float = 0.0) -> float:
    """Robust service time theta_i * p_i / g_k with an optional demand margin."""
    demand = patient.demand_level + demand_extra
    return float(patient.service_time) * demand / float(caregiver.skill_level)


def _demand_extra(patient: Patient, demand_deviation: float, gamma_p: float) -> float:
    return budget_sum([patient.demand_uncertainty * demand_deviation], gamma_p)


def decode(permutation: Sequence[int], instance: Instance) -> Solution:
    """Decode a permutation chromosome into per-caregiver routes (P8).

    Genes ``1..N`` are patients (patient ``g`` is ``instance.patients[g - 1]``);
    genes ``N+1..N+K`` are caregiver separators that open that caregiver's route
    (``instance.caregivers[g - N - 1]``). Patients before the first separator, or
    never mentioned, are returned as unassigned. Duplicate patient genes are
    preserved so the single-visit violation can be detected.
    """
    patient_by_gene = {index + 1: patient for index, patient in enumerate(instance.patients)}
    caregiver_by_gene = {
        len(instance.patients) + 1 + index: caregiver
        for index, caregiver in enumerate(instance.caregivers)
    }
    routes: list[Route] = []
    by_caregiver_id: dict[str | int, Route] = {}
    unassigned: list[Patient] = []
    current: Route | None = None
    for raw_gene in permutation:
        gene = int(raw_gene)
        if gene in caregiver_by_gene:
            caregiver = caregiver_by_gene[gene]
            current = by_caregiver_id.get(caregiver.id)
            if current is None:
                current = Route(caregiver=caregiver)
                by_caregiver_id[caregiver.id] = current
                routes.append(current)
        elif gene in patient_by_gene:
            patient = patient_by_gene[gene]
            if current is None:
                unassigned.append(patient)
            else:
                current.patients.append(patient)
        else:
            raise ValueError(
                f"Gene {gene} is outside the valid range "
                f"1..{len(instance.patients) + len(instance.caregivers)}"
            )
    mentioned = {patient.id for route in routes for patient in route.patients}
    mentioned.update(patient.id for patient in unassigned)
    for patient in instance.patients:
        if patient.id not in mentioned:
            unassigned.append(patient)
    return Solution(routes=routes, unassigned=unassigned)


def _schedule_route(
    route: Route,
    instance: Instance,
    gamma: tuple[float, float, float],
    travel_deviation: float,
    departure_deviation: float,
    demand_deviation: float,
) -> RouteDetail:
    """Schedule one route under worst-case budgeted deviations (P1/P3/P5)."""
    gamma_t, gamma_s, gamma_p = gamma
    caregiver = route.caregiver
    patients = route.patients
    speed = _speed(instance)
    has_lab = instance.laboratory is not None
    node_ids: list[str | int] = ["depot", *[patient.id for patient in patients]]
    if has_lab:
        node_ids.append("laboratory")

    arcs: list[dict[str, float]] = []
    for origin, destination in zip(node_ids, node_ids[1:]):
        distance = instance.distance(origin, destination)
        travel = distance / speed
        alpha = _arc_alpha(instance, origin, destination)
        arcs.append({
            "distance": distance,
            "travel": travel,
            "deviation": alpha * travel * travel_deviation,
            "distance_deviation": alpha * distance * travel_deviation,
        })

    time_margin = budget_sum([arc["deviation"] for arc in arcs], gamma_t)
    distance_margin = budget_sum([arc["distance_deviation"] for arc in arcs], gamma_t)
    worst_arc = -1
    if time_margin > 0.0 and arcs:
        worst_arc = max(range(len(arcs)), key=lambda index: arcs[index]["deviation"])

    departure_delay = budget_sum(
        [caregiver.departure_uncertainty * departure_deviation], gamma_s
    )
    departure_time = float(caregiver.start_time) + departure_delay
    current = departure_time

    detail = RouteDetail(caregiver=caregiver, departure_time=departure_time)
    for index, patient in enumerate(patients):
        arc = arcs[index]
        travel = arc["travel"]
        if index == worst_arc:
            travel += time_margin
        arrival = current + travel
        demand_extra = _demand_extra(patient, demand_deviation, gamma_p)
        service = service_minutes(patient, caregiver, demand_extra)
        window, waiting, tardiness = choose_window(arrival, patient.time_windows)
        if window is None:
            detail.window_missing += 1
            start = arrival
            waiting = tardiness = 0.0
        else:
            start = arrival + waiting
        finish = start + service
        current = finish
        detail.patient_ids.append(patient.id)
        detail.arrival.append(arrival)
        detail.service_start.append(start)
        detail.finish.append(finish)
        detail.waiting.append(waiting)
        detail.tardiness.append(tardiness)
        detail.windows.append(window)
        detail.service_minutes.append(service)
        detail.service_total += service
        detail.waiting_total += waiting
        detail.tardiness_total += tardiness

    if has_lab:
        lab_index = len(patients)
        lab_arc = arcs[lab_index]
        travel = lab_arc["travel"]
        if lab_index == worst_arc:
            travel += time_margin
        detail.return_time = current + travel
    else:
        detail.return_time = current
    detail.working_minutes = detail.return_time - departure_time
    detail.travel_minutes = sum(arc["travel"] for arc in arcs) + time_margin
    detail.travel_distance = sum(arc["distance"] for arc in arcs) + distance_margin
    return detail


def _lab_closing(instance: Instance) -> float | None:
    closing = getattr(instance, "lab_closing_time", None)
    if closing is not None:
        return float(closing)
    if instance.horizon is not None and len(instance.horizon) >= 2:
        return float(instance.horizon[1])
    return None


def evaluate(
    solution: Sequence[int] | Solution,
    instance: Instance,
    gamma: float | Sequence[float] | Mapping[str, float] = 0.0,
    *,
    costs: CostParameters | None = None,
) -> Evaluation:
    """Evaluate a chromosome or decoded solution for the three objectives.

    ``gamma`` is the P5 robustness budget; a scalar applies the same value to
    travel, departure and skill-demand budgets, a triple maps to (t, s, p), and a
    mapping uses keys ``t``/``s``/``p``. Gamma=0 reproduces the deterministic
    model exactly.
    """
    rates = costs or CostParameters()
    gammas = _resolve_gamma(gamma)
    decoded = solution if isinstance(solution, Solution) else decode(solution, instance)
    travel_deviation = float(
        getattr(instance, "travel_deviation_fraction", DEFAULT_TRAVEL_DEVIATION_FRACTION)
    )
    departure_deviation = float(
        getattr(instance, "departure_deviation_minutes", DEFAULT_DEPARTURE_DEVIATION_MINUTES)
    )
    demand_deviation = float(
        getattr(instance, "demand_deviation_level", DEFAULT_DEMAND_DEVIATION_LEVEL)
    )

    route_details = [
        _schedule_route(route, instance, gammas, travel_deviation, departure_deviation, demand_deviation)
        for route in decoded.routes
    ]

    service = sum(detail.service_total for detail in route_details)
    waiting = sum(detail.waiting_total for detail in route_details)
    tardiness = sum(detail.tardiness_total for detail in route_details)
    overtime = sum(
        max(0.0, detail.working_minutes - detail.caregiver.rated_minutes)
        for detail in route_details
    )
    travel = sum(detail.travel_distance for detail in route_details)
    fixed = float(len(decoded.routes))
    cost_breakdown = CostBreakdown(
        service=rates.service_rate * service,
        waiting=rates.waiting_rate * waiting,
        overtime=rates.overtime_rate * overtime,
        travel=rates.travel_rate * travel,
        fixed=rates.fixed_rate * fixed,
        tardiness=rates.tardiness_rate * tardiness,
    )

    z2 = 0.0
    skill_violations = 0.0
    incompatibility = 0.0
    workload = 0.0
    for route, detail in zip(decoded.routes, route_details):
        caregiver = route.caregiver
        for patient in route.patients:
            preference = float(patient.preferences.get(caregiver.id, 0.0))
            z2 += 1.0 + preference + caregiver.skill_level - patient.demand_level
            if float(caregiver.skill_level) + 1e-9 < float(patient.demand_level):
                skill_violations += 1.0
            if patient.rejections.get(caregiver.id) is True:
                incompatibility += 1.0
            elif (
                patient.care_type
                and caregiver.specialty
                and patient.care_type != caregiver.specialty
            ):
                incompatibility += 1.0
        workload += max(0.0, detail.working_minutes - caregiver.hard_daily_limit_minutes)

    lab_closing = 0.0
    closing = _lab_closing(instance)
    if closing is not None and instance.laboratory is not None:
        lab_closing = sum(
            max(0.0, detail.return_time - closing) for detail in route_details
        )

    counts = Counter(patient.id for route in decoded.routes for patient in route.patients)
    single_visit = float(sum(count - 1 for count in counts.values() if count > 1))
    window = float(sum(detail.window_missing for detail in route_details))

    notes_by_caregiver = {cg.id: 0.0 for cg in instance.caregivers}
    for detail in route_details:
        notes_by_caregiver[detail.caregiver.id] = notes_by_caregiver.get(detail.caregiver.id, 0.0) + detail.working_minutes / 60.0
    hours = list(notes_by_caregiver.values())
    mean_hours = sum(hours) / len(hours) if hours else 0.0
    z3 = sum((value - mean_hours) ** 2 for value in hours) / len(hours) if hours else 0.0

    violations = ConstraintViolation(
        single_visit=single_visit,
        incompatibility=incompatibility,
        skill=skill_violations,
        workload=workload,
        lab_closing=lab_closing,
        window=window,
    )
    return Evaluation(
        costs=cost_breakdown,
        z2=z2,
        z3=z3,
        violations=violations,
        routes=route_details,
        unassigned=decoded.unassigned,
        gamma=gammas,
    )


__all__ = [
    "CostBreakdown",
    "CostParameters",
    "ConstraintViolation",
    "Evaluation",
    "Route",
    "RouteDetail",
    "Solution",
    "budget_sum",
    "choose_window",
    "decode",
    "evaluate",
    "service_minutes",
]
