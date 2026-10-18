"""Read-only constraint checks and assignment-to-route conversion.

NSGA2.optimize() returns candidate mappings with keys "solution",
"objectives", and "summary". The "solution" value is an assignment list;
each current assignment has patient_id, caregiver_id, service_hours,
travel_time, and time_window. It does not include routes, depot/laboratory
nodes, start/arrival/departure/return schedules, or variable-domain declarations.

For independent route validation, this module also accepts a richer solution
mapping with "assignments", "routes" (caregiver ID to ordered node IDs),
"departure_times", "return_times", and "variable_domains". The instance is
expected to contain patient/caregiver records and, when relevant, depot_id,
laboratory_id, travel_times, lab_closing_time and T. Times are minutes unless
instance["time_unit"] is "hours". Missing evidence is UNVERIFIABLE, never a
violation. assignment_to_routes returns (routes, unassigned_patient_ids). This module
reports only and never repairs candidate data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

SATISFIED = "SATISFIED"
VIOLATED = "VIOLATED"
UNVERIFIABLE = "UNVERIFIABLE"


def _payload(solution: Any) -> Any:
    if isinstance(solution, Mapping) and "solution" in solution:
        return solution["solution"]
    return solution


def _assignment_data(solution: Any) -> tuple[bool, list[Mapping[str, Any]]]:
    payload = _payload(solution)
    if isinstance(payload, Mapping):
        if "assignments" not in payload:
            return False, []
        rows = payload["assignments"]
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False, []
    return True, [row for row in rows if isinstance(row, Mapping)]


def _assignments(solution: Any) -> list[Mapping[str, Any]]:
    return _assignment_data(solution)[1]


def _metadata(solution: Any) -> Mapping[str, Any]:
    payload = _payload(solution)
    if isinstance(solution, Mapping):
        # Keep optional route metadata supplied beside the optimizer assignment list.
        merged = dict(solution)
        if isinstance(payload, Mapping):
            merged.update(payload)
        return merged
    return payload if isinstance(payload, Mapping) else {}


def _routes(solution: Any) -> Mapping[Any, Any] | None:
    routes = _metadata(solution).get("routes")
    return routes if isinstance(routes, Mapping) else None


def _route_patients(route: Sequence[Any], instance: Mapping[str, Any]) -> list[Any]:
    """Return patient IDs from either patient-only routes or explicit depot/lab paths."""
    depot = instance.get("depot_id", 0)
    laboratory = instance.get("laboratory_id", len(instance.get("patients", [])) + 1)
    nodes = list(route)
    if nodes and nodes[0] == depot:
        nodes = nodes[1:]
    if nodes and nodes[-1] == laboratory:
        nodes = nodes[:-1]
    return nodes


def _route_is_explicit(route: Sequence[Any], instance: Mapping[str, Any]) -> bool:
    depot = instance.get("depot_id", 0)
    laboratory = instance.get("laboratory_id", len(instance.get("patients", [])) + 1)
    return depot in route or laboratory in route


def _earliest_window_start(patient: Mapping[str, Any], assignment: Mapping[str, Any]) -> float | None:
    windows = patient.get("time_windows")
    if windows:
        starts = []
        for window in windows:
            if isinstance(window, Mapping):
                value = window.get("start", window.get("ready_time"))
            else:
                try:
                    value = window[0]
                except (TypeError, IndexError):
                    continue
            number = _number(value)
            if number is not None:
                starts.append(number)
        return min(starts) if starts else None
    selected = assignment.get("time_window")
    if selected is not None:
        if isinstance(selected, Mapping):
            return _number(selected.get("start", selected.get("ready_time")))
        try:
            return _number(selected[0])
        except (TypeError, IndexError):
            return None
    return _number(patient.get("ready_time"))


def assignment_to_routes(
    assignment: Any, instance: Mapping[str, Any]
) -> tuple[dict[Any, list[Any]], list[Any]]:
    """Convert assignment rows to per-caregiver ordered patient-ID routes.

    Returns (routes, unassigned_patient_ids). The repository currently has no
    permutation/chromosome field or decoder, so candidates use the documented
    fallback: patients with time windows are sorted by earliest window start
    (stable for ties), patients without windows follow in their existing relative
    order, and assignment order is preserved when no assigned patient has a
    window. Unassigned IDs are reported in instance order and excluded from routes.
    """
    _, rows = _assignment_data(assignment)
    _, patients = _records(instance, "patients")
    _, caregivers = _records(instance, "caregivers")
    routes: dict[Any, list[Any]] = {caregiver_id: [] for caregiver_id in caregivers}
    original_orders: dict[Any, list[tuple[int, Any, float | None]]] = {}
    assigned_ids: set[Any] = set()
    for position, row in enumerate(rows):
        patient_id = row.get("patient_id")
        caregiver_id = row.get("caregiver_id")
        if patient_id is None or caregiver_id is None:
            continue
        routes.setdefault(caregiver_id, [])
        original_orders.setdefault(caregiver_id, []).append(
            (position, patient_id, _earliest_window_start(patients.get(patient_id, {}), row))
        )
        assigned_ids.add(patient_id)

    for caregiver_id, entries in original_orders.items():
        if any(start is not None for _, _, start in entries):
            entries.sort(key=lambda entry: (
                entry[2] is None,
                entry[2] if entry[2] is not None else 0.0,
                entry[0],
            ))
        routes[caregiver_id] = [patient_id for _, patient_id, _ in entries]

    patient_rows = instance.get("patients", [])
    unassigned = [
        patient["id"] for patient in patient_rows
        if isinstance(patient, Mapping) and patient.get("id") not in assigned_ids
    ]
    # Include explicit assignment rows with a patient but no caregiver.
    unassigned.extend(
        row["patient_id"] for row in rows
        if row.get("patient_id") is not None and row.get("caregiver_id") is None
        and row["patient_id"] not in unassigned
    )
    return routes, unassigned


def _records(instance: Mapping[str, Any], key: str) -> tuple[bool, dict[Any, Mapping[str, Any]]]:
    rows = instance.get(key)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False, {}
    return True, {row.get("id"): row for row in rows if isinstance(row, Mapping) and "id" in row}


def _result(status: str, message: str) -> tuple[str, str]:
    return status, message


def _travel(instance: Mapping[str, Any], i: Any, j: Any) -> float | None:
    matrix = instance.get("travel_times")
    if not isinstance(matrix, Mapping):
        return None
    value = matrix.get((i, j))
    if value is None:
        value = matrix.get(f"{i},{j}")
    if value is None and isinstance(matrix.get(i), Mapping):
        value = matrix[i].get(j)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _time_factor(instance: Mapping[str, Any]) -> float:
    return 60.0 if instance.get("time_unit") == "hours" else 1.0


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _service_time(patient: Mapping[str, Any], assignment: Mapping[str, Any]) -> float | None:
    for value in (assignment.get("service_minutes"), patient.get("service_minutes")):
        if value is not None:
            return _number(value)
    for value in (assignment.get("service_hours"), patient.get("service_hours")):
        if value is not None:
            number = _number(value)
            return None if number is None else number * 60
    for value in (assignment.get("service_duration"), patient.get("service_duration")):
        if value is not None:
            return _number(value)
    return None


def _route_minutes(solution: Any, instance: Mapping[str, Any], caregiver_id: Any) -> float | None:
    meta = _metadata(solution)
    departures, returns = meta.get("departure_times"), meta.get("return_times")
    if not isinstance(departures, Mapping) or not isinstance(returns, Mapping):
        return None
    if caregiver_id not in departures or caregiver_id not in returns:
        return None
    departure, returned = _number(departures[caregiver_id]), _number(returns[caregiver_id])
    if departure is None or returned is None:
        return None
    duration = (returned - departure) * _time_factor(instance)
    return duration if duration >= 0 else None


def check_5(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "caregiver routes are missing")
    counts: Counter[Any] = Counter()
    for route in routes.values():
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            return _result(UNVERIFIABLE, "a caregiver route is missing or malformed")
        counts.update(_route_patients(route, instance))
    if any(count > 1 for count in counts.values()):
        return _result(VIOLATED, "a patient is visited more than once")
    return _result(SATISFIED, "each patient is visited at most once across routes")


def check_6(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    present, assignments = _assignment_data(solution)
    routes = _routes(solution)
    if not present or routes is None:
        return _result(UNVERIFIABLE, "assignments or caregiver routes are missing")
    if any("patient_id" not in row or "caregiver_id" not in row for row in assignments):
        return _result(UNVERIFIABLE, "assignment patient/caregiver IDs are missing")
    assigned = Counter((row["caregiver_id"], row["patient_id"]) for row in assignments)
    visited: Counter[tuple[Any, Any]] = Counter()
    for caregiver_id, route in routes.items():
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            return _result(UNVERIFIABLE, "a caregiver route is missing or malformed")
        visited.update((caregiver_id, patient_id) for patient_id in _route_patients(route, instance))
    if assigned != visited:
        return _result(VIOLATED, "route visits and caregiver assignments fail flow conservation")
    return _result(SATISFIED, "route visits match assigned patients and caregivers")


def check_7(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "caregiver routes are missing")
    for route in routes.values():
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            return _result(UNVERIFIABLE, "a caregiver route is missing or malformed")
        if _route_is_explicit(route, instance) and (
            not route or route[0] != instance.get("depot_id", 0)
        ):
            return _result(VIOLATED, "an explicit route does not start at the depot")
    return _result(SATISFIED, "all represented routes start at or imply the depot")


def check_8(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "caregiver routes are missing")
    laboratory = instance.get("laboratory_id", len(instance.get("patients", [])) + 1)
    for route in routes.values():
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            return _result(UNVERIFIABLE, "a caregiver route is missing or malformed")
        if _route_is_explicit(route, instance) and (not route or route[-1] != laboratory):
            return _result(VIOLATED, "an explicit route does not end at the laboratory")
    return _result(SATISFIED, "all represented routes end at or imply the laboratory")


def check_9(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    present, rows = _assignment_data(solution)
    has_patients, patients = _records(instance, "patients")
    if routes is None or not present or not has_patients:
        return _result(UNVERIFIABLE, "routes, assignments, or patient records are missing")
    assignments = {(row.get("caregiver_id"), row.get("patient_id")): row for row in rows}
    meta = _metadata(solution)
    departures, returns = meta.get("departure_times"), meta.get("return_times")
    factor = _time_factor(instance)
    depot = instance.get("depot_id", 0)
    laboratory = instance.get("laboratory_id", len(patients) + 1)
    for caregiver_id, route in routes.items():
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            return _result(UNVERIFIABLE, "an ordered route is missing")
        for left, right in zip(route, route[1:]):
            travel = _travel(instance, left, right)
            if travel is None:
                return _result(UNVERIFIABLE, f"travel time for arc {left!r}->{right!r} is missing")
            if left == depot:
                if not isinstance(departures, Mapping) or caregiver_id not in departures:
                    return _result(UNVERIFIABLE, "depot departure time is missing")
                departure = _number(departures[caregiver_id])
                if departure is None:
                    return _result(UNVERIFIABLE, "depot departure time is not numeric")
                earliest = departure * factor + travel * factor
            else:
                previous = assignments.get((caregiver_id, left))
                patient = patients.get(left)
                if previous is None or patient is None:
                    return _result(UNVERIFIABLE, f"predecessor data for node {left!r} is missing")
                if previous.get("service_start") is None:
                    return _result(UNVERIFIABLE, f"service start for predecessor {left!r} is missing")
                service = _service_time(patient, previous)
                start = _number(previous["service_start"])
                if service is None or start is None:
                    return _result(UNVERIFIABLE, f"service time for predecessor {left!r} is missing")
                earliest = start * factor + service + travel * factor
            if right == laboratory:
                if not isinstance(returns, Mapping) or caregiver_id not in returns:
                    return _result(UNVERIFIABLE, "laboratory return time is missing")
                actual = _number(returns[caregiver_id])
                if actual is None:
                    return _result(UNVERIFIABLE, "laboratory return time is not numeric")
                actual *= factor
            else:
                following = assignments.get((caregiver_id, right))
                if following is None or following.get("service_start") is None:
                    return _result(UNVERIFIABLE, f"service start for successor {right!r} is missing")
                actual = _number(following["service_start"])
                if actual is None:
                    return _result(UNVERIFIABLE, f"service start for successor {right!r} is not numeric")
                actual *= factor
            if actual + 1e-9 < earliest:
                return _result(VIOLATED, f"time propagation fails on arc {left!r}->{right!r}")
    return _result(SATISFIED, "all represented route arcs satisfy time propagation")


def check_11(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "caregiver routes are missing")
    seen: set[Any] = set()
    for route in routes.values():
        if not isinstance(route, Sequence) or isinstance(route, (str, bytes)):
            return _result(UNVERIFIABLE, "a caregiver route is missing or malformed")
        nodes = _route_patients(route, instance)
        if len(nodes) != len(set(nodes)):
            return _result(VIOLATED, "a repeated patient node forms a subtour")
        if any(node in seen for node in nodes):
            return _result(VIOLATED, "a patient node occurs in multiple routes")
        seen.update(nodes)
    return _result(SATISFIED, "represented caregiver routes contain no patient subtour")


def check_12(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    present, assignments = _assignment_data(solution)
    has_patients, patients = _records(instance, "patients")
    if not present or not has_patients:
        return _result(UNVERIFIABLE, "assignments or patient records are missing")
    for row in assignments:
        patient = patients.get(row.get("patient_id"))
        if patient is None:
            return _result(UNVERIFIABLE, "assigned patient record is missing")
        windows = patient.get("time_windows")
        if not windows:
            if patient.get("ready_time") is None or patient.get("due_date") is None:
                windows = [(0.0, float("inf"))]
            else:
                windows = [(patient["ready_time"], patient["due_date"])]
        index, selected = row.get("window_index"), row.get("time_window")
        if index is not None:
            if not isinstance(index, int) or not 0 <= index < len(windows):
                return _result(VIOLATED, "selected time-window index is invalid")
            raw = windows[index]
        elif selected is not None:
            raw = selected
            if isinstance(raw, Mapping):
                raw = (raw.get("start", raw.get("ready_time")), raw.get("end", raw.get("due_date")))
            try:
                if tuple(raw) not in [tuple(w.items()) if isinstance(w, Mapping) else tuple(w) for w in windows]:
                    return _result(VIOLATED, "selected window is not one of the patient's windows")
            except (TypeError, ValueError):
                return _result(VIOLATED, "selected time window is malformed")
        else:
            return _result(UNVERIFIABLE, "selected time window is missing")
        if row.get("service_start") is None:
            return _result(UNVERIFIABLE, "service start is missing")
        if isinstance(raw, Mapping):
            start, end = raw.get("start", raw.get("ready_time")), raw.get("end", raw.get("due_date"))
        else:
            try:
                start, end = raw
            except (TypeError, ValueError):
                return _result(UNVERIFIABLE, "time-window bounds are missing")
        service_start = _number(row.get("service_start"))
        start, end = _number(start), _number(end)
        if service_start is None or start is None or end is None:
            return _result(UNVERIFIABLE, "service start or window bounds are not numeric")
        if not start <= service_start <= end:
            return _result(VIOLATED, "service start lies outside the selected time window")
    return _result(SATISFIED, "each represented service starts inside its selected window")


def check_13(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "route sequences are missing")
    if instance.get("T") is None:
        return _result(UNVERIFIABLE, "daily working-time limit T is missing")
    limit = _number(instance["T"])
    if limit is None:
        return _result(UNVERIFIABLE, "daily working-time limit T is not numeric")
    limit *= _time_factor(instance)
    for caregiver_id in routes:
        duration = _route_minutes(solution, instance, caregiver_id)
        if duration is None:
            return _result(UNVERIFIABLE, f"working duration for caregiver {caregiver_id!r} is missing")
        if duration > limit + 1e-9:
            return _result(VIOLATED, f"caregiver {caregiver_id!r} works beyond T")
    return _result(SATISFIED, "represented route durations do not exceed T")


def check_14(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "route sequences are missing")
    closing = _number(instance.get("lab_closing_time"))
    if closing is None:
        return _result(UNVERIFIABLE, "laboratory closing time is missing or invalid")
    factor = _time_factor(instance)
    returns = _metadata(solution).get("return_times")
    if not isinstance(returns, Mapping):
        return _result(UNVERIFIABLE, "laboratory return times are missing")
    for caregiver_id in routes:
        if caregiver_id not in returns:
            return _result(UNVERIFIABLE, f"return time for caregiver {caregiver_id!r} is missing")
        returned = _number(returns[caregiver_id])
        if returned is None:
            return _result(UNVERIFIABLE, f"return time for caregiver {caregiver_id!r} is invalid")
        if returned * factor > closing * factor + 1e-9:
            return _result(VIOLATED, f"caregiver {caregiver_id!r} returns after laboratory closing")
    return _result(SATISFIED, "all represented routes return before laboratory closing")


def check_15(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    present, assignments = _assignment_data(solution)
    has_patients, patients = _records(instance, "patients")
    has_caregivers, caregivers = _records(instance, "caregivers")
    if not present or not has_patients or not has_caregivers:
        return _result(UNVERIFIABLE, "assignments or patient/caregiver records are missing")
    for row in assignments:
        patient = patients.get(row.get("patient_id"))
        caregiver = caregivers.get(row.get("caregiver_id"))
        if patient is None or caregiver is None:
            return _result(UNVERIFIABLE, "assigned patient or caregiver record is missing")
        demand = _number(patient.get("required_skill_level", patient.get("demand_level")))
        skill = _number(caregiver.get("skill_level", caregiver.get("skill")))
        if demand is None or skill is None:
            return _result(UNVERIFIABLE, "patient demand or caregiver skill is missing/non-numeric")
        if skill < demand:
            return _result(VIOLATED, "assigned caregiver skill is below patient demand")
    return _result(SATISFIED, "all represented assignments meet the skill requirement")


def check_16(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "route sequences are missing")
    departures = _metadata(solution).get("departure_times")
    if not isinstance(departures, Mapping):
        return _result(UNVERIFIABLE, "depot departure times are missing")
    for caregiver_id in routes:
        if caregiver_id not in departures:
            return _result(UNVERIFIABLE, f"depot departure for caregiver {caregiver_id!r} is missing")
        departure = _number(departures[caregiver_id])
        if departure is None:
            return _result(UNVERIFIABLE, f"depot departure for caregiver {caregiver_id!r} is invalid")
        if departure < 0:
            return _result(VIOLATED, f"depot departure for caregiver {caregiver_id!r} is negative")
    return _result(SATISFIED, "all represented caregivers have a valid depot departure")


def _check_domain(solution: Any, equation: int) -> tuple[str, str]:
    domains = _metadata(solution).get("variable_domains")
    if not isinstance(domains, Mapping):
        return _result(UNVERIFIABLE, "variable-domain declarations are missing")
    declaration = domains.get(equation, domains.get(str(equation)))
    if not isinstance(declaration, Sequence) or isinstance(declaration, (str, bytes)) or not declaration:
        return _result(UNVERIFIABLE, "variable-domain declarations for this equation are missing")
    for variable in declaration:
        if not isinstance(variable, Mapping):
            return _result(UNVERIFIABLE, "explicit binary/integer variable metadata is missing")
        kind, values = variable.get("kind"), variable.get("values")
        if values is None:
            return _result(UNVERIFIABLE, "decision-variable values are missing")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            values = [values]
        if kind == "binary":
            if any(value not in (0, 1, False, True) for value in values):
                return _result(VIOLATED, "a binary decision variable is outside {0, 1}")
        elif kind == "integer":
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                return _result(VIOLATED, "an integer decision variable has a non-integer value")
        else:
            return _result(UNVERIFIABLE, "binary/integer variable kind is not declared")
    return _result(SATISFIED, "declared decision variables satisfy their binary/integer domains")


def check_17(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    return _check_domain(solution, 17)


def check_18(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    return _check_domain(solution, 18)


def check_19(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    return _check_domain(solution, 19)


def check_20(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    return _check_domain(solution, 20)


def check_21(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    return _check_domain(solution, 21)


def check_overtime_240(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    routes = _routes(solution)
    if routes is None:
        return _result(UNVERIFIABLE, "route sequences are missing")
    rated_time = _number(instance.get("T", 480))
    if rated_time is None:
        return _result(UNVERIFIABLE, "rated working time is missing or invalid")
    rated_minutes = rated_time * _time_factor(instance)
    for caregiver_id in routes:
        duration = _route_minutes(solution, instance, caregiver_id)
        if duration is None:
            return _result(UNVERIFIABLE, f"working duration for caregiver {caregiver_id!r} is missing")
        if max(0.0, duration - rated_minutes) > 240.0 + 1e-9:
            return _result(VIOLATED, f"caregiver {caregiver_id!r} exceeds the 240 minute overtime cap")
    return _result(SATISFIED, "represented overtime does not exceed 240 minutes")


def check_one_window_per_patient_per_day(solution: Any, instance: Mapping[str, Any]) -> tuple[str, str]:
    present, assignments = _assignment_data(solution)
    has_patients, patients = _records(instance, "patients")
    if not present or not has_patients:
        return _result(UNVERIFIABLE, "assignments or patient records are missing")
    selected: set[tuple[Any, Any]] = set()
    for row in assignments:
        patient_id = row.get("patient_id")
        patient = patients.get(patient_id)
        if patient is None:
            return _result(UNVERIFIABLE, "assigned patient record is missing")
        windows = patient.get("time_windows")
        if not windows:
            windows = [(patient.get("ready_time", 0), patient.get("due_date", float("inf")))]
        index, window = row.get("window_index"), row.get("time_window")
        if index is not None:
            if not isinstance(index, int) or not 0 <= index < len(windows):
                return _result(VIOLATED, "selected time-window index is invalid")
            choice = index
        elif window is not None:
            try:
                choice = tuple(window.items()) if isinstance(window, Mapping) else tuple(window)
            except TypeError:
                return _result(VIOLATED, "selected time window is malformed")
        else:
            return _result(UNVERIFIABLE, "selected time window is missing")
        patient_day = (patient_id, row.get("day", 0))
        if patient_day in selected:
            return _result(VIOLATED, "patient has more than one selected window on the same day")
        selected.add(patient_day)
    return _result(SATISFIED, "each represented patient has at most one window selection per day")


CHECKS = (
    ("(5)", check_5),
    ("(6)", check_6),
    ("(7)", check_7),
    ("(8)", check_8),
    ("(9)", check_9),
    ("(11)", check_11),
    ("(12)", check_12),
    ("(13)", check_13),
    ("(14)", check_14),
    ("(15)", check_15),
    ("(16)", check_16),
    ("(17)", check_17),
    ("(18)", check_18),
    ("(19)", check_19),
    ("(20)", check_20),
    ("(21)", check_21),
    ("overtime_240", check_overtime_240),
    ("time_window_once", check_one_window_per_patient_per_day),
)


def validate_solution(solution: Any, instance: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return one (constraint_id, status, message) row per check.

    Status is SATISFIED when data is present and the rule holds, VIOLATED when
    data is present and the rule fails, and UNVERIFIABLE when required candidate
    data is absent. Therefore a missing representation is never a violation.
    """
    return [
        (constraint_id, *check(solution, instance))
        for constraint_id, check in CHECKS
    ]
