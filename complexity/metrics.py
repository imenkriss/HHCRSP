"""Mesures de performance et indicateurs métier du pipeline HHCOP."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import numpy as np

try:
    from pymoo.indicators.hv import HyperVolume
except Exception:  # pragma: no cover - fallback when pymoo is unavailable
    HyperVolume = None


def hypervolume(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    """Hypervolume dominated by ``points`` with respect to ``reference``."""
    if not points:
        return 0.0

    ref = tuple(float(value) for value in reference)
    clean: list[tuple[float, ...]] = []
    for point in points:
        candidate = tuple(float(value) for value in point)
        if len(candidate) != len(ref):
            raise ValueError("point dimension does not match reference dimension")
        if all(candidate[i] < ref[i] for i in range(len(ref))):
            clean.append(candidate)
    if not clean:
        return 0.0

    if HyperVolume is not None:
        try:
            solution = np.asarray(clean, dtype=float)
            return float(HyperVolume(ref_point=np.asarray(ref, dtype=float)).compute(solution))
        except Exception:  # pragma: no cover - fallback for API drift
            pass

    dims = len(ref)

    if dims == 1:
        return max(0.0, ref[0] - max(point[0] for point in clean))

    if dims == 2:
        x_values = sorted({point[0] for point in clean} | {ref[0]})
        total = 0.0
        for index, x_value in enumerate(x_values[:-1]):
            next_x = x_values[index + 1]
            height = max((ref[1] - point[1]) for point in clean if point[0] <= x_value)
            total += max(0.0, next_x - x_value) * max(0.0, height)
        return float(total)

    if dims == 3:
        x_values = sorted({point[0] for point in clean} | {ref[0]})
        total = 0.0
        for index, x_value in enumerate(x_values[:-1]):
            next_x = x_values[index + 1]
            active = [point[1:] for point in clean if point[0] <= x_value]
            if not active:
                continue
            yz_area = hypervolume([(tuple(item)) for item in active], ref[1:])
            total += max(0.0, next_x - x_value) * yz_area
        return float(total)

    return 0.0


def nadir_reference(points: list[tuple[float, ...]], margin: float = 1.0) -> list[float]:
    """Reference point used for normalisation / ideal-point comparisons."""
    if not points:
        return []
    dimensions = len(points[0])
    return [max(point[idx] for point in points) + margin for idx in range(dimensions)]


class ComplexityMetrics:
    """Collecte des mesures dans un format stable pour le terminal et Streamlit."""

    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, Any]] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = round(time.perf_counter() - start, 6)
            self.metrics[name] = {"execution_time_sec": float(duration)}

    def start(self) -> float:
        """Retourne un repère temporel à transmettre à :meth:`record`."""
        return time.perf_counter()

    def record(self, component: str, started_at: float) -> float:
        """Enregistre et retourne la durée d'exécution d'un composant, en secondes."""
        duration = round(time.perf_counter() - started_at, 6)
        self.metrics[component] = {"execution_time_sec": duration}
        return duration

    def add_stage_time(self, name: str, seconds: float) -> None:
        self.metrics[name] = {"execution_time_sec": float(seconds)}

    def total_time(self) -> float:
        return sum(
            float(stage.get("execution_time_sec", 0.0))
            for stage in self.metrics.values()
            if isinstance(stage, dict) and "execution_time_sec" in stage
        )

    def problem_size(self, patients: list[dict], caregivers: list[dict]) -> None:
        patient_count = len(patients)
        caregiver_count = len(caregivers)
        self.metrics["problem_size"] = {
            "patients": patient_count,
            "caregivers": caregiver_count,
            "possible_assignments": patient_count * caregiver_count,
            "search_space_complexity": "O(P × C)",
        }

    def agent_interactions(
        self, patients: list[dict], caregivers: list[dict], observed_count: int | None = None
    ) -> None:
        patient_count = len(patients)
        caregiver_count = len(caregivers)
        self.metrics["agent_interactions"] = {
            "patient_agents": patient_count,
            "caregiver_agents": caregiver_count,
            "possible_interactions": patient_count * caregiver_count,
            "observed_messages": observed_count if observed_count is not None else 0,
            "complexity": "O(P × C)",
        }

    def caregiver_delay(self, caregivers: list[dict]) -> None:
        delays = [float(caregiver.get("delay", 0) or 0) for caregiver in caregivers]
        total = sum(delays)
        self.metrics["caregiver_delay"] = {
            "total_minutes": round(total, 2),
            "average_minutes": round(total / len(delays), 2) if delays else 0,
            "delayed_caregivers": sum(delay > 0 for delay in delays),
        }

    def workload_balance(self, caregivers: list[dict], assignments: list[dict] | None = None) -> None:
        """Mesure la variance des heures après les affectations, si elles sont fournies."""
        assigned_hours: dict[str, float] = {}
        for assignment in assignments or []:
            caregiver_id = assignment.get("caregiver_id")
            if caregiver_id:
                assigned_hours[caregiver_id] = assigned_hours.get(caregiver_id, 0) + float(
                    assignment.get("service_hours", 1) or 0
                )
        workloads = [
            float(caregiver.get("current_workload", 0) or 0) + assigned_hours.get(caregiver.get("id"), 0)
            for caregiver in caregivers
        ]
        average = sum(workloads) / len(workloads) if workloads else 0
        variance = sum((workload - average) ** 2 for workload in workloads) / len(workloads) if workloads else 0
        self.metrics["workload_balance"] = {
            "average_workload_hours": round(average, 2),
            "workload_variance_hours": round(variance, 4),
            "max_difference_hours": round(max(workloads) - min(workloads), 2) if workloads else 0,
        }

    def unassigned_patients(self, patients: list[dict], assignments: list[dict]) -> None:
        assigned = {assignment.get("patient_id") for assignment in assignments}
        count = sum(patient.get("id") not in assigned for patient in patients)
        self.metrics["unassigned_patients"] = {
            "count": count,
            "percentage": round(100 * count / len(patients), 2) if patients else 0,
            "served_count": len(patients) - count,
        }

    def patient_satisfaction(self, patients: list[dict], caregivers: list[dict], assignments: list[dict]) -> None:
        caregivers_by_id = {caregiver.get("id"): caregiver for caregiver in caregivers}
        assignments_by_patient = {assignment.get("patient_id"): assignment for assignment in assignments}
        scores: list[float] = []
        for patient in patients:
            assignment = assignments_by_patient.get(patient.get("id"))
            if not assignment:
                scores.append(0)
                continue
            caregiver = caregivers_by_id.get(assignment.get("caregiver_id"), {})
            score = 0.0
            if caregiver.get("id") == patient.get("preferred_caregiver"):
                score += 40
            score += min(5.0, max(0.0, float(caregiver.get("skill_level", 1) or 0))) * 12
            scores.append(score)
        served_scores = [score for score in scores if score]
        self.metrics["patient_satisfaction"] = {
            "score_percent": round(sum(scores) / len(scores), 2) if scores else 0,
            "served_score_percent": round(sum(served_scores) / len(served_scores), 2) if served_scores else 0,
        }

    def optimization_indicators(self, solution: dict | None) -> None:
        self.metrics["optimization"] = {"status": "no_solution"} if not solution else {
            "status": "ok", **solution.get("summary", {})
        }

    def uncertainty_indicators(self, llm_output: dict, rag_context: list[str]) -> None:
        """Expose les signaux d'incertitude observables sans inventer une probabilité."""
        missing_fields = sum(
            llm_output.get(field) is None or llm_output.get(field) == ""
            for field in ("patient_id", "urgency", "delayed")
        )
        signal_count = 3
        if not rag_context:
            confidence_score = 0.0
            uncertainty_score = 1.0
            status = "low_evidence"
        else:
            evidence_score = min(1.0, len(rag_context) / 3)
            completeness_score = 1 - (missing_fields / signal_count)
            confidence_score = round((evidence_score + completeness_score) / 2, 2)
            uncertainty_score = round(1 - confidence_score, 2)
            status = "low_evidence" if confidence_score < 0.5 else "evidence_available"

        self.metrics["uncertainty"] = {
            "confidence_score": confidence_score,
            "uncertainty_score": uncertainty_score,
            "rag_documents": len(rag_context),
            "missing_signals": missing_fields,
            "status": status,
        }

    def delays_and_overtime(self, routes: list[Any], caregivers: list[dict]) -> None:
        overtime_minutes = 0.0
        tardiness_minutes = 0.0
        waiting_minutes = 0.0
        delay_total = 0.0
        for route in routes:
            route_working = float(getattr(route, "working_minutes", 0.0) or 0.0)
            caregiver = getattr(route, "caregiver", None)
            rated_minutes = float(getattr(caregiver, "rated_minutes", 0.0) or 0.0)
            overtime_minutes += max(0.0, route_working - rated_minutes)
            tardiness_minutes += float(getattr(route, "tardiness_total", 0.0) or 0.0)
            waiting_minutes += float(getattr(route, "waiting_total", 0.0) or 0.0)
        for caregiver in caregivers:
            delay_total += float(caregiver.get("delay", 0) or 0)
        self.metrics["delays_and_overtime"] = {
            "routes_considered": len(routes),
            "overtime_minutes": round(overtime_minutes, 2),
            "tardiness_minutes": round(tardiness_minutes, 2),
            "waiting_minutes": round(waiting_minutes, 2),
            "caregiver_delay_total_minutes": round(delay_total, 2),
        }

    def pareto_quality(self, points: list[tuple[float, ...]], reference: tuple[float, ...], feasible_count: int | None = None) -> None:
        self.metrics["pareto_quality"] = {
            "non_dominated_solutions": len(points),
            "feasible_solutions": feasible_count if feasible_count is not None else len(points),
            "hypervolume": round(float(hypervolume(points, reference)), 6),
        }

    def balance_from_schedule(self, workload_variance_hours: float, route_workloads: list[float] | None = None) -> None:
        self.metrics["workload_balance"] = {
            "workload_variance_hours": float(workload_variance_hours),
            "route_workloads": list(route_workloads) if route_workloads is not None else [],
        }

    def satisfaction_from_schedule(self, satisfaction_total: float, served: int = 0, total: int = 0) -> None:
        pct = 0.0
        if total:
            pct = (served / total) * 100.0
        self.metrics["patient_satisfaction"] = {
            "served_patients": served,
            "total_patients": total,
            "percentage": float(pct),
            "satisfaction_total": float(satisfaction_total),
        }

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        return self.metrics.copy()
