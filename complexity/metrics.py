"""Mesures de performance et indicateurs métier du pipeline HHCOP."""

from __future__ import annotations

import time
from typing import Any


class ComplexityMetrics:
    """Collecte des mesures dans un format stable pour le terminal et Streamlit."""

    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, Any]] = {}

    def start(self) -> float:
        """Retourne un repère temporel à transmettre à :meth:`record`."""
        return time.perf_counter()

    def record(self, component: str, started_at: float) -> float:
        """Enregistre et retourne la durée d'exécution d'un composant, en secondes."""
        duration = round(time.perf_counter() - started_at, 6)
        self.metrics[component] = {"execution_time_sec": duration}
        return duration

    def problem_size(self, patients: list[dict], caregivers: list[dict]) -> None:
        patient_count = len(patients)
        caregiver_count = len(caregivers)
        self.metrics["problem_size"] = {
            "patients": patient_count,
            "caregivers": caregiver_count,
            "possible_assignments": patient_count * caregiver_count,
            "search_space_complexity": "O(P × C)",
        }

    def agent_interactions(self, patients: list[dict], caregivers: list[dict]) -> None:
        patient_count = len(patients)
        caregiver_count = len(caregivers)
        self.metrics["agent_interactions"] = {
            "patient_agents": patient_count,
            "caregiver_agents": caregiver_count,
            "possible_interactions": patient_count * caregiver_count,
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
            float(caregiver.get("current_workload", 0) or 0) + assigned_hours.get(caregiver_id, 0)
            for caregiver in caregivers
            for caregiver_id in [caregiver.get("id")]
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
            score = 70.0
            if caregiver.get("id") == patient.get("preferred_caregiver"):
                score += 20
            if caregiver.get("skill") == patient.get("care_type"):
                score += 10
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

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        return self.metrics.copy()
