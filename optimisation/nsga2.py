"""Optimisation multi-objectifs des affectations HHCOP par front de Pareto."""

from __future__ import annotations

import random
from typing import Any

from agents.compatibilite import caregiver_can_visit


class NSGA2:
    """Optimise les coûts, la prise en charge, la satisfaction et la charge.

    Les champs de coût, de trajet et de durée sont optionnels : les CSV actuels
    restent donc valides, tout en permettant d'enrichir progressivement le modèle.
    """

    def __init__(self, patients: list[dict], caregivers: list[dict], population_size: int = 20, generations: int = 10, seed: int | None = 42) -> None:
        self.patients = patients
        self.caregivers = caregivers
        self.population_size = max(1, population_size)
        self.generations = max(0, generations)
        self.random = random.Random(seed)
        self.patients_by_id = {patient["id"]: patient for patient in patients}
        self.caregivers_by_id = {caregiver["id"]: caregiver for caregiver in caregivers}

    def is_compatible(self, patient: dict, caregiver: dict) -> bool:
        return caregiver_can_visit(caregiver, patient.get("care_type", ""))

    @staticmethod
    def _service_hours(patient: dict) -> float:
        return float(patient.get("service_hours", patient.get("service_duration", 1)) or 1)

    @staticmethod
    def _skill_level(caregiver: dict) -> float:
        """Retourne le niveau de compétence, avec 1 comme valeur historique."""
        return max(0.0, float(caregiver.get("skill_level", 1) or 0))

    @classmethod
    def _patient_satisfaction(cls, patient: dict, caregiver: dict) -> float:
        """Calcule la satisfaction (préférence 40 %, compétence 60 %)."""
        preference_score = 40.0 if patient.get("preferred_caregiver") == caregiver["id"] else 0.0
        skill_score = min(cls._skill_level(caregiver), 5.0) * 12.0
        return preference_score + skill_score

    def create_solution(self) -> list[dict]:
        """Construit une affectation faisable (qualification et capacité respectées)."""
        solution: list[dict] = []
        added_hours = {caregiver["id"]: 0.0 for caregiver in self.caregivers}
        patients = sorted(self.patients, key=lambda patient: {"Critical": 0, "High": 1, "Medium": 2}.get(patient.get("priority"), 3))
        for patient in patients:
            hours = self._service_hours(patient)
            candidates = [
                caregiver for caregiver in self.caregivers
                if self.is_compatible(patient, caregiver)
                and float(caregiver.get("current_workload", 0) or 0) + added_hours[caregiver["id"]] + hours <= float(caregiver.get("max_work_hours", 0) or 0)
            ]
            if not candidates:
                continue
            caregiver = self.random.choice(candidates)
            added_hours[caregiver["id"]] += hours
            solution.append({"patient_id": patient["id"], "caregiver_id": caregiver["id"], "service_hours": hours, "travel_time": float(patient.get("travel_time", 0) or 0)})
        return solution

    def _summary(self, solution: list[dict]) -> dict[str, float | int]:
        assigned_ids = {assignment["patient_id"] for assignment in solution}
        used_caregivers = {assignment["caregiver_id"] for assignment in solution}
        costs = {key: 0.0 for key in ("service_cost", "waiting_cost", "overtime_cost", "travel_cost", "fixed_cost", "delay_penalty")}
        satisfaction = 0.0
        added_hours = {caregiver["id"]: 0.0 for caregiver in self.caregivers}
        for assignment in solution:
            patient = self.patients_by_id[assignment["patient_id"]]
            caregiver = self.caregivers_by_id[assignment["caregiver_id"]]
            hours, travel = float(assignment["service_hours"]), float(assignment["travel_time"])
            costs["service_cost"] += hours * float(patient.get("service_cost_rate", 50) or 0)
            costs["waiting_cost"] += max(0, float(caregiver.get("delay", 0) or 0) + travel) * float(patient.get("waiting_cost_rate", 0) or 0)
            costs["travel_cost"] += travel * float(patient.get("travel_cost_rate", 0.5) or 0)
            costs["delay_penalty"] += float(caregiver.get("delay", 0) or 0) * float(patient.get("delay_penalty_rate", 2) or 0)
            added_hours[caregiver["id"]] += hours
            satisfaction += self._patient_satisfaction(patient, caregiver)
        workloads = []
        for caregiver in self.caregivers:
            workload = float(caregiver.get("current_workload", 0) or 0) + added_hours[caregiver["id"]]
            workloads.append(workload)
            costs["overtime_cost"] += max(0, workload - float(caregiver.get("max_work_hours", 0) or 0)) * float(caregiver.get("overtime_cost_rate", 30) or 0)
        costs["fixed_cost"] = sum(float(self.caregivers_by_id[caregiver_id].get("fixed_cost", 10) or 0) for caregiver_id in used_caregivers)
        average = sum(workloads) / len(workloads) if workloads else 0
        variance = sum((workload - average) ** 2 for workload in workloads) / len(workloads) if workloads else 0
        served_count = len(assigned_ids)
        summary: dict[str, float | int] = {
            "served_patients": served_count,
            "unassigned_patients": len(self.patients) - served_count,
            # La couverture étant son propre objectif, on mesure ici la qualité
            # des affectations réellement réalisées.
            "satisfaction_percent": round(satisfaction / served_count, 2) if served_count else 0,
            "average_assigned_skill_level": round(
                sum(self._skill_level(self.caregivers_by_id[item["caregiver_id"]]) for item in solution) / served_count,
                2,
            ) if served_count else 0,
            "workload_variance": round(variance, 4),
            **{name: round(value, 2) for name, value in costs.items()},
        }
        summary["total_cost"] = round(sum(costs.values()), 2)
        return summary

    def evaluate(self, solution: list[dict]) -> tuple[list[float], dict[str, Any]]:
        summary = self._summary(solution)
        # NSGA-II minimise chaque objectif; ils restent séparés afin qu'aucune
        # pénalité arbitraire ne masque la couverture ou la satisfaction.
        return [
            summary["total_cost"],
            float(summary["unassigned_patients"]),
            -float(summary["satisfaction_percent"]),
            float(summary["workload_variance"]),
        ], summary

    @staticmethod
    def dominates(a: list[float], b: list[float]) -> bool:
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def pareto_front(self, population: list[dict]) -> list[dict]:
        front = [
            candidate for candidate in population
            if not any(
                other is not candidate
                and self.dominates(other["objectives"], candidate["objectives"])
                for other in population
            )
        ]
        # Plusieurs tirages peuvent produire exactement la même affectation.
        unique_front = []
        signatures = set()
        for candidate in front:
            signature = tuple(sorted(
                (assignment["patient_id"], assignment["caregiver_id"])
                for assignment in candidate["solution"]
            ))
            if signature not in signatures:
                signatures.add(signature)
                unique_front.append(candidate)
        return unique_front

    def optimize(self) -> list[dict]:
        population = []
        for _ in range(self.population_size):
            solution = self.create_solution()
            objectives, summary = self.evaluate(solution)
            population.append({"solution": solution, "objectives": objectives, "summary": summary})
        for _ in range(self.generations):
            offspring = []
            for _ in range(self.population_size):
                solution = self.create_solution()
                objectives, summary = self.evaluate(solution)
                offspring.append({"solution": solution, "objectives": objectives, "summary": summary})
            population = self.pareto_front(population + offspring)
            population.sort(key=lambda candidate: tuple(candidate["objectives"]))
            population = population[:self.population_size]
        return self.pareto_front(population)

    @staticmethod
    def select_compromise(pareto_solutions: list[dict]) -> dict | None:
        """Sélection : couverture, satisfaction, coût, puis charge."""
        if not pareto_solutions:
            return None
        return min(pareto_solutions, key=lambda candidate: (candidate["summary"]["unassigned_patients"], -candidate["summary"]["satisfaction_percent"], candidate["summary"]["total_cost"], candidate["summary"]["workload_variance"]))

    def complexity_analysis(self) -> dict[str, Any]:
        return {
            "patients": len(self.patients), "caregivers": len(self.caregivers),
            "population_size": self.population_size, "generations": self.generations,
            "solution_evaluation_complexity": "O(P + C)",
            "pareto_front_complexity": "O(N²)",
            "total_complexity": "O(G × N² × (P + C))",
            "evaluations": self.population_size * (self.generations + 1),
        }
