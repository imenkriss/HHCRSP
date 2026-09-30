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

    def __init__(
        self,
        patients: list[dict],
        caregivers: list[dict],
        population_size: int = 20,
        generations: int = 10,
        seed: int | None = 42,
        uncertainty_scenarios: int = 32,
    ) -> None:
        self.patients = patients
        self.caregivers = caregivers
        self.population_size = max(1, population_size)
        self.generations = max(0, generations)
        self.random = random.Random(seed)
        self.uncertainty_scenarios = max(1, uncertainty_scenarios)
        self.patients_by_id = {patient["id"]: patient for patient in patients}
        self.caregivers_by_id = {caregiver["id"]: caregiver for caregiver in caregivers}
        self.scenarios = self._build_scenarios(seed)

    def is_compatible(self, patient: dict, caregiver: dict) -> bool:
        return caregiver_can_visit(
            caregiver,
            patient.get("care_type", ""),
            int(patient.get("required_skill_level", 1) or 1),
        )

    @staticmethod
    def _service_hours(patient: dict) -> float:
        if patient.get("service_hours") is not None:
            return max(0.0, float(patient["service_hours"]))
        duration_minutes = float(patient.get("service_duration", 60) or 60)
        return max(0.0, duration_minutes / 60)

    @staticmethod
    def _range(record: dict, base: float, minimum_key: str, maximum_key: str) -> tuple[float, float]:
        minimum = float(record.get(minimum_key, base))
        maximum = float(record.get(maximum_key, base))
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"Invalid uncertainty range: {minimum_key}/{maximum_key}")
        return minimum, maximum

    @classmethod
    def _service_hours_range(cls, patient: dict) -> tuple[float, float]:
        return cls._range(
            patient,
            cls._service_hours(patient),
            "service_hours_min",
            "service_hours_max",
        )

    @classmethod
    def _travel_time_range(cls, patient: dict) -> tuple[float, float]:
        base = max(0.0, float(patient.get("travel_time", 0) or 0))
        return cls._range(patient, base, "travel_time_min", "travel_time_max")

    def _build_scenarios(self, seed: int | None) -> list[dict[str, dict[str, float]]]:
        ranges = {
            patient["id"]: (
                self._service_hours_range(patient),
                self._travel_time_range(patient),
            )
            for patient in self.patients
        }
        has_uncertainty = any(
            lower != upper
            for service_range, travel_range in ranges.values()
            for lower, upper in (service_range, travel_range)
        )
        scenario_count = self.uncertainty_scenarios if has_uncertainty else 1
        sampler = random.Random(seed)
        return [
            {
                patient_id: {
                    "service_hours": sampler.uniform(*service_range),
                    "travel_time": sampler.uniform(*travel_range),
                }
                for patient_id, (service_range, travel_range) in ranges.items()
            }
            for _ in range(scenario_count)
        ]

    @staticmethod
    def _time_windows(patient: dict) -> list[tuple[float, float]]:
        windows = patient.get("time_windows")
        if not windows:
            if patient.get("ready_time") is None or patient.get("due_date") is None:
                return [(0.0, float("inf"))]
            windows = [(patient["ready_time"], patient["due_date"])]

        normalized = []
        for window in windows:
            if isinstance(window, dict):
                start = window.get("start", window.get("ready_time"))
                end = window.get("end", window.get("due_date"))
            else:
                start, end = window
            start, end = float(start), float(end)
            if start < 0 or end < start:
                raise ValueError(f"Invalid patient time window: {window}")
            normalized.append((start, end))
        return normalized

    @classmethod
    def _feasible_window(cls, patient: dict, service_hours: float) -> tuple[float, float] | None:
        required_minutes = service_hours * 60
        return next(
            (
                window for window in cls._time_windows(patient)
                if window[1] - window[0] >= required_minutes
            ),
            None,
        )

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
        """Construit une affectation respectant les fenêtres et capacités déclarées."""
        solution: list[dict] = []
        added_max_hours = {caregiver["id"]: 0.0 for caregiver in self.caregivers}
        patients = sorted(self.patients, key=lambda patient: {"Critical": 0, "High": 1, "Medium": 2}.get(patient.get("priority"), 3))
        for patient in patients:
            service_min, service_max = self._service_hours_range(patient)
            hours = (service_min + service_max) / 2
            window = self._feasible_window(patient, service_max)
            if window is None:
                continue
            candidates = [
                caregiver for caregiver in self.caregivers
                if self.is_compatible(patient, caregiver)
                and float(caregiver.get("current_workload", 0) or 0)
                + added_max_hours[caregiver["id"]]
                + service_max
                <= float(caregiver.get("max_work_hours", 0) or 0)
                + float(caregiver.get("max_overtime_hours", 2) or 0)
            ]
            if not candidates:
                continue
            caregiver = self.random.choice(candidates)
            added_max_hours[caregiver["id"]] += service_max
            travel_min, travel_max = self._travel_time_range(patient)
            solution.append({
                "patient_id": patient["id"],
                "caregiver_id": caregiver["id"],
                "service_hours": hours,
                "travel_time": (travel_min + travel_max) / 2,
                "time_window": window,
            })
        return solution

    def _summary(self, solution: list[dict]) -> dict[str, float | int]:
        assigned_ids = {assignment["patient_id"] for assignment in solution}
        used_caregivers = {assignment["caregiver_id"] for assignment in solution}
        cost_names = ("service_cost", "waiting_cost", "overtime_cost", "travel_cost", "fixed_cost", "delay_penalty")
        expected_costs = {key: 0.0 for key in cost_names}
        satisfaction = 0.0
        for assignment in solution:
            patient = self.patients_by_id[assignment["patient_id"]]
            caregiver = self.caregivers_by_id[assignment["caregiver_id"]]
            satisfaction += self._patient_satisfaction(patient, caregiver)

        expected_overtime_hours = 0.0
        expected_workload_variance = 0.0
        for scenario in self.scenarios:
            scenario_costs = {key: 0.0 for key in cost_names}
            added_hours = {caregiver["id"]: 0.0 for caregiver in self.caregivers}
            for assignment in solution:
                patient = self.patients_by_id[assignment["patient_id"]]
                caregiver = self.caregivers_by_id[assignment["caregiver_id"]]
                sample = scenario[patient["id"]]
                hours, travel = sample["service_hours"], sample["travel_time"]
                scenario_costs["service_cost"] += hours * float(patient.get("service_cost_rate", 50) or 0)
                scenario_costs["waiting_cost"] += max(0, float(caregiver.get("delay", 0) or 0) + travel) * float(patient.get("waiting_cost_rate", 0) or 0)
                scenario_costs["travel_cost"] += travel * float(patient.get("travel_cost_rate", 0.5) or 0)
                scenario_costs["delay_penalty"] += float(caregiver.get("delay", 0) or 0) * float(patient.get("delay_penalty_rate", 2) or 0)
                added_hours[caregiver["id"]] += hours

            workloads = []
            scenario_overtime_hours = 0.0
            for caregiver in self.caregivers:
                workload = float(caregiver.get("current_workload", 0) or 0) + added_hours[caregiver["id"]]
                workloads.append(workload)
                overtime = max(0, workload - float(caregiver.get("max_work_hours", 0) or 0))
                scenario_overtime_hours += overtime
                scenario_costs["overtime_cost"] += overtime * float(caregiver.get("overtime_cost_rate", 30) or 0)
            scenario_costs["fixed_cost"] = sum(
                float(self.caregivers_by_id[caregiver_id].get("fixed_cost", 10) or 0)
                for caregiver_id in used_caregivers
            )
            average = sum(workloads) / len(workloads) if workloads else 0
            variance = sum((workload - average) ** 2 for workload in workloads) / len(workloads) if workloads else 0
            expected_overtime_hours += scenario_overtime_hours / len(self.scenarios)
            expected_workload_variance += variance / len(self.scenarios)
            for name in cost_names:
                expected_costs[name] += scenario_costs[name] / len(self.scenarios)

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
            "workload_variance": round(expected_workload_variance, 4),
            "expected_overtime_hours": round(expected_overtime_hours, 4),
            "uncertainty_scenarios": len(self.scenarios),
            **{name: round(value, 2) for name, value in expected_costs.items()},
        }
        summary["total_cost"] = round(sum(expected_costs.values()), 2)
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
            "uncertainty_scenarios": len(self.scenarios),
            "solution_evaluation_complexity": "O(S × (P + C))",
            "pareto_front_complexity": "O(N²)",
            "total_complexity": "O(G × N² × S × (P + C))",
            "evaluations": self.population_size * (self.generations + 1),
        }
