from __future__ import annotations

from typing import Any

from agents.compatibilite import caregiver_can_visit

DEFAULT_WEIGHTS = {"cost": 0.5, "satisfaction": 0.25, "balance": 0.25}


class DecisionEngine:
    def __init__(self, weights: dict[str, float] | None = None, method: str = "weighted_sum"):
        self.method = method
        if method not in {"weighted_sum", "ideal_point", "legacy"}:
            raise ValueError(f"Unsupported selection method: {method}")
        base = DEFAULT_WEIGHTS.copy() if weights is None else dict(weights)
        total = sum(float(value) for value in base.values())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = {name: float(base.get(name, 0.0)) / total for name in DEFAULT_WEIGHTS}

    def _candidate_summary(self, candidate: Any) -> tuple[int, float, float, float]:
        if isinstance(candidate, dict):
            summary = candidate.get("summary", {})
            return (
                int(summary.get("unassigned_patients", 0)),
                -float(summary.get("satisfaction_percent", 0.0)),
                float(summary.get("total_cost", 0.0)),
                float(summary.get("workload_variance", 0.0)),
            )
        if hasattr(candidate, "summary"):
            summary = candidate.summary
            return (
                int(summary.get("unassigned_patients", 0)),
                -float(summary.get("satisfaction_percent", 0.0)),
                float(summary.get("total_cost", 0.0)),
                float(summary.get("workload_variance", 0.0)),
            )
        objectives = getattr(candidate, "objectives", (0.0, 0.0, 0.0))
        return (
            0,
            -float(objectives[1]),
            float(objectives[0]),
            float(objectives[2]),
        )

    def choose_pareto_solution(self, pareto_solutions: list[dict]) -> dict | None:
        """Sélectionne une solution du front après l'optimisation.

        La règle est volontairement explicite : couvrir les patients, puis
        maximiser leur satisfaction, réduire les coûts et équilibrer la charge.
        """
        if not pareto_solutions:
            return None
        return min(
            pareto_solutions,
            key=lambda candidate: self._candidate_summary(candidate),
        )

    def _normalise(self, value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 0.0
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    def _select_from_candidates(self, candidates: list[Any]) -> Any:
        if not candidates:
            return None
        if self.method == "weighted_sum":
            candidate_scores = []
            for candidate in candidates:
                if hasattr(candidate, "objectives"):
                    z1, z2, z3 = candidate.objectives
                    objective_values = {"cost": z1, "satisfaction": -z2, "balance": z3}
                elif isinstance(candidate, dict) and "summary" in candidate:
                    summary = candidate["summary"]
                    objective_values = {
                        "cost": float(summary.get("total_cost", 0.0)),
                        "satisfaction": float(summary.get("satisfaction_percent", 0.0)),
                        "balance": float(summary.get("workload_variance", 0.0)),
                    }
                else:
                    continue
                score = sum(self.weights[name] * objective_values.get(name, 0.0) for name in self.weights)
                candidate_scores.append((score, candidate))
            if not candidate_scores:
                return candidates[0]
            _, picked = min(candidate_scores, key=lambda item: item[0])
            return picked

        if self.method == "ideal_point":
            values = []
            for candidate in candidates:
                if hasattr(candidate, "objectives"):
                    z1, z2, z3 = candidate.objectives
                    values.append({"cost": z1, "satisfaction": -z2, "balance": z3, "candidate": candidate})
                elif isinstance(candidate, dict) and "summary" in candidate:
                    summary = candidate["summary"]
                    values.append({
                        "cost": float(summary.get("total_cost", 0.0)),
                        "satisfaction": float(summary.get("satisfaction_percent", 0.0)),
                        "balance": float(summary.get("workload_variance", 0.0)),
                        "candidate": candidate,
                    })
            if not values:
                return candidates[0]
            mins = {name: min(item[name] for item in values) for name in ("cost", "satisfaction", "balance")}
            maxs = {name: max(item[name] for item in values) for name in ("cost", "satisfaction", "balance")}
            best = None
            best_distance = None
            for item in values:
                normalized = {
                    name: self._normalise(item[name], mins[name], maxs[name]) for name in ("cost", "satisfaction", "balance")
                }
                distance = sum(self.weights[name] * normalized[name] ** 2 for name in self.weights)
                if best_distance is None or distance < best_distance:
                    best = item["candidate"]
                    best_distance = distance
            return best

        return candidates[0]

    def select_from_pareto_set(self, pareto_set: list[Any] | Any) -> dict | None:
        if pareto_set is None or pareto_set == []:
            return None
        if hasattr(pareto_set, "solutions"):
            solutions = list(pareto_set.solutions)
            front_size = pareto_set.size
        else:
            solutions = list(pareto_set)
            front_size = len(solutions)
        candidate = self._select_from_candidates(solutions)
        if candidate is None:
            return None

        def _extract_objectives(item: Any) -> dict[str, float]:
            if hasattr(item, "z1") and hasattr(item, "z2") and hasattr(item, "z3"):
                return {"Z1": float(item.z1), "Z2": float(item.z2), "Z3": float(item.z3)}
            if hasattr(item, "objectives"):
                z1, z2, z3 = item.objectives
                return {"Z1": float(z1), "Z2": float(-z2), "Z3": float(z3)}
            if isinstance(item, dict) and "summary" in item:
                summary = item["summary"]
                return {
                    "Z1": float(summary.get("total_cost", 0.0)),
                    "Z2": float(summary.get("satisfaction_percent", 0.0)),
                    "Z3": float(summary.get("workload_variance", 0.0)),
                }
            return {"Z1": 0.0, "Z2": 0.0, "Z3": 0.0}

        objectives = _extract_objectives(candidate)
        assignments = []
        visit_plan = []
        evaluation = getattr(candidate, "evaluation", None)
        routes = getattr(evaluation, "routes", []) if evaluation is not None else []
        for route in routes:
            caregiver_id = getattr(route, "caregiver_id", None)
            for patient_id in getattr(route, "patient_ids", []):
                assignments.append({"patient_id": patient_id, "caregiver_id": caregiver_id})
                visit_plan.append({"patient_id": patient_id, "caregiver_id": caregiver_id, "service_start": float(len(visit_plan) + 1)})
        if not assignments:
            if hasattr(candidate, "permutation"):
                for patient_id in getattr(candidate, "permutation", tuple()):
                    visit_plan.append({"patient_id": patient_id, "caregiver_id": "unknown", "service_start": float(len(visit_plan) + 1)})
                    assignments.append({"patient_id": patient_id, "caregiver_id": "unknown"})

        score_map = {
            "cost": objectives["Z1"],
            "satisfaction": objectives["Z2"],
            "balance": objectives["Z3"],
        }
        normalized_scores = {
            name: self._normalise(score_map[name], min(score_map.values()), max(score_map.values()))
            for name in ("cost", "satisfaction", "balance")
        }
        if isinstance(pareto_set, dict):
            index = 0
        else:
            index = next((idx for idx, item in enumerate(solutions) if item is candidate), 0)
        selection = {
            "status": "ok",
            "index": index,
            "front_size": front_size,
            "candidate": candidate,
            "objects": objectives,
            "objectives": objectives,
            "assignments": assignments,
            "visit_plan": visit_plan,
            "normalised_scores": normalized_scores,
            "justification": (
                f"Selected candidate from {front_size} non-dominated solutions with Z1={objectives['Z1']:.2f}, "
                f"Z2={objectives['Z2']:.2f}, Z3={objectives['Z3']:.2f}; this is the best compromise among the non-dominated solutions."
            ),
        }
        return selection

    def choose_best_caregiver(
        self,
        patient_info: dict,
        caregiver_info: dict,
        candidate_caregivers: list[dict],
        urgent: bool = False,
    ) -> dict:
        if not patient_info:
            return {"status": "error", "message": "No patient information found."}

        if not urgent and caregiver_info and caregiver_can_visit(caregiver_info, patient_info.get("care_type")):
            return {
                "status": "assigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": caregiver_info["id"],
                "reason": "Original caregiver is compatible and available.",
            }

        if not candidate_caregivers:
            return {
                "status": "unassigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": None,
                "reason": "No available caregiver found.",
            }

        compatible_candidates = [
            caregiver
            for caregiver in candidate_caregivers
            if caregiver_can_visit(caregiver, patient_info.get("care_type"))
        ]
        if not compatible_candidates:
            return {
                "status": "unassigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": None,
                "reason": "No compatible caregiver found.",
            }

        preferred = patient_info.get("preferred_caregiver", "")
        if urgent:
            best = min(
                compatible_candidates,
                key=lambda x: (x.get("delay", 0), x.get("current_workload", 0), x.get("id") != preferred),
            )
            reason = "Urgent case assigned to the fastest compatible caregiver."
        else:
            preferred_candidates = [caregiver for caregiver in compatible_candidates if caregiver.get("id") == preferred]
            if preferred_candidates:
                best = preferred_candidates[0]
                reason = "Preferred caregiver is available."
            else:
                best = min(compatible_candidates, key=lambda x: x.get("current_workload", 0))
                reason = "Selected caregiver with the lowest workload among compatible candidates."

        return {
            "status": "assigned",
            "patient_id": patient_info["id"],
            "assigned_caregiver": best["id"],
            "reason": reason,
        }
