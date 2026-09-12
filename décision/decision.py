from agents.compatibilite import caregiver_can_visit


class DecisionEngine:
    def __init__(self):
        pass

    def choose_best_caregiver(
        self,
        patient_info: dict,
        caregiver_info: dict,
        candidate_caregivers: list[dict],
        urgent: bool = False,
    ) -> dict:
        if not patient_info:
            return {
                "status": "error",
                "message": "No patient information found."
            }

        # 1) garder le soignant initial s'il est encore utilisable
        if not urgent and caregiver_info and caregiver_can_visit(
            caregiver_info, patient_info["care_type"]
        ):
            return {
                "status": "assigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": caregiver_info["id"],
                "reason": "Original caregiver is compatible and available."
            }

        # 2) sinon chercher un candidat
        if not candidate_caregivers:
            return {
                "status": "unassigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": None,
                "reason": "No available caregiver found."
            }

        compatible_candidates = [
            caregiver
            for caregiver in candidate_caregivers
            if caregiver_can_visit(caregiver, patient_info["care_type"])
        ]
        if not compatible_candidates:
            return {
                "status": "unassigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": None,
                "reason": "No compatible caregiver found.",
            }

        preferred = patient_info.get("preferred_caregiver", "")

        # En urgence, le délai prime sur la préférence du patient.
        if urgent:
            best = min(
                compatible_candidates,
                key=lambda x: (
                    x["delay"],
                    x["current_workload"],
                    x["id"] != preferred,
                ),
            )
            reason = "Urgent case assigned to the fastest compatible caregiver."
        else:
            preferred_candidates = [
                caregiver
                for caregiver in compatible_candidates
                if caregiver["id"] == preferred
            ]
            if preferred_candidates:
                best = preferred_candidates[0]
                reason = "Preferred caregiver is available."
            else:
                best = min(
                    compatible_candidates,
                    key=lambda x: x["current_workload"],
                )
                reason = "Selected caregiver with the lowest workload among compatible candidates."

        return {
            "status": "assigned",
            "patient_id": patient_info["id"],
            "assigned_caregiver": best["id"],
            "reason": reason
        }