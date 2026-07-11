class DecisionEngine:
    def __init__(self):
        pass

    def choose_best_caregiver(self, patient_info: dict, caregiver_info: dict, candidate_caregivers: list[dict]) -> dict:
        if not patient_info:
            return {
                "status": "error",
                "message": "No patient information found."
            }

        # 1) garder le soignant initial s'il est encore utilisable
        if caregiver_info:
            if caregiver_info["available"] and caregiver_info["delay"] <= 20:
                return {
                    "status": "assigned",
                    "patient_id": patient_info["id"],
                    "assigned_caregiver": caregiver_info["id"],
                    "reason": "Original caregiver is available and delay is acceptable."
                }

        # 2) sinon chercher un candidat
        if not candidate_caregivers:
            return {
                "status": "unassigned",
                "patient_id": patient_info["id"],
                "assigned_caregiver": None,
                "reason": "No available caregiver found."
            }

        preferred = patient_info["preferred_caregiver"]

        # priorité au soignant préféré s'il est dans les candidats
        for cg in candidate_caregivers:
            if cg["id"] == preferred:
                return {
                    "status": "assigned",
                    "patient_id": patient_info["id"],
                    "assigned_caregiver": cg["id"],
                    "reason": "Preferred caregiver is available."
                }

        # sinon prendre celui avec la plus faible charge
        best = min(candidate_caregivers, key=lambda x: x["current_workload"])

        return {
            "status": "assigned",
            "patient_id": patient_info["id"],
            "assigned_caregiver": best["id"],
            "reason": "Selected caregiver with the lowest workload among compatible candidates."
        }