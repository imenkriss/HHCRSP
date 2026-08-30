import time


class ComplexityMetrics:

    def __init__(self):
        self.metrics = {}

    # Taille du problème
    def problem_size(self, patients, caregivers):
        self.metrics["problem_size"] = {
            "patients": len(patients),
            "caregivers": len(caregivers),
            "potential_assignments": len(patients) * len(caregivers)
        }

    # Retard total des soignants
    def caregiver_delay(self, caregivers):
        total = sum(c.get("delay", 0) for c in caregivers)

        self.metrics["caregiver_delay"] = {
            "total_minutes": total,
            "average_minutes": total / len(caregivers)
            if caregivers else 0
        }

    # Équilibre de charge
    def workload_balance(self, caregivers):
        workloads = [
            c.get("current_workload", 0)
            for c in caregivers
        ]

        difference = (
            max(workloads) - min(workloads)
            if workloads else 0
        )

        self.metrics["workload_balance"] = {
            "difference": difference
        }

    # Patients non affectés
    def unassigned_patients(self, patients, assignments):
        assigned = {
            a.get("patient_id")
            for a in assignments
        }

        count = sum(
            1 for p in patients
            if p.get("id") not in assigned
        )

        self.metrics["unassigned_patients"] = {
            "count": count
        }

    # Satisfaction des patients
    def patient_satisfaction(self, patients, assignments):
        if not patients:
            return 0

        score = 0

        for patient in patients:

            assignment = next(
                (
                    a for a in assignments
                    if a.get("patient_id") == patient.get("id")
                ),
                None
            )

            if assignment:
                if (
                    assignment.get("caregiver_id")
                    == patient.get("preferred_caregiver")
                ):
                    score += 100
                else:
                    score += 70

        satisfaction = score / len(patients)

        self.metrics["patient_satisfaction"] = {
            "score_percent": satisfaction
        }

    def get_metrics(self):
        return self.metrics