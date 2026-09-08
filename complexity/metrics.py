import time


class ComplexityMetrics:

    def __init__(self):
        self.metrics = {}

    # Temps d'exécution d'un composant
    def start(self):
        return time.perf_counter()

    def record(self, component, start):
        self.metrics[component] = {
            "execution_time_sec": round(
                time.perf_counter() - start, 6
            )
        }

    # Taille du problème HHCOP
    def problem_size(self, patients, caregivers):
        p = len(patients)
        c = len(caregivers)

        self.metrics["problem_size"] = {
            "patients": p,
            "caregivers": c,
            "possible_assignments": p * c,
            "search_space_complexity": f"O({p} × {c})"
        }

    # Complexité des interactions entre agents
    def agent_interactions(self, patients, caregivers):
        p = len(patients)
        c = len(caregivers)

        self.metrics["agent_interactions"] = {
            "patient_agents": p,
            "caregiver_agents": c,
            "possible_interactions": p * c,
            "complexity": "O(P × C)"
        }

    # Retard des soignants
    def caregiver_delay(self, caregivers):
        delays = [
            c.get("delay", 0)
            for c in caregivers
        ]

        total = sum(delays)

        self.metrics["caregiver_delay"] = {
            "total_minutes": total,
            "average_minutes": round(
                total / len(delays), 2
            ) if delays else 0,
            "delayed_caregivers": sum(
                d > 0 for d in delays
            )
        }

    # Équilibre de la charge de travail
    def workload_balance(self, caregivers):
        workloads = [
            c.get("current_workload", 0)
            for c in caregivers
        ]

        if not workloads:
            difference = 0
            average = 0
        else:
            difference = max(workloads) - min(workloads)
            average = sum(workloads) / len(workloads)

        self.metrics["workload_balance"] = {
            "average_workload": round(average, 2),
            "max_difference": difference
        }

    # Patients non affectés
    def unassigned_patients(self, patients, assignments):
        assigned = {
            a.get("patient_id")
            for a in assignments
            if a.get("patient_id")
        }

        unassigned = sum(
            p.get("id") not in assigned
            for p in patients
        )

        self.metrics["unassigned_patients"] = {
            "count": unassigned,
            "percentage": round(
                100 * unassigned / len(patients), 2
            ) if patients else 0
        }

    # Satisfaction des patients
    def patient_satisfaction(self, patients, assignments):
        if not patients:
            score = 0
        else:
            assignments_by_patient = {
                a.get("patient_id"): a
                for a in assignments
            }

            total = 0

            for patient in patients:
                assignment = assignments_by_patient.get(
                    patient.get("id")
                )

                if not assignment:
                    continue

                caregiver_id = assignment.get(
                    "caregiver_id",
                    assignment.get("assigned_caregiver")
                )

                if caregiver_id == patient.get(
                    "preferred_caregiver"
                ):
                    total += 100
                else:
                    total += 70

            score = total / len(patients)

        self.metrics["patient_satisfaction"] = {
            "score_percent": round(score, 2)
        }

    def get_metrics(self):
        return self.metrics