from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import CaregiverAgent
from agents.compatibilite import caregiver_can_visit
from agents.messages import MessageLog


class Orchestrator:
    def __init__(self, patients: list[dict], caregivers: list[dict], patient_clusters: dict[int, list[str]] | None = None):
        self.patients = patients
        self.caregivers = caregivers
        self.patients_by_id = {patient["id"]: patient for patient in patients}
        self.caregivers_by_id = {caregiver["id"]: caregiver for caregiver in caregivers}
        self.patient_agents = {patient["id"]: PatientAgent(patient) for patient in patients}
        self.caregiver_agents = {caregiver["id"]: CaregiverAgent(caregiver) for caregiver in caregivers}
        self.assignments: dict[str, str] = {}
        self.patient_clusters = dict(patient_clusters or {})
        self.patient_cluster_of = {
            patient_id: cluster_id
            for cluster_id, members in self.patient_clusters.items()
            for patient_id in members
        }
        self.log = MessageLog()
        self.interaction_count = 0

    def set_patient_clusters(self, clusters: dict[int, list[str]]):
        self.patient_clusters = {int(cluster_id): list(members) for cluster_id, members in clusters.items()}
        self.patient_cluster_of = {
            patient_id: cluster_id
            for cluster_id, members in self.patient_clusters.items()
            for patient_id in members
        }

    def patients_in_cluster(self, cluster_id: int):
        members = self.patient_clusters.get(int(cluster_id), [])
        return [self.patients_by_id[patient_id] for patient_id in members if patient_id in self.patients_by_id]

    def grouped_patients(self):
        result = {}
        for cluster_id, members in self.patient_clusters.items():
            result[int(cluster_id)] = [self.patients_by_id[patient_id] for patient_id in members if patient_id in self.patients_by_id]
        return result

    def find_patient(self, patient_id: str):
        patient = self.patients_by_id.get(patient_id)
        return PatientAgent(patient) if patient else None

    def find_caregiver(self, caregiver_id: str):
        caregiver = self.caregivers_by_id.get(caregiver_id)
        return CaregiverAgent(caregiver) if caregiver else None

    def get_candidate_caregivers(
        self,
        care_type: str,
        required_skill_level: int = 1,
        exclude_caregiver_id: str | None = None,
    ) -> list[dict]:
        """Retourne une seule fois chaque soignant compatible, disponible et
        possédant le niveau de compétence requis, en excluant éventuellement
        le soignant déjà assigné."""
        return [
            CaregiverAgent(caregiver).get_information()
            for caregiver in self.caregivers
            if caregiver["id"] != exclude_caregiver_id
            and caregiver_can_visit(caregiver, care_type, required_skill_level)
        ]

    def handle_caregiver_delay(self, caregiver_id: str, delay: int = 0, patient_id: str | None = None):
        self.log.record(caregiver_id, "orchestrator", "caregiver_delay", {"delay": delay, "patient_id": patient_id})
        if patient_id is None:
            self.interaction_count = self.log.count
            return {"status": "not_reassigned", "patient_id": patient_id, "replacement": None}

        patient = self.patients_by_id.get(patient_id)
        if patient is None:
            self.interaction_count = self.log.count
            return {"status": "not_reassigned", "patient_id": patient_id, "replacement": None}

        replacement = next(
            (
                caregiver["id"]
                for caregiver in self.caregivers
                if caregiver["id"] != caregiver_id
                and caregiver.get("available", False)
                and caregiver.get("delay", 0) <= 20
                and caregiver_can_visit(
                    caregiver,
                    patient["care_type"],
                    patient.get("required_skill_level", 1),
                )
            ),
            None,
        )
        if replacement:
            self.assignments[patient_id] = replacement
            self.log.record("orchestrator", "orchestrator", "assignment", {"patient_id": patient_id, "caregiver_id": replacement})
            self.interaction_count = self.log.count
            return {"status": "reassigned", "replacement": replacement, "patient_id": patient_id}
        self.interaction_count = self.log.count
        return {"status": "not_reassigned", "patient_id": patient_id, "replacement": replacement}

    def handle_patient_cancellation(self, patient_id: str):
        removed = self.assignments.pop(patient_id, None)
        self.log.record("patient", "orchestrator", "appointment_removed", {"patient_id": patient_id, "removed_caregiver": removed})
        self.interaction_count = self.log.count
        return {"status": "cancelled", "patient_id": patient_id, "removed_caregiver": removed}

    def handle_vehicle_breakdown(self, caregiver_id: str, patient_id: str | None = None):
        caregiver = self.caregivers_by_id.get(caregiver_id)
        if caregiver:
            caregiver["available"] = False
        self.log.record(caregiver_id, "orchestrator", "vehicle_breakdown", {"patient_id": patient_id})

        if patient_id is None:
            impacted = [pid for pid, assigned in self.assignments.items() if assigned == caregiver_id]
            if impacted:
                patient_id = impacted[0]

        replacement = None
        if patient_id is not None:
            patient = self.patients_by_id.get(patient_id)
            replacement = next(
                (
                    other["id"]
                    for other in self.caregivers
                    if other["id"] != caregiver_id
                    and other.get("available", False)
                    and caregiver_can_visit(
                        other,
                        patient.get("care_type", "") if patient else "",
                        patient.get("required_skill_level", 1) if patient else 1,
                    )
                ),
                None,
            )
        if patient_id and replacement:
            self.assignments[patient_id] = replacement
            self.log.record("orchestrator", "orchestrator", "assignment", {"patient_id": patient_id, "caregiver_id": replacement})
            self.interaction_count = self.log.count
            return {"status": "replaced", "replacements": {patient_id: replacement}, "patient_id": patient_id}
        self.interaction_count = self.log.count
        return {"status": "replaced", "replacements": {}, "patient_id": patient_id}

    def handle_critical_conflict(self, patient_a: str, patient_b: str, caregiver_id: str):
        priorities = {patient: self.patients_by_id[patient].get("priority", "Normal") for patient in (patient_a, patient_b)}
        is_a_critical = priorities[patient_a] == "Critical"
        is_b_critical = priorities[patient_b] == "Critical"
        if is_a_critical and not is_b_critical:
            winner, loser = patient_a, patient_b
        elif is_b_critical and not is_a_critical:
            winner, loser = patient_b, patient_a
        elif patient_a < patient_b:
            winner, loser = patient_a, patient_b
        else:
            winner, loser = patient_b, patient_a
        
        self.assignments[winner] = caregiver_id
        # Correction : 4 arguments passés à record pour éviter le TypeError
        self.log.record("orchestrator", "system", "critical_conflict", {"winner": winner, "loser": loser, "caregiver_id": caregiver_id})
        self.interaction_count = self.log.count
        return {"status": "conflict_resolved", "winner": winner, "loser": loser, "caregiver_id": caregiver_id}

    def handle_caregiver_exchange(self, patient_id: str, from_caregiver_id: str, to_caregiver_id: str):
        from_patient = self.patients_by_id.get(patient_id)
        to_caregiver = self.caregivers_by_id.get(to_caregiver_id)
        if from_patient and to_caregiver and caregiver_can_visit(to_caregiver, from_patient.get("care_type", ""), from_patient.get("required_skill_level", 1)):
            self.assignments[patient_id] = to_caregiver_id
            self.log.record("orchestrator", "orchestrator", "exchange_confirmed", {"patient_id": patient_id, "from": from_caregiver_id, "to": to_caregiver_id})
            self.interaction_count = self.log.count
            return {"status": "exchanged", "patient_id": patient_id, "caregiver_id": to_caregiver_id}
        self.log.record("orchestrator", "orchestrator", "exchange_rejected", {"patient_id": patient_id, "from": from_caregiver_id, "to": to_caregiver_id})
        self.interaction_count = self.log.count
        return {"status": "rejected", "patient_id": patient_id, "caregiver_id": from_caregiver_id}

    def process(self, llm_output: dict) -> dict:
        if not isinstance(llm_output, dict):
            raise ValueError("llm_output doit être un dictionnaire")

        request_type = llm_output.get("request_type", "assignment")
        valid_request_types = {
            "patient_info",
            "patient_urgent",
            "caregiver_info",
            "vehicle_info",
            "assignment",
            "unknown",
        }
        if request_type not in valid_request_types:
            request_type = "unknown"

        if request_type == "unknown":
            return {
                "request_type": request_type,
                "routed_to": None,
                "agent_response": {
                    "status": "unsupported_request",
                    "message": "La demande ne correspond pas à une fonction reconnue.",
                },
                "patient_info": None,
                "caregiver_info": None,
                "candidate_caregivers": [],
                "llm_output": llm_output,
            }

        patient_info = None
        caregiver_info = None
        candidate_caregivers = []
        agent_response = None
        routed_to = None

        patient_id = llm_output.get("patient_id")
        if request_type in ("patient_info", "patient_urgent", "assignment") and patient_id:
            patient_agent = self.find_patient(patient_id)
            if patient_agent:
                patient_info = patient_agent.get_information()
                routed_to = "patient"
                if request_type in ("patient_info", "patient_urgent"):
                    agent_response = patient_agent.handle_request(
                        request_type,
                        llm_urgency=bool(llm_output.get("urgency")),
                    )

        caregiver_id = llm_output.get("caregiver_id")
        if request_type in ("caregiver_info", "vehicle_info", "assignment") and caregiver_id:
            caregiver_agent = self.find_caregiver(caregiver_id)
            if caregiver_agent:
                caregiver_info = caregiver_agent.get_information()
                routed_to = "caregiver"
                if request_type in ("caregiver_info", "vehicle_info"):
                    agent_response = caregiver_agent.handle_request(request_type)

        if request_type in ("patient_info", "patient_urgent") and not patient_info:
            agent_response = {
                "status": "not_found",
                "message": "Identifiant patient absent ou introuvable.",
            }
        elif request_type in ("caregiver_info", "vehicle_info") and not caregiver_info:
            agent_response = {
                "status": "not_found",
                "message": "Identifiant soignant absent ou introuvable.",
            }

        if patient_info and request_type in ("assignment", "patient_urgent"):
            candidate_caregivers = self.get_candidate_caregivers(
                patient_info["care_type"],
                required_skill_level=patient_info["required_skill_level"],
                exclude_caregiver_id=caregiver_id,
            )

        if request_type == "patient_urgent" and patient_info:
            agent_response["candidate_caregivers"] = candidate_caregivers

        return {
            "request_type": request_type,
            "routed_to": routed_to,
            "agent_response": agent_response,
            "patient_info": patient_info,
            "caregiver_info": caregiver_info,
            "candidate_caregivers": candidate_caregivers,
            "llm_output": llm_output,
        }