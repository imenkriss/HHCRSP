from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import CaregiverAgent
from agents.compatibilite import caregiver_can_visit


class Orchestrator:
    def __init__(self, patients: list[dict], caregivers: list[dict]):
        self.patients = patients
        self.caregivers = caregivers
        self.patients_by_id = {patient["id"]: patient for patient in patients}
        self.caregivers_by_id = {
            caregiver["id"]: caregiver for caregiver in caregivers
        }

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