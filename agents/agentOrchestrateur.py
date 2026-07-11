from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import CaregiverAgent

class Orchestrator:
    def __init__(self, patients: list[dict], caregivers: list[dict]):
        self.patients = patients
        self.caregivers = caregivers

    def find_patient(self, patient_id: str):
        for patient in self.patients:
            if patient["id"] == patient_id:
                return PatientAgent(patient)
        return None

    def find_caregiver(self, caregiver_id: str):
        for caregiver in self.caregivers:
            if caregiver["id"] == caregiver_id:
                return CaregiverAgent(caregiver)
        return None

    def get_candidate_caregivers(self, care_type: str) -> list[dict]:
        candidates = []

        for caregiver in self.caregivers:
            # on garde les disponibles
            if caregiver["available"] and caregiver["skill"] == care_type:
                agent = CaregiverAgent(caregiver)
                candidates.append(agent.get_information())

        # si aucun soignant exact, on peut chercher les "General"
        if not candidates:
            for caregiver in self.caregivers:
                if caregiver["available"] and caregiver["skill"] == "General":
                    agent = CaregiverAgent(caregiver)
                    candidates.append(agent.get_information())

        return candidates

    def process(self, llm_output: dict) -> dict:
        patient_info = None
        caregiver_info = None
        candidate_caregivers = []

        # récupérer patient
        if llm_output["patient_id"]:
            patient_agent = self.find_patient(llm_output["patient_id"])
            if patient_agent:
                patient_info = patient_agent.get_information()

        # récupérer soignant mentionné dans la requête
        if llm_output["caregiver_id"]:
            caregiver_agent = self.find_caregiver(llm_output["caregiver_id"])
            if caregiver_agent:
                caregiver_info = caregiver_agent.get_information()

        # si on a un patient, chercher des candidats pour réaffectation
        if patient_info:
            candidate_caregivers = self.get_candidate_caregivers(patient_info["care_type"])

        return {
            "patient_info": patient_info,
            "caregiver_info": caregiver_info,
            "candidate_caregivers": candidate_caregivers,
            "llm_output": llm_output
        }