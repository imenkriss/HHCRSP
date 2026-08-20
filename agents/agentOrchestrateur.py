from agents.agentsPatients import PatientAgent, PatientRegistry
from agents.agentsSoignants import SoignantAgent

CaregiverAgent = SoignantAgent

class Orchestrator:
    def __init__(self, patients: list[dict], caregivers: list[dict]):
        self.patient_registry = PatientRegistry(patients)
        self.caregivers = caregivers

    def receive_patients(self, patients: list[dict]) -> None:
        self.patient_registry.register_many(patients)

    def get_patients_by_care_type(self, care_type: str) -> list[dict]:
        return [patient.get_information()
                for patient in self.patient_registry.by_care_type(care_type)]

    def get_patients_by_priority(self) -> list[dict]:
        return [patient.get_information()
                for patient in self.patient_registry.by_priority()]

    def find_patient(self, patient_id: str):
        return self.patient_registry.get(patient_id)

    def find_caregiver(self, caregiver_id: str):
        for caregiver in self.caregivers:
            if caregiver["id"] == caregiver_id:
                return caregiver
        return None

    def get_candidate_caregivers(self, care_type: str) -> list[dict]:
        candidates = []

        for caregiver in self.caregivers:
            # on garde les disponibles
            if caregiver["available"] and caregiver["skill"] == care_type:
                candidates.append(caregiver)

        # si aucun soignant exact, on peut chercher les "General"
        if not candidates:
            for caregiver in self.caregivers:
                if caregiver["available"] and caregiver["skill"] == "General":
                    candidates.append(caregiver)

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
                caregiver_info = caregiver_agent

        # si on a un patient, chercher des candidats pour réaffectation
        if patient_info:
            candidate_caregivers = self.get_candidate_caregivers(patient_info["care_type"])

        return {
            "patient_info": patient_info,
            "caregiver_info": caregiver_info,
            "candidate_caregivers": candidate_caregivers,
            "llm_output": llm_output
        }