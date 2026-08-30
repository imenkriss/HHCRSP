from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import CaregiverAgent

# R002 : un soignant dont le retard dépasse ce seuil doit être remplacé
MAX_ACCEPTABLE_DELAY = 20


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

    def _is_eligible(self, caregiver: dict) -> bool:
        """
        Un soignant est éligible s'il est disponible, pas trop en retard (R002)
        et pas déjà au maximum de ses heures (C002).
        """
        if not caregiver["available"]:
            return False

        if caregiver["delay"] > MAX_ACCEPTABLE_DELAY:
            return False

        if caregiver["current_workload"] >= caregiver["max_work_hours"]:
            return False

        return True

    def get_candidate_caregivers(self, care_type: str) -> list[dict]:
        candidates = []

        # C004 : seuls les soignants qualifiés pour ce type de soin
        for caregiver in self.caregivers:
            if self._is_eligible(caregiver) and caregiver["skill"] == care_type:
                candidates.append(CaregiverAgent(caregiver).get_information())

        # repli : les soignants polyvalents ("General")
        if not candidates:
            for caregiver in self.caregivers:
                if self._is_eligible(caregiver) and caregiver["skill"] == "General":
                    candidates.append(CaregiverAgent(caregiver).get_information())

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