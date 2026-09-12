from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import CaregiverAgent
from agents.compatibilite import caregiver_can_visit, caregiver_is_eligible


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

    def _is_eligible(self, caregiver: dict) -> bool:
        """
        Un soignant est éligible s'il est disponible.
        """
        return caregiver_is_eligible(caregiver)

    def get_candidate_caregivers(self, care_type: str) -> list[dict]:
        candidates = []

        # C004 : seuls les soignants qualifiés pour ce type de soin
        for caregiver in self.caregivers:
            if caregiver_can_visit(caregiver, care_type):
                candidates.append(CaregiverAgent(caregiver).get_information())

        # repli : les soignants polyvalents ("General")
        if not candidates:
            for caregiver in self.caregivers:
                if caregiver_can_visit(caregiver, care_type):
                    candidates.append(CaregiverAgent(caregiver).get_information())

        return candidates

    def process(self, llm_output: dict) -> dict:
        patient_info = None
        caregiver_info = None
        candidate_caregivers = []

        # récupérer patient
        if llm_output.get("patient_id"):
            patient_agent = self.find_patient(llm_output["patient_id"])
            if patient_agent:
                patient_info = patient_agent.get_information()

        # récupérer soignant mentionné dans la requête
        if llm_output.get("caregiver_id"):
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