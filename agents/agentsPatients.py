class PatientAgent:
    def __init__(self, patient_data: dict):
        self.patient_data = patient_data

    def get_information(self) -> dict:
        return {
            "id": self.patient_data["id"],
            "care_type": self.patient_data["care_type"],
            "priority": self.patient_data["priority"],
            "preferred_caregiver": self.patient_data["preferred_caregiver"]
        }