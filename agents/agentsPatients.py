class PatientAgent:
    MEDICAL_FIELDS = (
        "Age",
        "Chronic_Health_Conditions",
        "Diabetic",
        "Dementia",
        "HeartRate",
        "BloodOxygenLevel",
        "BodyTemperature",
        "Medication_History",
        "Prescription",
        "Dosage in mg",
        "Hospitalization_History",
    )

    def __init__(self, patient_data: dict):
        self.patient_data = patient_data

    def get_information(self) -> dict:
        return {
            "id": self.patient_data["id"],
            "care_type": self.patient_data["care_type"],
            "priority": self.patient_data["priority"],
            "preferred_caregiver": self.patient_data.get("preferred_caregiver", ""),
            "required_skill_level": self.patient_data.get("required_skill_level", 1),
        }

    def handle_request(self, request_type: str, llm_urgency: bool = False) -> dict:
        information = self.get_information()
        information["coordinates"] = {
            "x": self.patient_data.get("x"),
            "y": self.patient_data.get("y"),
        }
        information["medical_information"] = {
            field: self.patient_data[field]
            for field in self.MEDICAL_FIELDS
            if field in self.patient_data and self.patient_data[field] not in (None, "")
        }
        information["is_urgent"] = bool(
            llm_urgency or self.patient_data.get("priority") == "Critical"
        )
        information["request_type"] = request_type
        return information
