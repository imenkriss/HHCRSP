class PatientAgent:
    def __init__(self, patient_data: dict):
        self.patient_data = dict(patient_data)
        self.proposals = []

    def get_information(self) -> dict:
        return {
            "id": self.patient_data["id"],
            "care_type": self.patient_data["care_type"],
            "priority": self.patient_data["priority"],
            "preferred_caregiver": self.patient_data["preferred_caregiver"]
        }

    def receive_proposal(self, proposal):
        self.proposals.append(proposal)

    def show_proposals(self) -> list[dict]:
        return list(self.proposals)


class PatientRegistry:
    PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

    def __init__(self, patients: list[dict] | None = None):
        self._patients: dict[str, PatientAgent] = {}
        if patients:
            self.register_many(patients)

    def register(self, patient_data: dict) -> PatientAgent:
        patient_id = patient_data.get("id")
        if not patient_id:
            raise ValueError("Un patient doit posséder un identifiant 'id'.")

        patient = PatientAgent(patient_data)
        existing_patient = self._patients.get(patient_id)
        if existing_patient:
            patient.proposals = existing_patient.proposals
        self._patients[patient_id] = patient
        return patient

    def register_many(self, patients: list[dict]) -> None:
        for patient_data in patients:
            self.register(patient_data)

    def get(self, patient_id: str) -> PatientAgent | None:
        return self._patients.get(patient_id)

    def all(self) -> list[PatientAgent]:
        return list(self._patients.values())

    def by_care_type(self, care_type: str) -> list[PatientAgent]:
        return [
            patient for patient in self._patients.values()
            if patient.patient_data.get("care_type") == care_type
        ]

    def by_priority(self) -> list[PatientAgent]:
        return sorted(
            self._patients.values(),
            key=lambda patient: self.PRIORITY_ORDER.get(
                patient.patient_data.get("priority"),
                len(self.PRIORITY_ORDER)
            )
        )