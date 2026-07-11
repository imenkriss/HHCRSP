class CaregiverAgent:
    def __init__(self, caregiver_data: dict):
        self.caregiver_data = caregiver_data

    def get_information(self) -> dict:
        return {
            "id": self.caregiver_data["id"],
            "skill": self.caregiver_data["skill"],
            "max_work_hours": self.caregiver_data["max_work_hours"],
            "current_workload": self.caregiver_data["current_workload"],
            "available": self.caregiver_data["available"],
            "delay": self.caregiver_data["delay"]
        }