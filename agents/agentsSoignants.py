class CaregiverAgent:
    def __init__(self, caregiver_data: dict):
        self.caregiver_data = caregiver_data

    def get_information(self) -> dict:
        return {
            "id": self.caregiver_data["id"],
            "skill": self.caregiver_data.get("skill", ""),
            "max_work_hours": self.caregiver_data.get("max_work_hours", 0),
            "current_workload": self.caregiver_data.get("current_workload", 0),
            "available": self.caregiver_data.get("available", False),
            "delay": self.caregiver_data.get("delay", 0)
        }