class CaregiverAgent:
    def __init__(self, caregiver_data: dict):
        self.caregiver_data = caregiver_data

    def get_information(self) -> dict:
        return {
            "id": self.caregiver_data["id"],
            "skill": self.caregiver_data.get("skill", ""),
            "max_work_hours": self.caregiver_data.get("max_work_hours", 0),
            "current_workload": self.caregiver_data.get("current_workload", 0),
            "skill_level": self.caregiver_data.get("skill_level", 1),
            "available": self.caregiver_data.get("available", False),
            "delay": self.caregiver_data.get("delay", 0)
        }

    def handle_request(self, request_type: str) -> dict:
        information = self.get_information()
        information["request_type"] = request_type
        if request_type == "vehicle_info":
            information["vehicle"] = {
                "x": self.caregiver_data.get("x"),
                "y": self.caregiver_data.get("y"),
                "capacity": self.caregiver_data.get("vehicle_capacity"),
            }
        return information
