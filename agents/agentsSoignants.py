from agents.base_agents import BaseAgent


class SoignantAgent(BaseAgent):
    """
    Agent representing a caregiver in the HHCOP system.
    """

    def __init__(
        self,
        caregiver_id,
        name,
        skills,
        departure_time,
        workload,
        location,
        availability=True,
        delay=0
    ):

        super().__init__(
            agent_id=caregiver_id,
            name=name
        )

        # -----------------------------
        # Caregiver information
        # -----------------------------

        self.skills = skills

        self.departure_time = departure_time

        self.workload = workload

        self.location = location

        self.availability = availability

        self.delay = delay

        # Proposals generated for patients
        self.proposals = []

    # ==========================================
    # INFORMATION
    # ==========================================

    def get_information(self):

        return {
            "caregiver_id": self.agent_id,
            "name": self.name,
            "skills": self.skills,
            "departure_time": self.departure_time,
            "workload": self.workload,
            "location": self.location,
            "availability": self.availability,
            "delay": self.delay
        }

    # ==========================================
    # SKILL MANAGER
    # ==========================================

    def has_required_skill(self, required_skill):

        return required_skill in self.skills

    # ==========================================
    # AVAILABILITY MANAGER
    # ==========================================

    def is_available(self):

        return self.availability

    # ==========================================
    # PATIENT EVALUATION
    # ==========================================

    def evaluate_patient(self, patient_information):

        required_skill = patient_information["required_skill"]

        # Check availability
        if not self.is_available():

            return {
                "accepted": False,
                "reason": "Caregiver is not available"
            }

        # Check required skill
        if not self.has_required_skill(required_skill):

            return {
                "accepted": False,
                "reason": "Required skill not available"
            }

        return {
            "accepted": True,
            "reason": "Caregiver is compatible"
        }

    # ==========================================
    # PROPOSAL GENERATOR
    # ==========================================

    def generate_proposal(self, patient_information):

        evaluation = self.evaluate_patient(patient_information)

        # If caregiver cannot provide the service
        if not evaluation["accepted"]:

            return {
                "caregiver_id": self.agent_id,
                "patient_id": patient_information["patient_id"],
                "accepted": False,
                "reason": evaluation["reason"]
            }

        # Simple estimation for the moment
        travel_time = 20 + self.delay

        estimated_cost = travel_time * 0.5

        expected_satisfaction = 1.0

        # Preferred caregiver
        if (
            patient_information["preferred_caregiver"]
            == self.agent_id
        ):

            expected_satisfaction = 1.0

        else:

            expected_satisfaction = 0.8

        proposal = {

            "caregiver_id": self.agent_id,

            "patient_id":
                patient_information["patient_id"],

            "accepted": True,

            "travel_time":
                travel_time,

            "estimated_cost":
                estimated_cost,

            "expected_satisfaction":
                expected_satisfaction,

            "predicted_workload":
                self.workload,

            "expected_delay":
                self.delay
        }

        self.proposals.append(proposal)

        return proposal