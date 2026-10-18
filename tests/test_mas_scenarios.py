import pytest

from agents.agentOrchestrateur import Orchestrator
from agents.messages import Message, MessageLog


@pytest.fixture
def scenario():
    patients = [
        {"id": "P1", "care_type": "Cardio", "priority": "High", "required_skill_level": 2},
        {"id": "P2", "care_type": "Diabetes", "priority": "Critical", "required_skill_level": 1},
        {"id": "P3", "care_type": "Cardio", "priority": "Medium", "required_skill_level": 1},
    ]
    caregivers = [
        {"id": "S1", "skill": "Cardio", "skill_level": 2, "available": True, "current_workload": 1, "max_work_hours": 8, "delay": 0},
        {"id": "S2", "skill": "Cardio", "skill_level": 3, "available": True, "current_workload": 1, "max_work_hours": 8, "delay": 0},
        {"id": "S3", "skill": "General", "skill_level": 1, "available": True, "current_workload": 0, "max_work_hours": 8, "delay": 0},
    ]
    return Orchestrator(patients, caregivers)


# --- message primitives ---------------------------------------------------

def test_message_object_and_log():
    log = MessageLog()
    message = log.record("S1", "orchestrator", "caregiver_delay", {"delay": 30})

    assert isinstance(message, Message)
    assert (message.sender, message.receiver, message.type) == ("S1", "orchestrator", "caregiver_delay")
    assert message.as_dict()["payload"] == {"delay": 30}
    assert log.interactions() == 1
    assert log.by_type() == {"caregiver_delay": 1}


def test_orchestrator_creates_one_agent_per_record(scenario):
    assert set(scenario.patient_agents) == {"P1", "P2", "P3"}
    assert set(scenario.caregiver_agents) == {"S1", "S2", "S3"}


# --- A4 scenario handlers -------------------------------------------------

def test_caregiver_delay_redistributes_to_compatible_caregiver(scenario):
    result = scenario.handle_caregiver_delay("S1", delay=45, patient_id="P1")

    assert result["status"] == "reassigned"
    assert result["replacement"] == "S2"  # only skill-compatible available caregiver
    assert scenario.assignments["P1"] == "S2"
    types = scenario.log.by_type()
    assert types["caregiver_delay"] == 1
    assert types["assignment"] == 1
    assert scenario.interaction_count >= 3


def test_patient_cancellation_removes_appointment(scenario):
    scenario.assignments["P2"] = "S3"

    result = scenario.handle_patient_cancellation("P2")

    assert result == {"status": "cancelled", "patient_id": "P2", "removed_caregiver": "S3"}
    assert "P2" not in scenario.assignments
    assert scenario.log.by_type()["appointment_removed"] == 1


def test_vehicle_breakdown_assigns_replacement(scenario):
    scenario.assignments["P1"] = "S1"

    result = scenario.handle_vehicle_breakdown("S1")

    assert result["status"] == "replaced"
    assert result["replacements"] == {"P1": "S2"}
    assert scenario.caregivers_by_id["S1"]["available"] is False
    assert scenario.assignments["P1"] == "S2"


def test_critical_conflict_most_urgent_wins(scenario):
    result = scenario.handle_critical_conflict("P1", "P2", caregiver_id="S2")

    assert result["status"] == "conflict_resolved"
    assert result["winner"] == "P2"  # Critical beats High
    assert result["loser"] == "P1"
    assert scenario.assignments["P2"] == "S2"


def test_caregiver_exchange_reassigns_patient(scenario):
    result = scenario.handle_caregiver_exchange("P1", from_caregiver_id="S1", to_caregiver_id="S2")

    assert result["status"] == "exchanged"
    assert scenario.assignments["P1"] == "S2"
    assert scenario.log.by_type()["exchange_confirmed"] == 1


def test_caregiver_exchange_rejected_when_incompatible(scenario):
    result = scenario.handle_caregiver_exchange("P2", from_caregiver_id="S3", to_caregiver_id="S1")

    assert result["status"] == "rejected"
    assert scenario.log.by_type()["exchange_rejected"] == 1


def test_interaction_count_covers_all_scenarios(scenario):
    assert scenario.interaction_count == 0

    scenario.handle_caregiver_delay("S1", 45, "P1")
    scenario.handle_patient_cancellation("P1")
    scenario.handle_vehicle_breakdown("S2", "P1")
    scenario.handle_critical_conflict("P1", "P2", "S3")
    scenario.handle_caregiver_exchange("P1", "S3", "S2")

    assert scenario.interaction_count == scenario.log.count > 0
