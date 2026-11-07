from agents.agentsPatients import PatientAgent
from agents.agentsSoignants import CaregiverAgent
from agents.agentOrchestrateur import Orchestrator

def test_patient_agent():
    # Test de la classe PatientAgent
    data = {
        "id": "p1",
        "care_type": "Diabete",
        "priority": "Critical",
        "x": 10.0,
        "y": 20.0,
        "Age": 65,
        "Diabetic": True
    }
    agent = PatientAgent(data)
    info = agent.handle_request("patient_urgent")
    
    assert info["id"] == "p1"
    assert info["is_urgent"] is True
    assert info["medical_information"]["Diabetic"] is True
    print("Test PatientAgent réussi avec succès !")

def test_caregiver_agent():
    # Test de la classe CaregiverAgent
    data = {
        "id": "c1",
        "skill": "Diabete",
        "skill_level": 2,
        "available": True,
        "x": 12.0,
        "y": 22.0,
        "vehicle_capacity": 5
    }
    agent = CaregiverAgent(data)
    info = agent.handle_request("vehicle_info")
    
    assert info["id"] == "c1"
    assert info["available"] is True
    assert info["vehicle"]["capacity"] == 5
    print("Test CaregiverAgent réussi avec succès !")

def test_orchestrator_scenarios():
    # Test de l'Orchestrateur face aux événements dynamiques
    patients = [{"id": "p1", "care_type": "Diabete", "priority": "Critical", "required_skill_level": 2}]
    caregivers = [{"id": "c1", "skill": "Diabete", "skill_level": 2, "available": True}]
    
    orchestrator = Orchestrator(patients, caregivers)
    
    # Test de gestion de retard / réassignation
    result = orchestrator.handle_caregiver_delay(caregiver_id="c1", delay=25, patient_id="p1")
    assert result["status"] in ["reassigned", "not_reassigned"]
    
    # Vérification du journal d'interactions (M1)
    assert orchestrator.log.count > 0
    print(f"Test Orchestrateur réussi ! Nombre total d'interactions enregistrées : {orchestrator.log.count}")

if __name__ == "__main__":
    test_patient_agent()
    test_caregiver_agent()
    test_orchestrator_scenarios()