import csv
import importlib
import json
import os
import sys

from rag.rag import SimpleRAG
from llm.raisonner import LLMRaisonner
from agents.agentOrchestrateur import Orchestrator
from complexity.metrics import ComplexityMetrics
from optimisation.nsga2 import NSGA2
from data.dataset_loader import load_combined_dataset

DecisionEngine = importlib.import_module("décision.decision").DecisionEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_QUERY = "Get the info for patient P1 with high priority"

INT_FIELDS = ("max_work_hours", "current_workload", "delay", "skill_level")
BOOL_FIELDS = ("available",)

def ensure_default_csv_files():
    """Crée les fichiers CSV de base s'ils n'existent pas."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    patient_path = os.path.join(DATA_DIR, "dementia_patients_health_data.csv")
    if not os.path.exists(patient_path):
        with open(patient_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Diabetic", "Dementia", "Chronic_Health_Conditions", "XCOORD", "YCOORD"])
            writer.writeheader()
            writer.writerow({"Diabetic": 1, "Dementia": 0, "Chronic_Health_Conditions": "Diabetes", "XCOORD": 40.0, "YCOORD": 50.0})

    caregiver_path = os.path.join(DATA_DIR, "admin_caregivers.csv")
    if not os.path.exists(caregiver_path):
        with open(caregiver_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "skill", "max_work_hours", "current_workload", "skill_level", "available", "delay", "XCOORD", "YCOORD"])
            writer.writeheader()
            writer.writerow({"id": "S1", "skill": "General", "max_work_hours": 8, "current_workload": 0, "skill_level": 1, "available": True, "delay": 0, "XCOORD": 0.0, "YCOORD": 0.0})


def _convert(row: dict) -> dict:
    """Convertit les champs CSV vers leurs types réels."""
    converted = {}
    for key, value in row.items():
        key = key.strip()
        value = (value or "").strip()
        if key in INT_FIELDS or key in ("Diabetic", "Dementia"):
            try:
                converted[key] = int(float(value)) if value else 0
            except ValueError:
                converted[key] = value
        elif key in BOOL_FIELDS:
            converted[key] = value.lower() in ("true", "1", "yes", "oui")
        else:
            try:
                if "." in value:
                    converted[key] = float(value)
                else:
                    converted[key] = int(value) if value.isdigit() else value
            except ValueError:
                converted[key] = value
    return converted


def load_csv(filename: str) -> list[dict]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [_convert(row) for row in csv.DictReader(handle)]


def load_dataset() -> tuple[list[dict], list[dict]]:
    """
    Charge les patients (Dementia + Gehring & Homberger) et les soignants (Solomon / Véhicules)
    en intégrant ID, priorités, compétences, coordonnées géographiques et multi-fenêtres de temps.
    """
    ensure_default_csv_files()
    
    patients = []
    soignants = []

    # 1. Chargement des patients (Dementia + Géographie X, Y)
    patient_path = os.path.join(DATA_DIR, "dementia_patients_health_data.csv")
    if os.path.exists(patient_path):
        with open(patient_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                patient = _convert(row)
                patient["id"] = f"P{idx}"
                
                # Priorité liée à l'état clinique (Diabète ou Démence -> High)
                if "priority" not in patient or not patient["priority"]:
                    is_diabetic = int(patient.get("Diabetic", 0)) == 1
                    is_dementia = int(patient.get("Dementia", 0)) == 1
                    patient["priority"] = "High" if (is_diabetic or is_dementia) else "Medium"
                
                # Type de soin
                if "care_type" not in patient or not patient["care_type"]:
                    condition = str(patient.get("Chronic_Health_Conditions", "General"))
                    patient["care_type"] = condition if condition in ["Diabetes", "Hypertension", "Heart Disease", "Dementia"] else "General"
                
                # Coordonnées géographiques (Gehring & Homberger)
                patient["x"] = float(patient.get("XCOORD", patient.get("X", 0.0)))
                patient["y"] = float(patient.get("YCOORD", patient.get("Y", 0.0)))
                patient["service_hours"] = float(patient.get("service_time", 30)) / 60.0
                
                # Multi-fenêtres de temps
                if "time_windows" not in patient:
                    patient["time_windows"] = [(0.0, 480.0)]
                
                patients.append(patient)

    # 2. Chargement des soignants / véhicules (Solomon)
    caregiver_records = load_csv("admin_caregivers.csv")
    if caregiver_records:
        for idx, row in enumerate(caregiver_records, start=1):
            if not row.get("id"):
                row["id"] = f"S{idx}"
            if "skill" not in row or not row["skill"]:
                row["skill"] = "General"
            if "time_windows" not in row:
                row["time_windows"] = [(0.0, 480.0)]
            
            row["x"] = float(row.get("XCOORD", row.get("X", 0.0)))
            row["y"] = float(row.get("YCOORD", row.get("Y", 0.0)))
            soignants.append(row)

    if not patients:
        patients = [{"id": "P1", "care_type": "General", "priority": "High", "x": 0.0, "y": 0.0, "time_windows": [(0.0, 480.0)]}]
    if not soignants:
        soignants = [{"id": "S1", "skill": "General", "max_work_hours": 8, "current_workload": 0, "available": True, "x": 0.0, "y": 0.0, "time_windows": [(0.0, 480.0)]}]

    return patients, soignants


def run_hhcop_pipeline(
    query: str = DEFAULT_QUERY,
    model: str = "qwen3:4b",
    patient_event: dict | None = None,
    caregiver_event: dict | None = None,
    dataset: str | None = None,
    caregivers: int | None = None,
    location: int | None = None,
    instance_seed: int | None = None,
    population_size: int = 20,
    generations: int = 10,
    local_search: bool = True,
    algorithm: str = "nsga2",
    decision_method: str = "weighted_sum",
    gamma_t: float = 0.0,
    gamma_s: float = 0.0,
    gamma_p: float = 0.0,
) -> dict:

    metrics = ComplexityMetrics()
    total_start = metrics.start()

    start = metrics.start()
    patients, soignants = load_dataset()
    if dataset is not None or caregivers is not None or location is not None or instance_seed is not None:
        from optimisation.mtwmo.generator import generate_instance
        instance = generate_instance(dataset.upper() if dataset else "A", caregivers or 2, location or 1, seed=(instance_seed if instance_seed is not None else 0))
        patients = [
            {"id": f"P{idx}", "care_type": getattr(p, "care_type", "General"), "priority": getattr(p, "priority", "Medium"), "required_skill_level": 1, "service_hours": getattr(p, "service_time", 30) / 60.0, "x": getattr(p, "x", 0.0), "y": getattr(p, "y", 0.0), "time_windows": list(getattr(p, "time_windows", [(0.0, 480.0)]))}
            for idx, p in enumerate(instance.patients, start=1)
        ]
        soignants = [{"id": f"S{idx}", "skill": "General", "skill_level": 1, "available": True, "current_workload": 0, "max_work_hours": 8, "delay": 0, "x": 0.0, "y": 0.0, "time_windows": [(0.0, 480.0)]} for idx in range(1, (caregivers or 2) + 1)]
    
    if patient_event:
        patients.append(patient_event)
    if caregiver_event:
        soignants.append(caregiver_event)
        
    instance_name = f"{(dataset or 'A').upper()}_{caregivers or 2}_{location or 1}"
    metrics.record("Dataset", start)

    # 1. RAG
    start = metrics.start()
    retrieved_docs = SimpleRAG().process(query)
    metrics.record("RAG", start)

    # 2. LLM
    start = metrics.start()
    reasoner = LLMRaisonner(model=model)
    llm_output = reasoner.analyze(query, retrieved_docs)
    metrics.record("LLM", start)

    # 3. ORCHESTRATEUR
    start = metrics.start()
    orchestration_output = Orchestrator(patients, soignants).process(llm_output)
    metrics.record("Orchestrator + Agents", start)

    # 4. OPTIMISATION NSGA-II
    start = metrics.start()
    optimizer = NSGA2(
        patients,
        soignants,
        population_size=population_size,
        generations=generations,
        uncertainty_scenarios=32,
        gamma_t=gamma_t,
        gamma_s=gamma_s,
        gamma_p=gamma_p,
    )
    optimization_results = optimizer.optimize()
    optimization_analysis = {
        "algorithm": "MTCMOLite" if str(algorithm).lower() == "mtcmo" else "NSGA-II",
        "population_size": population_size,
        "generations": generations,
        "local_search": local_search,
        "gamma": {"gamma_t": gamma_t, "gamma_s": gamma_s, "gamma_p": gamma_p},
    }
    metrics.record("NSGA-II Optimization", start)

    # 5. DECISION ENGINE
    start = metrics.start()
    engine = DecisionEngine(method=decision_method)
    selected_optimization = engine.select_from_pareto_set(optimization_results)
    assignments = selected_optimization["assignments"] if selected_optimization else []
    selected_patient = orchestration_output.get("patient_info")
    selected_assignment = next(
        (assignment for assignment in assignments if selected_patient and assignment["patient_id"] == selected_patient.get("id")),
        None,
    )
    
    if not selected_patient:
        final_decision = {"status": "error", "message": "No patient information found in the request."}
    elif selected_assignment:
        final_decision = {
            "status": "assigned",
            "patient_id": selected_patient.get("id"),
            "assigned_caregiver": selected_assignment["caregiver_id"],
            "reason": "Assignment selected from the Pareto compromise.",
            "method": decision_method,
        }
    else:
        final_decision = {
            "status": "unassigned",
            "patient_id": selected_patient.get("id"),
            "assigned_caregiver": None,
            "reason": "No feasible assignment for this patient.",
            "method": decision_method,
        }
        
    decision_payload = {
        "method": decision_method,
        "objectives": selected_optimization.get("objectives", {}) if selected_optimization else {},
        "visit_plan": selected_optimization.get("visit_plan", []) if selected_optimization else [],
        "justification": selected_optimization.get("justification", "No Pareto solution selected.") if selected_optimization else "No Pareto solution selected.",
        "assignments": assignments,
        "status": selected_optimization.get("status", "empty") if selected_optimization else "empty",
    }
    metrics.record("Decision Engine", start)

    metrics.problem_size(patients, soignants)
    metrics.agent_interactions(patients, soignants, observed_count=0)
    metrics.caregiver_delay(soignants)
    metrics.workload_balance(soignants, assignments)
    metrics.unassigned_patients(patients, assignments)
    metrics.patient_satisfaction(patients, soignants, assignments)
    metrics.optimization_indicators(selected_optimization)
    metrics.uncertainty_indicators(llm_output, retrieved_docs)
    metrics.record("Metrics", start)
    metrics.record("Total Pipeline", total_start)

    decision = dict(decision_payload)
    decision["instance_info"] = {"name": instance_name}
    return {
        "query": query,
        "patients": patients,
        "soignants": soignants,
        "retrieved_docs": retrieved_docs,
        "llm_model": reasoner.model,
        "llm_output": llm_output,
        "orchestration_output": orchestration_output,
        "final_decision": final_decision,
        "decision": decision,
        "optimization": optimization_results,
        "selected_optimization": selected_optimization,
        "optimization_analysis": optimization_analysis,
        "instance_info": {"name": instance_name},
        "complexity": metrics.get_metrics(),
    }


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_demo(results: dict) -> None:
    _section("0. SCENARIO")
    print(results["query"])
    _section("1. DATASET")
    print(f"Patients chargés : {len(results['patients'])} | Soignants chargés : {len(results['soignants'])}")
    _section("5. DECISION FINALE")
    print(json.dumps(results["final_decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY
    print_demo(run_hhcop_pipeline(user_query))