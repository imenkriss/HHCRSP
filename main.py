
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
# le paquet "décision" contient un caractère accentué : import dynamique
DecisionEngine = importlib.import_module("décision.decision").DecisionEngine


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_QUERY = "Analyze patient need care with High priority"

INT_FIELDS = ("max_work_hours", "current_workload", "delay", "skill_level")
BOOL_FIELDS = ("available",)


def _convert(row: dict) -> dict:
    
    #Convertit les champs CSV(chaines)02vers leurs types réels.
    
    converted = {}

    for key, value in row.items():

        key = key.strip()
        value = (value or "").strip()

        if key in INT_FIELDS:
            converted[key] = int(value) if value else 0

        elif key in BOOL_FIELDS:
            converted[key] = value.lower() in ("true", "1", "yes", "oui")

        else:
            converted[key] = value

    return converted


def load_csv(filename: str) -> list[dict]:
    
    # Charge un fichier CSV du dossier data(en liste dictonnaire)
    
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with open(path, newline="", encoding="utf-8") as handle:
        return [_convert(row) for row in csv.DictReader(handle)]


def load_dataset() -> tuple[list[dict], list[dict]]:
    """Load benchmark-backed patients, clinical profiles, and caregivers."""
    patients, soignants = load_combined_dataset()
    for filename, records in (
        ("admin_patients.csv", patients),
        ("admin_caregivers.csv", soignants),
    ):
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            records.extend(load_csv(filename))
    return patients, soignants


def append_csv_record(filename: str, record: dict) -> bool:
    """Ajoute un enregistrement au CSV si son identifiant n'existe pas."""
    path = os.path.join(DATA_DIR, filename)
    rows = load_csv(filename) if os.path.exists(path) else []

    if any(row.get("id") == record.get("id") for row in rows):
        return False

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record.keys())
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerow(record)

    return True


def save_patient(patient: dict) -> bool:
    return append_csv_record(
        "admin_patients.csv",
        {
            "id": patient["id"],
            "care_type": patient["care_type"],
            "priority": patient["priority"],
            "preferred_caregiver": patient.get("preferred_caregiver", ""),
        },
    )


def save_caregiver(caregiver: dict) -> bool:
    return append_csv_record(
        "admin_caregivers.csv",
        {
            "id": caregiver["id"],
            "skill": caregiver["skill"],
            "max_work_hours": caregiver.get("max_work_hours", 8),
            "current_workload": caregiver.get("current_workload", 0),
            "skill_level": caregiver.get("skill_level", 1),
            "available": caregiver.get("available", True),
            "delay": caregiver.get("delay", 0),
        },
    )

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

    # DATASET
    start = metrics.start()
    patients, soignants = load_dataset()
    if dataset is not None or caregivers is not None or location is not None or instance_seed is not None:
        from optimisation.mtwmo.generator import generate_instance
        instance = generate_instance(dataset or "A", caregivers or 2, location or 1, seed=(instance_seed if instance_seed is not None else 0))
        patients = [
            {"id": str(p.id), "care_type": getattr(p, "care_type", "General"), "priority": getattr(p, "priority", "Medium"), "required_skill_level": 1, "service_hours": p.service_time / 60.0, "travel_time": 0.0, "travel_time_min": 0.0, "travel_time_max": 0.0, "time_windows": list(getattr(p, "time_windows", []))}
            for p in instance.patients
        ]
        soignants = [{"id": f"K{idx}", "skill": "General", "skill_level": 1, "available": True, "current_workload": 0, "max_work_hours": 8, "delay": 0} for idx in range(1, (caregivers or 2) + 1)]
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
    llm_output = reasoner.analyze(
        query,
        retrieved_docs
    )

    metrics.record("LLM", start)

    # 3. ORCHESTRATEUR + AGENTS
    

    start = metrics.start()

    orchestration_output = Orchestrator(
        patients,
        soignants
    ).process(llm_output)

    metrics.record(
        "Orchestrator + Agents",
        start
    )

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

    # 5. DECISION ENGINE : sélection d'un compromis du front de Pareto.
    start = metrics.start()
    engine = DecisionEngine(method=decision_method)
    selected_optimization = engine.select_from_pareto_set(optimization_results)
    assignments = selected_optimization["assignments"] if selected_optimization else []
    selected_patient = orchestration_output["patient_info"]
    selected_assignment = next(
        (
            assignment for assignment in assignments
            if selected_patient and assignment["patient_id"] == selected_patient["id"]
        ),
        None,
    )
    if not selected_patient:
        final_decision = {"status": "error", "message": "No patient information found in the request."}
    elif selected_assignment:
        final_decision = {
            "status": "assigned",
            "patient_id": selected_patient["id"],
            "assigned_caregiver": selected_assignment["caregiver_id"],
            "reason": "Assignment selected from the Pareto compromise.",
            "method": decision_method,
        }
    else:
        final_decision = {
            "status": "unassigned",
            "patient_id": selected_patient["id"],
            "assigned_caregiver": None,
            "reason": "No feasible assignment for this patient.",
            "method": decision_method,
        }
    decision_payload = {
        "method": decision_method,
        "objectives": selected_optimization["objectives"] if selected_optimization else {},
        "visit_plan": selected_optimization["visit_plan"] if selected_optimization else [],
        "justification": selected_optimization["justification"] if selected_optimization else "No Pareto solution selected.",
        "assignments": assignments,
        "status": selected_optimization["status"] if selected_optimization else "empty",
    }
    metrics.record("Decision Engine", start)

    # 6. COMPLEXITÉ ET INDICATEURS : calculés après l'optimisation, sur la
    # solution réellement retenue et ses paramètres d'exécution.
    metrics.problem_size(patients, soignants)
    metrics.agent_interactions(patients, soignants, observed_count=0)
    metrics.caregiver_delay(soignants)
    metrics.workload_balance(soignants, assignments)
    metrics.unassigned_patients(patients, assignments)
    metrics.patient_satisfaction(patients, soignants, assignments)
    metrics.optimization_indicators(selected_optimization)
    metrics.uncertainty_indicators(llm_output, retrieved_docs)
    metrics.delays_and_overtime([], soignants)
    metrics.pareto_quality([], (0.0, 0.0), feasible_count=0)
    metrics.record("Metrics", start)
    metrics.record("Total Pipeline", total_start)

    # keep compatibility with tests that expect the decision info at the top level
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
    """
    Affiche le déroulé complet du pipeline dans le terminal.
    """

    _section("0. SCENARIO")
    print(results["query"])

    _section("1. DATASET")

    print("Patients :")
    for patient in results["patients"]:
        print(f"  {patient['id']:4} type={patient['care_type']:10}"
              f" priorité={patient['priority']:7}"
              f" préféré={patient['preferred_caregiver']}")

    print("\nSoignants :")
    for soignant in results["soignants"]:
        print(f"  {soignant['id']:4} skill={soignant['skill']:10} "
              f" charge={soignant['current_workload']}/{soignant['max_work_hours']}"
              f" dispo={str(soignant['available']):5}"
              f" retard={soignant['delay']} min")

    _section("2. RAG - CONNAISSANCES RECUPEREES")
    if results["retrieved_docs"]:
        for index, doc in enumerate(results["retrieved_docs"], start=1):
            print(f"  {index}. {doc}")
    else:
        print("  (aucun document pertinent)")

    _section("3. LLM - ANALYSE")
    llm = results["llm_output"]
    print(f"  patient_id  : {llm['patient_id']}")
    print(f"  soignant_id : {llm['caregiver_id']}")
    print(f"  urgency     : {llm['urgency']}")
    print(f"  delayed     : {llm['delayed']}")

    _section("4. ORCHESTRATEUR")
    orchestration = results["orchestration_output"]
    print("  Patient concerné  :", json.dumps(orchestration["patient_info"], ensure_ascii=False))
    print("  Soignant mentionné:", json.dumps(orchestration["caregiver_info"], ensure_ascii=False))
    print("  Candidats         :")

    if orchestration["candidate_caregivers"]:
        for candidate in orchestration["candidate_caregivers"]:
            print(f"    - {candidate['id']} (charge={candidate['current_workload']},"
                  f" retard={candidate['delay']})")
    else:
        print("    (aucun)")

    _section("5. DECISION FINALE")
    print(json.dumps(results["final_decision"], indent=2, ensure_ascii=False))
    print()
    _section("6. COMPLEXITY")

    complexity = results.get("complexity", {})

    if complexity:

        for component, data in complexity.items():

            if "execution_time_sec" in data:

                print(
                    f"  {component:25} : "
                    f"{data['execution_time_sec']:.6f} secondes"
                )

            elif component == "agent_interactions":

                print(f"  Patients              : {data['patient_agents']}")
                print(f"  Soignants             : {data['caregiver_agents']}")
                print(f"  Interactions          : {data['possible_interactions']}")
                print(f"  Complexité théorique  : {data['complexity']}")

    else:

        print("  Aucune métrique disponible.")

    print()

    _section("7. OPTIMISATION NSGA-II")
    optimization = results.get("optimization", [])

    if optimization:
        for index, candidate in enumerate(optimization, start=1):
            print(f"  Solution {index} - objectifs : {candidate['objectives']}")
            print(f"    Affectations : {candidate['solution']}")
    else:
        print("  Aucune solution d'optimisation disponible.")


if __name__ == "__main__":

    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY
    print_demo(run_hhcop_pipeline(user_query))
