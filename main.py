
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
# le paquet "décision" contient un caractère accentué : import dynamique
DecisionEngine = importlib.import_module("décision.decision").DecisionEngine


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_QUERY = "Patient P2 is urgent and caregiver S2 is delayed."

INT_FIELDS = ("max_work_hours", "current_workload", "delay")
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
    #Charge les patients et les soignants.

    return load_csv("patients.csv"), load_csv("soignants.csv")


def append_csv_record(filename: str, record: dict) -> bool:
    """Ajoute un enregistrement au CSV si son identifiant n'existe pas."""
    path = os.path.join(DATA_DIR, filename)
    rows = load_csv(filename)

    if any(row.get("id") == record.get("id") for row in rows):
        return False

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record.keys())
        writer.writerow(record)

    return True


def save_patient(patient: dict) -> bool:
    return append_csv_record(
        "patients.csv",
        {
            "id": patient["id"],
            "care_type": patient["care_type"],
            "priority": patient["priority"],
            "preferred_caregiver": patient.get("preferred_caregiver", ""),
        },
    )


def save_caregiver(caregiver: dict) -> bool:
    return append_csv_record(
        "soignants.csv",
        {
            "id": caregiver["id"],
            "skill": caregiver["skill"],
            "max_work_hours": caregiver.get("max_work_hours", 8),
            "current_workload": caregiver.get("current_workload", 0),
            "available": caregiver.get("available", True),
            "delay": caregiver.get("delay", 0),
        },
    )

def run_hhcop_pipeline(
    query: str = DEFAULT_QUERY,
    model: str = "qwen3:4b",
    patient_event: dict | None = None,
    caregiver_event: dict | None = None,
) -> dict:


    metrics = ComplexityMetrics()
    total_start = metrics.start()

    # DATASET
    start = metrics.start()
    patients, soignants = load_dataset()

    if patient_event:
        patients.append(patient_event)

    if caregiver_event:
        soignants.append(caregiver_event)

    metrics.record("Dataset", start)

    # Complexité
    metrics.problem_size(patients, soignants)
    metrics.caregiver_delay(soignants)
    metrics.workload_balance(soignants)

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

   
    # interactions entre agents
    metrics.agent_interactions(patients, soignants)


    # 4. DECISION ENGINE

    start = metrics.start()

    final_decision = DecisionEngine().choose_best_caregiver(
        orchestration_output["patient_info"],
        orchestration_output["caregiver_info"],
        orchestration_output["candidate_caregivers"]
    )

    metrics.record(
        "Decision Engine",
        start
    )

    assignments = []
    if final_decision.get("status") == "assigned":
        assignments.append({
            "patient_id": final_decision.get("patient_id"),
            "caregiver_id": final_decision.get("assigned_caregiver"),
        })

    metrics.unassigned_patients(patients, assignments)
    metrics.patient_satisfaction(patients, assignments)

    # 5. OPTIMISATION NSGA-II
    start = metrics.start()

    optimizer = NSGA2(
        patients,
        soignants
    )
    optimization_results = optimizer.optimize()
    optimization_analysis = optimizer.complexity_analysis()

    metrics.record("NSGA-II Optimization", start)

    metrics.record(
        "Total Pipeline",
        total_start
    )


    return {
        "query": query,
        "patients": patients,
        "soignants": soignants,
        "retrieved_docs": retrieved_docs,
        "llm_model": reasoner.model,
        "llm_output": llm_output,
        "orchestration_output": orchestration_output,
        "final_decision": final_decision,
        "optimization": optimization_results,
        "optimization_analysis": optimization_analysis,
        "complexity": metrics.get_metrics()
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
