from pathlib import Path

import pandas as pd

from agents.agentOrchestrateur import Orchestrator
from décision.decision import DecisionEngine
from llm.raisonner import LLMRaisonner
from rag.retriever import Retriever


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data() -> tuple[list[dict], list[dict]]:
    patients = pd.read_csv(DATA_DIR / "patients.csv").fillna("").to_dict("records")
    caregivers = pd.read_csv(DATA_DIR / "soignants.csv").fillna("").to_dict("records")

    for caregiver in caregivers:
        caregiver.update({
            "available": True,
            "delay": 0,
            "current_workload": caregiver.get("max_work_hours", 0),
        })

    return patients, caregivers


def run_hhcop_pipeline(query: str) -> dict:
    patients, caregivers = load_data()
    retrieved_results = Retriever().retrieve(query)
    retrieved_docs = [result["document"]["description"] for result in retrieved_results]
    llm_output = LLMRaisonner().analyze(query, retrieved_docs)
    orchestration_output = Orchestrator(patients, caregivers).process(llm_output)
    final_decision = DecisionEngine().choose_best_caregiver(
        orchestration_output["patient_info"],
        orchestration_output["caregiver_info"],
        orchestration_output["candidate_caregivers"],
    )

    return {
        "query": query,
        "patients": patients,
        "caregivers": caregivers,
        "retrieved_docs": retrieved_docs,
        "llm_output": llm_output,
        "orchestration_output": orchestration_output,
        "final_decision": final_decision,
    }