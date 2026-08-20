import streamlit as st
import pandas as pd
from pathlib import Path
from main import run_hhcop_pipeline


st.set_page_config(page_title="HHCOP RAG Multi-Agent System", layout="wide")

PATIENTS_FILE = Path(__file__).resolve().parent.parent / "data" / "patients.csv"
CAREGIVERS_FILE = Path(__file__).resolve().parent.parent / "data" / "soignants.csv"


def add_patient(patient: dict) -> tuple[bool, str]:
    patients = pd.read_csv(PATIENTS_FILE)

    if patient["id"] in patients["id"].astype(str).values:
        return False, f"Le patient {patient['id']} existe déjà."

    patients = pd.concat([patients, pd.DataFrame([patient])], ignore_index=True)
    patients.to_csv(PATIENTS_FILE, index=False)
    return True, f"Le patient {patient['id']} a été ajouté."

st.title("HHCOP - RAG + LLM + Multi-Agent System")
st.write("Interface de visualisation des différentes étapes du système.")

with st.sidebar:
    st.header("Ajouter un patient")
    caregivers = pd.read_csv(CAREGIVERS_FILE)

    with st.form("add_patient_form", clear_on_submit=True):
        patient_id = st.text_input("Identifiant", placeholder="P5").strip()
        care_type = st.selectbox("Type de soins", ["Cardio", "Diabetes", "General"])
        priority = st.selectbox("Priorité", ["High", "Medium", "Low"])
        preferred_caregiver = st.selectbox(
            "Soignant préféré", caregivers["id"].astype(str).tolist()
        )
        submitted = st.form_submit_button("Ajouter le patient")

    if submitted:
        if not patient_id:
            st.error("L'identifiant du patient est obligatoire.")
        else:
            was_added, message = add_patient({
                "id": patient_id,
                "care_type": care_type,
                "priority": priority,
                "preferred_caregiver": preferred_caregiver,
            })
            (st.success if was_added else st.error)(message)
            if was_added:
                st.rerun()

# Entrée utilisateur
query = st.text_input(
    "Entrez un scénario HHCOP :",
    "Patient P1 is urgent and caregiver S2 is delayed"
)

if st.button("Exécuter le système"):
    results = run_hhcop_pipeline(query)

    # 1. Dataset
    st.header("1. Dataset HHCOP")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patients")
        st.dataframe(pd.DataFrame(results["patients"]))

    with col2:
        st.subheader("Caregivers")
        st.dataframe(pd.DataFrame(results["caregivers"]))

    # 2. RAG
    st.header("2. RAG - Connaissances récupérées")
    st.write("**Requête :**", results["query"])

    for i, doc in enumerate(results["retrieved_docs"], start=1):
        st.write(f"{i}. {doc}")

    # 3. LLM
    st.header("3. LLM - Analyse de la requête")
    st.json(results["llm_output"])

    # 4. Orchestrateur
    st.header("4. Orchestrateur")

    orchestration = results["orchestration_output"]

    st.subheader("Patient concerné")
    st.json(orchestration["patient_info"])

    st.subheader("Soignant mentionné")
    st.json(orchestration["caregiver_info"])

    st.subheader("Soignants candidats")
    st.json(orchestration["candidate_caregivers"])

    # 5. Décision finale
    st.header("5. Décision finale")
    st.json(results["final_decision"])