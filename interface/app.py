import streamlit as st
import pandas as pd
from main import run_hhcop_pipeline


st.set_page_config(page_title="HHCOP RAG Multi-Agent System", layout="wide")

st.title("HHCOP - RAG + LLM + Multi-Agent System")
st.write("Interface de visualisation des différentes étapes du système.")

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