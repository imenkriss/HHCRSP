import os
import sys

# Permet de lancer "streamlit run interface/app.py" depuis la racine du projet :
# on ajoute le dossier parent au sys.path pour retrouver le module main.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from main import DEFAULT_QUERY, run_hhcop_pipeline


st.set_page_config(page_title="HHCOP RAG Multi-Agent System", layout="wide")

st.title("HHCOP - RAG + LLM + Multi-Agent System")
st.write("Interface de visualisation des différentes étapes du système.")

# Entrée utilisateur
query = st.text_input("Entrez un scénario HHCOP :", DEFAULT_QUERY)

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

    if results["retrieved_docs"]:
        for i, doc in enumerate(results["retrieved_docs"], start=1):
            st.write(f"{i}. {doc}")
    else:
        st.info("Aucun document pertinent trouvé pour cette requête.")

    # 3. LLM
    st.header("3. LLM - Analyse de la requête")
    st.json(results["llm_output"])

    # 4. Orchestrateur
    st.header("4. Orchestrateur")

    orchestration = results["orchestration_output"]

    st.subheader("Patient concerné")
    if orchestration["patient_info"]:
        st.json(orchestration["patient_info"])
    else:
        st.warning("Aucun patient identifié dans la requête.")

    st.subheader("Soignant mentionné")
    if orchestration["caregiver_info"]:
        st.json(orchestration["caregiver_info"])
    else:
        st.warning("Aucun soignant identifié dans la requête.")

    st.subheader("Soignants candidats")
    if orchestration["candidate_caregivers"]:
        st.dataframe(pd.DataFrame(orchestration["candidate_caregivers"]))
    else:
        st.warning("Aucun soignant candidat disponible.")

    # 5. Décision finale
    st.header("5. Décision finale")

    decision = results["final_decision"]

    if decision["status"] == "assigned":
        st.success(f"Soignant affecté : {decision['assigned_caregiver']}")
    elif decision["status"] == "unassigned":
        st.error("Aucun soignant n'a pu être affecté.")

    st.json(decision)
    # 6. Complexité du système
    st.header("6. Analyse de complexité")

    complexity = results.get("complexity", {})

    if complexity:

        st.subheader("Temps d'exécution par composant")

        execution_data = []

        for component, data in complexity.items():

            if isinstance(data, dict) and "execution_time" in data:

                execution_data.append({
                    "Composant": component,
                    "Temps d'exécution (s)": data["execution_time"]
                })

        if execution_data:

            df_complexity = pd.DataFrame(execution_data)

            st.dataframe(
                df_complexity,
                use_container_width=True
            )

            st.bar_chart(
                df_complexity.set_index("Composant")
            )

        # Interactions entre agents
        if "Agent Interactions" in complexity:

            st.subheader("Interactions entre agents")

            interactions = complexity["Agent Interactions"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Patients",
                    interactions["patients"]
                )

            with col2:
                st.metric(
                    "Soignants",
                    interactions["caregivers"]
                )

            with col3:
                st.metric(
                    "Interactions potentielles",
                    interactions["interactions"]
                )

            st.info(
                f"Complexité théorique de la coordination : "
                f"{interactions['complexity']}"
            )

    else:

        st.info(
            "Aucune métrique de complexité disponible."
        )