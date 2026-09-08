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

st.title("Home Healthcare optimization problem under uncertainty using Rag and muti agent system")
st.write("Interface de visualisation du système.")

# Événements transmis aux agents, au RAG et au MAS
st.header("Nouveaux événements")
patient_event_type = st.selectbox(
    "Événement patient",
    ["Aucun", "Urgence patient existant", "Nouveau patient"]
)

patient_event = None
patient_id = None

if patient_event_type == "Urgence patient existant":
    patient_id = st.text_input("Identifiant du patient", "P2")
    patient_event_query = st.text_input(
        "Détail de l'urgence",
        "Le patient demande une prise en charge urgente."
    )
elif patient_event_type == "Nouveau patient":
    patient_id = st.text_input("Identifiant du nouveau patient", "P10")
    patient_care_type = st.selectbox(
        "Type de soin du nouveau patient",
        ["Cardio", "Diabetes", "General"]
    )
    patient_priority = st.selectbox(
        "Priorité",
        ["High", "Medium", "Low"]
    )
    patient_preferred = st.text_input(
        "Soignant préféré (optionnel)",
        ""
    )
    patient_event_query = st.text_input(
        "Demande du nouveau patient",
        "Le nouveau patient demande une prise en charge."
    )
    patient_event = {
        "id": patient_id,
        "care_type": patient_care_type,
        "priority": patient_priority,
        "preferred_caregiver": patient_preferred,
    }

caregiver_event_type = st.selectbox(
    "Événement soignant",
    ["Aucun", "Soignant disponible", "Nouveau soignant disponible"]
)

caregiver_event = None
caregiver_id = None

if caregiver_event_type == "Soignant disponible":
    caregiver_id = st.text_input("Identifiant du soignant", "S5")
    caregiver_event_query = st.text_input(
        "Demande du soignant",
        "Le soignant est disponible et demande un nouveau travail."
    )
elif caregiver_event_type == "Nouveau soignant disponible":
    caregiver_id = st.text_input("Identifiant du nouveau soignant", "S8")
    caregiver_skill = st.selectbox(
        "Compétence du nouveau soignant",
        ["Cardio", "Diabetes", "General"]
    )
    caregiver_event_query = st.text_input(
        "Demande du nouveau soignant",
        "Le nouveau soignant est disponible et demande un travail."
    )
    caregiver_event = {
        "id": caregiver_id,
        "skill": caregiver_skill,
        "max_work_hours": 8,
        "current_workload": 0,
        "available": True,
        "delay": 0,
    }

# Entrée utilisateur
query = st.text_input("Entrez un scénario HHCOP :", DEFAULT_QUERY)
model = st.text_input("Modèle Ollama :", "qwen3:4b")

event_parts = []
if patient_event_type != "Aucun":
    event_parts.append(
        f"Patient {patient_id}: {patient_event_query}"
    )
if caregiver_event_type != "Aucun":
    event_parts.append(
        f"Caregiver {caregiver_id}: {caregiver_event_query}"
    )
if event_parts:
    query = query + "\n\n" + "\n".join(event_parts)
    st.info("Requête envoyée au RAG et au MAS : " + query)

if st.button("Exécuter le système"):

    try:
        with st.spinner(f"Analyse en cours avec Ollama ({model})..."):
            results = run_hhcop_pipeline(
                query,
                model=model,
                patient_event=patient_event,
                caregiver_event=caregiver_event,
            )
    except Exception as error:
        st.error(
            "Impossible d'exécuter le pipeline. Vérifiez qu'Ollama est ouvert "
            f"et que le modèle '{model}' est installé."
        )
        st.exception(error)
        st.stop()

    # 1. Dataset
    st.header("1. Dataset HHCOP")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patients")
        st.dataframe(pd.DataFrame(results["patients"]))

    with col2:
        st.subheader("Soignants")
        st.dataframe(pd.DataFrame(results["soignants"]))

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
    st.caption(f"Modèle Ollama utilisé : `{results['llm_model']}`")
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

        st.subheader("Métriques calculées")
        st.json(complexity)

        st.subheader("Temps d'exécution par composant")

        execution_data = []

        for component, data in complexity.items():

            if isinstance(data, dict) and "execution_time_sec" in data:

                execution_data.append({
                    "Composant": component,
                    "Temps d'exécution (s)": data["execution_time_sec"]
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
        else:
            st.info("Aucun temps d'exécution n'a été enregistré.")

        # Interactions entre agents
        if "agent_interactions" in complexity:

            st.subheader("Interactions entre agents")

            interactions = complexity["agent_interactions"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Patients",
                    interactions["patient_agents"]
                )

            with col2:
                st.metric(
                    "Soignants",
                    interactions["caregiver_agents"]
                )

            with col3:
                st.metric(
                    "Interactions potentielles",
                    interactions["possible_interactions"]
                )

            st.info(
                f"Complexité théorique de la coordination : "
                f"{interactions['complexity']}"
            )

        quality_metrics = [
            ("problem_size", "Taille du problème"),
            ("caregiver_delay", "Retard des soignants"),
            ("workload_balance", "Équilibre de charge"),
            ("unassigned_patients", "Patients non affectés"),
            ("patient_satisfaction", "Satisfaction des patients"),
        ]
        available_quality_metrics = {
            label: complexity[key]
            for key, label in quality_metrics
            if key in complexity
        }

        if available_quality_metrics:
            st.subheader("Indicateurs opérationnels")
            st.json(available_quality_metrics)

    else:

        st.info(
            "Aucune métrique de complexité disponible."
        )

    # 7. Optimisation NSGA-II
    st.header("7. Optimisation NSGA-II")

    optimization = results.get("optimization", [])

    if optimization:
        optimization_data = []

        for index, candidate in enumerate(optimization, start=1):
            optimization_data.append({
                "Solution": index,
                "Retard total": candidate["objectives"][0],
                "Différence de charge": candidate["objectives"][1],
                "Satisfaction": -candidate["objectives"][2],
                "Affectations": candidate["solution"],
            })

        st.dataframe(
            pd.DataFrame(optimization_data),
            use_container_width=True
        )
    else:
        st.info("Aucune solution d'optimisation disponible.")
