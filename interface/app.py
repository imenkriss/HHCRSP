import os
import sys

# on ajoute le dossier parent au sys.path pour retrouver le module main.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import streamlit as st

from main import (
    DEFAULT_QUERY,
    run_hhcop_pipeline,
)


st.set_page_config(page_title="HHCOP RAG Multi-Agent System", layout="wide")

st.title("Home Healthcare Multi-AGENT Systems")


# Scénario principal et modèle utilisés par le pipeline.
query = st.text_input("Entrez un scénario HHCOP :", DEFAULT_QUERY)
model = st.text_input("Modèle Ollama :", "qwen3:4b")

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if st.button("Exécuter le système") or st.session_state.analysis_results:

    try:
        with st.spinner(f"Analyse en cours avec Ollama ({model})..."):
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
            else:
                results = run_hhcop_pipeline(
                    query,
                    model=model,
                )
                st.session_state.analysis_results = results
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

        st.subheader("Résultats théoriques")
        st.markdown(
            "La complexité théorique décrit le coût prévu selon la taille "
            "du problème. Les temps ci-dessus sont les mesures pratiques "
            "de cette exécution."
        )
        theoretical_data = {
            "Recherche des candidats": "O(P × C)",
            "Interactions entre agents": "O(P × C)",
            "Évaluation d'une solution": "O(N × C)",
            "Front de Pareto": "O(P²)",
        }
        st.dataframe(
            pd.DataFrame(
                theoretical_data.items(),
                columns=["Opération", "Complexité"]
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Aucune métrique de complexité."
        )

    # 7. Optimisation NSGA-II
    st.header("7. Optimisation NSGA-II")

    optimization = results.get("optimization", [])

    if optimization:
        analysis = results.get("optimization_analysis", {})
        if analysis:
            st.subheader("Analyse théorique de l'optimisation")
            st.json(analysis)

        optimization_data = []

        for index, candidate in enumerate(optimization, start=1):
            summary = candidate.get("summary", {})
            optimization_data.append({
                "Solution": index,
                "Coût total": summary.get("total_cost"),
                "Patients servis": summary.get("served_patients"),
                "Patients non affectés": summary.get("unassigned_patients"),
                "Variance de charge": summary.get("workload_variance"),
                "Satisfaction": summary.get("satisfaction_percent"),
                "Affectations": candidate["solution"],
            })

        st.dataframe(
            pd.DataFrame(optimization_data),
            use_container_width=True
        )

        selected_optimization = results.get("selected_optimization")
        if selected_optimization:
            st.subheader("Compromis retenu par le moteur de décision")
            st.json(selected_optimization.get("summary", {}))
    else:
        st.info("Aucune solution d'optimisation disponible.")
