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
    save_caregiver,
    save_patient,
)


st.set_page_config(page_title="HHCOP RAG Multi-Agent System", layout="wide")

st.title("Home Healthcare optimization problem under uncertainty using Rag and muti agent system")
st.write("Interface de visualisation du système.")

# Tableau de bord administrateur : les demandes acceptées sont transmises
# au RAG, aux agents et au moteur de décision.
if "admin_requests" not in st.session_state:
    st.session_state.admin_requests = []

with st.sidebar:
    st.header("Administration")
    st.caption("Créer, prioriser et valider les demandes.")

    with st.form("new_request_form", clear_on_submit=True):
        request_type = st.selectbox(
            "Type de demande",
            ["Nouveau patient", "Urgence patient", "Nouveau soignant"]
        )
        request_id = st.text_input("Identifiant", "P10")
        request_priority = st.selectbox(
            "Priorité d'urgence",
            ["Critique", "Haute", "Normale", "Basse"]
        )

        if request_type in ["Nouveau patient", "Urgence patient"]:
            request_care_type = st.selectbox(
                "Type de soin",
                ["Cardio", "Diabetes", "General"]
            )
            request_preferred = st.text_input(
                "Soignant préféré",
                ""
            )
            request_message = st.text_input(
                "Message du patient",
                "Le patient demande une prise en charge urgente."
            )
        else:
            request_skill = st.selectbox(
                "Compétence du soignant",
                ["Cardio", "Diabetes", "General"]
            )
            request_message = st.text_input(
                "Message du soignant",
                "Le soignant est disponible et demande un travail."
            )

        add_request = st.form_submit_button("Ajouter la demande")

    if add_request:
        request = {
            "type": request_type,
            "id": request_id.strip(),
            "priority": request_priority,
            "message": request_message,
            "status": "En attente",
        }
        if request_type in ["Nouveau patient", "Urgence patient"]:
            request.update({
                "care_type": request_care_type,
                "preferred_caregiver": request_preferred.strip(),
            })
        else:
            request["skill"] = request_skill
        st.session_state.admin_requests.append(request)
        st.session_state.analysis_results = []
        st.rerun()

    st.subheader("Demandes en attente")
    accepted_requests = []
    for index, request in enumerate(st.session_state.admin_requests):
        st.markdown(
            f"**{request['type']} · {request['id']}**  "
            f"({request['priority']})"
        )
        request["status"] = st.selectbox(
            "Décision administrateur",
            ["En attente", "Acceptée", "Refusée"],
            index=["En attente", "Acceptée", "Refusée"].index(
                request["status"]
            ),
            key=f"request_status_{index}",
        )
        if request["status"] == "Acceptée":
            accepted_requests.append(request)

    if st.button("Vider les demandes"):
        st.session_state.admin_requests = []
        st.session_state.analysis_results = []
        st.rerun()

# Scénario principal et modèle utilisés par le pipeline.
query = st.text_input("Entrez un scénario HHCOP :", DEFAULT_QUERY)
model = st.text_input("Modèle Ollama :", "qwen3:4b")

event_parts = []
saved_messages = []
requests_to_process = []

for request in accepted_requests:
    if request["type"] == "Nouveau patient":
        event_parts.append(
            f"{request['type']} {request['id']} ({request['priority']}): "
            f"{request['message']}"
        )
        patient_record = {
            "id": request["id"],
            "care_type": request.get("care_type", "General"),
            "priority": request["priority"],
            "preferred_caregiver": request.get("preferred_caregiver", ""),
        }
        requests_to_process.append({
            "label": f"Patient {request['id']} - nouvelle demande",
            "query": (
                f"Patient {request['id']} requests {request.get('care_type', 'General')} "
                f"care with priority {request['priority']}. "
                f"{request['message']}"
            ),
        })
        if not request.get("saved_to_csv"):
            if save_patient(patient_record):
                saved_messages.append(
                    f"Patient {request['id']} ajouté à patients.csv."
                )
            else:
                saved_messages.append(
                    f"Patient {request['id']} existe déjà dans patients.csv."
                )
            request["saved_to_csv"] = True
    elif request["type"] == "Urgence patient":
        event_parts.append(
            f"{request['type']} {request['id']} ({request['priority']}): "
            f"{request['message']}"
        )
        # L'urgence concerne un patient déjà chargé depuis le CSV.
        # Elle est transmise dans la requête sans créer de doublon.
        requests_to_process.append({
            "label": f"Patient {request['id']} - urgence",
            "query": (
                f"Patient {request['id']} has an urgent request for "
                f"{request.get('care_type', 'General')} care. "
                f"Priority: {request['priority']}. {request['message']}"
            ),
        })
    else:
        event_parts.append(
            f"{request['type']} {request['id']} ({request['priority']}): "
            f"{request['message']}"
        )
        caregiver_record = {
            "id": request["id"],
            "skill": request["skill"],
            "max_work_hours": 8,
            "current_workload": 0,
            "available": True,
            "delay": 0,
        }
        if not request.get("saved_to_csv"):
            if save_caregiver(caregiver_record):
                saved_messages.append(
                    f"Soignant {request['id']} ajouté à soignants.csv."
                )
            else:
                saved_messages.append(
                    f"Soignant {request['id']} existe déjà dans soignants.csv."
                )
            request["saved_to_csv"] = True


for message in saved_messages:
    st.success(message)

if event_parts:
    query = query + "\n\n" + "\n".join(event_parts)
    st.info("Demandes acceptées envoyées au RAG et au MAS : " + query)

if not requests_to_process:
    requests_to_process.append({
        "label": "Scénario manuel",
        "query": query,
    })

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if st.button("Exécuter le système") or st.session_state.analysis_results:

    try:
        with st.spinner(f"Analyse en cours avec Ollama ({model})..."):
            if st.session_state.analysis_results:
                all_results = st.session_state.analysis_results
            else:
                all_results = []
                for request_to_process in requests_to_process:
                    all_results.append(
                        (
                            request_to_process["label"],
                            run_hhcop_pipeline(
                                request_to_process["query"],
                                model=model,
                            ),
                        )
                    )
                st.session_state.analysis_results = all_results

            if len(all_results) > 1:
                selected_label = st.selectbox(
                    "Résultat à afficher",
                    [label for label, _ in all_results],
                )
                results = next(
                    result
                    for label, result in all_results
                    if label == selected_label
                )
                st.success(
                    f"{len(all_results)} demandes analysées."
                )
            else:
                results = all_results[0][1]
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
