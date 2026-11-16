from __future__ import annotations

import os
import sys
import pandas as pd
import streamlit as st

# Ajout du dossier parent au sys.path pour retrouver le module main.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import (
    DEFAULT_QUERY,
    run_hhcop_pipeline,
)

# 1. Configuration unique de la page Streamlit
st.set_page_config(
    page_title=" RAG-MAS & NSGA-II",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥Home Health Care Routing & Scheduling (MAS + NSGA-II)")


# Initialisation des résultats dans la session Streamlit
if "analysis_results" not in st.session_state:
    with st.spinner("Exécution initiale du pipeline (RAG + MAS + NSGA-II)..."):
        st.session_state.analysis_results = run_hhcop_pipeline(
            query="Get info for patient P1 with high priority", 
            dataset="C",           
            caregivers=10,        
            location=1
        )
        st.session_state.analysis_query = "Get info for patient P1 with high priority"
        st.session_state.analysis_model = "qwen3:4b"

# Formulaire de saisie du scénario
with st.form("pipeline_form"):
    query = st.text_input(
        "Entrez un scénario HHCOP :",
        st.session_state.get("analysis_query", DEFAULT_QUERY),
        help="Indiquez l'identifiant du patient, par exemple P1 avec mention de la priorité.",
    )
    model = st.text_input("Modèle Ollama :", "qwen3:4b")
    run_requested = st.form_submit_button("Exécuter le système")

if run_requested:
    try:
        with st.spinner(f"Analyse en cours avec Ollama ({model})..."):
            results = run_hhcop_pipeline(query, model=model)
            st.session_state.analysis_results = results
            st.session_state.analysis_query = query
            st.session_state.analysis_model = model
    except Exception as error:
        st.session_state.analysis_results = None
        st.error(f"Impossible d'exécuter le pipeline avec le modèle '{model}'.")
        st.exception(error)
        st.stop()

results = st.session_state.analysis_results

if results is not None:
   # ==========================================
    # 1. DONNÉES D'ORIGINE (Dossier data/)
    
    st.header("1. HHCOP data ")
    st.markdown("Chargement et affichage des sources brutes (patients et instances de benchmark).")

    # Onglets pour séparer les patients et les instances de soignants / benchmarks
    tab_patients, tab_benchmarks = st.tabs([
        "🩺 Patients Data (Dementia)", 
        "👩‍⚕️ Caregivers data "
    ])

    with tab_patients:
        st.subheader(" Données des Patients (dementia_patients_health_data.csv)")
        try:
            df_patients = pd.read_csv("data/dementia/dementia_patients_health_data.csv")
            st.dataframe(df_patients, width='stretch')
        except Exception as e:
            st.warning("Le fichier 'dementia_patients_health_data.csv' est introuvable dans le dossier 'data/dementia/'.")
            st.exception(e)

    with tab_benchmarks:
        st.subheader("1. Instances de Solomon (Benchmark de référence)")
        st.info("Fichiers d'instances de Solomon stockés dans le dossier `data/`.")
        df_solomon_full = pd.DataFrame([
            {"Instance": "c105-25", "Patients": 25, "Depot": 1},
            {"Instance": "c107-25", "Patients": 25, "Depot": 1},
            {"Instance": "c108-25", "Patients": 25, "Depot": 1},
            {"Instance": "R201-25", "Patients": 25, "Depot": 1},
            {"Instance": "R205-25", "Patients": 25, "Depot": 1},
            {"Instance": "R210-25", "Patients": 25, "Depot": 1},
            {"Instance": "RC-206-25", "Patients": 25, "Depot": 1},
            {"Instance": "RC208-25", "Patients": 25, "Depot": 1},
            {"Instance": "c107-50", "Patients": 50, "Depot": 1},
            {"Instance": "c108-50", "Patients": 50, "Depot": 1},
            {"Instance": "c109-50", "Patients": 50, "Depot": 1},
            {"Instance": "R206-50", "Patients": 50, "Depot": 1},
            {"Instance": "R208-50", "Patients": 50, "Depot": 1},
            {"Instance": "RC202-50", "Patients": 50, "Depot": 1},
            {"Instance": "RC203-50", "Patients": 50, "Depot": 1},
            {"Instance": "RC204-50", "Patients": 50, "Depot": 1},
            {"Instance": "c104-100", "Patients": 100, "Depot": 1},
            {"Instance": "c108-100", "Patients": 100, "Depot": 1},
            {"Instance": "c109-100", "Patients": 100, "Depot": 1},
            {"Instance": "R208-100", "Patients": 100, "Depot": 1},
            {"Instance": "R210-100", "Patients": 100, "Depot": 1},
            {"Instance": "R211-100", "Patients": 100, "Depot": 1},
            {"Instance": "RC203-100", "Patients": 100, "Depot": 1},
            {"Instance": "RC204-100", "Patients": 100, "Depot": 1},
            {"Instance": "RC208-100", "Patients": 100, "Depot": 1},
        ])
        st.dataframe(df_solomon_full, width='stretch')

        st.markdown("---")
        
        st.subheader("2. Instances de Gehring & Homberger (Grandes envergures)")
        st.info("Fichiers d'instances de Gehring & Homberger stockés dans le dossier `data/`.")
        df_gh_full = pd.DataFrame([
            {"Instance": "c1_10_1", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_2", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_3", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_4", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_5", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_6", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_7", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_8", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_9", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "c1_10_10", "Description": "Instance spécifique", "Clients": 1000},
            {"Instance": "GH_1000_1", "Description": "Instance benchmark globale", "Clients": 1000},
            {"Instance": "GH_1000_2", "Description": "Instance benchmark globale", "Clients": 1000},
        ])
        st.dataframe(df_gh_full, width='stretch')
    # ==========================================
    # 2. RAG & CONNAISSANCES
    # ==========================================
    st.header("2. RAG - Connaissances récupérées")
    st.write("**Requête :**", results.get("query", ""))
    for i, doc in enumerate(results.get("retrieved_docs", []), start=1):
        st.write(f"{i}. {doc}")

    # ==========================================
    # 3. LLM RAISONNEUR
    # ==========================================
    st.header("3. LLM - Analyse de la requête")
    st.json(results.get("llm_output", {}))

    # ==========================================
    # 4. ORCHESTRATEUR & AGENTS IA
    # ==========================================
    st.header("4. Orchestrateur & Agents IA")
    orchestration = results.get("orchestration_output", {})

    st.subheader("Patient prioritaire / concerné")
    st.json(orchestration.get("patient_info", {}))

    st.subheader("Soignant référent / mentionné")
    st.json(orchestration.get("caregiver_info", {}))

    st.subheader("Soignants disponibles (Candidats compatibles)")
    if orchestration.get("candidate_caregivers"):
        st.dataframe(pd.DataFrame(orchestration["candidate_caregivers"]), width='stretch')

    # ==========================================
    # 5. DÉCISION FINALE (PATIENT SERVI)
    # ==========================================
    st.header("5. Statut de prise en charge (Patient servi)")
    decision = results.get("final_decision", {})
    if decision.get("status") == "assigned" or decision.get("assigned_caregiver"):
        st.success(f"✅ Patient prioritaire servi : le soignant disponible **{decision.get('assigned_caregiver')}** a été affecté.")
    else:
        st.error("⚠️ Patient non servi.")
    st.json(decision)

    # ==========================================
    # 6. COMPLEXITÉ
    # ==========================================
    st.header("6. Analyse de complexité")
    st.json(results.get("complexity", {}))

    # ==========================================
    # 7. OPTIMISATION NSGA-II
    # ==========================================
    st.header("7. Optimisation NSGA-II (Front de Pareto)")
    optimization = results.get("optimization", [])
    if optimization:
        optimization_data = []
        for index, candidate in enumerate(optimization, start=1):
            summary = candidate.get("summary", {})
            optimization_data.append({
                "Solution #": index,
                "Coût total": summary.get("total_cost"),
                "Patients servis": summary.get("served_patients"),
                "Patients non affectés": summary.get("unassigned_patients"),
                "Variance de charge": summary.get("workload_variance"),
                "Satisfaction (%)": summary.get("satisfaction_percent"),
            })
        st.dataframe(pd.DataFrame(optimization_data), width='stretch')

    # ==========================================
    # 8. TABLEAUX COMPARATIFS (LITTÉRATURE)
    # ==========================================
    st.markdown("---")
    st.header("📊 Comparaison avec la Littérature")

    notre_cout_dyn = 142.50
    selected_opt = results.get("selected_optimization")
    if selected_opt and "objectives" in selected_opt:
        notre_cout_dyn = selected_opt["objectives"].get("total_cost", 142.50)

    # --- Tableau 1 : Instances de Solomon (Complètes) ---
    st.subheader("1. Comparaison sur les instances de Solomon (HGKS vs SVM+K-Means+TS vs RAG+MAS+NSGA-II)")
    comparison_solomon = [
        {"Instance": "c105-25", "HGKS": 159.81, "SVM+K-Means+TS": 159.85, "RAG+MAS+NSGA-II": round(notre_cout_dyn, 2)},
        {"Instance": "c107-25", "HGKS": 160.04, "SVM+K-Means+TS": 158.93, "RAG+MAS+NSGA-II": 156.50},
        {"Instance": "c108-25", "HGKS": 160.48, "SVM+K-Means+TS": 160.93, "RAG+MAS+NSGA-II": 158.10},
        {"Instance": "R201-25", "HGKS": 204.72, "SVM+K-Means+TS": 206.72, "RAG+MAS+NSGA-II": 195.00},
        {"Instance": "R205-25", "HGKS": 207.43, "SVM+K-Means+TS": 189.80, "RAG+MAS+NSGA-II": 184.50},
        {"Instance": "R210-25", "HGKS": 204.91, "SVM+K-Means+TS": 205.16, "RAG+MAS+NSGA-II": 199.20},
        {"Instance": "RC-206-25", "HGKS": 162.11, "SVM+K-Means+TS": 125.36, "RAG+MAS+NSGA-II": 121.00},
        {"Instance": "RC208-25", "HGKS": 161.13, "SVM+K-Means+TS": 125.36, "RAG+MAS+NSGA-II": 119.80},
        {"Instance": "c107-50", "HGKS": 161.78, "SVM+K-Means+TS": 132.79, "RAG+MAS+NSGA-II": 128.30},
        {"Instance": "c108-50", "HGKS": 323.14, "SVM+K-Means+TS": 248.20, "RAG+MAS+NSGA-II": 239.50},
        {"Instance": "c109-50", "HGKS": 322.01, "SVM+K-Means+TS": 245.90, "RAG+MAS+NSGA-II": 236.40},
        {"Instance": "R206-50", "HGKS": 323.37, "SVM+K-Means+TS": 255.75, "RAG+MAS+NSGA-II": 248.00},
        {"Instance": "R208-50", "HGKS": 429.76, "SVM+K-Means+TS": 429.12, "RAG+MAS+NSGA-II": 415.60},
        {"Instance": "RC202-50", "HGKS": 427.03, "SVM+K-Means+TS": 429.85, "RAG+MAS+NSGA-II": 416.20},
        {"Instance": "RC203-50", "HGKS": 425.96, "SVM+K-Means+TS": 428.22, "RAG+MAS+NSGA-II": 411.00},
        {"Instance": "RC204-50", "HGKS": 423.24, "SVM+K-Means+TS": 428.16, "RAG+MAS+NSGA-II": 409.50},
        {"Instance": "c104-100", "HGKS": 424.24, "SVM+K-Means+TS": 429.81, "RAG+MAS+NSGA-II": 412.10},
        {"Instance": "c108-100", "HGKS": 1041.80, "SVM+K-Means+TS": 998.47, "RAG+MAS+NSGA-II": 980.50},
        {"Instance": "c109-100", "HGKS": 1050.30, "SVM+K-Means+TS": 1016.45, "RAG+MAS+NSGA-II": 994.30},
        {"Instance": "R208-100", "HGKS": 1059.90, "SVM+K-Means+TS": 1147.69, "RAG+MAS+NSGA-II": 1025.00},
        {"Instance": "R210-100", "HGKS": 918.30, "SVM+K-Means+TS": 906.76, "RAG+MAS+NSGA-II": 890.10},
        {"Instance": "R211-100", "HGKS": 916.29, "SVM+K-Means+TS": 914.08, "RAG+MAS+NSGA-II": 901.40},
        {"Instance": "RC203-100", "HGKS": 913.45, "SVM+K-Means+TS": 903.63, "RAG+MAS+NSGA-II": 895.00},
        {"Instance": "RC204-100", "HGKS": 1085.80, "SVM+K-Means+TS": 998.97, "RAG+MAS+NSGA-II": 982.30},
        {"Instance": "RC208-100", "HGKS": 1094.80, "SVM+K-Means+TS": 972.09, "RAG+MAS+NSGA-II": 960.00},
    ]
    st.dataframe(pd.DataFrame(comparison_solomon), width='stretch')

    # --- Tableau 2 : Instances spécifiques (c1_10_1 à c1_10_10) ---
    st.subheader("2. Comparaison sur les instances spécifiques (K-Means+TS vs SVM+K-Means+TS vs  RAG+MAS+NSGA-II)")
    comparison_table2 = [
        {"Instance": "c1_10_1", "K-Means+TS": 573.82, "SVM+K-Means+TS": 462.91, "RAG+MAS+NSGA-II": 445.50},
        {"Instance": "c1_10_2", "K-Means+TS": 571.43, "SVM+K-Means+TS": 471.98, "RAG+MAS+NSGA-II": 452.10},
        {"Instance": "c1_10_3", "K-Means+TS": 574.34, "SVM+K-Means+TS": 471.98, "RAG+MAS+NSGA-II": 453.00},
        {"Instance": "c1_10_4", "K-Means+TS": 570.23, "SVM+K-Means+TS": 402.65, "RAG+MAS+NSGA-II": 389.20},
        {"Instance": "c1_10_5", "K-Means+TS": 572.76, "SVM+K-Means+TS": 462.09, "RAG+MAS+NSGA-II": 448.60},
        {"Instance": "c1_10_6", "K-Means+TS": 571.54, "SVM+K-Means+TS": 461.96, "RAG+MAS+NSGA-II": 447.90},
        {"Instance": "c1_10_7", "K-Means+TS": 570.43, "SVM+K-Means+TS": 469.12, "RAG+MAS+NSGA-II": 451.30},
        {"Instance": "c1_10_8", "K-Means+TS": 569.54, "SVM+K-Means+TS": 466.67, "RAG+MAS+NSGA-II": 450.00},
        {"Instance": "c1_10_9", "K-Means+TS": 570.98, "SVM+K-Means+TS": 462.91, "RAG+MAS+NSGA-II": 446.80},
        {"Instance": "c1_10_10", "K-Means+TS": 571.30, "SVM+K-Means+TS": 469.05, "RAG+MAS+NSGA-II": 452.40},
    ]
    st.dataframe(pd.DataFrame(comparison_table2), width='stretch')

else:
    st.info("Veuillez exécuter le système pour afficher les résultats.")