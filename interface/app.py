"""Streamlit dashboard for the HHCOP / HHCRSP pipeline.

Design rules enforced here (see docs/REPAIR_REPORT.md):

* no hard-coded result or comparison number — every cell comes from a pipeline
  result or from a measured CSV on disk;
* the dataset panel always shows the *working set* actually used by the run and
  the source-file fingerprint, so a one-row table is immediately explainable;
* all tables are built with :mod:`experiments.comparison`, keep a stable
  ``item_id``, and support column selection, text filtering, sorting,
  pagination and CSV/JSON export;
* the UI never writes into ``data/`` — exports go to ``outputs/`` on demand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import dataset_loader  # noqa: E402
from data.integrity import DatasetIntegrityError, save_csv_atomic, save_json_atomic  # noqa: E402
from experiments import comparison  # noqa: E402
from main import (  # noqa: E402
    DEFAULT_PATIENT_LIMIT,
    DEFAULT_QUERY,
    dataset_status,
    run_hhcop_pipeline,
)
from optimisation.mtwmo.generator import DATASET_TABLE, LOCATION_SIDE  # noqa: E402

st.set_page_config(
    page_title="HHCRSP - MAS & NSGA-II",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Tableau de Bord - Home Health Care Routing & Scheduling (MAS + NSGA-II)")
st.caption(
    "Système multi-agents & optimisation multi-objectifs sous incertitude. "
    "Toutes les valeurs affichées sont mesurées; aucune table n'est codée en dur."
)

DISPLAY_COLUMNS = [
    "id", "source_row", "priority", "care_type", "x", "y",
    "service_hours", "required_skill_level",
]
CAREGIVER_COLUMNS = [
    "id", "skill", "skill_level", "available", "max_work_hours",
    "current_workload", "delay", "source",
]


def _sidebar_configuration() -> dict[str, Any]:
    """Collect every configurable input; no machine-specific paths."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        source_mode = st.radio(
            "Source des patients",
            ["teacher_csv", "generator"],
            format_func=lambda code: (
                "Jeu de données réel (teacher CSV, 1000 lignes)"
                if code == "teacher_csv"
                else "Instance générée P11 (A–F)"
            ),
            help=(
                "Le teacher CSV est lu en lecture seule. La limite ne change "
                "que le jeu de travail d'une exécution, jamais le fichier."
            ),
        )

        dataset_args: dict[str, Any] = {}
        if source_mode == "teacher_csv":
            patient_limit = st.slider(
                "Patients dans l'exécution",
                min_value=10,
                max_value=1000,
                value=min(50, DEFAULT_PATIENT_LIMIT),
                step=10,
                help="1000 patients restent sur disque; seuls les N premiers entrent dans l'optimisation.",
            )
            area = float(
                st.selectbox(
                    "Zone géographique (côté)",
                    options=[50.0, 100.0, 200.0],
                    index=0,
                    format_func=lambda value: f"{int(value)} × {int(value)}",
                )
            )
            dataset_args.update(
                patient_limit=int(patient_limit), area=area, caregiver_count=None
            )
            dataset_label = f"CSV_{patient_limit}p"
        else:
            dataset_code = st.selectbox("Dataset P11", options=list(DATASET_TABLE))
            patient_count, caregiver_options, location_options = DATASET_TABLE[dataset_code]
            caregivers = st.selectbox("Soignants", options=list(caregiver_options))
            location = st.selectbox(
                "Localisation",
                options=list(location_options),
                format_func=lambda code: f"{code} ({int(LOCATION_SIDE[code])}×{int(LOCATION_SIDE[code])})",
            )
            instance_seed = st.number_input("Graine de l'instance", min_value=0, value=0, step=1)
            dataset_args.update(
                dataset=dataset_code,
                caregivers=int(caregivers),
                location=int(location),
                instance_seed=int(instance_seed),
            )
            dataset_label = f"{dataset_code}_{caregivers}_{location}"

        st.subheader("Optimisation")
        population_size = st.slider("Taille de population", 4, 100, 12, step=2)
        generations = st.slider("Générations", 1, 200, 5, step=1)
        algorithm = st.selectbox(
            "Algorithme principal",
            ["nsga2", "mtcmo"],
            format_func=lambda code: "NSGA-II" if code == "nsga2" else "MTCMOLite (baseline P10)",
        )
        decision_method = st.selectbox(
            "Règle de décision",
            ["weighted_sum", "ideal_point", "legacy"],
        )
        gamma_t = st.slider(
            "Budget d'incertitude Γ",
            0.0, 5.0, 0.0, step=0.5,
            help="Γ=0 reproduit le modèle déterministe (P5).",
        )

        st.subheader("RAG")
        rag_top_k = st.number_input(
            "Documents RAG max (0 = tous)", min_value=0, max_value=20, value=0, step=1
        )

        st.subheader("Comparaison mesurée (P10)")
        compare_algorithms = st.checkbox(
            "Exécuter NSGA-II vs MTCMOLite sur la même instance",
            value=False,
            help="Nécessite une instance générée (A–F). Les deux algorithmes sont mesurés, pas simulés.",
        )
        comparison_runs = st.slider("Répétitions par algorithme", 1, 5, 2)
        comparison_population = st.slider("Population (comparaison)", 6, 60, 20, step=2)
        comparison_generations = st.slider("Générations (comparaison)", 2, 100, 15)

        st.subheader("Scénario")
        query = st.text_area("Requête HHCOP", value=DEFAULT_QUERY, height=80)
        model = st.text_input("Modèle Ollama (optionnel)", value="qwen3:4b")
        run_requested = st.button("▶ Exécuter le système", type="primary", width="stretch")

        st.divider()
        st.caption(
            "Ollama n'est appelé que si USE_OLLAMA=1; sinon le raisonneur utilise "
            "son fallback déterministe validé."
        )

    return {
        "dataset_args": dataset_args,
        "dataset_label": dataset_label,
        "population_size": int(population_size),
        "generations": int(generations),
        "algorithm": algorithm,
        "decision_method": decision_method,
        "gamma_t": float(gamma_t),
        "rag_top_k": int(rag_top_k) or None,
        "compare_algorithms": bool(compare_algorithms),
        "comparison_runs": int(comparison_runs),
        "comparison_population": int(comparison_population),
        "comparison_generations": int(comparison_generations),
        "query": query,
        "model": model,
        "run_requested": run_requested,
    }


def _execute(config: dict[str, Any]) -> None:
    """Run the pipeline and keep results + history in session state."""
    kwargs = {
        **config["dataset_args"],
        "population_size": config["population_size"],
        "generations": config["generations"],
        "algorithm": config["algorithm"],
        "decision_method": config["decision_method"],
        "gamma_t": config["gamma_t"],
        "rag_top_k": config["rag_top_k"],
        "compare_algorithms": config["compare_algorithms"],
        "comparison_runs": config["comparison_runs"],
        "comparison_population": config["comparison_population"],
        "comparison_generations": config["comparison_generations"],
    }
    with st.spinner("Pipeline en cours : RAG → LLM → agents → optimisation → décision…"):
        results = run_hhcop_pipeline(config["query"], model=config["model"], **kwargs)
    st.session_state.analysis_results = results
    st.session_state.analysis_config = {
        key: value for key, value in config.items() if key != "run_requested"
    }
    history = st.session_state.setdefault("history", [])
    history.append(results)
    st.session_state.history = history[-25:]


def _render_table(
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
    columns: Sequence[str] | None = None,
    empty_message: str = "Aucune donnée mesurée disponible.",
) -> None:
    """Consistent, scalable table: select columns, filter, sort, paginate, export."""
    if not rows:
        st.info(empty_message)
        return
    available = [column for column in (columns or list(rows[0].keys())) if column in rows[0]]
    chosen = st.multiselect(
        "Colonnes affichées", available, default=available, key=f"{key}_columns"
    )
    left, middle, right = st.columns([2, 1, 1])
    with left:
        query = st.text_input("Filtrer (toutes colonnes)", key=f"{key}_filter")
    with middle:
        sort_column = st.selectbox(
            "Trier par", ["—"] + available, key=f"{key}_sort"
        )
    with right:
        ascending = st.radio(
            "Ordre", ["Croissant", "Décroissant"], horizontal=True, key=f"{key}_order"
        ) == "Croissant"

    filtered = comparison.filter_rows(rows, query)
    ordered = comparison.sort_rows(
        filtered, None if sort_column == "—" else sort_column, ascending
    )
    page_size = st.select_slider(
        "Lignes par page", options=[10, 25, 50, 100], value=25, key=f"{key}_page_size"
    )
    pages = max(1, math.ceil(len(ordered) / page_size))
    page_key = f"{key}_page"
    if page_key not in st.session_state or st.session_state[page_key] > pages:
        st.session_state[page_key] = 1
    page = st.number_input(
        f"Page (1–{pages})", min_value=1, max_value=pages, step=1, key=page_key
    )
    page_data = comparison.paginate(ordered, int(page), int(page_size))
    st.caption(
        f"{page_data['total']} ligne(s) · page {page_data['page']}/{page_data['pages']} · "
        f"affichage {page_data['start']}–{page_data['end']}"
    )
    display_columns = chosen or available
    st.dataframe(
        pd.DataFrame(page_data["rows"])[display_columns],
        width="stretch",
        hide_index=True,
    )

    download_left, download_middle, download_right = st.columns(3)
    with download_left:
        st.download_button(
            "⬇ CSV (toutes les lignes filtrées)",
            data=comparison.export_csv(ordered),
            file_name=f"{key}.csv",
            mime="text/csv",
            key=f"{key}_csv",
        )
    with download_middle:
        st.download_button(
            "⬇ JSON",
            data=comparison.export_json(ordered),
            file_name=f"{key}.json",
            mime="application/json",
            key=f"{key}_json",
        )
    with download_right:
        if st.button("💾 Enregistrer dans outputs/", key=f"{key}_save"):
            try:
                output_dir = PROJECT_ROOT / "outputs"
                save_csv_atomic(ordered, output_dir / f"{key}.csv")
                save_json_atomic(ordered, output_dir / f"{key}.json")
                st.success(f"Écrit dans outputs/{key}.csv et outputs/{key}.json")
            except Exception as error:  # surface, never fail silently
                st.error(f"Échec de l'enregistrement : {error}")


def _render_dataset(results: dict[str, Any]) -> None:
    info = results.get("dataset_info", {})
    st.header("1. Jeu de données")
    first, second, third, fourth = st.columns(4)
    first.metric("Lignes sur disque", info.get("source_rows", "—"))
    second.metric("Patients (exécution)", len(results.get("patients", [])))
    third.metric("Soignants", len(results.get("soignants", [])))
    fourth.metric("Intégrité", "OK" if info.get("integrity", {}).get("ok") else "À vérifier")

    st.caption(
        f"Source : `{info.get('source')}` · instance `{info.get('instance')}` · "
        f"sha256 `{str(info.get('source_sha256', ''))[:16]}`"
    )
    st.dataframe(
        pd.DataFrame(results["patients"])[
            [column for column in DISPLAY_COLUMNS if column in results["patients"][0]]
        ],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Soignants (agents)"):
        st.dataframe(
            pd.DataFrame(results["soignants"])[
                [column for column in CAREGIVER_COLUMNS if column in results["soignants"][0]]
            ],
            width="stretch",
            hide_index=True,
        )
    with st.expander("Contrôles d'intégrité"):
        st.json({
            "dataset_info": info,
            "records": info.get("records", {}),
            "complexity_data_integrity": results["complexity"].get("data_integrity", {}),
        })


def _render_rag(results: dict[str, Any]) -> None:
    st.header("2. RAG — connaissances récupérées")
    st.write("**Requête :**", results["query"])
    documents = results.get("retrieved_docs", [])
    rag = results.get("rag", {})
    stats = rag.get("embedding_stats") or {}
    st.caption(
        f"Documents indexés : {rag.get('chunks', 0)} chunks · "
        f"embeddings en cache : {stats.get('cached_vectors', '—')} · "
        f"hits/miss : {stats.get('hits', '—')}/{stats.get('misses', '—')}"
    )
    if not documents:
        st.info("Aucun document pertinent pour cette requête.")
        return
    st.dataframe(
        pd.DataFrame([
            {
                "ID": document.get("id"),
                "Catégorie": document.get("category"),
                "Score": document.get("score"),
                "Source": document.get("source"),
                "Description": document.get("description") or document.get("text"),
            }
            for document in documents
        ]),
        width="stretch",
        hide_index=True,
    )


def _render_llm_and_agents(results: dict[str, Any]) -> None:
    st.header("3. LLM & orchestration multi-agents")
    st.caption(
        f"Modèle : `{results.get('llm_model')}` · source : "
        f"`{results.get('llm_output', {}).get('reasoning_source')}`"
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Sortie structurée du raisonneur")
        st.json(results.get("llm_output", {}))
    orchestration = results.get("orchestration_output", {}) or {}
    with right:
        st.subheader("Patient identifié")
        if orchestration.get("patient_info"):
            st.json(orchestration["patient_info"])
        else:
            st.warning("Aucun patient identifié dans la requête.")
        st.subheader("Soignant mentionné")
        if orchestration.get("caregiver_info"):
            st.json(orchestration["caregiver_info"])
        else:
            st.info("Aucun soignant mentionné.")
    candidates = orchestration.get("candidate_caregivers") or []
    if candidates:
        st.subheader("Soignants candidats")
        st.dataframe(pd.DataFrame(candidates), width="stretch", hide_index=True)


def _render_decision(results: dict[str, Any]) -> None:
    st.header("4. Décision finale")
    decision = results.get("final_decision", {}) or {}
    if decision.get("status") == "assigned":
        st.success(f"Soignant affecté : {decision.get('assigned_caregiver')}")
    elif decision.get("status") == "unassigned":
        st.error("Aucun soignant n'a pu être affecté.")
    else:
        st.warning(decision.get("message", "Décision indisponible."))
    st.json(decision)
    plan = results.get("decision", {}).get("visit_plan", [])
    if plan:
        st.subheader("Plan de visite")
        st.dataframe(pd.DataFrame(plan), width="stretch", hide_index=True)


def _render_pareto(results: dict[str, Any]) -> None:
    st.header("5. Optimisation multi-objectifs")
    analysis = results.get("optimization_analysis", {})
    st.caption(
        f"Algorithme : {analysis.get('algorithm')} · "
        f"population {analysis.get('population_size')} · "
        f"générations {analysis.get('generations')} · "
        f"front de Pareto : {analysis.get('pareto_solutions')} solutions"
    )
    rows = results.get("comparison", {}).get("solutions", [])
    if not rows:
        st.info("Aucune solution non dominée pour cette exécution.")
        return
    chart_frame = pd.DataFrame([
        {
            "Z1 (coût)": row["z1_total_cost"],
            "Z2 (satisfaction)": row["z2_satisfaction"],
            "Z3 (variance charge)": row["z3_workload_variance"],
        }
        for row in rows
        if row["z1_total_cost"] is not None
    ])
    if not chart_frame.empty:
        st.scatter_chart(
            chart_frame, x="Z1 (coût)", y="Z2 (satisfaction)",
            color="Z3 (variance charge)", size="Z3 (variance charge)",
        )
    _render_table(rows, key="pareto_solutions", columns=comparison.SOLUTION_COLUMNS)


def _render_comparison(results: dict[str, Any]) -> None:
    st.markdown("---")
    st.header("📊 Comparaison (mesurée)")
    st.caption(
        "Les tableaux ci-dessous proviennent uniquement de calculs exécutés. "
        "Une source absente est signalée; aucune valeur de littérature n'est recopiée."
    )
    tables = results.get("comparison", {})
    history = st.session_state.get("history", [])

    tab_solutions, tab_runs, tab_algorithms, tab_ml = st.tabs([
        "Solutions du run",
        "Configurations exécutées",
        "NSGA-II vs MTCMOLite",
        "Baseline ML (D3)",
    ])
    with tab_solutions:
        _render_table(
            tables.get("solutions", []),
            key="solutions_table",
            columns=comparison.SOLUTION_COLUMNS,
            empty_message="Aucune solution mesurée pour cette exécution.",
        )
    with tab_runs:
        run_rows = comparison.configurations_table(history)
        st.caption(
            f"{len(run_rows)} exécution(s) dans cette session. Chaque ligne est une "
            "configuration distincte (jeu de données, tailles, algorithme, modèle)."
        )
        _render_table(
            run_rows,
            key="configurations_table",
            columns=comparison.CONFIGURATION_COLUMNS,
            empty_message="Aucune exécution enregistrée dans la session.",
        )
    with tab_algorithms:
        status = tables.get("algorithm_status", {})
        if not status.get("available"):
            st.warning(status.get("message", "Comparaison non disponible."))
        else:
            st.write(status.get("message"))
            algorithm_rows = tables.get("algorithms", [])
            _render_table(
                algorithm_rows,
                key="algorithm_comparison",
                columns=comparison.ALGORITHM_COLUMNS,
                empty_message="Aucune mesure d'algorithme disponible.",
            )
            deltas = comparison.algorithm_deltas(algorithm_rows)
            if deltas:
                st.subheader("Écarts mesurés par instance")
                st.dataframe(pd.DataFrame(deltas), width="stretch", hide_index=True)
            runs = tables.get("algorithm_runs", [])
            if runs:
                with st.expander("Détail des répétitions"):
                    st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
            st.caption(
                "MTCMOLite est une ré-implémentation du principe décrit par Qiao et al. "
                "(2022), jamais présentée comme l'algorithme MTCMO original."
            )
    with tab_ml:
        ml = tables.get("ml_baseline", {"rows": [], "status": {}})
        st.caption(ml.get("status", {}).get("message", ""))
        _render_table(
            ml.get("rows", []),
            key="ml_baseline",
            columns=comparison.ML_COLUMNS,
            empty_message=(
                "Aucune mesure ML disponible. Exécutez `python -m ml.evaluation` "
                "puis rechargez cette page."
            ),
        )


def _render_complexity(results: dict[str, Any]) -> None:
    st.header("6. Complexité & ressources")
    complexity = results.get("complexity", {}) or {}
    timing_rows = [
        {"Composant": component, "Temps (s)": data["execution_time_sec"]}
        for component, data in complexity.items()
        if isinstance(data, dict) and "execution_time_sec" in data
    ]
    if timing_rows:
        frame = pd.DataFrame(timing_rows).sort_values("Temps (s)", ascending=False)
        st.dataframe(frame, width="stretch", hide_index=True)
        st.bar_chart(frame.set_index("Composant"))

    resources = complexity.get("resources", {}) or {}
    if resources:
        st.subheader("Empreinte CPU / mémoire")
        left, middle, right, last = st.columns(4)
        left.metric("Temps total (s)", resources.get("wall_time_sec", "—"))
        middle.metric("Pic Python (MB)", resources.get("peak_python_mb", "—"))
        right.metric("RSS fin (MB)", resources.get("rss_mb_stop", "—"))
        last.metric("Cœurs", resources.get("cpu_count", "—"))

    interactions = complexity.get("agent_interactions")
    if interactions:
        st.subheader("Agents")
        left, middle, right = st.columns(3)
        left.metric("Agents patients", interactions.get("patient_agents", 0))
        middle.metric("Agents soignants", interactions.get("caregiver_agents", 0))
        right.metric("Interactions potentielles", interactions.get("possible_interactions", 0))

    with st.expander("Métriques complètes (JSON)"):
        st.json(complexity)


def main() -> None:
    config = _sidebar_configuration()

    if config["run_requested"]:
        try:
            _execute(config)
        except DatasetIntegrityError as error:
            st.error("Intégrité du jeu de données compromise — exécution interrompue.")
            st.exception(error)
            return
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            st.error("Impossible d'exécuter le pipeline.")
            st.exception(error)
            return

    results = st.session_state.get("analysis_results")
    if results is None:
        st.info(
            "Configurez le scénario dans la barre latérale, puis cliquez sur "
            "« Exécuter le système ». La première exécution peut prendre quelques secondes."
        )
        st.subheader("Aperçu de la source de données")
        st.json(dataset_status())
        return

    run_config = st.session_state.get("analysis_config", {})
    if run_config and not config["run_requested"]:
        st.caption(
            f"Résultats de la dernière exécution — instance `{run_config.get('dataset_label')}` · "
            f"algorithme `{run_config.get('algorithm')}` · "
            f"population {run_config.get('population_size')} · "
            f"générations {run_config.get('generations')}`."
        )

    _render_dataset(results)
    _render_rag(results)
    _render_llm_and_agents(results)
    _render_decision(results)
    _render_pareto(results)
    _render_comparison(results)
    _render_complexity(results)


main()
