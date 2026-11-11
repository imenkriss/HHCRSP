"""HHCOP / HHCRSP pipeline entry point.

Execution pipeline (must match the architecture diagram):

    Streamlit UI -> RAG retriever -> LLM reasoner -> multi-agent orchestration
    -> multi-objective optimisation -> decision engine
    (transversal: complexity/performance metrics)

Dataset policy
--------------
Source data is **read-only**. Patients come from
``data/dementia/new_dataset_with_predictions.csv`` (1000 rows) and caregivers
from ``data/admin_caregivers.csv`` when it is non-empty, otherwise from a
deterministic scaffold. The old ``ensure_default_csv_files`` helper wrote a
one-row CSV *into* ``data/`` and that is what made a run appear to replace the
dataset with a single row; it has been removed. Every run snapshots the source
files (sha256 + rows + columns) and asserts they are unchanged at the end.

Generated artefacts (embedding cache, exports) go to ``data/generated/`` or
``outputs/`` only.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from agents.agentOrchestrateur import Orchestrator
from complexity.metrics import ComplexityMetrics, ResourceMonitor
from data import dataset_loader
from data.integrity import (
    GENERATED_DIR,
    DatasetIntegrityError,
    assert_unchanged,
    check_records,
    snapshot_files,
)
from experiments import comparison
from llm.raisonner import LLMRaisonner
from optimisation.nsga2 import NSGA2
from rag.embeddings import CachedEmbedder, HashingEmbedder
from rag.rag import RAGPipeline

DecisionEngine = importlib.import_module("décision.decision").DecisionEngine

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_QUERY = "Get the info for patient P1 with high priority"
DEFAULT_PATIENT_LIMIT = dataset_loader.DEFAULT_PATIENT_LIMIT
DEFAULT_AREA = dataset_loader.LOCATION_AREA[1]

#: Files whose integrity is verified before/after every run.
SOURCE_FILES: tuple[Path, ...] = (
    dataset_loader.TEACHER_CSV,
    dataset_loader.CAREGIVER_CSV,
    DATA_DIR / "dementia_patients_health_data.csv",
)

INT_FIELDS = ("max_work_hours", "current_workload", "delay", "skill_level")
BOOL_FIELDS = ("available",)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset access (read-only)
# --------------------------------------------------------------------------- #
def dataset_status() -> dict[str, Any]:
    """Fingerprint every source file without modifying it."""
    snapshots = snapshot_files(SOURCE_FILES)
    return {path: snapshot.as_dict() for path, snapshot in snapshots.items()}


def load_csv(filename: str) -> list[dict]:
    """Read one auxiliary CSV from ``data/`` (read-only, kept for compatibility)."""
    path = DATA_DIR / filename
    if not path.exists():
        return []
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [_convert(row) for row in csv.DictReader(handle)]


def _convert(row: dict) -> dict:
    """Convert CSV strings into int/bool/float where unambiguous."""
    converted: dict[str, Any] = {}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            continue
        key = raw_key.strip()
        value = (raw_value or "").strip()
        if key in INT_FIELDS or key in ("Diabetic", "Dementia"):
            try:
                converted[key] = int(float(value)) if value else 0
            except ValueError:
                converted[key] = value
        elif key in BOOL_FIELDS:
            converted[key] = value.lower() in ("true", "1", "yes", "oui")
        else:
            try:
                if "." in value:
                    converted[key] = float(value)
                else:
                    converted[key] = int(value) if value.isdigit() else value
            except ValueError:
                converted[key] = value
    return converted


def load_dataset(
    patient_limit: int | None = DEFAULT_PATIENT_LIMIT,
    caregiver_count: int | None = None,
    area: float = DEFAULT_AREA,
    seed: int = 42,
    source: str | Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load patients + caregivers from the canonical, read-only sources.

    ``patient_limit=None`` uses all 1000 teacher rows. The default working set
    is 100 patients with ``ceil(100/4) = 25`` caregivers, which keeps a run
    practical on a CPU-only laptop; the full file is never truncated on disk.
    """
    patients = dataset_loader.load_hhcop_patients(
        limit=patient_limit, area=area, seed=seed, source=source
    )
    caregivers = dataset_loader.load_hhcop_caregivers(
        count=caregiver_count, patient_count=len(patients), area=area, seed=seed
    )
    return patients, caregivers


def _validate_records(
    patients: Sequence[dict], caregivers: Sequence[dict]
) -> dict[str, dict[str, Any]]:
    """Validate both record sets and refuse duplicate identifiers."""
    patient_report = check_records(
        patients, label="patients", required=("id", "x", "y", "time_windows")
    )
    caregiver_report = check_records(
        caregivers,
        label="caregivers",
        required=("id", "max_work_hours", "available"),
    )
    problems = patient_report.problems + caregiver_report.problems
    if problems:
        raise DatasetIntegrityError(
            "Invalid working dataset:\n" + "\n".join(f"  - {item}" for item in problems)
        )
    return {
        "patients": patient_report.as_dict(),
        "caregivers": caregiver_report.as_dict(),
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _pareto_points(pareto_front: Any) -> list[tuple[float, ...]]:
    solutions = getattr(pareto_front, "solutions", pareto_front) or []
    points: list[tuple[float, ...]] = []
    for candidate in solutions:
        objectives = getattr(candidate, "objectives", None)
        if objectives is None and isinstance(candidate, dict):
            objectives = candidate.get("objectives")
        if objectives is None:
            continue
        try:
            points.append(tuple(float(value) for value in objectives))
        except (TypeError, ValueError):
            continue
    return points


def _selected_route_details(selection: dict | None) -> list[Any]:
    """Route details of the selected candidate, when the optimiser exposes them."""
    candidate = (selection or {}).get("candidate")
    evaluation = getattr(candidate, "evaluation", None)
    if evaluation is not None and getattr(evaluation, "routes", None):
        return list(evaluation.routes)
    if isinstance(candidate, dict):
        routes = candidate.get("routes")
        if isinstance(routes, list):
            return routes
    return []


def _pareto_size(pareto_front: Any) -> int:
    size = getattr(pareto_front, "size", None)
    if size is not None:
        return int(size)
    return len(pareto_front or [])


def run_hhcop_pipeline(
    query: str = DEFAULT_QUERY,
    model: str = "qwen3:4b",
    patient_event: dict | None = None,
    caregiver_event: dict | None = None,
    dataset: str | None = None,
    caregivers: int | None = None,
    location: int | None = None,
    instance_seed: int | None = None,
    population_size: int = 20,
    generations: int = 10,
    local_search: bool = True,
    algorithm: str = "nsga2",
    decision_method: str = "weighted_sum",
    gamma_t: float = 0.0,
    gamma_s: float = 0.0,
    gamma_p: float = 0.0,
    patient_limit: int | None = DEFAULT_PATIENT_LIMIT,
    caregiver_count: int | None = None,
    area: float | None = None,
    rag_top_k: int | None = None,
    compare_algorithms: bool = False,
    comparison_runs: int = 2,
    comparison_population: int | None = None,
    comparison_generations: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the complete pipeline and return every stage's output.

    ``dataset`` (A-F) switches to the P11 seeded generator; without it the
    teacher CSV is used. ``algorithm`` selects ``"nsga2"`` (legacy assignment
    sampler used by the UI) or ``"mtcmo"`` (``MTCMOLite`` on a generated
    instance — the honest label for the Qiao et al. idea baseline).
    """
    monitor = ResourceMonitor().start()
    metrics = ComplexityMetrics()
    total_start = metrics.start()

    # 0. Dataset (with integrity snapshot taken *before* touching anything)
    start = metrics.start()
    snapshots_before = snapshot_files(SOURCE_FILES)
    generated_instance = None
    if dataset is not None or caregivers is not None or location is not None or instance_seed is not None:
        from optimisation.mtwmo.generator import generate_instance

        dataset_code = (dataset or "A").upper()
        caregiver_total = int(caregivers or 2)
        location_code = int(location or 1)
        generated_instance = generate_instance(
            dataset_code,
            caregiver_total,
            location_code,
            seed=(instance_seed if instance_seed is not None else 0),
        )
        patients = [
            {
                "id": f"P{index}",
                "care_type": getattr(patient, "care_type", None) or "General",
                "priority": getattr(patient, "priority", "Medium"),
                "required_skill_level": int(getattr(patient, "demand_level", 1)),
                "service_hours": getattr(patient, "service_time", 30) / 60.0,
                "x": getattr(patient, "x", 0.0),
                "y": getattr(patient, "y", 0.0),
                "time_windows": list(getattr(patient, "time_windows", [(0.0, 480.0)])),
                "source": "generator",
            }
            for index, patient in enumerate(generated_instance.patients, start=1)
        ]
        soignants = [
            {
                "id": f"S{index}",
                "skill": getattr(caregiver, "specialty", None) or "General",
                "skill_level": int(getattr(caregiver, "skill_level", 1)),
                "available": True,
                "current_workload": 0,
                "max_work_hours": 8,
                "max_overtime_hours": 4,
                "delay": 0,
                "x": float(generated_instance.depot["x"]),
                "y": float(generated_instance.depot["y"]),
                "time_windows": [(0.0, 480.0)],
                "source": "generator",
            }
            for index, caregiver in enumerate(generated_instance.caregivers, start=1)
        ]
        instance_name = generated_instance.name
        dataset_info: dict[str, Any] = {
            "source": "generator",
            "source_path": None,
            "source_rows": len(generated_instance.patients),
            "requested_patient_limit": None,
            "working_patients": len(patients),
            "working_caregivers": len(soignants),
            "caregiver_source": "generator",
            "instance": instance_name,
            "seed": seed if instance_seed is None else instance_seed,
        }
    else:
        work_area = float(area) if area is not None else DEFAULT_AREA
        patients, soignants = load_dataset(
            patient_limit=patient_limit,
            caregiver_count=caregiver_count,
            area=work_area,
            seed=seed,
        )
        instance_name = f"CSV_{len(patients)}p_{len(soignants)}c"
        source_snapshot = snapshots_before[str(dataset_loader.TEACHER_CSV)]
        dataset_info = {
            "source": "teacher_csv",
            "source_path": str(dataset_loader.TEACHER_CSV),
            "source_rows": source_snapshot.rows,
            "source_sha256": source_snapshot.sha256,
            "requested_patient_limit": patient_limit,
            "working_patients": len(patients),
            "working_caregivers": len(soignants),
            "caregiver_source": soignants[0].get("source") if soignants else None,
            "instance": instance_name,
            "area": work_area,
            "seed": seed,
        }

    if patient_event:
        patients.append(dict(patient_event))
    if caregiver_event:
        soignants.append(dict(caregiver_event))

    records = _validate_records(patients, soignants)
    dataset_info["records"] = records
    metrics.record("Dataset", start)

    # 1. RAG (keyword retriever + cached embeddings)
    start = metrics.start()
    embedder = CachedEmbedder(HashingEmbedder(), cache_dir=GENERATED_DIR / "embeddings")
    rag_pipeline = RAGPipeline(embedder=embedder)
    rag_output = rag_pipeline.run(query, top_k=rag_top_k)
    retrieved_documents = rag_output["documents"]
    llm_context = [item["document"] for item in rag_output["retrieved"]]
    metrics.record("RAG", start)

    # 2. LLM (structured, validated fallback when Ollama is off/unavailable)
    start = metrics.start()
    reasoner = LLMRaisonner(model=model)
    llm_output = reasoner.analyze(query, llm_context)
    metrics.record("LLM", start)

    # 3. Orchestrator + agents
    start = metrics.start()
    orchestration_output = Orchestrator(patients, soignants).process(llm_output)
    metrics.record("Orchestrator + Agents", start)

    # 4. Multi-objective optimisation
    start = metrics.start()
    algorithm_name = "MTCMOLite" if str(algorithm).lower() == "mtcmo" else "NSGA-II"
    if algorithm_name == "MTCMOLite":
        if generated_instance is None:
            raise ValueError(
                "algorithm='mtcmo' requires a generated dataset (A-F with caregivers "
                "and location); use algorithm='nsga2' for teacher-CSV runs."
            )
        from optimisation.mtwmo.algorithms import MTCMOLite

        optimizer = MTCMOLite(population_size=population_size, generations=generations)
        optimization_results: Any = optimizer.run(
            generated_instance,
            seed=(instance_seed if instance_seed is not None else 0),
            gamma={"t": gamma_t, "s": gamma_s, "p": gamma_p},
        )
    else:
        optimizer = NSGA2(
            patients,
            soignants,
            population_size=population_size,
            generations=generations,
            uncertainty_scenarios=32,
            gamma_t=gamma_t,
            gamma_s=gamma_s,
            gamma_p=gamma_p,
        )
        optimization_results = optimizer.optimize()
    optimization_analysis = {
        "algorithm": algorithm_name,
        "population_size": population_size,
        "generations": generations,
        "local_search": local_search,
        "gamma": {"gamma_t": gamma_t, "gamma_s": gamma_s, "gamma_p": gamma_p},
        "pareto_solutions": _pareto_size(optimization_results),
    }
    metrics.record("NSGA-II Optimization", start)

    # 5. Decision engine
    start = metrics.start()
    engine = DecisionEngine(method=decision_method)
    selected_optimization = engine.select_from_pareto_set(optimization_results)
    assignments = selected_optimization["assignments"] if selected_optimization else []
    selected_patient = (orchestration_output or {}).get("patient_info")
    selected_assignment = next(
        (
            assignment
            for assignment in assignments
            if selected_patient and assignment["patient_id"] == selected_patient.get("id")
        ),
        None,
    )

    if not selected_patient:
        final_decision = {
            "status": "error",
            "message": "No patient information found in the request.",
        }
    elif selected_assignment:
        final_decision = {
            "status": "assigned",
            "patient_id": selected_patient.get("id"),
            "assigned_caregiver": selected_assignment["caregiver_id"],
            "reason": "Assignment selected from the Pareto compromise.",
            "method": decision_method,
        }
    else:
        final_decision = {
            "status": "unassigned",
            "patient_id": selected_patient.get("id"),
            "assigned_caregiver": None,
            "reason": "No feasible assignment for this patient.",
            "method": decision_method,
        }

    decision_payload = {
        "method": decision_method,
        "objectives": selected_optimization.get("objectives", {}) if selected_optimization else {},
        "visit_plan": selected_optimization.get("visit_plan", []) if selected_optimization else [],
        "justification": (
            selected_optimization.get("justification", "No Pareto solution selected.")
            if selected_optimization else "No Pareto solution selected."
        ),
        "assignments": assignments,
        "status": selected_optimization.get("status", "empty") if selected_optimization else "empty",
    }
    metrics.record("Decision Engine", start)

    # 6. Cross-cutting metrics
    start = metrics.start()
    metrics.problem_size(patients, soignants)
    metrics.agent_interactions(patients, soignants, observed_count=0)
    metrics.caregiver_delay(soignants)
    metrics.workload_balance(soignants, assignments)
    metrics.unassigned_patients(patients, assignments)
    metrics.patient_satisfaction(patients, soignants, assignments)
    metrics.delays_and_overtime(_selected_route_details(selected_optimization), soignants)
    metrics.optimization_indicators(selected_optimization)
    metrics.uncertainty_indicators(llm_output, rag_output["context"])
    metrics.metrics["data_integrity"] = {
        "records": records,
        "source_snapshot": {
            path: snapshot.as_dict() for path, snapshot in snapshots_before.items()
        },
    }
    points = _pareto_points(optimization_results)
    if points:
        reference = tuple(
            max(point[index] for point in points) + 1.0 for index in range(len(points[0]))
        )
        metrics.pareto_quality(points, reference, feasible_count=len(points))
    metrics.record("Metrics", start)

    # 7. Comparison tables (measured only)
    comparison_tables: dict[str, Any] = {
        "solutions": comparison.solution_table(optimization_results),
        "configurations": [],
        "algorithms": [],
        "algorithm_runs": [],
        "ml_baseline": comparison.ml_baseline_table(),
        "algorithm_status": {
            "available": False,
            "message": (
                "Live algorithm comparison requires a generated P11 instance "
                "(dataset A-F)."
            ),
        },
    }
    if compare_algorithms and generated_instance is not None:
        try:
            comparison_tables["algorithm_status"] = {
                "available": True,
                "message": "NSGA-II vs MTCMOLite measured on the same generated instance.",
            }
            live = comparison.algorithm_comparison(
                dataset_info["instance"].split("_")[0],
                len(soignants),
                int(location or 1),
                seed=(instance_seed if instance_seed is not None else 0),
                population_size=int(comparison_population or population_size),
                generations=int(comparison_generations or generations),
                runs=int(comparison_runs),
                gamma=gamma_t,
            )
            comparison_tables["algorithms"] = live["rows"]
            comparison_tables["algorithm_runs"] = live["runs"]
        except Exception as error:  # keep the pipeline usable, report the failure
            logger.warning("Algorithm comparison failed: %s", error)
            comparison_tables["algorithm_status"] = {
                "available": False,
                "message": f"Algorithm comparison failed: {error}",
            }

    # 8. Integrity: source files must be byte-identical to the start of the run
    snapshots_after = snapshot_files(SOURCE_FILES)
    assert_unchanged(snapshots_before, snapshots_after, label="source dataset")
    dataset_info["integrity"] = {
        "ok": True,
        "checked_files": list(snapshots_after),
        "sha256": {
            path: snapshot.sha256 for path, snapshot in snapshots_after.items()
        },
    }

    metrics.resources(monitor.stop())
    metrics.record("Total Pipeline", total_start)

    decision = dict(decision_payload)
    decision["instance_info"] = {"name": instance_name}

    result: dict[str, Any] = {
        "query": query,
        "patients": patients,
        "soignants": soignants,
        "dataset_info": dataset_info,
        "retrieved_docs": retrieved_documents,
        "rag": {
            "documents": retrieved_documents,
            "context": rag_output["context"],
            "chunks": rag_output["chunks"],
            "embedding_stats": rag_output["embedding_stats"],
        },
        "llm_model": reasoner.model,
        "llm_output": llm_output,
        "orchestration_output": orchestration_output,
        "final_decision": final_decision,
        "decision": decision,
        "optimization": optimization_results,
        "selected_optimization": selected_optimization,
        "optimization_analysis": optimization_analysis,
        "instance_info": {"name": instance_name},
        "complexity": metrics.get_metrics(),
        "comparison": comparison_tables,
    }
    # The configuration row is filled last so it can carry runtime/memory.
    result["comparison"]["configurations"] = comparison.configurations_table([result])
    return result


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #
def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_demo(results: dict) -> None:
    dataset_info = results.get("dataset_info", {})
    _section("0. SCENARIO")
    print(results["query"])
    _section("1. DATASET")
    print(
        f"Source          : {dataset_info.get('source')} "
        f"({dataset_info.get('source_rows')} rows on disk)"
    )
    print(
        f"Working set     : {len(results['patients'])} patients | "
        f"{len(results['soignants'])} caregivers"
    )
    print(f"Integrity       : {dataset_info.get('integrity', {}).get('ok')}")
    _section("2. RAG")
    for document in results.get("retrieved_docs", []):
        print(f"  {document.get('id')}: {document.get('description') or document.get('text')}")
    _section("3. LLM")
    print(f"  source : {results['llm_output'].get('reasoning_source')}")
    print(f"  patient: {results['llm_output'].get('patient_id')}")
    _section("4. OPTIMISATION")
    analysis = results.get("optimization_analysis", {})
    print(f"  algorithm : {analysis.get('algorithm')}")
    print(f"  front     : {analysis.get('pareto_solutions')} solutions")
    _section("5. DECISION")
    print(json.dumps(results["final_decision"], indent=2, ensure_ascii=False))
    _section("6. COMPARISON (measured)")
    for row in results.get("comparison", {}).get("solutions", [])[:5]:
        print(
            f"  {row['item_id']}: Z1={row['z1_total_cost']} "
            f"Z2={row['z2_satisfaction']} Z3={row['z3_workload_variance']} "
            f"served={row['patients_served']}/{row['visits']}"
        )
    _section("7. RESOURCES")
    resources = results.get("complexity", {}).get("resources", {})
    print(
        f"  wall={resources.get('wall_time_sec')}s "
        f"peak_python={resources.get('peak_python_mb')}MB "
        f"rss={resources.get('rss_mb_stop')}MB"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY
    print_demo(run_hhcop_pipeline(user_query))
