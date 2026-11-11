"""Measured comparison tables for the HHCOP pipeline.

Every value produced here comes from an actual computation:

* ``solution_table``     — one row per non-dominated solution of the run that
  just finished (legacy ``NSGA2`` candidates or ``optimisation.mtwmo``
  ``ParetoSet`` candidates).
* ``configuration_row`` / ``configurations_table`` — one row per executed
  pipeline configuration (dataset, sizes, algorithm, LLM model, RAG documents,
  objectives, runtime, peak memory). The UI keeps the session history and calls
  ``configurations_table`` with it.
* ``algorithm_comparison`` — NSGA-II vs MTCMOLite on the **same generated
  instance and seeds**, aggregated with :func:`experiments.run_benchmark.aggregate`
  (P10). ``MTCMOLite`` is never presented as the published MTCMO.
* ``ml_baseline_table`` — the measured ``ml/results/ml_benchmark.csv`` rows for
  SVM+K-Means+Tabu vs K-Means+Tabu (D3), when that file exists.

Nothing is invented: a missing source yields an empty table plus a status
message, never placeholder numbers. All tables are lists of dicts with a
stable ``item_id``, identical keys per row, missing values kept as ``None``
(rendered as an em dash by the UI) and exported to CSV/JSON unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data.integrity import rows_to_csv_text, rows_to_json_text, validate_exports

NA = None
ML_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "ml" / "results" / "ml_benchmark.csv"

SOLUTION_COLUMNS = [
    "item_id", "algorithm", "source", "feasible", "z1_total_cost",
    "service_cost", "waiting_cost", "overtime_cost", "travel_cost",
    "fixed_cost", "tardiness_cost", "z2_satisfaction", "z3_workload_variance",
    "patients_served", "unassigned_patients", "caregivers_used", "visits",
]

CONFIGURATION_COLUMNS = [
    "item_id", "instance", "dataset_source", "patients", "caregivers",
    "algorithm", "decision_method", "llm_model", "llm_source", "rag_documents",
    "pareto_size", "z1_total_cost", "z2_satisfaction", "z3_workload_variance",
    "total_runtime_sec", "peak_python_mb", "peak_rss_mb", "final_status",
]

ALGORITHM_COLUMNS = [
    "item_id", "instance", "algorithm", "gamma", "runs",
    "best_z1", "best_z2", "best_z3", "mean_z1", "mean_z2", "mean_z3",
    "mean_first_feasible_gen", "mean_caregivers", "mean_pareto_solutions",
    "non_dominated_solutions", "feasible_runs", "no_feasible_solution",
    "mean_runtime_sec",
]

ML_COLUMNS = [
    "item_id", "family", "instance", "size", "k",
    "svm_kmeans_tabu_cost", "kmeans_tabu_cost", "improvement_percent",
    "svm_runtime_s", "baseline_runtime_s",
]

#: Columns whose values are textual, used by the UI to right-align numbers.
TEXT_COLUMNS = {"item_id", "algorithm", "source", "instance", "family",
                "dataset_source", "decision_method", "llm_model", "llm_source",
                "final_status"}


def _round(value: Any, digits: int = 4) -> Any:
    if value is None:
        return NA
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return NA


def _stable_id(*parts: Any) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# 1. Pareto solutions of the current run
# --------------------------------------------------------------------------- #
def _solution_row_legacy(index: int, candidate: Mapping[str, Any]) -> dict[str, Any]:
    summary = candidate.get("summary", {}) or {}
    solution = candidate.get("solution", []) or []
    caregivers_used = len({item.get("caregiver_id") for item in solution
                           if item.get("caregiver_id") is not None})
    unassigned = int(summary.get("unassigned_patients", 0) or 0)
    return {
        "item_id": f"S{index:02d}",
        "algorithm": "NSGA-II",
        "source": "current_run",
        "feasible": unassigned == 0,
        "z1_total_cost": _round(summary.get("total_cost"), 2),
        "service_cost": _round(summary.get("service_cost"), 2),
        "waiting_cost": _round(summary.get("waiting_cost"), 2),
        "overtime_cost": _round(summary.get("overtime_cost"), 2),
        "travel_cost": _round(summary.get("travel_cost"), 2),
        "fixed_cost": _round(summary.get("fixed_cost"), 2),
        "tardiness_cost": _round(summary.get("delay_penalty"), 2),
        "z2_satisfaction": _round(summary.get("satisfaction_percent"), 2),
        "z3_workload_variance": _round(summary.get("workload_variance"), 4),
        "patients_served": int(summary.get("served_patients", 0) or 0),
        "unassigned_patients": unassigned,
        "caregivers_used": caregivers_used,
        "visits": len(solution),
    }


def _solution_row_mtwmo(index: int, candidate: Any) -> dict[str, Any]:
    evaluation = getattr(candidate, "evaluation", None)
    costs = getattr(evaluation, "costs", None)
    routes = getattr(evaluation, "routes", []) or []
    return {
        "item_id": f"S{index:02d}",
        "algorithm": "MTCMOLite",
        "source": "current_run",
        "feasible": bool(getattr(candidate, "feasible", False)),
        "z1_total_cost": _round(getattr(candidate, "z1", None), 4),
        "service_cost": _round(getattr(costs, "service", None), 4),
        "waiting_cost": _round(getattr(costs, "waiting", None), 4),
        "overtime_cost": _round(getattr(costs, "overtime", None), 4),
        "travel_cost": _round(getattr(costs, "travel", None), 4),
        "fixed_cost": _round(getattr(costs, "fixed", None), 4),
        "tardiness_cost": _round(getattr(costs, "tardiness", None), 4),
        "z2_satisfaction": _round(getattr(candidate, "z2", None), 4),
        "z3_workload_variance": _round(getattr(candidate, "z3", None), 6),
        "patients_served": int(sum(len(getattr(detail, "patient_ids", []) or []) for detail in routes)),
        "unassigned_patients": len(getattr(evaluation, "unassigned", []) or []) if evaluation else NA,
        "caregivers_used": int(getattr(candidate, "caregivers_used", 0) or 0),
        "visits": int(sum(len(getattr(detail, "patient_ids", []) or []) for detail in routes)),
    }


def solution_table(pareto_front: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Build one row per non-dominated solution, with a stable ``item_id``.

    Accepts the legacy ``list[dict]`` front and
    :class:`optimisation.mtwmo.algorithms.ParetoSet`. Rows are ordered by
    coverage first, then satisfaction (descending), then cost. The table is
    never collapsed to a single row: an empty front yields ``[]``.
    """
    if pareto_front is None:
        return []
    solutions = getattr(pareto_front, "solutions", pareto_front)
    if not isinstance(solutions, Sequence) or isinstance(solutions, (str, bytes)):
        raise TypeError(f"Unsupported Pareto front type: {type(pareto_front)!r}")
    solutions = list(solutions)
    if not solutions:
        return []

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(solutions, start=1):
        if isinstance(candidate, Mapping):
            rows.append(_solution_row_legacy(index, candidate))
        else:
            rows.append(_solution_row_mtwmo(index, candidate))

    rows.sort(key=lambda row: (
        row["unassigned_patients"] if row["unassigned_patients"] is not None else 10**9,
        -(row["z2_satisfaction"] or 0.0),
        row["z1_total_cost"] if row["z1_total_cost"] is not None else float("inf"),
    ))
    # Re-number after sorting so S01 is always the best compromise, keeping ids
    # stable for a given front.
    for position, row in enumerate(rows, start=1):
        row["item_id"] = f"S{position:02d}"
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    validate_exports(rows, label="solution_table")
    return rows


# --------------------------------------------------------------------------- #
# 2. Executed pipeline configurations
# --------------------------------------------------------------------------- #
def configuration_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise one pipeline result as a comparison row (measured values only)."""
    analysis = result.get("optimization_analysis", {}) or {}
    dataset_info = result.get("dataset_info", {}) or {}
    selected = result.get("selected_optimization") or {}
    objectives = selected.get("objectives", {}) if isinstance(selected, Mapping) else {}
    complexity = result.get("complexity", {}) or {}
    resources = complexity.get("resources", {}) or {}
    instance = (result.get("instance_info", {}) or {}).get("name")
    llm_output = result.get("llm_output", {}) or {}
    decision = result.get("final_decision", {}) or {}
    optimization = result.get("optimization")
    pareto_size = getattr(optimization, "size", None)
    if pareto_size is None and isinstance(optimization, Sequence):
        pareto_size = len(optimization)

    config_signature = (
        instance, dataset_info.get("source"), len(result.get("patients", []) or []),
        len(result.get("soignants", []) or []), analysis.get("algorithm"),
        analysis.get("population_size"), analysis.get("generations"),
        result.get("llm_model"), decision.get("method"),
    )
    return {
        "item_id": f"cfg-{_stable_id(*config_signature)}",
        "instance": instance,
        "dataset_source": dataset_info.get("source"),
        "patients": len(result.get("patients", []) or []),
        "caregivers": len(result.get("soignants", []) or []),
        "algorithm": analysis.get("algorithm"),
        "decision_method": decision.get("method"),
        "llm_model": result.get("llm_model"),
        "llm_source": llm_output.get("reasoning_source"),
        "rag_documents": len(result.get("retrieved_docs", []) or []),
        "pareto_size": pareto_size,
        "z1_total_cost": _round(objectives.get("Z1"), 2),
        "z2_satisfaction": _round(objectives.get("Z2"), 2),
        "z3_workload_variance": _round(objectives.get("Z3"), 4),
        "total_runtime_sec": _round(
            (complexity.get("Total Pipeline", {}) or {}).get("execution_time_sec"), 3
        ),
        "peak_python_mb": _round(resources.get("peak_python_mb"), 2),
        "peak_rss_mb": _round(resources.get("peak_rss_mb"), 2),
        "final_status": decision.get("status"),
    }


def configurations_table(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per pipeline execution, with a unique ``item_id`` per row.

    Re-running the exact same configuration keeps the same base id; repeats get
    a ``#n`` suffix so a table never contains duplicate identifiers.
    """
    rows = [configuration_row(result) for result in results]
    seen: dict[str, int] = {}
    for row in rows:
        base = row["item_id"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            row["item_id"] = f"{base}#{seen[base]}"
    if rows:
        validate_exports(rows, label="configurations_table")
    return rows


# --------------------------------------------------------------------------- #
# 3. NSGA-II vs MTCMOLite on the same instance (P10)
# --------------------------------------------------------------------------- #
def algorithm_comparison(
    dataset: str,
    caregivers: int,
    location: int,
    *,
    seed: int = 0,
    population_size: int = 30,
    generations: int = 20,
    runs: int = 2,
    gamma: float = 0.0,
    algorithms: Sequence[str] = ("NSGA-II", "MTCMOLite"),
    progress: Any = None,
) -> dict[str, Any]:
    """Run both algorithms on identical instances/seeds and aggregate (P10).

    The instance is regenerated identically for every run because
    ``generate_instance`` is seeded and deterministic. ``MTCMOLite`` is
    reported under its own name; see docs/ASSUMPTIONS.md.
    """
    if int(runs) < 1:
        raise ValueError("runs must be >= 1")
    from experiments.run_benchmark import aggregate, run_one, ALGORITHMS
    from optimisation.mtwmo.generator import generate_instance

    unknown = [name for name in algorithms if name not in ALGORITHMS]
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}; expected {sorted(ALGORITHMS)}")

    instance_info = {
        "dataset": dataset.upper(),
        "caregivers": int(caregivers),
        "location": int(location),
        "name": f"{dataset.upper()}_{int(caregivers)}_{int(location)}",
    }
    # Fail early with the generator's own validation message.
    generate_instance(instance_info["dataset"], instance_info["caregivers"],
                      instance_info["location"], seed)

    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        pareto_sets = []
        for run_index in range(int(runs)):
            run_seed = int(seed) + run_index
            outcome = run_one(
                instance_info, algorithm, run_seed, float(gamma),
                int(population_size), int(generations),
            )
            result = outcome["result"]
            pareto_sets.append(result)
            run_rows.append({
                "item_id": f"{instance_info['name']}|{algorithm}|r{run_index + 1}",
                "instance": instance_info["name"],
                "algorithm": algorithm,
                "gamma": float(gamma),
                "run": run_index + 1,
                "seed": run_seed,
                "pareto_size": result.size,
                "feasible": result.has_feasible,
                "best_z1": _round(result.best_z1),
                "best_z2": _round(result.best_z2),
                "best_z3": _round(result.best_z3),
                "runtime_seconds": _round(result.runtime_seconds),
            })
            if progress is not None:
                progress(algorithm, run_index + 1, int(runs), result)
        summary = aggregate(
            instance_info["name"], instance_info["dataset"], algorithm,
            float(gamma), pareto_sets,
        )
        summary["item_id"] = f"{instance_info['name']}|{algorithm}|g{float(gamma)}"
        rows.append(summary)

    validate_exports(rows, label="algorithm_comparison")
    return {
        "instance": instance_info,
        "rows": rows,
        "runs": run_rows,
        "runtime_seconds": _round(sum(row["mean_runtime_sec"] for row in rows
                                      if isinstance(row.get("mean_runtime_sec"), (int, float))), 3),
    }


# --------------------------------------------------------------------------- #
# 4. ML baseline (D3, measured CSV written by ml.evaluation)
# --------------------------------------------------------------------------- #
def ml_baseline_table(path: str | Path | None = None) -> dict[str, Any]:
    """Load the measured ML benchmark rows, never inventing missing numbers."""
    source = Path(path) if path is not None else ML_BENCHMARK_PATH
    status: dict[str, Any] = {"path": str(source), "available": False, "rows": 0}
    if not source.exists() or source.stat().st_size == 0:
        status["message"] = (
            "No measured ML benchmark file. Run "
            "`python -m ml.evaluation` to create it."
        )
        return {"rows": [], "status": status}

    import pandas as pd

    try:
        frame = pd.read_csv(source)
    except Exception as error:  # pragma: no cover - corrupt user file
        status["message"] = f"Cannot read {source}: {error}"
        return {"rows": [], "status": status}

    required = {"family", "instance", "size"}
    if not required <= set(frame.columns):
        status["message"] = (
            f"{source} is missing required columns {sorted(required - set(frame.columns))}"
        )
        return {"rows": [], "status": status}

    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        rows.append({
            "item_id": f"ml|{record['family']}|{record['instance']}|n{int(record['size'])}",
            "family": str(record["family"]),
            "instance": str(record["instance"]),
            "size": int(record["size"]),
            "k": int(record["k"]) if "k" in frame.columns and pd.notna(record.get("k")) else NA,
            "svm_kmeans_tabu_cost": _round(record.get("svm_total_cost"), 4),
            "kmeans_tabu_cost": _round(record.get("baseline_total_cost"), 4),
            "improvement_percent": _round(record.get("improvement_percent"), 4),
            "svm_runtime_s": _round(record.get("svm_runtime_s"), 4),
            "baseline_runtime_s": _round(record.get("baseline_runtime_s"), 4),
        })
    validate_exports(rows, label="ml_baseline_table")
    status.update({"available": True, "rows": len(rows),
                   "message": f"{len(rows)} measured rows from {source.name}"})
    return {"rows": rows, "status": status}


# --------------------------------------------------------------------------- #
# 5. Discussion summary (measured deltas, no ranking without data)
# --------------------------------------------------------------------------- #
def algorithm_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair algorithms per instance and report measured Z1/Z2/Z3 differences.

    Only pairs where both rows carry numeric objectives are compared; missing
    values are skipped rather than treated as zero.
    """
    import statistics

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("instance")), {})[str(row.get("algorithm"))] = row
    pairs: list[dict[str, Any]] = []
    for instance, by_algorithm in sorted(grouped.items()):
        baseline = by_algorithm.get("NSGA-II")
        candidate = by_algorithm.get("MTCMOLite")
        if baseline is None or candidate is None:
            continue
        def numeric(name: str) -> tuple[float, float] | None:
            a, b = baseline.get(name), candidate.get(name)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(a), float(b)
            return None
        z1 = numeric("best_z1")
        z2 = numeric("best_z2")
        z3 = numeric("best_z3")
        pairs.append({
            "item_id": f"delta|{instance}",
            "instance": instance,
            "z1_nsga2": _round(z1[0]) if z1 else NA,
            "z1_mtcmolite": _round(z1[1]) if z1 else NA,
            "z1_delta_pct": _round(100.0 * (z1[1] - z1[0]) / z1[0], 2) if z1 and z1[0] else NA,
            "z2_nsga2": _round(z2[0]) if z2 else NA,
            "z2_mtcmolite": _round(z2[1]) if z2 else NA,
            "z2_delta_pct": _round(100.0 * (z2[1] - z2[0]) / z2[0], 2) if z2 and z2[0] else NA,
            "z3_nsga2": _round(z3[0], 6) if z3 else NA,
            "z3_mtcmolite": _round(z3[1], 6) if z3 else NA,
            "z3_delta_pct": _round(100.0 * (z3[1] - z3[0]) / z3[0], 2) if z3 and z3[0] else NA,
        })
    return pairs


# --------------------------------------------------------------------------- #
# 6. Table utilities (sorting / filtering / pagination / exports)
# --------------------------------------------------------------------------- #
def sort_rows(
    rows: Sequence[Mapping[str, Any]], column: str | None, ascending: bool = True
) -> list[dict[str, Any]]:
    """Sort rows on one column; ``None`` values always sort last."""
    ordered = [dict(row) for row in rows]
    if not column:
        return ordered

    def key(row: Mapping[str, Any]) -> tuple[int, Any]:
        value = row.get(column)
        if value is None:
            return (1, "")
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float)):
            return (0, float(value))
        return (0, str(value).casefold())

    return sorted(ordered, key=key, reverse=not ascending)


def filter_rows(rows: Sequence[Mapping[str, Any]], text: str) -> list[dict[str, Any]]:
    """Case-insensitive substring filter across all values."""
    needle = (text or "").strip().casefold()
    if not needle:
        return [dict(row) for row in rows]
    return [
        dict(row) for row in rows
        if any(needle in str(value).casefold() for value in row.values() if value is not None)
    ]


def paginate(
    rows: Sequence[Mapping[str, Any]], page: int = 1, page_size: int = 25
) -> dict[str, Any]:
    """Return one page plus its bounds (1-based page, clamped)."""
    page_size = max(1, int(page_size))
    total = len(rows)
    pages = max(1, (total + page_size - 1) // page_size)
    current = min(max(1, int(page)), pages)
    start = (current - 1) * page_size
    return {
        "rows": [dict(row) for row in rows[start:start + page_size]],
        "page": current,
        "pages": pages,
        "page_size": page_size,
        "total": total,
        "start": start + 1 if total else 0,
        "end": min(start + page_size, total),
    }


def export_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """CSV text for ``st.download_button`` (all rows, stable column order)."""
    return rows_to_csv_text(rows)


def export_json(rows: Sequence[Mapping[str, Any]]) -> str:
    """JSON text for ``st.download_button``."""
    return rows_to_json_text(rows)
