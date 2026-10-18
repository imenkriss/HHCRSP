"""P10 benchmark runner: NSGA-II vs MTCMOLite over the P11 instance sets.

For every selected instance it performs ``--runs`` independent runs (distinct
seeds) of both algorithms and writes CSVs reproducing the paper's table
structure:

* ``benchmark_summary.csv``  -- best-of-N per objective (min Z1, max Z2,
  min Z3), mean of the non-dominated solutions per objective, mean generation
  of the first feasible solution, mean caregivers used, mean number of Pareto
  solutions, number of runs that found a feasible solution. Missing values are
  written as ``"—"`` (em dash).
* ``benchmark_runs.csv``     -- one row per (instance, algorithm, seed).
* ``robustness.csv``         -- gamma=0 versus gamma>0 and the cost of
  robustness on Z1.

Default scope is the small A/B instances; ``--full`` adds C-F. Every run is
seeded, so results are reproducible.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimisation.mtwmo.algorithms import MTCMOLite, NSGAII  # noqa: E402
from optimisation.mtwmo.generator import DATASET_TABLE, generate_instance  # noqa: E402

NA = "\u2014"  # em dash, used for "no feasible solution"
ALGORITHMS = {"NSGA-II": NSGAII, "MTCMOLite": MTCMOLite}
SMALL_DATASETS = ("A", "B")
FULL_DATASETS = ("A", "B", "C", "D", "E", "F")


def build_instance_list(datasets: tuple[str, ...], location: int = 1) -> list[dict]:
    """One instance per (dataset, caregiver count) from the P11 care table."""
    instances: list[dict] = []
    for dataset in datasets:
        _, caregiver_options, _ = DATASET_TABLE[dataset]
        for caregivers in caregiver_options:
            instances.append({
                "dataset": dataset,
                "caregivers": caregivers,
                "location": location,
                "name": f"{dataset}_{caregivers}_{location}",
            })
    return instances


def _algorithm(name: str, population: int, generations: int):
    if name == "NSGA-II":
        return NSGAII(population_size=population, generations=generations)
    return MTCMOLite(population_size=population, generations=generations)


def run_one(instance_info: dict, algorithm_name: str, seed: int, gamma: float,
            population: int, generations: int) -> dict:
    instance = generate_instance(
        instance_info["dataset"], instance_info["caregivers"],
        instance_info["location"], seed,
    )
    result = _algorithm(algorithm_name, population, generations).run(
        instance, seed=seed, gamma=gamma,
    )
    return {"result": result, "instance": instance}


def _fmt(value, digits: int = 4):
    if value is None:
        return NA
    if isinstance(value, float):
        return round(value, digits)
    return value


def aggregate(instance_name: str, dataset: str, algorithm: str, gamma: float,
              pareto_sets: list) -> dict:
    """Reduce the per-run Pareto sets to one P10 summary row."""
    feasible = [c for ps in pareto_sets for c in ps.solutions if c.feasible]
    feasible_runs = sum(1 for ps in pareto_sets if ps.has_feasible)
    first_feasible = [ps.first_feasible_generation for ps in pareto_sets
                      if ps.first_feasible_generation is not None]
    pareto_sizes = [ps.size for ps in pareto_sets]
    runtimes = [ps.runtime_seconds for ps in pareto_sets]

    if feasible:
        row = {
            "best_z1": _fmt(min(c.z1 for c in feasible)),
            "best_z2": _fmt(max(c.z2 for c in feasible)),
            "best_z3": _fmt(min(c.z3 for c in feasible)),
            "mean_z1": _fmt(statistics.fmean(c.z1 for c in feasible)),
            "mean_z2": _fmt(statistics.fmean(c.z2 for c in feasible)),
            "mean_z3": _fmt(statistics.fmean(c.z3 for c in feasible)),
            "mean_caregivers": _fmt(statistics.fmean(c.caregivers_used for c in feasible), 2),
            "non_dominated_solutions": len(feasible),
        }
    else:
        row = {key: NA for key in (
            "best_z1", "best_z2", "best_z3", "mean_z1", "mean_z2", "mean_z3",
            "mean_caregivers", "non_dominated_solutions")}

    return {
        "dataset": dataset,
        "instance": instance_name,
        "algorithm": algorithm,
        "gamma": gamma,
        "runs": len(pareto_sets),
        **row,
        "mean_first_feasible_gen": _fmt(statistics.fmean(first_feasible), 2)
        if first_feasible else NA,
        "mean_pareto_solutions": _fmt(statistics.fmean(pareto_sizes), 2) if pareto_sizes else NA,
        "feasible_runs": feasible_runs,
        "no_feasible_solution": feasible_runs == 0,
        "mean_runtime_sec": _fmt(statistics.fmean(runtimes), 4) if runtimes else NA,
    }


SUMMARY_FIELDS = [
    "dataset", "instance", "algorithm", "gamma", "runs",
    "best_z1", "best_z2", "best_z3", "mean_z1", "mean_z2", "mean_z3",
    "mean_first_feasible_gen", "mean_caregivers", "mean_pareto_solutions",
    "non_dominated_solutions", "feasible_runs", "no_feasible_solution",
    "mean_runtime_sec",
]
RUN_FIELDS = [
    "dataset", "instance", "algorithm", "gamma", "run", "seed",
    "z1", "z2", "z3", "first_feasible_gen", "caregivers_used",
    "pareto_size", "feasible", "runtime_seconds",
]
ROBUSTNESS_FIELDS = [
    "dataset", "instance", "algorithm", "gamma_nominal", "gamma_robust",
    "best_z1_nominal", "best_z1_robust", "delta_best_z1", "relative_best_z1_pct",
    "mean_z1_nominal", "mean_z1_robust", "delta_mean_z1", "relative_mean_z1_pct",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HHCRSP-MAS P10 benchmark")
    parser.add_argument("--datasets", nargs="+", default=list(SMALL_DATASETS))
    parser.add_argument("--full", action="store_true", help="include C-F datasets")
    parser.add_argument("--location", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--robustness-gamma", type=float, default=0.5)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results"))
    args = parser.parse_args(argv)

    datasets = FULL_DATASETS if args.full else tuple(args.datasets)
    instances = build_instance_list(tuple(datasets), location=args.location)
    out_dir = Path(args.out)
    seeds = [args.seed + index for index in range(args.runs)]

    summary_rows: list[dict] = []
    run_rows: list[dict] = []
    robustness_rows: list[dict] = []

    for instance_info in instances:
        for algorithm in ALGORITHMS:
            pareto_sets = []
            for run_index, seed in enumerate(seeds):
                outcome = run_one(instance_info, algorithm, seed, 0.0,
                                  args.population, args.generations)
                result = outcome["result"]
                pareto_sets.append(result)
                run_rows.append({
                    "dataset": instance_info["dataset"],
                    "instance": instance_info["name"],
                    "algorithm": algorithm,
                    "gamma": 0.0,
                    "run": run_index + 1,
                    "seed": seed,
                    "z1": _fmt(result.best_z1),
                    "z2": _fmt(result.best_z2),
                    "z3": _fmt(result.best_z3),
                    "first_feasible_gen": result.first_feasible_generation
                    if result.first_feasible_generation is not None else NA,
                    "caregivers_used": _fmt(result.caregivers_used_mean, 2),
                    "pareto_size": result.size,
                    "feasible": result.has_feasible,
                    "runtime_seconds": round(result.runtime_seconds, 4),
                })
                print(
                    f"[{instance_info['name']:>9}] {algorithm:>9} run {run_index + 1}/{args.runs} "
                    f"seed={seed} front={result.size} feasible={result.has_feasible} "
                    f"({result.runtime_seconds:.3f}s)"
                )
            summary_rows.append(aggregate(
                instance_info["name"], instance_info["dataset"], algorithm, 0.0, pareto_sets,
            ))

    if not args.skip_robustness and args.robustness_gamma > 0.0:
        for instance_info in instances:
            for algorithm in ALGORITHMS:
                nominal = [
                    run_one(instance_info, algorithm, seed, 0.0, args.population, args.generations)["result"]
                    for seed in seeds
                ]
                robust = [
                    run_one(instance_info, algorithm, seed, args.robustness_gamma,
                            args.population, args.generations)["result"]
                    for seed in seeds
                ]
                nominal_feasible = [c for ps in nominal for c in ps.solutions if c.feasible]
                robust_feasible = [c for ps in robust for c in ps.solutions if c.feasible]
                best_nominal = min((c.z1 for c in nominal_feasible), default=None)
                best_robust = min((c.z1 for c in robust_feasible), default=None)
                mean_nominal = statistics.fmean(c.z1 for c in nominal_feasible) if nominal_feasible else None
                mean_robust = statistics.fmean(c.z1 for c in robust_feasible) if robust_feasible else None
                robustness_rows.append({
                    "dataset": instance_info["dataset"],
                    "instance": instance_info["name"],
                    "algorithm": algorithm,
                    "gamma_nominal": 0.0,
                    "gamma_robust": args.robustness_gamma,
                    "best_z1_nominal": _fmt(best_nominal),
                    "best_z1_robust": _fmt(best_robust),
                    "delta_best_z1": _fmt(best_robust - best_nominal)
                    if best_nominal is not None and best_robust is not None else NA,
                    "relative_best_z1_pct": _fmt(100.0 * (best_robust - best_nominal) / best_nominal, 2)
                    if best_nominal not in (None, 0) and best_robust is not None else NA,
                    "mean_z1_nominal": _fmt(mean_nominal),
                    "mean_z1_robust": _fmt(mean_robust),
                    "delta_mean_z1": _fmt(mean_robust - mean_nominal)
                    if mean_nominal is not None and mean_robust is not None else NA,
                    "relative_mean_z1_pct": _fmt(100.0 * (mean_robust - mean_nominal) / mean_nominal, 2)
                    if mean_nominal not in (None, 0) and mean_robust is not None else NA,
                })
                print(
                    f"[{instance_info['name']:>9}] {algorithm:>9} robustness "
                    f"gamma={args.robustness_gamma}: best Z1 {_fmt(best_nominal)} -> {_fmt(best_robust)}"
                )

    _write_csv(out_dir / "benchmark_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "benchmark_runs.csv", RUN_FIELDS, run_rows)
    _write_csv(out_dir / "robustness.csv", ROBUSTNESS_FIELDS, robustness_rows)
    print(f"\nWrote {len(summary_rows)} summary rows, {len(run_rows)} run rows, "
          f"{len(robustness_rows)} robustness rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
