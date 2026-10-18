"""D3 benchmark evaluation on Solomon and Gehring-Homberger subsets.

Compares the ``SVM + K-Means + Tabu`` pipeline against the ``K-Means + Tabu``
baseline on identical customer subsets. Only measured numbers are written.

ASSUMPTION (docs/ASSUMPTIONS.md, D3): benchmark customers carry no clinical
features, so the SVM care type is attached positionally from the completed
teacher CSV (row i <-> customer i). D4's coordinate-mapping hypothesis was
checked and is weak (match_rate 0.12), so positional attachment is the
documented simplest reading.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from data import loader

from .care_type import CareTypeClassifier, load_teacher_dataframe
from .pipeline import compare_svm_vs_baseline

ROOT = Path(__file__).resolve().parents[1]
SOLOMON_DIR = ROOT / "data" / "Solomon" / "In"
GH_DIR = ROOT / "data" / "Gehing&Homberger"
RESULTS_DIR = ROOT / "ml" / "results"

DEFAULT_TABU = {"iterations": 60, "tabu_tenure": 8}


def _depot_and_points(parsed: dict) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    depot = (float(parsed["depot"]["x"]), float(parsed["depot"]["y"]))
    points = [
        (float(row["x"]), float(row["y"])) for row in parsed["customers"]
    ]
    return depot, points


def solomon_subset(path: Path | str, size: int):
    parsed = loader.parse_solomon_file(Path(path))
    depot, points = _depot_and_points(parsed)
    return depot, points[:size]


def gehring_subset(path: Path | str, size: int):
    parsed = loader.parse_gehring_homberger_file(Path(path))
    depot, points = _depot_and_points(parsed)
    return depot, points[:size]


def build_care_pool(classifier: CareTypeClassifier | None = None) -> list[int]:
    """Completed (given + SVM) care-type codes, used positionally."""
    if classifier is None:
        classifier = CareTypeClassifier().fit(load_teacher_dataframe())
    completed = classifier.complete(load_teacher_dataframe())
    return [int(code) for code in completed["care_type_code"].tolist()]


def attach_care_codes(n: int, pool: Sequence[int]) -> list[int]:
    if not pool:
        raise ValueError("care pool is empty")
    return [int(pool[i % len(pool)]) for i in range(n)]


def default_specs() -> list[dict]:
    return [
        {"family": "Solomon", "instance": "c101", "path": SOLOMON_DIR / "c101.txt",
         "sizes": [25, 50, 100]},
        {"family": "Solomon", "instance": "r101", "path": SOLOMON_DIR / "r101.txt",
         "sizes": [25, 50, 100]},
        {"family": "Gehring-Homberger", "instance": "C1_2_1",
         "path": GH_DIR / "200-custmers" / "C1_2_1.TXT", "sizes": [25, 50, 100, 200]},
        {"family": "Gehring-Homberger", "instance": "C1_4_1",
         "path": GH_DIR / "400-customers" / "C1_4_1.TXT",
         "sizes": [25, 50, 100, 200, 400]},
    ]


def evaluate(
    specs: Sequence[dict] | None = None,
    k: int = 3,
    seed: int = 42,
    tabu_kwargs: dict | None = None,
    out_path: Path | str | None = None,
    classifier: CareTypeClassifier | None = None,
) -> list[dict]:
    """Run every (instance, size) spec and return the summary rows."""
    specs = list(specs) if specs is not None else default_specs()
    options = dict(DEFAULT_TABU)
    if tabu_kwargs:
        options.update(tabu_kwargs)
    pool = build_care_pool(classifier)

    rows: list[dict] = []
    for spec in specs:
        parser = solomon_subset if spec["family"] == "Solomon" else gehring_subset
        for size in spec["sizes"]:
            depot, points = parser(spec["path"], size)
            care = attach_care_codes(len(points), pool)
            comparison = compare_svm_vs_baseline(
                points, depot, care, k=k, seed=seed, tabu_kwargs=options
            )
            rows.append(
                {
                    "family": spec["family"],
                    "instance": spec["instance"],
                    "size": len(points),
                    "k": comparison["k"],
                    "svm_total_cost": round(comparison["svm_total_cost"], 6),
                    "baseline_total_cost": round(comparison["baseline_total_cost"], 6),
                    "improvement_percent": round(comparison["improvement_percent"], 6),
                    "svm_runtime_s": round(comparison["svm"].runtime_seconds, 6),
                    "baseline_runtime_s": round(
                        comparison["baseline"].runtime_seconds, 6
                    ),
                }
            )
    if out_path is not None:
        write_rows(rows, Path(out_path))
    return rows


def write_rows(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the D3 ml benchmark")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=DEFAULT_TABU["iterations"])
    parser.add_argument("--tabu-tenure", type=int, default=DEFAULT_TABU["tabu_tenure"])
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR / "ml_benchmark.csv"
    )
    args = parser.parse_args(argv)

    classifier = CareTypeClassifier().fit(load_teacher_dataframe())
    print("SVM care-type metrics (cross-validated):", classifier.metrics)
    rows = evaluate(
        k=args.k,
        seed=args.seed,
        tabu_kwargs={"iterations": args.iterations, "tabu_tenure": args.tabu_tenure},
        out_path=args.out,
        classifier=classifier,
    )
    for row in rows:
        print(
            f"{row['family']:>18} {row['instance']:>7} n={row['size']:>4} "
            f"svm={row['svm_total_cost']:.3f} base={row['baseline_total_cost']:.3f} "
            f"impr={row['improvement_percent']:+.3f}%"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
