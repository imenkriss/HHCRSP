# Repair report — dataset collapsed to one row, comparison table incorrect

Date: 2026-11-11 · Machine: Intel i5-3320M (4 threads), 10 GB RAM, no CUDA,
Windows, Python 3.11.4, CPU-only.

Every number below was produced by a command on this machine; commands are
quoted with their real output. No result was fabricated.

---

## 1. Reported problem and confirmed root cause

> "After executing the project, the dataset appears to be changed or replaced by
> a table containing only one row. The comparison table is not correct."

Reproduced before any change:

```
$ python -c "import main; p, c = main.load_dataset(); print(len(p), len(c))"
load_dataset -> patients: 1 caregivers: 1
pipeline patients: 1 caregivers: 1
```

and on disk:

```
$ wc -l data/dementia_patients_health_data.csv data/admin_caregivers.csv \
        data/dementia/new_dataset_with_predictions.csv
2    data/dementia_patients_health_data.csv      (81 bytes)
0    data/admin_caregivers.csv                   (0 bytes)
1001 data/dementia/new_dataset_with_predictions.csv   (1000 data rows)
```

Three independent causes, all in the data path:

**(a) Wrong source file (primary).** `main.load_dataset()` read
`data/dementia_patients_health_data.csv` — an 81-byte placeholder with **one
hard-coded record** — instead of the real 1000-row teacher dataset
`data/dementia/new_dataset_with_predictions.csv`. The patient panel therefore
had exactly one row, and `data/admin_caregivers.csv` being 0 bytes left a single
fallback caregiver `S1`.

**(b) The application manufactured the 1-row dataset.** The removed
`ensure_default_csv_files()` wrote that header + one record into `data/`
whenever the file was missing. On a clean checkout, running the project created
the one-row "dataset" itself (and wrote into raw data, violating the data
contract). This is the "replaced by a one-row table" behaviour: the code
*generated* the replacement.

**(c) The UI silently switched to it.** `interface/app.py` ran the initial
pipeline with `dataset="C", caregivers=10, location=1` (50 generated patients),
but the "Exécuter" button called `run_hhcop_pipeline(query, model=model)` with
no dataset arguments. Pressing Run fell through to `load_dataset()` → 1 patient.
So the table visibly changed from 50 rows to 1 row *after executing*.

**Comparison table.** The final section of `interface/app.py` was 100 %
hard-coded: 10 rows "K-Means+TS vs SVM+K-Means+TS vs Notre Approche" and 27 rows
"HGKS vs SVM+K-Means+TS", with invented numbers, a duplicated block of the same
10 rows, and a typo key `"NSGA-II"` → `"NSga-II"` producing an extra column. None
of those values came from a computation. Separately,
`experiments/comparison.py` did not exist, so nothing built a measured table.

**Blocking defect found during investigation.** The test suite could not even be
collected:

```
E ImportError: cannot import name '_parse_instance' from 'data.dataset_loader'
2 errors in 15.26s
```

`optimisation/mtwmo/loaders.py` imported a parser that had disappeared from
`data/dataset_loader.py`.

---

## 2. Files inspected

`readme.md`, `requirements.txt`, `AGENTS.md`, `main.py`, `interface/app.py`,
`data/` (all CSVs, `dataset_loader.py`, `loader.py`, `test_data.py`), `rag/`
(`rag.py`, `retriever.py`, `embeddings.py`, `vector_store.py`, `test_rag.py`),
`connaissance/` (`base_connaissance.py`, `regles.py`, `contraintes.py`),
`llm/raisonner.py`, `agents/` (`agentOrchestrateur.py`, `agentsPatients.py`,
`agentsSoignants.py`, `compatibilite.py`, `messages.py`),
`optimisation/` (`nsga2.py`, `validators.py`, `mtwmo/*`), `décision/decision.py`,
`complexity/metrics.py`, `ml/*`, `experiments/run_benchmark.py`, all `tests/`,
`docs/` (`AUDIT.md`, `FIX_PLAN.md`, `DATA_CONTRACT.md`, `ASSUMPTIONS.md`).

---

## 3. Files modified / created

| File | Type | Why |
|---|---|---|
| `data/dataset_loader.py` | rewritten | canonical read-only loader (teacher CSV, caregiver scaffold), `_parse_instance` restored, `generate_random_instance` delegate |
| `data/integrity.py` | **new** | sha256 snapshots, `assert_unchanged`, record validation, atomic CSV/JSON writers |
| `main.py` | rewritten | no writes into `data/`, canonical loading, integrity before/after, cached RAG, real `mtcmo` path, resource metrics, measured comparison payload |
| `rag/retriever.py` | fixed | R-series rules only (R2); constraints no longer leak into retrieval |
| `rag/rag.py` | extended | chunking with metadata, per-document scores, embedding stats, no vector duplication |
| `rag/embeddings.py` | extended | `EmbeddingCache` + `CachedEmbedder` (persistent, incremental, atomic) |
| `complexity/metrics.py` | extended | `ResourceMonitor` (tracemalloc peak + RSS), `resources()` |
| `experiments/comparison.py` | **new** | measured solution / configuration / algorithm / ML tables, sort/filter/paginate/export |
| `interface/app.py` | rewritten | data-driven dashboard; hard-coded tables removed; pagination + CSV/JSON + `outputs/` export |
| `agents/agentOrchestrateur.py` | fixed | caregiver-delay flow notifies the patient agent (A4) |
| `llm/raisonner.py` | fixed | fallback no longer invents patient `P1` |
| `rag/test_rag.py` | aligned | legacy expectations contradicted R2/`tests/test_rag_rules.py`; see §7 |
| `requirements.txt` | rewritten | removed unused `langchain*` and invalid PyPI names `llm`, `nsga2` |
| `.gitignore` | extended | `data/generated/`, `outputs/`, `.pytest_cache/` |
| `readme.md` | rewritten | correct data contract, install/run/test on the target laptop |
| `docs/REPAIR_REPORT.md` | **new** | this report |
| `docs/ASSUMPTIONS.md` | appended | new loader/RAG/working-set assumptions |
| `tests/test_dataset_integrity.py` | **new** | Tests A, C, E |
| `tests/test_comparison_tables.py` | **new** | Test B |
| `tests/test_rag_cache.py` | **new** | Test D |
| `tests/test_performance.py` | **new** | Test F |

**Not touched:** every file under `data/` (including the 1-row placeholder and
the empty caregiver CSV), `optimisation/*`, `décision/*`, `ml/*`,
`connaissance/*`, `experiments/run_benchmark.py`, `docs/AUDIT.md`.

---

## 4. Dataset integrity resolution

1. **One source of truth.** Patients are read from
   `data/dementia/new_dataset_with_predictions.csv`; `data/` is opened read-only.
   `ensure_default_csv_files()` and every write into `data/` were removed.
2. **Fingerprints.** `data/integrity.snapshot_csv()` records size, sha256, row
   count, columns, duplicate ids and missing values. `run_hhcop_pipeline` takes
   snapshots before and after and calls `assert_unchanged`, which raises
   `DatasetIntegrityError` on any change.
3. **Separated layers.** Source data (`data/`), generated artefacts
   (`data/generated/embeddings/`), exports (`outputs/`) never share a directory;
   `ensure_output_dir` refuses `data/`.
4. **Atomic writes.** Exports and the embedding cache are written to a temporary
   file in the target directory then `os.replace`d, so an interrupted run cannot
   leave a half-written file.
5. **Record validation.** `check_records` rejects empty sets, duplicate ids and
   missing `id`/`x`/`y`/`time_windows` (patients) or
   `id`/`max_work_hours`/`available` (caregivers) before optimisation starts.
6. **Repeatable work set.** A prefix of the source rows (not a random sample) is
   used, keyed by the original row (`P7` is always source row 7). Re-running does
   not shrink the file, duplicate patients or re-embed unchanged text.

Record counts before / after:

| Artefact | Before repair | After repair |
|---|---:|---:|
| `data/dementia/new_dataset_with_predictions.csv` | 1000 | **1000** (unchanged, sha256 verified) |
| `data/dementia_patients_health_data.csv` | 1 | **1** (kept, no longer read) |
| `data/admin_caregivers.csv` | 0 | **0** (kept, scaffold generated in memory) |
| Patients in a default run | **1** | **100** (1000 available via `patient_limit=None`) |
| Caregivers in a default run | **1** | **25** (`ceil(patients/4)`; 250 at full size) |

---

## 5. Comparison table correction

Removed entirely: the 10 hard-coded "K-Means+TS vs SVM+…" rows, the 27
hard-coded "HGKS vs SVM+…" rows, their duplicate block and the inconsistent
`NSga-II` key.

`experiments/comparison.py` now builds all final tables from measured data:

* **Solutions** — one row per non-dominated solution of the run, stable
  `item_id` `S01…`, Z1 total + six cost components, Z2, Z3, served/unassigned,
  caregivers used. Empty front ⇒ empty table (never a fake 1-row table).
* **Configurations** — one row per executed pipeline configuration; repeats get
  a `#n` suffix so ids stay unique; includes dataset source, sizes, algorithm,
  decision rule, LLM model/source, RAG documents, objectives, runtime, peak RAM.
* **NSGA-II vs MTCMOLite** — both algorithms run on the *same* generated
  instance and seeds, aggregated with the existing P10 `aggregate()`; MTCMOLite
  is labelled as a re-implementation, never as the published MTCMO.
* **ML baseline (D3)** — read from `ml/results/ml_benchmark.csv` when present
  (15 measured rows on this machine); otherwise an explicit message telling the
  user to run `python -m ml.evaluation`. No invented fallback rows.

Every table has consistent columns (`validate_exports` raises otherwise), keeps
missing values as `None` (shown as "—" by Streamlit, preserved in exports),
supports column selection, text filtering, sorting, pagination and CSV/JSON
export/download, and can be saved to `outputs/`.

Measured example (A_2_1, population 20, 15 generations, 2 runs, Γ=0):

| Algorithm | best Z1 | best Z2 | best Z3 | mean Z1 | front | feasible runs | mean runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSGA-II | 87.97 | 38.0 | 0.0 | 548.32 | 65 | 2/2 | 0.225 s |
| MTCMOLite | 65.47 | 38.0 | 0.0 | 506.47 | 80 | 2/2 | 0.347 s |

One instance only — this demonstrates the table works, it is not a literature
claim about either algorithm.

---

## 6. Verification

### Automated tests

```
$ python -m pytest -q
........................................................................ [ 27%]
........................................................................ [ 55%]
........................................................................ [ 83%]
..........................................                               [100%]
258 passed, 1 warning in 15.91s
```

The single warning is the generator's informational report of 145 downgraded
patients across 50 seeds for large F instances (P2 window feasibility), not a
failure. Before the repair the suite could not be collected (2 import errors).

New coverage: A `test_pipeline_leaves_source_dataset_unchanged`; B
`test_three_test_records_become_three_rows`; C
`test_repeated_execution_is_idempotent`, `test_embedding_cache_is_reused_across_runs`;
D `test_every_knowledge_document_is_ingested_once`,
`test_embeddings_are_persisted_and_reused`, `test_retrieval_returns_expected_ids_and_scores`;
E eight invalid-input tests; F `test_loading_a_representative_working_set_is_bounded`,
`test_pipeline_reports_time_and_memory`.

### End-to-end and UI

```
$ python -c "import main; print(len(main.load_dataset()[0]))"
100
```

Streamlit `AppTest` (headless, real click on "Exécuter"):
initial render 0 exceptions; after run 0 exceptions; 7 sections rendered;
`('Lignes sur disque','1000')`, `('Patients (exécution)','50')`,
`('Soignants','13')`, `('Intégrité','OK')`; 10 dataframes, 8 download buttons.
The server was also started for real on `http://localhost:8501` (HTTP 200) and
registered in the preview panel.

### Measurements (real output)

```
ui_default : patients=50  caregivers=13 front=9  wall=0.879 s  peak_python=2.23 MB  rss=94.76 MB
cli_default: patients=100 caregivers=25 front=14 wall=5.560 s  peak_python=2.32 MB  rss=98.30 MB
full load  : patients=1000 caregivers=250            wall=0.238 s  peak_python=1.37 MB  rss=94.33 MB
A_2_1 comparison (2 algorithms x 2 runs, pop 20, gen 15): 1.174 s
```

Stage breakdown (CLI default): Dataset 0.053 s, RAG 0.009 s, LLM 0.0002 s,
agents 0.001 s, **NSGA-II 5.442 s**, decision 0.002 s.

---

## 7. Optimisations and trade-offs

* **CPU-only, no new heavy dependency.** No model download, no GPU package; the
  RAG embedder is the existing deterministic hashing embedder.
* **Bounded working set, not a truncated dataset.** Default 100 patients keeps a
  run at ~5.6 s; the full 1000-row file and `patient_limit=None` remain
  available and are loaded in 0.24 s. Nothing is deleted or downsampled on disk.
* **Embedding persistence.** `CachedEmbedder` writes one JSON cache per
  embedder/dimension under `data/generated/embeddings/`; a second run has
  `hits == len(chunks)` and `misses == 0`, so unchanged knowledge is never
  re-embedded. Cache is atomic and a corrupt file is ignored/rebuilt.
* **No vector-index rebuild cost.** The knowledge base is 14 documents/chunks;
  the in-memory store is rebuilt from cached vectors in ~10 ms, which is cheaper
  than maintaining a persistent index. This is documented rather than
  introducing a new dependency.
* **Memory.** Full 1000-row load peaks at 1.37 MB of Python allocations and
  94 MB RSS; pandas reads the 260 KB CSV once and the frame is cached by
  path+mtime+size.
* **Low-core execution.** No worker pool/parallelism is introduced; the
  optimiser is single-threaded, which avoids oversubscribing 4 threads.
* **Explicit progress/errors.** Every pipeline stage is timed; the UI shows a
  spinner, catches `DatasetIntegrityError` separately, and surfaces exceptions
  with `st.exception` instead of a generic message.

---

## 8. Remaining limitations / conflicts

1. **Legacy vs documented RAG behaviour (CONFLICT, resolved in favour of R2).**
   `rag/test_rag.py` expected the retriever to return constraint `C003`; R2,
   `docs/ASSUMPTIONS.md` and `tests/test_rag_rules.py` require rules only.
   `rag/test_rag.py` was updated to `R002` with an explanatory docstring. No
   production assertion was weakened: the authoritative R2 tests still pass.
2. **Teacher CSV lacks clinical/operational fields.** `service_hours = 0.5`,
   `required_skill_level = 1`, one window `(0, 480)` and `care_type = "General"`
   for the 700 unlabelled rows are documented assumptions — not measurements.
3. **Caregivers are a scaffold** because `data/admin_caregivers.csv` is empty.
   The file is left untouched; generated caregivers are labeled
   `source: "generated"` in the UI.
4. **`optimisation/nsga2.py` is an assignment sampler**, not canonical NSGA-II
   (no crossover/mutation of a permutation chromosome). The P10-compatible
   NSGA-II is `optimisation/mtwmo/algorithms.py::NSGAII`, used by the comparison
   tab and `experiments/run_benchmark.py`.
5. **`MTCMO` is not reproduced exactly** — `MTCMOLite` is a two-task
   epsilon-relaxed re-implementation; labelled as such everywhere.
6. **`data/dementia_patients_health_data.csv` (1 row) and
   `data/admin_caregivers.csv` (empty) still exist.** They are no longer read
   but were not deleted (project rule: list, don't delete). Removing them is a
   user decision.
7. **Streamlit UI is tested with `AppTest` and a real server health check**, not
   by a full visual/interaction audit of every widget.

---

## 9. Exact commands

```powershell
# install
python -m pip install -r requirements.txt

# tests
python -m pytest -q

# CLI pipeline (teacher CSV, 100 patients / 25 caregivers)
python main.py "Get the info for patient P1 with high priority"

# full 1000-patient working set
python -c "import main; p,c = main.load_dataset(patient_limit=None); print(len(p), len(c))"

# UI
python -m streamlit run interface/app.py

# measured P10 benchmark
python experiments/run_benchmark.py --datasets A B --runs 5

# measured D3 ML baseline (creates ml/results/ml_benchmark.csv)
python -m ml.evaluation
```
