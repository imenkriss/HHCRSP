# ASSUMPTIONS

Documented explicit readings made when a paper/course detail is unavailable or
ambiguous (AGENTS.md working rule). Each entry is labelled `ASSUMPTION:`.

## D1 — teacher CSV and the SVM care-type completion

`ASSUMPTION:` `data/dementia/new_dataset_with_predictions.csv` is loaded
verbatim (1000 rows, 26 columns) and is never edited. `data/loader.py` adds two
derived columns only: `care_type` (mapped from `Predictions`) and
`care_type_source`.

`ASSUMPTION:` On the 300 labelled rows, `Predictions == 1 - Diabetic` exactly.
The SVM (One-vs-Rest, features = the 23 columns excluding `Predictions`,
`XCOORD.`, `YCOORD.`) therefore effectively **learns the deterministic
`1 - Diabetic` rule**; because cross-validated accuracy/precision/F1 are near
1.0, the metric describes this leak, not a clinical classifier.

`ASSUMPTION:` The file has only **2 classes**, whereas the ICCCI paper describes
4 departments (diabetes, heart disease, hypertension, dementia). The two labels
are therefore kept as opaque `type_0` / `type_1`; no clinical mapping is
invented for the unlabelled 700 rows. The 700 NaN predictions are filled by the
SVM and marked `care_type_source == "svm"`.

## Coordinates

`ASSUMPTION:` `XCOORD.`/`YCOORD.` are min-max normalised to `[0, 1]`. Rescaling
to 50×50 / 100×100 / 200×200 is a uniform affine map (`x * size`), and rescaling
to the Gehring-Homberger range uses that file's own min/max. The D4 comparison
is computed, not assumed: `data.loader.check_d4_hypothesis` prints the
row-order and sorted correlations plus the nearest-neighbour match rate.

## P2/P4/P11 — instance generator and `care_type`

`ASSUMPTION:` `care_type` (from the CSV) influences the instance only through
**caregiver specialty compatibility** (`g_k >= p_i` remains the skill rule and a
specialty match is an additional compatibility dimension) and as information
fed to the MAS (patient `care_type`/`priority` are carried on the agent-facing
records). It is **not** mapped to the paper's skill demand `p_i`: `p_i` keeps the
paper's 50/35/15 generator and `g_k` stays uniform on `{1, 2, 3}`.

`ASSUMPTION:` The instance horizon `l0` is 720 minutes and the depot earliest
time `e0` is 0; time-window centres follow P2 with `t0i = d(depot, i) / speed`.
When patients come from the CSV, depot and laboratory coordinates stay random in
the location square because the teacher CSV has no depot/lab locations.

`ASSUMPTION:` The generated uncertainty levels `alpha_ij`, `beta_k`, `sigma_i`
(P5) are independent draws in `[0, 1]`; the nominal caregiver departure is 0
(`s_k = 0`). The robust budgets `Gamma_t`, `Gamma_s`, `Gamma_p` are out of scope
for the generator phase (see `docs/AUDIT.md`, P5).

## P5/P6/P7 — `optimisation/mtwmo/model.py` evaluation

`ASSUMPTION:` The Bertsimas-Sim budget is applied **per route** (see the module
docstring): a route's travel margin is the budgeted sum of its own arc
deviations, the departure margin uses the single caregiver's deviation, and the
skill-demand margin uses the single patient's deviation. `Gamma = 0` therefore
reproduces the deterministic model exactly. A scalar `gamma` sets all three
budgets; a triple is `(t, s, p)`; a mapping uses keys `t`/`s`/`p`.

`ASSUMPTION:` The worst-case travel deviation is realised on the route arc with
the largest deviation and delays every node after it. Unlisted `alpha_ij` default
to 0; missing `travel_deviation_fraction` / `departure_deviation_minutes` /
`demand_deviation_level` attributes default to `0.2` / `30.0` / `1.0`.

`ASSUMPTION:` The Z1 service cost uses `theta_ik = theta_i * p_i / g_k` (P4), so
the extra `p_i` factor written in the paper's first Z1 term is treated as already
folded into `theta_ik`. In Z2 the coverage term `o_i` is read as "patient `i` is
served" (1 if assigned). Patient satisfaction uses nominal `p_i`, not the robust
demand.

`ASSUMPTION:` Caregiver-patient incompatibility (constraint 10) is recorded when
`patient.rejections[caregiver.id]` is `True` **or** when both a non-null
`patient.care_type` and a non-null `caregiver.specialty` differ. Working-time
violation (13) uses `rated_minutes + max_overtime_minutes` (480 + 240). The
laboratory-closing violation (14) falls back to `instance.horizon[1]` when no
`lab_closing_time` attribute is present, and is skipped when the instance has no
laboratory.

## P8/P9 — permutation algorithms (`optimisation/mtwmo/algorithms.py`)

`ASSUMPTION:` The heuristic half of the initial population assigns each patient
to its highest-skill **compatible** caregiver (ties broken by preference, then
randomly) and orders each route by earliest time-window start; patients with no
compatible caregiver are placed before the first separator so they decode as
unassigned. Diversity within the heuristic half comes from random tie-breaking.
Genes are the exact set `1..N+K`, so every chromosome is a permutation and all
crossover/mutation operators preserve that invariant.

`ASSUMPTION:` P9 lists only `pc = 0.9`; the mutation probability defaults to
`0.1`. Three crossovers are provided and chosen uniformly when crossover fires:
PMX (required), order crossover (OX1) and position-based crossover (PBX). Local
search uses a constraint-first lexicographic key
`(infeasible?, total_violation, Z1, -Z2, Z3)` to accept a deletion+insertion
move; the global GA selection still uses constraint-domination.

## P10 — MTCMOLite (Qiao et al. 2022 idea)

`ASSUMPTION:` The original Qiao et al. (2022) MTCMO paper/code was not available
in this environment, so `MTCMOLite` re-implements only the described idea and
**must never be reported as the original MTCMO**. Faithful elements: two
cooperating tasks (a main task using true constraint-domination and an auxiliary
task using an `epsilon`-constraint relaxation where any individual with total
violation `<= epsilon` counts as feasible), a decaying `epsilon` schedule, and
periodic knowledge transfer (elite migration) between the two populations.

`ASSUMPTION:` Deviations from the original MTCMO: (a) `epsilon` decays linearly
from the initial maximum violation to 0 over `epsilon_fraction` (default 0.5) of
the generations instead of an adaptive schedule; (b) transfer moves the best
10% of each population into the other every `transfer_interval` (default 10)
generations and re-selects by constraint-domination; (c) no problem-specific
auxiliary objective transformation is applied beyond the `epsilon` relaxation;
(d) the same permutation operators (PMX/OX/PBX, swap/insert) are shared with
NSGA-II. `MTCMOLite` therefore measures a budget-relaxed two-task search, not
the published MTCMO algorithm.

## R1–R4 — RAG and LLM reasoning

`ASSUMPTION:` The R-series rules live in `connaissance/rules.json` and the
C-series validation constraints stay in `connaissance/contraintes.py`. The
retriever searches **only the R-series rules**, so the R2 example
("Patient P4 is urgent and caregiver S2 is delayed.") returns exactly `R001` and
`R002`. This deliberately replaces the previous behaviour where constraints
(`C001`, `C003`) leaked into the result; `rag/test_rag.py` was updated
accordingly.

`ASSUMPTION:` Keyword matching is token-based with accent/case folding and a
prefix rule: a keyword token of length >= 4 matches a query token that starts
with it (so "delayed" matches "delay"), shorter keywords require an exact token.
A multi-word keyword (e.g. "time window") matches only when all its tokens are
present. Score = number of matched keywords; results are sorted by
`(-score, id)` and filtered to `score > 0`.

`ASSUMPTION:` Embeddings and the vector store are optional. The default path
(`embedder=None`) is keyword-only and fully offline. `HashingEmbedder` +
`InMemoryVectorStore` provide a deterministic, dependency-free hybrid signal
(cosine bonus weighted by `embedding_weight`, default 0.5) when enabled.

`ASSUMPTION:` The Ollama client is imported lazily; `LLMRaisonner(timeout=30.0)`
passes the timeout to `ollama.Client`. Any import/connection/timeout/schema
failure falls back to a deterministic rule-based classifier that returns the
validated `ReasoningResult` shape (same JSON keys) with
`reasoning_source == "fallback"`. Tests never require Ollama.

## A1–A5 — message-passing multi-agent scenarios

`ASSUMPTION:` Messages are immutable `Message(sender, receiver, type, payload)`
objects appended to a `MessageLog`; `interaction_count` = number of messages.
Scenario handlers live on the existing `Orchestrator` (its LLM-request
`process` path is unchanged). Replacement caregiver selection picks the
highest-skill eligible caregiver (tie-break by id). Patient urgency ranking is
`Critical < High < Medium < Low`. A vehicle breakdown marks the caregiver
unavailable before searching replacements. These handler semantics are a
documented simple reading of the A4 scenario descriptions, not a published
protocol.

## M1 / U1 / U2 / P10 / P11 — metrics, decision, benchmarks, UI

`ASSUMPTION:` Stage timers use `ComplexityMetrics.stage(name)` context managers
holding no global state (each instance owns its own dict). The hypervolume is an
exact deterministic algorithm over minimisation points; when no reference point
is supplied the nadir + 1 is used. Pareto-quality points are the archive with
Z2 negated, i.e. `(Z1, -Z2, Z3)`.

`ASSUMPTION:` The decision engine's post-optimisation rule is configurable:
`weighted_sum` (default weights cost 0.4 / satisfaction 0.4 / balance 0.2) or
`ideal_point`, both computed on objectives min-max normalised over the front
(satisfaction inverted so lower is better). Weights never enter NSGA-II; they
apply only to the already-computed Pareto set. The justification text is built
from the selected candidate's real Z1/Z2/Z3, its six cost components, the number
of non-dominated solutions and the normalised scores.

`ASSUMPTION:` P10 "mean of non-dominated solutions per objective" is read as the
pooled mean of each objective over all non-dominated feasible solutions found
across the independent runs (computed on feasible solutions only). "Best per
objective" is the min Z1 / max Z2 / min Z3 over those same solutions. An
em-dash `"—"` marks an instance/algorithm where no feasible solution was found.

`ASSUMPTION:` The "cost of robustness" is reported as the Z1 change when the
Bertsimas–Sim budget moves from `Γ=0` to `Γ>0` (same instances, same seeds), as
absolute and relative differences on the best and mean Z1.

`ASSUMPTION:` The end-to-end pipeline runs the optimiser on the canonically
generated P11 instance (e.g. `A_2_1`) rather than feeding the teacher CSV
coordinates into the generator: the CSV/benchmark coordinates can fall outside
the 50×50 location square and empty a time-window centre interval. The clinical
CSV still feeds the RAG/LLM/MAS stages. The LLM stage degrades to the validated
fallback when Ollama is absent, so the whole pipeline runs offline.

## D3 — SVM → K-Means → Tabu pipeline (`ml/`)

`ASSUMPTION:` The SVM target is the teacher `Predictions` column (D1/D2). Because
that label is a deterministic function of `Diabetic`, the cross-validated
accuracy/precision/F1 measured here are `1.0`; this leak is reported, not hidden,
and no paper figure (0.8184/0.7936/0.8171) is copied. Metrics are macro-averaged
over stratified folds.

`ASSUMPTION:` Features for K-Means are standardised before clustering. On raw
scale the coordinates (up to hundreds of units) would drown the 0/1 care-type
column, contradicting the paper's `[care_type, X, Y]` intent. This is the
documented simplest reading.

`ASSUMPTION:` Benchmark customers (Solomon/G&H) carry no clinical features, so
the SVM care type is attached positionally: teacher row `i` ↔ customer `i`
(cyclic for subsets larger than 1000). D4's coordinate-mapping hypothesis was
checked and is weak (`match_rate ≈ 0.12`), so positional attachment is preferred.

`ASSUMPTION:` The per-cluster router is a single-vehicle closed tour
(depot → customers → depot) solved by a deterministic Tabu Search: a
nearest-neighbour seed, best improving swap, tabu tenure on the swapped pair, and
the aspiration criterion. It is a routing heuristic for the D3 comparison, not
the P6/P7 multi-objective HHCRSP model.

`MEASURED (no paper number copied):` on the local Solomon/G&H subsets the
`SVM + K-Means + Tabu` pipeline produced **longer** total tours than the
`K-Means + Tabu` baseline (improvement −8% to −67%; see
`ml/results/ml_benchmark.csv` and the real command output in the phase report).
The extra `care_type` column splits spatially coherent clusters, which is the
measured cause. This negative result is reported as-is.

## A1/A3 — optional MAS pre-clustering (`ml/mas_integration.py`)

`ASSUMPTION:` Caregivers/patients can be pre-grouped with K-Means on
`[care_type, X, Y]` before assignment (`Orchestrator(patient_clusters=...)`,
`main.py --precluster`). Grouping is only a hint: the orchestrator still enforces
skill and availability constraints, so a cluster never overrides feasibility.



