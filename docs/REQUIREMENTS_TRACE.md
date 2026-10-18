# Requirements trace audit

Audit scope: repository source files and supplied specifications in `docs/specs/`. The architecture image and full ODT text were inspected. The Annals paper was checked through its publisher abstract and bibliographic page; its local PDF was not text-extractable in this environment, so Eqs. 1–21, the full IMTCMO section, and benchmark tables could not be verified. `Problématique - IK.pdf` was present but likewise not text-extractable. No claims about its contents are inferred. Statuses measure implementation evidence against confirmed requirements; inaccessible details are noted as such.

Repository structure (excluding `.git`, caches and benchmark/data file contents): `.vscode/`; `agents/`; `complexity/`; `connaissance/`; `data/` (including `dementia/`, `Gehing&Homberger/` instance sets, `Solomon/In/`); `docs/specs/`; `décision/`; `interface/`; `llm/`; `optimisation/`; `rag/`. Main files: `main.py`, `readme.md`, `requirements.txt`, agent classes and compatibility checks, knowledge rules/constraints, dataset loader, optimizer, decision engine, Streamlit interface, RAG/retriever, LLM reasoner, complexity metrics. Tests exist under `agents/`, `complexity/`, `data/`, `optimisation/`, and `rag/`.

## CORE — Section 3 optimization model (item-by-item)

The matrix checks the elements supplied from Section 3 against the optimizer and its invocation from main.py. “Reachable” means exercised by the normal run_hhcop_pipeline path, not merely that a helper exists. Section 3 is the source for every row; equation numbers are included where applicable. The local paper PDF remains unavailable for independent equation rendering, so these checks use the Section 3 statements supplied in the audit request.

### Sets and parameters

| Item | Required element | Implemented? | Formula/value correct? | Status | Reachable in main run? | Code evidence and finding |
|---|---|---|---|---|---|
| S-01 | Caregiver set K | Partial | N/A | PARTIAL | Yes | Caregiver records/list and ID map: optimisation/nsga2.py:27-35; dataset constructs caregivers at data/dataset_loader.py:172-188. Not modeled as a formal set with route variables. Optimizer runs at main.py:180-185. |
| S-02 | Patient set N | Partial | N/A | PARTIAL | Yes | Patient records/list and ID map: optimisation/nsga2.py:27-35; dataset creates records at data/dataset_loader.py:140-170. No formal routing set/decision variables. |
| S-03 | Depot node 0 | No | N/A | MISSING | No | Dataset retains depot coordinates/source metadata (data/dataset_loader.py:126-136, 149-168), but optimizer has no depot node or depot arc variables (optimisation/nsga2.py:143-175). |
| S-04 | Laboratory node N+1 | No | N/A | MISSING | No | No laboratory node, closing time, or route return representation found in optimizer. |
| S-05 | Pairwise travel time t_ij | No | N/A | MISSING | No | Optimizer stores one scalar travel time per patient and scores it at optimisation/nsga2.py:167-173, 193-201; no i-to-j travel matrix or leg sequence. |
| S-06 | Pairwise distance d_ij | No | N/A | MISSING | No | Dataset computes patient-to-depot Euclidean distance only at data/dataset_loader.py:167; no pairwise d_ij used in optimization. |
| S-07 | Patient service time θ_i | Partial | Partial | PARTIAL | Yes | Service duration/hours are loaded at data/dataset_loader.py:165-167 and read at optimisation/nsga2.py:44-71; ranges can be sampled. No service-start schedule/arc propagation. |
| S-08 | Patient demand level p_i ∈ {p1,p2,p3} | Partial | No | PARTIAL | Yes | Dataset has benchmark vehicle demand at data/dataset_loader.py:159 and a separate clinical skill requirement derived as 5/3/1 at :114-123, 148-155. It does not encode the specified three demand levels as the model demand class. |
| S-09 | Caregiver skill g_k ∈ {g1,g2,g3} | Partial | No | PARTIAL | Yes | Numeric caregiver skill_level is generated as 3, 4, or 5 at data/dataset_loader.py:172-180; optimizer accepts numeric values at optimisation/nsga2.py:131-141. It is not the stated g1/g2/g3 domain. |
| S-10 | Skill rule g_k ≥ p_i | Partial | No (not the specified domains/fields) | PARTIAL | Yes | Numeric comparison exists in agents/compatibilite.py:18-20, used at optimisation/nsga2.py:37-42, 154-160; it compares skill_level against clinical required_skill_level, not p_i in {p1,p2,p3}. |
| S-11 | Maximum daily working time T | Partial | No | PARTIAL | Yes | max_work_hours is checked with an overtime allowance at optimisation/nsga2.py:154-161; current data sets it to 8 hours (data/dataset_loader.py:175-180). This is not the hard constraint total working time ≤ T. |
| S-12 | Rated time 480 minutes | Partial | Partial | PARTIAL | Yes | Dataset uses 8 hours as the maximum (data/dataset_loader.py:175-180), equivalent numerically to 480 minutes, while work/service quantities are maintained in hours (optimisation/nsga2.py:44-49, 207-211). The 480-minute parameter is not explicitly represented. |
| S-13 | Overtime cap 240 minutes | Partial | Partial | CONTRADICTS | Yes | A 2-hour default cap is used at optimisation/nsga2.py:154-161, equivalent to 120 minutes, not the requested 240 minutes. Overtime is costed at :204-211; code units are hours. |

### Constraints (5)–(21)

| Eq. | Required constraint | Implemented? | Formula correct? | Reachable in main run? | Status | Code evidence and finding |
|---|---|---|---|---|---|
| (5) | Each patient visited at most once | Yes | Yes for assignment count; no route visitation variable | PARTIAL | Yes | create_solution loops once over each patient and appends at most one assignment (optimisation/nsga2.py:143-175). Called through optimize at :279-294, from main.py:180-185. |
| (6) | Flow conservation: each caregiver leaves the patient it visits | No | No | MISSING | No | No route arcs, successor variables, or flow equations in optimisation/nsga2.py:143-175, 279-294. |
| (7) | Every caregiver route starts at depot 0 | No | No | MISSING | No | No caregiver route or depot departure arc in optimizer (optimisation/nsga2.py:143-175). |
| (8) | Every caregiver route ends at laboratory N+1 | No | No | MISSING | No | No laboratory node/return arc in optimizer (optimisation/nsga2.py:143-175). |
| (9) | Arrival propagation: start_j ≥ start_i + t_ij + θ_i (big-M) | No | No | MISSING | No | No visit start variables, pairwise t_ij, predecessor relation, or big-M constraint in optimizer. |
| (10) | Rejected patient-caregiver pairs forbidden: z_ik * y_ik = 0 | No | No | MISSING | No | Patient preference is a positive score only (optimisation/nsga2.py:137-141); no rejected-pair field or z/y variables. Preference can be omitted when forming candidates. |
| (11) | Subtour elimination | No | No | MISSING | No | No routing arcs or subtour constraints; assignment list only (optimisation/nsga2.py:143-175). |
| (12) | Service start lies in requested window [e_i^π, l_i^π] | Partial | No | PARTIAL | Yes | Windows are parsed and a window wide enough to contain service duration is selected (optimisation/nsga2.py:100-129, 149-153); no service start is chosen or checked against window bounds. |
| (13) | Caregiver working time ≤ T | Partial | No | PARTIAL | Yes | Candidate feasibility allows workload + assigned service time ≤ max_work_hours + max_overtime_hours (optimisation/nsga2.py:154-161), which can exceed T and omits route travel/return time. |
| (14) | Return to laboratory before its closing time | No | No | MISSING | No | No laboratory closing-time parameter, route return, or return-time check in optimizer (optimisation/nsga2.py:143-175). |
| (15) | Skill: p_i ≤ maximum skill of assigned caregiver | Partial | No (specified p_i and domain absent) | PARTIAL | Yes | Numeric skill compatibility check (agents/compatibilite.py:18-20) is called by optimizer (optimisation/nsga2.py:37-42, 154-160), but checks clinical required_skill_level, not model p_i. |
| (16) | Departure time from depot | No | No | MISSING | No | No depot departure-time variable or parameter in optimizer (optimisation/nsga2.py:143-175). |
| (17) | Binary/integer domain for assignment variables | No | No | MISSING | No | Assignments are ordinary Python dictionaries/lists; no declared binary/integer decision variables (optimisation/nsga2.py:143-175). |
| (18) | Binary/integer domain for routing variables | No | No | MISSING | No | No routing decision variables or domains in optimizer. |
| (19) | Binary/integer domain for time-window selection variables | No | No | MISSING | No | A window tuple is attached to an assignment (optimisation/nsga2.py:168-174); no window-selection variable/domain. |
| (20) | Binary/integer domain for remaining model variables | No | No | MISSING | No | No mathematical-programming variable declarations or domain constraints in optimizer. |
| (21) | Binary/integer domain for remaining model variables | No | No | MISSING | No | No mathematical-programming variable declarations or domain constraints in optimizer. |

### Objectives

| Objective | Required formula | Implemented? | Formula correct? | Reachable in main run? | Status | Code evidence and finding |
|---|---|---|---|---|---|
| Z1 | Minimize total cost: service + early-arrival waiting + overtime + travel + fixed + late-arrival tardiness penalty | Partial | No | PARTIAL | Yes | Six cost buckets are summed at optimisation/nsga2.py:180-181, 198-221, 237-240; total cost is objective 1 at :242-251. Service/overtime/travel/fixed exist. “Waiting” uses caregiver delay + scalar travel (:199), not early arrival against a scheduled start. Tardiness penalty uses caregiver delay (:201), not late arrival. No scheduled arrivals exist. Reached via main.py:180-185. |
| Z2 | Maximize served patients + Σ(preference + caregiver skill − patient demand) | Partial | No | PARTIAL | Yes | Served count and preference/skill score appear at optimisation/nsga2.py:182-186, 223-233; score formula at :137-141. Demand is not subtracted. Coverage is a separate objective; satisfaction is average per served patient and evaluate minimizes its negative at :242-251. Used, but not with the specified formula. |
| Z3 | Minimize workload variance E[st_k²] − (E[st_k])² | Partial | No | PARTIAL | Yes | Scenario workload variance is computed at optimisation/nsga2.py:204-219, averaged across scenarios, and minimized as objective 4 at :242-251. This is mean within-scenario population variance, not the stated variance of expected caregiver workload. Reached via main.py:180-185. |

**Optimizer reachability and unused objectives.** The main pipeline constructs NSGA2, calls optimize(), then selects a Pareto compromise (main.py:177-195); assignment generation, cost summary, satisfaction score, workload variance, and the four-element objective vector are therefore reached. No paper objective is implemented exactly and then omitted from the optimizer: cost, satisfaction, and workload all feed its objective vector, with altered formulas. expected_overtime_hours and average_assigned_skill_level are summary metrics, not separate optimization objectives. Unassigned-patient count is an additional fourth objective, beyond the paper’s three.

### Multiple time windows: storage, selection, and cost verification

Required profile: each patient has 1, 2, or 3 windows; across patients the specified mix is 50% / 30% / 20%. A daily assignment uses exactly one selected window. Early arrival incurs c_w per minute; late arrival incurs c_tw per minute.

| Check | Finding | Status | Evidence |
|---|---|---|
| Storage shape | Optimizer accepts patient.time_windows as a list of either (start, end) tuples/lists or dictionaries with start/end (or ready_time/due_date) keys. It normalizes entries to a list of (float start, float end) tuples. If absent/empty, it falls back to a single ready_time/due_date tuple, or [0, infinity] if either is missing. | PARTIAL | optimisation/nsga2.py:99-118 |
| Loading | dataset_loader copies clinical time_windows if truthy; otherwise each generated patient receives a one-element list containing its benchmark ready_time/due_date. The checked dementia CSV header has no time_windows column, so the normal bundled source data follows the one-window fallback. | PARTIAL | data/dataset_loader.py:87-102, 140-166; data/dementia/dementia_patients_health_data.csv:1 |
| 1, 2, or 3 windows and mixed counts | Parser accepts lists of length 1, 2, or 3 and also does not limit the length to 3. Different patients can carry different lengths. No generation/validation of the required population distribution (50%, 30%, 20%) was found; bundled data produces one window per patient. | PARTIAL | optimisation/nsga2.py:100-118; data/dataset_loader.py:140-166 |
| Required population distribution 50% / 30% / 20% | No generation or validation of the target proportions; the checked bundled dataset falls back to one window per patient. | MISSING | data/dataset_loader.py:87-102, 140-166; tests/test_instance_generation.py:4-9 |
| Window selection | _feasible_window returns the first window whose total span is at least the maximum service duration. create_solution stores that one tuple in the assignment field time_window and appends at most one assignment while iterating once per patient. This selects one window per assignment, but it does not schedule an arrival/start time or ensure that the selected window is feasible in the route. | PARTIAL | optimisation/nsga2.py:120-129, 143-175 |
| Early waiting cost | Not implemented against the selected window: _summary does not read assignment arrival_time or time_window when computing waiting_cost. Its current formula uses caregiver delay plus scalar travel time. | MISSING | optimisation/nsga2.py:193-201 |
| Late tardiness penalty | Not implemented against the selected window: _summary does not compare arrival_time to the selected window end. Its delay_penalty uses caregiver delay directly. | MISSING | optimisation/nsga2.py:193-201 |

The tests added at tests/test_time_windows.py cover 1/2/3 windows, different window counts across patients, one selected window/one assignment, and arrival-cost cases. Result: 6 passed, 2 failed. The early-arrival test expected 10 waiting-cost units (5 minutes × c_w=2) but obtained 0. The late-arrival test expected 15 penalty units (5 minutes × c_tw=3) but obtained 0. These failures confirm that arrival relative to the window is ignored. The inside-window test passes with zero costs, but does not demonstrate a window-aware formula: the current implementation also returns zero because the fixture uses zero caregiver delay and zero travel time. The one-window-use test confirms at most one assignment and one attached tuple; the optimizer does not represent a visit sequence in which a time window could be reused.

### Uncertainty handling: Bertsimas–Sim verification

| Required parameter / check | Finding | Status | Evidence |
|---|---|---|
| Caregiver departure time s_k: nominal 0, deviation ≤30 minutes, budget Γ^s | Not modeled or randomized. Caregivers only carry a current scalar delay field (dataset default 0); it is not a departure-time deviation and has no uncertainty bound/budget. | MISSING | data/dataset_loader.py:172-182; optimiser has no departure-time input or sampling in optimisation/nsga2.py:18-35, 73-97. |
| Travel time t_ij: nominal ±20%, budget Γ^t ∈ [0,L] | Partially randomized, not robustified. Code samples a per-patient scalar travel_time from optional patient travel_time_min/max with a uniform RNG. It does not construct pairwise arc times, compute deviations as 0.2 × nominal, enforce Γ^t or budget uncertainty. | PARTIAL | optimisation/nsga2.py:68-97. Travel samples affect cost summary only at :193-221; route feasibility does not use them. |
| Patient demand skill p_i: nominal ±1 level with σ_i and Γ^p_k | Not modeled or randomized. A fixed required_skill_level is created from clinical thresholds and checked against caregiver skill. No σ_i perturbation or Γ^p_k. | MISSING | data/dataset_loader.py:114-123, 148-155; agents/compatibilite.py:18-20; optimisation/nsga2.py:37-42. |
| Robustness budgets Γ^t, Γ^s, Γ^p | No parameters found. NSGA2 constructor accepts population_size, generations, seed, and uncertainty_scenarios only. | MISSING | optimisation/nsga2.py:18-26. Main constructs it without robustness settings at main.py:180-185. |
| Effect of increasing robustness on caregiver count / feasible set | Not implemented. No Γ values can increase; caregiver count is sourced from benchmark vehicle_count and candidate eligibility is based on availability, delay, care type, skill and workload/overtime. | MISSING | data/dataset_loader.py:172-188; agents/compatibilite.py:4-30; optimisation/nsga2.py:154-165. |
| Deterministic model at Γ=0 | A fixed-value run is possible when optional ranges collapse or are omitted: _build_scenarios then uses one scenario. However, a robust model parameterized by Γ and its Γ=0 counterpart are absent, so deterministic equivalence at all Γ=0 is not reachable as specified. | PARTIAL | optimisation/nsga2.py:73-97; main.py:180-185 passes no Γ. |

Added tests/test_robust.py to assert equality between deterministic and Γ=0 optimizer results on the same instance. Run result: **1 failed** at the prerequisite assertion because the NSGA2 constructor has no gamma_t, gamma_s, or gamma_p parameters; therefore the robust-zero result cannot be constructed or compared. The failure is retained and documents the missing model interface.


### Optimization algorithm and comparison metrics

The paper identifies IMTCMO as the proposed method and NSGA-II as a comparison baseline; the architecture diagram instead labels NSGA-II as the main optimizer. The main pipeline imports and executes optimisation.nsga2.NSGA2 (main.py:12, 177-185). That class performs seeded assignment generation, repeated new-solution sampling, nondominance filtering, lexicographic objective sorting, and truncation (optimisation/nsga2.py:279-294). It does not implement standard NSGA-II's nondominated sorting/crowding-distance selection, and it is not IMTCMO. Thus, code follows the diagram label at the class-name level, while contradicting the paper’s proposed-algorithm role; it is not a faithful implementation of either IMTCMO or a standard NSGA-II baseline.

| Component | Status | File:line | Finding |
|---|---|---|---|
| Algorithm actually run | CONTRADICTS | main.py:12, 177-185; optimisation/nsga2.py:11, 279-294 | Main run executes the NSGA2-named class. Paper says IMTCMO is proposed and NSGA-II is baseline; diagram calls NSGA-II main. Implementation is a simplified custom Pareto sampler, not IMTCMO or canonical NSGA-II. |
| Two-population search: feasible population plus infeasible/auxiliary task; epsilon constraint narrows over generations | MISSING | optimisation/nsga2.py:279-294 | One population is generated; no feasibility/infeasibility populations, auxiliary task, epsilon value, or schedule. |
| Initial population: 50% skill-first/time-window-ordered heuristic and 50% random | MISSING | optimisation/nsga2.py:143-175, 279-284 | Every individual calls the same create_solution procedure, which priority-sorts patients, selects the first duration-feasible window and randomly chooses a compatible caregiver. No 50/50 split or stated heuristic/random constructors. |
| Permutation encoding: patient genes 1..N and caregiver separators N+1..N+K | MISSING | optimisation/nsga2.py:143-175 | Solutions are lists of patient/caregiver assignment dictionaries; no permutation genes or separator encoding. |
| Multimodal crossover: OX / PMX / Elite Strategy by r1 thirds, p_c=0.9 | MISSING | optimisation/nsga2.py:279-294 | No crossover operators or crossover probability. Offspring are fresh calls to create_solution. |
| Elite Strategy with four variants | MISSING | optimisation/nsga2.py:279-294 | No elite selection or segment/node insertion, replacement, reversal, or append operators. |
| Local search: three deletion operators plus shortest-route, skill-sorted insertion | MISSING | optimisation/nsga2.py:279-294 | No route neighborhood, deletion, or insertion search. |
| External Pareto archive, maximum size 100 | PARTIAL | optimisation/nsga2.py:257-277, 279-294 | Computes a nondominated front and carries it between generations, but this is the current population, not a separate external archive; size is limited by population_size (default 20), not archive max 100. |
| Population 100; maximum generations 500 | CONTRADICTS | optimisation/nsga2.py:18-30; main.py:180-185 | Class defaults are population_size=20 and generations=10; main passes no overrides. |

#### NSGA-II baseline and comparison metrics

| Check | Status | File:line | Finding |
|---|---|---|---|
| Separate NSGA-II baseline implementation | MISSING | Repository search: no separate baseline module; optimisation/nsga2.py:11-294 | Only one optimizer class is present. Its name is NSGA2, but its generation/selection logic is not canonical NSGA-II. |
| Can a baseline be run on the same instances? | PARTIAL | data/dataset_loader.py:67-84, 140-189; main.py:180-185 | The NSGA2-named optimizer receives the dataset loaded by the main pipeline, so that implementation runs on those data. No independent baseline and no side-by-side experiment runner were found. |
| GD (Generational Distance) | MISSING | Repository search: no implementation found | No comparison metric code found. |
| IGD (Inverted Generational Distance) | MISSING | Repository search: no implementation found | No comparison metric code found. |
| HV (Hypervolume) | MISSING | Repository search: no implementation found | No comparison metric code found. |
| Spacing | MISSING | Repository search: no implementation found | No comparison metric code found. |
| Spread | MISSING | Repository search: no implementation found | No comparison metric code found. |


### CORE status counts

Counts classify each of the 33 Section 3 rows by implementation coverage: “OK” means implemented as specified; “PARTIAL” means related data/logic exists but is not equivalent; “MISSING” means absent; “CONTRADICTS” means the coded default/value conflicts with the stated requirement. Formula mismatches are detailed per row.

| Group | OK | PARTIAL | MISSING | CONTRADICTS | Total |
|---|---:|---:|---:|---:|---:|
| CORE Section 3 items | 1 | 14 | 17 | 1 | 33 |

## PIPELINE — architecture diagram and ArchitectureMAS.odt

| ID | Requirement | Source (doc + section/page) | Expected implementation | Status | Code evidence (file:line) |
|---|---|---|---|---|---|
| P-01 | User interface accepts scenario and displays results. | `image.png`, block 1; ODT overall architecture | Interactive scenario entry and result display. | OK | `interface/app.py:19-45, 66-128`. |
| P-02 | RAG retrieves rules/constraints and enriches context; ODT describes embeddings/vector storage. | `image.png`, block 2; ODT “RAG classique” | Knowledge construction, embeddings/vector storage, retrieval, context enrichment. | PARTIAL | Keyword scoring at `rag/retriever.py:14-21, 23-65`; context at `rag/rag.py:10-21`. No embedding/vector-store implementation found. |
| P-03 | LLM analyzes scenario and returns structured information/recommendations. | `image.png`, block 3; ODT “Raisonnement du LLM” | Analyze scenario with retrieved context. | OK | `llm/raisonner.py:7-28, 31-45, 52-83`. |
| P-04 | Orchestrator coordinates patient and caregiver agents. | `image.png`, block 4; ODT “Système Multi Agents” | Route requests and coordinate agent collaboration. | PARTIAL | Request routing/candidate discovery at `agents/agentOrchestrateur.py:39-125`; no message-based collaboration. |
| P-05 | Each patient has an agent considering care need and priority. | `image.png`, block 4; ODT patient-agent section | Patient-specific agent with priority/urgency handling. | PARTIAL | `agents/agentsPatients.py:16-43`; reports urgency/priority but does not send cancellations or initiate redistribution. |
| P-06 | Each caregiver has an agent reflecting skills, availability/workload and handling delay/vehicle events. | `image.png`, block 4; ODT caregiver-agent and collaboration sections | Caregiver agent plus event notification and coordination. | PARTIAL | `agents/agentsSoignants.py:1-25` exposes skills, availability, delay, vehicle information; no alert/replacement event flow. |
| P-07 | Optimization returns Pareto solutions for cost, service/satisfaction, workload, uncertainty. | `image.png`, block 5 | Multi-objective optimizer producing selectable solutions. | PARTIAL | `optimisation/nsga2.py:242-251, 279-294` returns Pareto set, but is not IMTCMO and has deviations C-01/C-02/C-08. |
| P-08 | Decision engine selects final solution and produces plan/justification. | `image.png`, block 6 | Select from Pareto set and report assignment. | PARTIAL | Wired at `main.py:189-223`; selector at `décision/decision.py:8-25`. Route ordering/visit planning is absent (`readme.md:32-39`). |
| P-09 | Complexity/performance indicators are collected and presented. | `image.png`, block 7 | Timings, interactions/problem size, delay, satisfaction and workload metrics. | OK | Collection `complexity/metrics.py:15-23, 25-104`; pipeline `main.py:225-257`; UI `interface/app.py:129-239`. |
| P-10 | Final output is balanced work routes and satisfied patients. | ODT “Résultat finale”; `image.png`, blocks 5–6 | Produce routes and evaluate balance/satisfaction. | PARTIAL | Workload/satisfaction summaries at `optimisation/nsga2.py:177-251`; routes/visit order are not generated (`readme.md:32-39`). |

## EXTRA — implementation features not required by inspected specifications

Here, OK means the feature exists, not that a specification requires it.

| ID | Requirement | Source (doc + section/page) | Expected implementation | Status | Code evidence (file:line) |
|---|---|---|---|---|---|
| E-01 | Four objectives, including unassigned patients, beyond the paper abstract's three. | Implementation addition; `optimisation/nsga2.py` | Track coverage separately as an objective. | OK | `optimisation/nsga2.py:242-251`. |
| E-02 | Uniform sampling over optional service-time/travel-time ranges, 32 scenarios by default. | Implementation addition; `readme.md:9-23` | Generate repeatable scenarios and aggregate objective summaries. | OK | `optimisation/nsga2.py:18-35, 73-97, 189-239`. |
| E-03 | Configurable, penalized overtime. | Implementation addition; `readme.md:25-30` | Configurable overtime cap and rate. | OK | `optimisation/nsga2.py:154-161, 204-211`. |
| E-04 | Ollama JSON-schema reasoning and request classifications. | `llm/raisonner.py` (implementation detail) | Structured parsing with a local Ollama model. | OK | `llm/raisonner.py:31-45, 48-83`. |

## Status counts

| Group | OK | PARTIAL | MISSING | CONTRADICTS | Total |
|---|---:|---:|---:|---:|---:|
| CORE (all optimization checks) | 1 | 19 | 34 | 3 | 57 |
| CORE Section 3 items | 1 | 14 | 17 | 1 | 33 |
| PIPELINE | 3 | 7 | 0 | 0 | 10 |
| EXTRA | 4 | 0 | 0 | 0 | 4 |


## Validation harness results

Added tests/test_feasibility.py with validate_solution(solution, instance), which returns constraint-numbered violations. It checks duplicate patient assignments, assignment compatibility/rejection data, window/start data, work limits when routes are available, and flags absent route, depot/laboratory, scheduling, and variable-domain representations instead of treating missing evidence as feasible. The random test instance has 10 patients and 2 caregivers; optimizer candidates are checked against constraints (5)–(21). Current results: the optimizer-feasibility assertion fails because returned assignments omit route flow, depot/laboratory, arrival/start, rejection, and variable-domain data. The duplicate-assignment checker test passes.

Added tests/test_metrics.py with test-local implementations of GD, IGD, two-objective minimization HV, Spacing, and Spread. On the exact toy front [(0,1),(1,0)] against itself, expected values are GD=0, IGD=0, HV=3 for reference point (2,2), Spacing=0, Spread=0. This verifies the test utility implementations analytically; no production metric implementation was found.

Instance-generation check: tests/test_instance_generation.py fails because data.dataset_loader has no generate_random_instance function. The current loader reads benchmark files and maps rows into records (data/dataset_loader.py:67-84, 126-189); it does not generate locations in 50×50/100×100/200×200, patient-to-caregiver ratios of 3/4/5, 50/30/20 time-window counts, U[30,90] service times, or U[60,120] window widths.

### Test run record

- Full command: python -m pytest -q. Collection stopped on data/test_data.py because it reads a hard-coded CSV path that does not exist in this workspace. Added tests/__init__.py to avoid the duplicate test_metrics module-name collision.
- Targeted command for the new harness plus the earlier window/robust tests: 5 failed, 8 passed. Failures: optimizer feasibility (missing constraint representation), Table 1 generator absent, early-window waiting cost, late-window penalty, and Γ=0 robust equivalence (no Γ parameters). The analytic metric test, duplicate-assignment checker, and remaining selected tests passed.
- Full suite with only the incompatible data test excluded: 6 failed, 25 passed. This includes one pre-existing complexity metrics failure (expected uncertainty_score=1, actual 0.67), the five failures listed above, and excludes data/test_data.py.
- Assertions were not changed after failures. No optimizer implementation was modified.
