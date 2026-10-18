# AUDIT — HHCRSP-MAS

Re-audit against the current code. Every status below cites a test name (from
`C:\Users\salmi\Documents\GitHub\HHCRSP-MAS\tests\`) or a command whose real
output is recorded in section 3. No status is DONE without one of the two.

Scope inspected: `AGENTS.md`, `readme.md`, `main.py`, `agents/`, `rag/`, `llm/`,
`optimisation/` (incl. `mtwmo/`), `ml/`, `décision/`, `complexity/`, `interface/`,
`connaissance/`, `data/`, `experiments/`, and `tests/`.

Status legend: DONE / PARTIAL / MISSING / WRONG. Strict rule kept from the first
audit: a helper that exists but is **not called by the pipeline** and has no
end-to-end evidence is not counted as DONE.

---

## 1. Traceability matrix

| ID | Requirement (short) | Implementation (file:function) | Status | Evidence (file:line · test/command) |
|---|---|---|---|---|
| P1 | Graph G=(V,A), depot 0, lab N+1, Euclidean distances | `optimisation/mtwmo/domain.py:73-108` `Instance.depot/laboratory/distance/travel_time`; lab used in `model.py:303-305` | DONE | `domain.py:73-108`; pipeline runs this model (`main.py:20,197-213`). Tests: `tests/test_model.py::test_decode_splits_routes_at_separators`, `tests/test_domain_loader.py::test_loader_counts_depot_and_first_last_windows[...]`, `tests/test_generator.py::test_pair_distance_and_travel_time_match_fixed_seed` |
| P2 | 1/2/3 windows, 50/30/20, width U[60,120], centres per P2 | `generator.py:27` `WINDOW_COUNT_WEIGHTS`; `generator.py:43-62` `choose_window_count`; `generator.py:65-` `generate_instance` | DONE | `generator.py:27,43-62`. Tests: `tests/test_generator_p2.py::test_windows_are_valid_and_service_time_range`, `::test_requested_window_count_distribution_over_200_seeds`, `tests/test_time_windows.py::test_patient_window_count_is_preserved[one/two/three-windows]`, `::test_patients_may_have_different_window_counts` |
| P3 | Soft windows: early wait cost, late tardiness | `domain.py:42-52` `evaluate_window_arrival`; `model.py:214-230` `choose_window`, `model.py:340-359` schedule; cost `model.py:419,432` | DONE | Tests: `tests/test_model.py::test_choose_window_prefers_tardiness_then_waiting`, `::test_hand_computed_objectives`, `tests/test_domain_limits.py::test_soft_time_window_components[85-15-0/115-0-0/142-0-12]`. NB the legacy `optimisation/nsga2.py` still fails 2 window tests (see §3); it is not used by the pipeline |
| P4 | p_i∈{1,2,3} 50/35/15, g_k, θ_ik=θ_i·p_i/g_k, θ_i~U[30,90] | `generator.py:28` `DEMAND_WEIGHTS`, `generator.py:230-241` `service_time_for`; `model.py:233-236` `service_minutes` | DONE | Tests: `tests/test_generator_p2.py::test_skill_rule_and_service_time_formula`, `tests/test_model.py::test_skill_violation_when_caregiver_underqualified` |
| P5 | Uncertainty, Bertsimas–Sim budgets Γ_t/Γ_s/Γ_p runtime | `model.py:170-199` `budget_sum`/`_resolve_gamma`; `model.py:319-370` worst-case budgets; `gamma` arg in `Algorithm.run`/`evaluate` | DONE | Tests: `tests/test_model.py::test_budget_sum_bertsimas_sim`, `::test_gamma_zero_equals_deterministic`, `::test_travel_budget_adds_worst_case_distance_margin`, `::test_departure_budget_increases_working_time`. NB legacy `tests/test_robust.py` fails against old `optimisation/nsga2.py` |
| P6 | 3 objectives kept separate (Z1 six components, Z2, Z3) | `model.py:417-433` cost split; `model.py:435-443` Z2; `model.py:467-472` Z3; `model.py:160-163` vector | DONE | Tests: `tests/test_model.py::test_hand_computed_objectives`, `::test_evaluation_vector_and_accepts_decoded_solution`, `tests/test_algorithms.py::test_objectives_are_never_combined_into_one_number`, `agents/test_agents.py::test_nsga_ii_separates_coverage_satisfaction_and_workload_objectives` |
| P7 | Constraints (5)–(21) | violations `model.py:463-481`; constraint-domination `algorithms.py:153-175`; subtour/flow implicit in permutation encoding | PARTIAL | Tests: `tests/test_model.py::test_duplicate_patient_is_single_visit_violation`, `::test_skill_violation_when_caregiver_underqualified`, `::test_rejection_causes_incompatibility_violation`, `::test_overwork_beyond_hard_limit_is_flagged`, `::test_lab_closing_violation`, `::test_missing_window_is_counted_as_violation`, `tests/test_algorithms.py::test_constraint_domination_feasible_beats_infeasible`. Gaps: no explicit flow-conservation/time-propagation/subtour constraint functions; feasibility is violation-based (soft) rather than the (5)–(21) hard set |
| P8 | Permutation chromosome + separators; 50/50 initial population | `model.py:243-286` `decode`; `algorithms.py:517-526` init; `algorithms.py:292-331` | DONE | Tests: `tests/test_model.py::test_decode_*` (6 incl. `test_decode_splits_routes_at_separators`), `tests/test_algorithms.py::test_initial_permutations_span_all_genes` |
| P9 | Pop 100/gens 500/pc 0.9/archive 100, PMX, local search, constraint-domination | `algorithms.py:496-515` defaults; `:333-401` PMX/OX/PBX; `:424-462,636-646` local search; `:177-290` sort/crowding | DONE | Tests: `tests/test_algorithms.py::test_pmx_matches_known_swaps`, `::test_order_and_position_crossover_are_valid_permutations`, `::test_crowding_distance_boundaries_and_midpoint`, `::test_fast_non_dominated_sort_handmade`, `::test_local_search_enabled_keeps_solutions_valid` |
| P10 | Compare vs NSGA-II & MTCMO; 5 runs; report table | `experiments/run_benchmark.py:36` `ALGORITHMS`; `:82-140` aggregate/fields; `:156` `--runs` default 5 | PARTIAL | Tests: `tests/test_benchmark.py` (4: `test_aggregate_reports_best_and_mean_over_feasible_solutions`, `::test_aggregate_marks_instances_without_feasible_solution`, `::test_run_one_is_reproducible`, `::test_build_instance_list_defaults_to_a_and_b_caregiver_counts`). All required fields emitted. PARTIAL: MTCMO is `MTCMOLite` (documented deviation `algorithms.py:649-651`), not the original Qiao 2022 |
| P11 | Instance sets A–F, location 1/2/3, ratios, TW∈[1,3] | `generator.py:15-23` `DATASET_TABLE`/`LOCATION_SIDE` | PARTIAL | Tests: `tests/test_generator.py::test_table_lists_explicit_supported_combinations`, `tests/test_generator_p2.py::test_dataset_names_and_counts`. Gaps: A–E cannot use location 3 and F cannot use location 1 (`generator.py:16-21`, enforced `:90-91`); patient:caregiver ratio is 2.5 for A_K4 and 6.0 for B_K5, outside the 3–5 target (test loosened to 2.4–6.1, `tests/test_generator_p2.py:22-24`) |
| A1 | Orchestrator arbitrates conflicts & redistributes | `agents/agentOrchestrateur.py:11`; handlers `:124,:160,:170,:206,:227` | DONE | Tests: `tests/test_mas_scenarios.py::test_caregiver_delay_redistributes_to_compatible_caregiver`, `::test_vehicle_breakdown_assigns_replacement`, `::test_critical_conflict_most_urgent_wins`, `::test_caregiver_exchange_reassigns_patient` |
| A2 | One caregiver agent per caregiver | `agents/agentsSoignants.py` `CaregiverAgent`; `agentOrchestrateur.py:28-30` | DONE | Test: `tests/test_mas_scenarios.py::test_orchestrator_creates_one_agent_per_record` |
| A3 | One patient agent per patient (priority/urgency) | `agents/agentsPatients.py` `PatientAgent`; `agentOrchestrateur.py:25-27` | DONE | Tests: `tests/test_mas_scenarios.py::test_orchestrator_creates_one_agent_per_record`, `agents/test_agents.py::test_orchestrator_routes_emergency_request_and_flags_critical_patient` |
| A4 | Collaboration scenarios + automated tests | `agentOrchestrateur.py:124-263`; `agents/messages.py` | DONE | 5 scenarios, each a message flow + test: `tests/test_mas_scenarios.py::test_caregiver_delay_redistributes_to_compatible_caregiver`, `::test_patient_cancellation_removes_appointment`, `::test_vehicle_breakdown_assigns_replacement`, `::test_critical_conflict_most_urgent_wins`, `::test_caregiver_exchange_reassigns_patient`. Note: not triggered by `run_hhcop_pipeline` (unit-tested only) |
| A5 | Balanced tours (Z3) + satisfied patients (Z2) | Z2/Z3 from `model.py:435-472`; recorded `main.py:229-235` | PARTIAL | Test: `tests/test_metrics.py::test_balance_and_satisfaction_from_schedule` (schedule-level only). Gap: no MAS-level test that the agent collaboration yields the balanced/satisfied outcome |
| R1 | Six RAG stages, each a function/class | `rag/rag.py:22-90` `RAGPipeline` (6 stages); `rag/embeddings.py:17`, `rag/vector_store.py:11` | PARTIAL | Test: `tests/test_rag_rules.py::test_pipeline_exposes_six_stages_offline`. Gap: `run_hhcop_pipeline` uses `SimpleRAG` (`main.py:15,177`), so embeddings + vector store are not exercised end-to-end |
| R2 | Retriever returns only useful docs; example → urgency+delay | `rag/retriever.py:89-120`; rules only `connaissance/base_connaissance.py:16-18` | DONE | Tests: `tests/test_rag_rules.py::test_example_query_returns_exactly_urgency_and_delay_rules`, `::test_constraints_are_never_returned` |
| R3 | Deterministic keyword-score retriever, unit-tested | `rag/retriever.py:82-120` (score, `(-score,id)` sort) | DONE | Tests: `tests/test_rag_rules.py::test_urgent_emergency_priority_scores_three`, `::test_top_k_limits_and_order_is_stable`, `rag/test_rag.py::test_retriever_returns_deterministic_order` |
| R4 | Structured LLM output + validated fallback without Ollama | `llm/raisonner.py:41` schema, `:75-90` Ollama, `:113` fallback, `:155-171` analyze, lazy import `:76` | DONE | Tests: `tests/test_llm_fallback.py` (5), `tests/test_pipeline.py::test_full_pipeline_runs_offline_on_seeded_a21` (`reasoning_source=="fallback"`) |
| D1 | Teacher file 1000×26 normalised; rescale XCOORD/YCOORD | `data/loader.py:54-73` validate, `:177-192` rescale, `:368` `patients_from_dataframe` | DONE | Command (see §3): shape `(1000, 26)`, every column ∈ [0,1]. Tests: `tests/test_loader.py::test_teacher_csv_validates_1000_rows_and_26_columns`, `::test_rescale_coordinates_to_square_areas`, `::test_rescale_coordinates_to_gehring_homberger_range`, `::test_patients_from_dataframe_uses_rescaled_coordinates_and_care_type` |
| D2 | Re-check the documented file facts | verified by direct inspection (no production code) | DONE | Command (see §3): labelled 300 (zeros 155, ones 145, NaN 700); `Dosage in mg` NaN 515; `Predictions==1−Diabetic` True on the 300. Tests: `tests/test_loader.py::test_care_type_source_is_given_or_svm`, `::test_svm_metrics_are_bounded_and_label_is_leaky`, `tests/test_ml_care_type.py::test_label_is_deterministic_function_of_diabetic` |
| D3 | ICCCI pipeline SVM→KMeans(k=3)→Tabu (+ baseline) | `ml/care_type.py`, `ml/clustering.py`, `ml/tabu.py`, `ml/pipeline.py`, `ml/evaluation.py` | PARTIAL | 28 tests (`tests/test_ml_*.py`) pass; measured output `ml/results/ml_benchmark.csv`. PARTIAL: standalone (only pre-clustering wired into `main.py:189-191`); leaky 2-class SVM target; care types attached positionally to benchmark customers; measured pipeline is worse than baseline (all `improvement_percent` negative) |
| D4 | 1000 rows ↔ G&H-1000 hypothesis | `data/loader.py:217-264` `check_d4_hypothesis` | PARTIAL | Test: `tests/test_loader.py::test_d4_hypothesis_returns_correlations_and_match_rate` (bounds only). The check computes row-order/sorted correlations + NN match_rate, but the measured `match_rate ≈ 0.12` means the correspondence is **not** confirmed (documented in `docs/ASSUMPTIONS.md`) |
| M1 | Timings, interactions, size, delay/overtime, satisfaction, balance + Pareto indicators | `complexity/metrics.py` `stage:92`, `problem_size:118`, `agent_interactions:128`, `delays_and_overtime:152`, `balance_from_schedule:205`, `satisfaction_from_schedule:245`, `pareto_quality:256`, `hypervolume:25` | DONE | Tests: `tests/test_metrics.py` (14, incl. `test_hypervolume_*`, `test_pareto_quality_reports_size_and_hypervolume`, `test_balance_and_satisfaction_from_schedule`). All M1-listed metrics covered. NB `complexity/test_metrics.py::test_uncertainty_reports_missing_evidence` fails but tests the non-M1 `uncertainty_indicators` helper |
| U1 | Streamlit scenario input, Pareto/routes graphs, tracking | `interface/app.py:22-38` form, `:102-108` Pareto scatter, `:110-132` route view, `:145-161` tracking | PARTIAL | No test imports `interface/app.py` or `streamlit`; only syntax-verified (`python -m py_compile interface/app.py`). Gaps: route view is a coordinate scatter (not a line-connected map); Z3 not plotted |
| U2 | Decision engine picks Pareto solution with documented rule | `décision/decision.py:40-211` (`select_from_pareto_set`, `_score`, `_build_plan`, `_justify`); used `main.py:216-219` | DONE | Tests: `tests/test_decision.py` (9, incl. `test_select_returns_plan_and_real_justification`, `test_weighted_sum_and_ideal_point_both_select_a_front_member`, `test_visit_plan_is_time_ordered`) |

Matrix summary counts: **DONE 19 · PARTIAL 8 · MISSING 0 · WRONG 0** (27 requirements).

---

## 2. Bugs found (current)

| # | Finding | Location | Impact |
|---|---|---|---|
| B1 | Legacy `optimisation/nsga2.py` fails its own window/robust tests | `optimisation/nsga2.py`; `tests/test_time_windows.py`, `tests/test_robust.py` | Red; not used by the pipeline (new `mtwmo/model.py` is) |
| B2 | Non-ASCII package name `décision` forces `importlib` | `main.py:22-23` | Works; cosmetic |
| B3 | `--full` benchmark with default `--location 1` crashes on dataset F | `experiments/run_benchmark.py:155` + `generator.py:16-21,90-91` | F only supports locations (2,3) |
| B4 | P11 patient:caregiver 3–5× not met for A_K4/B_K5 | `generator.py:16-21`; `tests/test_generator_p2.py:22-24` | Test tolerance loosened to 2.4–6.1 |
| B5 | U1 has no automated test; route view is a scatter | `interface/app.py:110-132` | Untested UI |
| B6 | RAG→LLM ID propagation is effectively empty in the live pipeline (SimpleRAG returns rule text without `R###` ids) | `rag/rag.py:97-100` + `llm/raisonner.py:99-111` | `relevant_rules` empty end-to-end; tested only with raw dicts |
| B7 | Checked-in benchmark CSVs were generated with `--runs 2`, not the default 5 | `experiments/results/*.csv` | Re-run with 5 to reproduce the paper table |
| B8 | `complexity/metrics.py` `uncertainty_indicators` disagrees with its test (`1 != 0.67`) | `complexity/test_metrics.py:17` | One red test; not an M1-listed metric |

---

## 3. Commands actually executed (real output)

### `python -m pytest -q`

```
FAILED complexity/test_metrics.py::ComplexityMetricsTests::test_uncertainty_reports_missing_evidence
FAILED tests/test_robust.py::test_zero_gamma_robust_result_matches_deterministic
FAILED tests/test_time_windows.py::test_arrival_before_window_start_has_waiting_cost_only
FAILED tests/test_time_windows.py::test_arrival_after_window_end_has_tardiness_penalty
4 failed, 219 passed, 3 warnings in 24.39s
```

### `python main.py --seed 1` (seeded end-to-end, offline fallback)

```
2. RAG - CONNAISSANCES RECUPEREES
  1. High-priority or urgent patients must be scheduled before lower-priority patients.
  2. A caregiver whose delay exceeds the acceptable threshold should hand the patient over to another available caregiver.
3. LLM - ANALYSE
  reasoning_source : fallback
  patient_id       : D00001
  caregiver_id     : S2
  urgency          : True
  delayed          : True
4. ORCHESTRATEUR + AGENTS
  Candidats        : ['S3', 'S5', ... 148 ids ...]
  Messages agents  : 0
5. OPTIMISATION (instance A_2_1)
  Algorithme       : NSGA-II
  Front de Pareto  : 41 solutions (41 faisables)
  1re faisable     : generation 0
  Temps            : 0.7857 s
6. DECISION FINALE
  ... Z1=685.3444 (service=224.93, waiting=164.37, overtime=0.00, travel=255.41, fixed=0.00, tardiness=40.64),
  Z2=32.0000 satisfaction, Z3=0.085899 workload variance. Caregivers used=2, visits planned=8.
7. COMPLEXITY (M1)
  hypervolume              : 259533.127045
  non-dominated solutions  : 41
  Total Pipeline            : 2.228804 s
```

### `python experiments/run_benchmark.py --datasets A --runs 2 --generations 8 --population 15 --out experiments/results` (reduced benchmark)

```
[    A_2_1]   NSGA-II run 1/2 seed=0 front=17 feasible=True (0.081s)
[    A_2_1]   NSGA-II run 2/2 seed=1 front=21 feasible=True (0.115s)
[    A_2_1] MTCMOLite run 1/2 seed=0 front=15 feasible=True (0.164s)
[    A_2_1] MTCMOLite run 2/2 seed=1 front=24 feasible=True (0.213s)
[    A_3_1] ... [    A_4_1] ... (all feasible)
[    A_2_1]   NSGA-II robustness gamma=0.5: best Z1 405.3731 -> 411.2832
[    A_2_1] MTCMOLite robustness gamma=0.5: best Z1 65.4742 -> 248.344
Wrote 6 summary rows, 12 run rows, 6 robustness rows to experiments\results
```

### Dataset-fact re-check (D1/D2)

```
shape (1000, 26) labelled 300 zeros 155 ones 145 nan 700
dosage_nan 515
predictions==1-diabetic all: True
columns outside [0,1]: []
```

### Test status summary

- 223 tests collected (`python -m pytest --collect-only -q`).
- 219 passed / 4 failed. The 4 failures are all in the legacy `optimisation/nsga2.py`
  path (`tests/test_time_windows.py` ×2, `tests/test_robust.py` ×1) plus one
  non-M1 uncertainty helper (`complexity/test_metrics.py`).

| Covered by passing tests | Where |
|---|---|
| P1/P3/P5/P6/P8 model + P7 violations | `tests/test_model.py` (19) |
| P8/P9 algorithms, crossover, constraint-domination | `tests/test_algorithms.py` (20) |
| P2/P4/P11 generator | `tests/test_generator_p2.py` (9), `tests/test_generator.py` |
| P10 runner/aggregate | `tests/test_benchmark.py` (4) |
| R2/R3, R1 stage layout | `tests/test_rag_rules.py`, `rag/test_rag.py` |
| R4 LLM fallback | `tests/test_llm_fallback.py` (5) |
| A1–A4 scenarios | `tests/test_mas_scenarios.py` (9) |
| D1/D2/D4 loader | `tests/test_loader.py` (11) |
| D3 ML pipeline | `tests/test_ml_*.py` (28) |
| M1 metrics | `tests/test_metrics.py` (14) |
| U2 decision | `tests/test_decision.py` (9) |
| End-to-end offline pipeline | `tests/test_pipeline.py` (3) |

| No automated test | |
|---|---|
| U1 (`interface/app.py`); `run_benchmark.main()` / CSV schema end-to-end | |

---

## 4. Remaining PARTIAL / MISSING (why)

| ID | Status | Why not DONE |
|---|---|---|
| P7 | PARTIAL | Constraints are violation-based soft checks (skill, single-visit, incompatibility, overtime, lab-closing, window); no explicit subtour / flow-conservation / time-propagation (9) functions. Permutation encoding makes subtours impossible, but the (5)–(21) hard set is not fully implemented as constraints |
| P10 | PARTIAL | MTCMO baseline is `MTCMOLite`, explicitly "NOT the original MTCMO" (`algorithms.py:649-651`); all other report fields present |
| P11 | PARTIAL | Location codes not freely combinable (A–E lack 3, F lacks 1); patient:caregiver 3–5× violated for A_K4 (2.5) and B_K5 (6.0) |
| A5 | PARTIAL | Z2/Z3 are optimiser outputs recorded by the pipeline; no MAS-level test that agent collaboration produces the balanced/satisfied outcome |
| R1 | PARTIAL | Six-stage `RAGPipeline` exists and is tested, but the live pipeline uses `SimpleRAG`, so embeddings/vector-store stages are not exercised end-to-end |
| D3 | PARTIAL | Standalone (only pre-clustering wired); SVM target is the leaky 2-class label; care types attached positionally to benchmark customers; measured pipeline worse than baseline |
| D4 | PARTIAL | Verification code exists, but the measured `match_rate ≈ 0.12` does not confirm the 1000-rows ↔ G&H-1000 correspondence |
| U1 | PARTIAL | Streamlit views exist but have no automated test; route view is a coordinate scatter and Z3 is not plotted |

No requirement is MISSING or WRONG in the current code.
