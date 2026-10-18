# Conformity report

## Scope and scoring

This report summarizes the trace in REQUIREMENTS_TRACE.md. Status scores are OK = 1, PARTIAL = 0.5, MISSING = 0, CONTRADICTS = 0. Percentage = sum of status scores divided by the number of requirement rows. The group inventories count unique checks: CORE includes the 33 Section 3 rows, six uncertainty checks, nine algorithm checks, seven baseline/comparison-metric checks, and two separate window-distribution/count checks (57 rows); overlapping detail in the window verification table is not counted again. PIPELINE includes the ten architecture rows. EXTRA is a catalog of four observed implementation additions and measures presence of those additions, not adherence to a requirement.

| Group | OK | PARTIAL | MISSING | CONTRADICTS | Conformity |
|---|---:|---:|---:|---:|---:|
| CORE | 1 | 19 | 34 | 3 | 18.42% |
| PIPELINE | 3 | 7 | 0 | 0 | 65.00% |
| EXTRA | 4 | 0 | 0 | 0 | 100.00% |

## CONTRADICTS

| Item | Exact specification | Code behavior | Evidence |
|---|---|---|---|
| Proposed optimization algorithm | Paper specifies IMTCMO as the proposed algorithm and NSGA-II as a comparison baseline. The architecture diagram instead labels NSGA-II as the main optimizer. | Main path runs the NSGA2-named class. It is a simplified Pareto sampler, not IMTCMO or canonical NSGA-II. It agrees with the diagram label but conflicts with the paper’s proposed-vs-baseline role. | main.py:12, 177-185; optimisation/nsga2.py:11, 279-294; trace algorithm table. |
| Algorithm defaults | Population 100 and maximum 500 generations. | NSGA2 defaults to population_size=20 and generations=10; main passes neither override. | optimisation/nsga2.py:18-30; main.py:180-185. |
| Overtime cap | Required maximum overtime cap: 240 minutes. | In the main run, missing max_overtime_hours falls back to 2 hours (120 minutes), half the specified cap. | optimisation/nsga2.py:154-161; data/dataset_loader.py:172-188. |

The diagram/paper disagreement is itself a specification conflict: the diagram depicts NSGA-II as primary while the paper describes IMTCMO as proposed and NSGA-II as baseline. The code’s class choice follows the diagram label.

## Prioritized fix plan

P1 items address correctness of the mathematical model/objectives. P2 items add missing algorithm, benchmark, and architecture capabilities. Effort estimates are relative: S = under a day, M = several days, L = multi-day design and implementation.

| Priority | Fix item | File(s) to change | Effort | Test that demonstrates the fix |
|---|---|---|---|---|
| P1 | Implement route representation and constraints (6)–(9), (11), (14), (16)–(21), with a valid route schedule and assignment uniqueness (5). | optimisation/nsga2.py; data/dataset_loader.py | L | tests/test_feasibility.py: generate seeded instances, require validate_solution returns no violations for every candidate; also mutate a route to violate each equation and require the checker to identify it. |
| P1 | Correct Z1/Z2/Z3 and time-window cost calculations: early waiting, late tardiness, served/preference/skill-minus-demand satisfaction, and the specified workload-variance formula. | optimisation/nsga2.py | L | Add tests/test_objectives.py with hand-computed assignments; require Z1 component values at early/on-time/late arrivals and exact Z2/Z3 values. Existing tests/test_time_windows.py must pass unchanged. |
| P1 | Implement the Bertsimas–Sim uncertainty sets and Γ^s/Γ^t/Γ^p controls for departure time, pairwise travel, and patient demand skill. | optimisation/nsga2.py; data/dataset_loader.py | L | tests/test_robust.py: Γ=0 equals deterministic; maximum-budget adversarial cases stay within bounds and cannot enlarge the feasible set relative to Γ=0. |
| P1 | Represent demand/skill levels and patient-rejected caregiver pairs so skill matching and interpersonal exclusion use the model’s p_i, g_k, and z_ik/y_ik semantics. | data/dataset_loader.py; agents/compatibilite.py; optimisation/nsga2.py | M | Add focused tests for each demand/skill level and rejected pair; rejected assignments must fail validate_solution and be absent from generated feasible solutions. |
| P2 | Implement the proposed IMTCMO workflow: two populations/epsilon schedule, specified initial mix and encoding, OX/PMX/elite crossover, four elite variants, local search, external archive, and 100/500 settings. | optimisation/nsga2.py | L | Add tests/test_imtcmo.py covering population separation, epsilon monotonicity, encoding validity, operator branch selection/variants, archive cap, and configured population/generation defaults. |
| P2 | Add the Annals Table 1 synthetic instance generator, including map sizes, patient/caregiver ratios, window-count proportions, service-time ranges, and window-width ranges. | data/dataset_loader.py or a new generator module under data/ | M | Replace the existence-only check in tests/test_instance_generation.py with seeded generation checks for allowed dimensions/ratios and exact or tolerance-bounded 50/30/20 counts and U-range bounds. |
| P2 | Add a separate canonical NSGA-II baseline and same-instance comparison runner; implement GD, IGD, HV, Spacing, and Spread as reusable production metrics. | optimisation/ (baseline, metrics, and benchmark runner modules); tests/test_metrics.py | M | Keep tests/test_metrics.py’s analytic toy-front checks; add a benchmark test requiring both algorithms to consume the same seeded instance and emit all five metrics. |
| P2 | Complete the architecture’s retrieval stack with embeddings/vector storage rather than keyword-only retrieval. | rag/retriever.py; knowledge-base/storage module | L | Add retrieval tests where semantically relevant documents have no exact query keyword and must be returned, plus deterministic top-k checks. |
| P2 | Implement agent event messages and orchestration for patient cancellation, caregiver delay, vehicle failure, and reassignment. | agents/agentOrchestrateur.py; agents/agentsPatients.py; agents/agentsSoignants.py | M | Add scenario tests for each event, checking the expected notification, reassignment/continuity outcome, and unchanged unaffected assignments. |
| P3 | Resolve the paper/diagram algorithm naming conflict and document which algorithm is proposed versus baseline; update defaults in user-facing documentation after P1/P2 implementation. | docs/specs/image.png; docs/specs/ArchitectureMAS.odt; readme.md | S | Documentation review: diagram, README, runtime labels, and algorithm description agree on primary optimizer and baseline. |

## Validation caveats

The current trace records a targeted validation run with five failures and eight passes. A full pytest run stops at collection because data/test_data.py references a CSV at a machine-specific path. Excluding that file gives six failures and 25 passes, including a pre-existing complexity-metrics assertion failure. These are documented in REQUIREMENTS_TRACE.md; no source implementation was changed while preparing this report.
