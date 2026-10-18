# HHCRSP-MAS — project rules and requirements

## Project
Python project solving HHCOP/HHCRSP (home health care routing & scheduling) with
RAG + LLM (Ollama) + Multi-Agent System + multi-objective optimisation (NSGA-II, MTCMO baseline).
Pipeline (must match the architecture diagram):
Streamlit UI (interface/) -> RAG retriever (rag/retriever.py, knowledge base in connaissance/)
-> LLM reasoner (llm/raisonner.py, Ollama) -> Multi-agents (agents/agentOrchestrateur.py,
agentsPatients.py, agentsSoignants.py) -> NSGA-II (optimisation/nsga2.py, mtwmo/) ->
Decision engine (décision/decision.py) ; transversal: complexity/metrics.py.
Data: data/dementia (patient CSV), data/Gehing&Homberger (200/400/1000 customers), data/Solomon/In.
Language of code/comments: English; domain words may stay French. Windows + PowerShell + conda base.

## Working rules (strict)
- Never fabricate results, metrics, tables or "tests passed". Only report what you actually ran; paste the command and its real output.
- Before changing code, read it. Prefer the smallest change that satisfies a requirement. No rewrites of working modules.
- Every requirement below has an ID. Each change must name the requirement IDs it addresses. Each requirement ends as: DONE (with test), PARTIAL, MISSING, or CONFLICT (explain).
- Add/extend pytest tests in tests/ for every behaviour you add. Run `python -m pytest -q` before saying a phase is done.
- Do not delete files or edit raw data under data/. If something must be renamed/deleted, list it and ask.
- All randomness seeded and configurable. No hard-coded absolute paths; use pathlib relative to repo root.
- If a requirement is ambiguous or a paper detail is unavailable, implement the documented simplest reading, label it `ASSUMPTION:` in code + docs/ASSUMPTIONS.md, and continue. Do not silently invent.
- Stop at the end of the phase and report: files changed, tests run (real output), remaining gaps.

## REQUIREMENTS DIGEST

### P — Problem model (MTWMO-UnHHCRSP, Annals of Operations Research paper + course "Problématique")
P1  Directed graph G=(V,A). Node 0 = HHC agency (departure), node N+1 = laboratory (every caregiver must END there), nodes 1..N = patient requests. Locations of all nodes known; Euclidean distances.
P2  Each patient has 1, 2 or 3 time windows [e_i^π, l_i^π]; exactly ONE service per patient, inside ONE of its windows. Patients do not all have the same number of windows. Generation: 50% one window, 30% two, 20% three. Window width ~ U[60,120] min. Window centre: 1 window ~ U[e0+t0i, l0−ti0−θi]; 2 windows ~ U[e0+t0i, l0/2] and U[l0/2, l0−ti0−θi]; 3 windows ~ U[e0+t0i, l0/3], U[l0/3, 2l0/3], U[2l0/3, l0−ti0−θi].
P3  Soft windows: arriving before e -> waiting cost; arriving after l -> tardiness penalty (not infeasible).
P4  Skills: patient required level p_i in {1,2,3} (generated 50%/35%/15%); caregiver skill g_k; constraint g_k >= p_i for the serving caregiver. Service time θ_ik = θ_i · y_ik · p_i / g_k, with θ_i ~ U[30,90] min.
P5  Uncertainty (3 sources, robust optimisation, Bertsimas–Sim budget sets):
    - caregiver departure time: s̃_k = s_k + β_k·ŝ_k  (nominal departure 0; caregivers may be late)
    - travel time: t̃_ij = t_ij ± α_ij·t̂_ij, α_ij in [0,1]; budget Γ_t in [0, L]; integer part of Γ_t arcs at max deviation, fractional part on one more arc; Γ_t=0 => deterministic, Γ_t=L => absolute robust
    - patient skill demand: p̃_i = p_i ± σ_i·p̂_i with budget Γ_p; departure budget Γ_s
    The deterministic model is the Γ=0 special case. Γ must be a runtime parameter.
P6  THREE objectives, kept separate (NO weighted-sum scalarisation, NSGA-II must see a vector):
    Z1 (min) total cost = service cost + waiting cost + overtime cost + travel cost + fixed cost (caregiver use) + tardiness penalty (six components: cs·θ_ik·p_i·y_ik ; cw·w_k ; b_k·max(0, st_k − st)·x_0jk ; ct·d_ij·x_ijk ; r_k·x_0jk ; ctw·max(0, tt_i − l_i^π)).
    Z2 (max) patient satisfaction = number of patients served first, then caregiver preference q_ik in {0,1,2,3} and skill margin: Σ o_i + Σ_i Σ_k (q_ik + g_k − p_i)·y_ik  (o_i read as "patient i served"; verify with paper, mark ASSUMPTION if unclear).
    Z3 (min) workload balance = variance of caregivers' total working hours: E[st_k²] − (E[st_k])².
P7  Constraints: (5) each patient visited at most once; (6) flow conservation per caregiver; (7)(8) each caregiver leaves agency 0 and ends at laboratory N+1; (9) time propagation tr_jk >= tt_ik + t_ij + θ_ik (big-M); (10) patient–caregiver incompatibility z_ik·y_ik = 0 (interpersonal relations); (11) no sub-tours; (12) service inside a chosen time window; (13) working time <= T (rated 480 min, max overtime 240 min); (14) return to lab before lab closing time; (15) caregiver skill >= patient requirement; (16) departure->first patient time relation; (17)–(21) binary/integer domains (q_ik in {0..3}).
P8  Encoding: permutation chromosome; genes 1..N = patients, N+1..N+K = caregiver separators (patients between two separators = route of the first caregiver, order = visit order). Initial population: 50% heuristic (assign max-compatible skill, order by time windows), 50% random.
P9  Algorithm parameters: population 100, generations 500, crossover prob 0.9, archive max 100. Crossover multimodal incl. PMX; local search with deletion + insertion operators. Constraint handling must be constraint-domination (feasible beats infeasible), not a blind penalty that hides infeasibility.
P10 Benchmarks & tests (course "Benchmarks et Tests"): compare our method against NSGA-II (Deb et al. 2002) and MTCMO (Qiao et al. 2022). 5 independent runs per instance. Report: best value per objective over the 5 runs (min Z1, max Z2, min Z3); mean of non-dominated solutions per objective; generation of first feasible solution; mean number of caregivers used; number of Pareto solutions; mark instances where an algorithm finds no feasible solution.
P11 Instance sets (Dataset_K_loc, loc 1/2/3 = 50×50 / 100×100 / 200×200): A 10 patients, K=2,3,4; B 30 patients K=5,8; C 50 K=10,12; D 75 K=15,18; E 100 K=20,25; F 200 K=45,58. Patients ≈ 3–5× caregivers. Time windows count in [1,3].

### A — Multi-agent system (course file ArchitectureMAS)
A1  Orchestrator agent: controls relations/communication between all agents; arbitrates conflicts; redistributes patients.
A2  One Caregiver (Soignant) agent per caregiver: differs by skills and availability/workload.
A3  One Patient agent per patient: handled by priority (urgency) and required care.
A4  Collaboration scenarios, each a real message flow + automated test:
    - Caregiver delay (traffic): caregiver informs orchestrator (or patient); orchestrator redistributes the patient to another caregiver.
    - Patient cancellation: patient agent notifies caregiver agent / orchestrator; appointment removed.
    - Vehicle breakdown: caregiver agent alerts orchestrator; orchestrator assigns a replacement to keep continuity of care.
    - Conflict between critical patients requesting the same caregiver: orchestrator prioritises the most urgent.
    - Conflict between caregivers / patient exchange: caregivers talk to orchestrator; orchestrator replaces/reassigns.
    - (extensible table: more scenarios may be added.)
A5  Final result: balanced tours (Z3) and satisfied patients (Z2).

### R — RAG (course file ArchitectureMAS)
R1  Six stages, each a separate function/class: knowledge-base construction, embeddings, vector storage, retrieval, context enrichment, LLM reasoning.
R2  The Retriever is "the brain of the RAG": it answers "which knowledge documents are useful for THIS query?" It must NOT return all rules.
    Example: query "Patient P4 is urgent and caregiver S2 is delayed." -> returns only R001, R002 (the urgency rule and the delay rule).
R3  Relevance-score retriever: each rule has an id (R001…), text, keywords (e.g. urgent, priority, emergency). Pipeline: tokenise query -> compare with keywords -> compute score (count of matching keywords: "urgent patient"=1, "urgent emergency priority"=3) -> sort -> return top-k above a threshold. Embedding/vector search may be added as a hybrid second signal, but the keyword-score retriever is required and must be deterministic and unit-tested.
R4  Retrieved rules become the context passed to the LLM (llm/raisonner.py, Ollama local). The LLM output must be structured (JSON schema) with a validated fallback when the LLM is unavailable (tests must run without Ollama).

### D — Data (teacher dataset + ICCCI 2025 paper "Multi Objective Optimization of the HHCOP Using ML and TS")
D1  Teacher file data/dementia/new_dataset_with_predictions.csv: 1000 rows, 26 columns, all features min-max normalised to [0,1]; XCOORD./YCOORD. are normalised coordinates (must be rescaled to the instance area / Gehring–Homberger coordinate range before computing distances); column `Predictions` = "care type" (the SVM output of the paper).
D2  Verified facts about this file (do not re-derive blindly, but re-check): only 300 of 1000 `Predictions` are filled (155 zeros, 145 ones; rows 700 are NaN); on those 300 rows `Predictions == 1 − Diabetic` exactly (so it is a deterministic label, an SVM trained on it is trivial/leaky); only 2 classes whereas the paper describes 4 departments (diabetes, heart disease, hypertension, dementia); `Dosage in mg` is NaN in 515 rows (mostly Prescription≈0.667).
D3  Paper pipeline (ICCCI): SVM (One-vs-Rest) classifies patients to a care type -> K-Means (k=3) on [care type, X, Y] -> Tabu Search per cluster; baseline = K-Means + Tabu Search without SVM; reported SVM accuracy 0.8184, precision 0.7936, F1 0.8171. Benchmarks used: Gehring–Homberger geography + this medical dataset; Solomon (25/50/100 patients).
D4  1000 rows probably correspond to the Gehring–Homberger 1000-customer instance (HYPOTHESIS — verify by comparing min-max normalised G&H coordinates with XCOORD./YCOORD.). Subsets of 200/400 customers map to the folders 200/400.

### M — Metrics & complexity (diagram block 7)
M1  Measure: execution time per stage (RAG, LLM, agents, NSGA-II, total); number of agent interactions (patients/caregivers); problem size; delays and overtime; patient satisfaction; caregiver workload balance. Add Pareto-quality indicators (hypervolume, number of non-dominated solutions) for the algorithm comparison.

### U — UI & decision (diagram blocks 1 and 6)
U1  Streamlit: scenario input, results/graphs (Pareto front, routes), process tracking.
U2  Decision engine: select the best solution from the NSGA-II Pareto set with an explicit, documented rule (e.g. configurable objective weights or ideal-point distance applied AFTER optimisation), final assignments, visit plan, justification text.