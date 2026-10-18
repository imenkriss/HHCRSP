# Permutation Encoding and Route Validation Design

Status: design proposal only. No optimizer code was changed for this document.

## 1. Current candidate and loaded-data audit

### Current optimizer representation

In optimisation/nsga2.py, create_solution (around line 144) returns a list of assignment dictionaries. Each assigned patient row contains patient_id, caregiver_id, service_hours, travel_time, and time_window. It does not return a chromosome or visit sequence. Patients are first sorted by priority; the caregiver is chosen randomly from compatible caregivers. Patients without a feasible window or compatible caregiver are omitted from the assignment list.

The optimizer has no permutation decoder, crossover, mutation, or repair function. optimize creates assignment candidates from create_solution and evaluates the assignment list. Candidate records contain solution, objectives, summary, routes, and unassigned_patients. _candidate_record attaches routes using assignment_to_routes before evaluate, but evaluate still scores the assignment list. The current route order comes from assignment_to_routes, which sorts by earliest available window start and retains assignment order as a stable tie-breaker.

### Patient fields exposed by data/dataset_loader.py

The loader parses the clinical CSV, adds a benchmark customer location, and constructs each patient at approximately lines 87–169. The current CSV header has these clinical fields:

| Loaded clinical fields |
|---|
| Diabetic, AlcoholLevel, HeartRate, BloodOxygenLevel, BodyTemperature, Weight, MRI_Delay, Prescription, Dosage in mg, Age, Education_Level, Dominant_Hand, Gender, Family_History, Smoking_Status, APOE_ε4, Physical_Activity, Depression_Status, Cognitive_Test_Scores, Medication_History, Nutrition_Diet, Sleep_Quality, Chronic_Health_Conditions, Dementia |

The patient dictionary also contains the fields below. Some are copied clinical values; the rest are computed or copied from the Solomon benchmark row.

| Field(s) | Available? | Meaning / limit |
|---|---|---|
| id | Yes | Generated clinical ID, e.g. D00001. |
| x, y | Yes | Patient coordinates copied from benchmark customer. |
| service_duration | Yes | Benchmark service duration, in minutes. |
| service_hours | Yes | Derived duration in hours, with a 0.25 hour minimum. The optimizer uses this field when present. |
| time_windows | Yes, but currently one window | The current clinical CSV has no time_windows column. Loader falls back to one window from benchmark ready_time and due_date. The optimizer parser accepts multiple windows if supplied by another source. |
| ready_time, due_date | Yes | Benchmark customer values. |
| demand | Yes | Benchmark vehicle-demand quantity; it is distinct from required_skill_level. |
| required_skill_level | Yes | Derived from clinical priority as 5, 3, or 1. This is the field used by caregiver skill compatibility. |
| care_type, priority | Yes | Derived from clinical data. |
| preferred_caregiver | Yes, but blank | Loader initializes it to an empty string; no rejection history is loaded. |
| travel_time | Yes, but not an arc time | Euclidean distance from the patient to that benchmark's depot. No speed/unit conversion makes this a pairwise travel-time matrix. |
| source_instance, source_customer_id | Yes | Benchmark provenance fields. |

Current time windows are therefore present as data, but the bundled dataset does not supply the paper's mixed one/two/three-window profile.

### Caregiver fields exposed by data/dataset_loader.py

The loader builds caregivers around lines 173–188:

| Field | Available? | Meaning / limit |
|---|---|---|
| id | Yes | Generated caregiver ID S1, S2, etc. |
| skill | Yes | Cycles through Cardio, Diabetes, and General care types. |
| skill_level | Yes | Numeric level 3, 4, or 5. It is not represented as symbolic g1/g2/g3. |
| available | Yes | Boolean; currently true for each loaded caregiver. |
| delay | Yes | Current delay; initialized to zero. |
| current_workload | Yes | Current work; initialized to zero. |
| max_work_hours | Yes | Daily work capacity; currently hard-coded to 8 hours. |
| vehicle_capacity | Yes | Copied from the first loaded benchmark's vehicle capacity and applied to all constructed caregivers. |
| x, y | Yes | All caregivers are placed at that benchmark's depot coordinates. |
| source_instance | Yes | Benchmark provenance. |

### Required routing data that is missing or incomplete

- No caregiver shift-start/depot-departure time or shift-end field exists.
- No laboratory node coordinates or laboratory closing-time field is loaded.
- No pairwise distance d_ij or travel-time t_ij matrix exists. The patient travel_time field is only depot-to-patient Euclidean distance and has no travel-speed conversion.
- No rejected-caregiver set or interpersonal-restriction history is loaded. preferred_caregiver is blank and is not equivalent to rejection data.
- The current clinical file supplies no multiple-window list; the loader creates one benchmark window per patient.
- A vehicle-capacity quantity and patient demand quantity exist, but the optimizer currently does not use vehicle_capacity for route capacity. The units/compatibility of benchmark demand and vehicle capacity should be confirmed before treating them as a clinical load limit.
- The optimizer has service duration and max_work_hours, but route working time cannot be computed without leg travel times and departure/return times.
- The paper's exact variable-to-domain mapping for constraints (17)–(21) is not available in the current assignment records.

## 2. Proposed permutation representation and decoding

### Chromosome

Use a typed chromosome, so separators can never be mistaken for patient IDs. Preserve caregiver input order as the canonical segment order.

For K caregivers, the chromosome contains every patient ID exactly once and K−1 caregiver separator tokens. Each separator divides consecutive patient genes into the next caregiver's segment. The first segment belongs to caregivers[0], the second to caregivers[1], and so on. Empty segments are allowed.

Example for caregivers K1, K2, K3:

    [P4, P1, SEP, P2, SEP, P3]

decodes to K1: [P4, P1], K2: [P2], K3: [P3].

The permutation invariant is: each patient ID occurs exactly once, each separator occurs exactly K−1 times, and no other gene occurs. A validator should reject an unknown ID, duplicate patient gene, missing patient gene, or incorrect separator count. This encoding guarantees a unique route order and route membership for each patient gene.

The current optimizer has no such chromosome. Until an encoding and decoder are implemented, assignment_to_routes must use its documented fallback: earliest window start, stable by existing assignment order for ties; if no patient has window data, retain the assignment order. A chromosome-based order must not be claimed for the current candidates.

### Decoding feasibility

The segment determines the assigned caregiver; the patient order within the segment is the route order. Decoding should not reorder or silently move genes to another caregiver.

For each segment, the decoder can check available skill fields: caregiver care type and skill_level against patient care_type and required_skill_level. It can also accumulate patient demand against vehicle_capacity, once the benchmark units are confirmed. A failed check should remain visible in the decoded candidate or be represented as explicitly unassigned; silently moving that patient would make the route differ from the chromosome.

Time-window scheduling would process each route in chromosome order. With the required travel data available, compute each arrival from the previous location, then choose a feasible window and service start. The current loader has window bounds and service durations, but no pairwise travel time or caregiver start time. Therefore this scheduling rule cannot currently produce a verifiable arrival schedule. The paper's Eq. (12) constrains service start to [e_i^π, l_i^π]; whether service must also finish by l_i^π should be resolved before implementation.

Daily work/capacity would accumulate service and route travel against max_work_hours (and the specified overtime cap, if modeled). Vehicle load would separately accumulate demand against vehicle_capacity. These are different limits and must not be conflated.

### Unassigned patients

The base chromosome includes every patient in a caregiver segment, so it has no representation for an intentionally unassigned patient. Two compatible designs are available:

1. Add a typed UNASSIGNED separator/bucket to the chromosome, with all patient genes still appearing once; or
2. Keep the chromosome as specified and let the decoder report infeasible segment assignments as unassigned without relocating them.

Option 2 can change the effective assignment represented by the genes and may hide raw offspring infeasibility if treated as repair. For stress auditing, retain the raw decoded route and report violations before any feasibility filtering. Select an unassigned representation before implementation, especially because the current optimizer has an unassigned-patient objective.

## 3. Permutation-preserving operators

No crossover or mutation operators exist in the current optimizer. The following are proposals only.

| Operator | Permutation-preserving construction | Feasibility constraints it can affect after decoding |
|---|---|---|
| Patient swap | Swap two patient genes; separator count and patient multiset are unchanged. A swap across a separator also changes caregiver assignment. | Within one route: (9), (12), (13), (14). Across routes: additionally (10), (15), and (13). (10) cannot be checked until rejection data exists. |
| Patient insertion / relocation | Remove one patient gene and insert it at another patient slot, including a slot in another caregiver segment. No gene is added or lost. | (9), (12), (13), (14); across segments additionally (10), (15). |
| Separator shift | Move a caregiver separator across an adjacent patient gene; the patient moves to a neighboring caregiver segment while all genes remain present. | (10), (13), (15), and the route-dependent (9), (12), (14). |
| Order crossover (OX) | Project each parent to its patient-gene sequence, apply OX to those unique patient IDs, then place the child patient order into the patient slots of a selected parent template. Keep the template's separator positions unchanged. Produce the reciprocal child with the other parent's template. | (9), (12), (13), (14); changed segment membership can also affect (10), (15). |
| Partially mapped crossover (PMX) | Apply PMX to the patient-only permutation projection, then refill patient slots in a parent separator template. Do not apply ordinary PMX to mixed patient/separator tokens. | Same as OX. |

These operators preserve the chromosome's patient multiset and separator count, so a valid genotype cannot duplicate a patient. With a decoder that forms one linear path per segment, (5), (6), (7), (8), and (11) remain structurally satisfied. The operators can still make the decoded phenotype infeasible for time, load, skill, or interpersonal rules. Keep raw offspring available to the validator before any repair or feasibility filtering.

## 4. Constraint status under this encoding and current data

Statuses below refer to a structurally valid chromosome decoded into routes, without inventing missing data. SATISFIED is a structural guarantee or a condition the decoder can enforce from loaded fields. VIOLATED is possible when a raw chromosome assigns a patient incompatibly. UNVERIFIABLE means necessary data is absent.

| Check | Status | Basis |
|---|---|---|
| (5) Each patient visited at most once | SATISFIED | A valid chromosome contains each patient ID once; a decoder must not duplicate genes into routes. |
| (6) Flow conservation | SATISFIED | Each segment is a single ordered path; every routed patient has the preceding and following node implied by that sequence. |
| (7) Route starts at depot | SATISFIED | Decoder defines the depot as the implicit predecessor of the first patient in each segment. Depot coordinates exist in benchmark data, though not as part of the current optimizer instance record. |
| (8) Route ends at laboratory | SATISFIED structurally | Decoder defines laboratory as the implicit successor of the final patient. A lab node/arc is not loaded; this is structural interpretation, not evidence that a real return trip is feasible. |
| (9) Time propagation | UNVERIFIABLE | Service duration exists, but pairwise travel times and caregiver departure/service-start times do not. |
| (10) Rejected patient-caregiver pair | UNVERIFIABLE | No rejection set/history is loaded; blank preferred_caregiver does not provide it. |
| (11) Subtour elimination | SATISFIED | Each caregiver route is one linear segment of a duplicate-free permutation; no disconnected cycles can be represented. |
| (12) Service start in a requested window | UNVERIFIABLE | Window bounds are loaded, but no arrival, departure, or service-start schedule can be computed without pairwise travel and caregiver start time. |
| (13) Daily working time ≤ T | UNVERIFIABLE | Service time and max_work_hours exist; route travel, caregiver start, and return schedule are missing. Vehicle demand capacity is a separate quantity. |
| (14) Return before laboratory closing | UNVERIFIABLE | Neither laboratory closing time nor route return time is loaded. |
| (15) Skill compatibility | SATISFIED if the decoder admits a patient only when existing care-type/skill checks pass; otherwise VIOLATED for an incompatible raw segment | Both patient required_skill_level and caregiver skill_level/care type are loaded, so this check is verifiable. Operators that change segment membership can make it fail. |
| (16) Depot departure time | UNVERIFIABLE | No caregiver departure/start-time field is loaded. |
| (17) Binary/integer domains | UNVERIFIABLE for the paper variable | Patient IDs and typed separator tokens form a valid discrete chromosome, but this does not establish the domain of the paper's assignment variable without its variable mapping. |
| (18) Binary/integer domains | UNVERIFIABLE for the paper variable | Same limitation: chromosome validity is not proof of the corresponding mathematical variable domain. |
| (19) Binary/integer domains | UNVERIFIABLE for the paper variable | Same limitation. |
| (20) Binary/integer domains | UNVERIFIABLE for the paper variable | Same limitation. |
| (21) Binary/integer domains | UNVERIFIABLE for the paper variable | Same limitation. |

The 240-minute overtime cap is also UNVERIFIABLE from loaded caregiver records: max_work_hours is present, but there is no overtime-cap field or route duration. No change is proposed to the objective or search in this design.

## Decision before implementation

Confirm whether to add an explicit unassigned bucket to the chromosome, and confirm what travel-time source/speed and caregiver start time should be used. Without those choices, the permutation can generate ordered caregiver routes, but complete time-window, working-time, return, and overtime validation remains unavailable.
