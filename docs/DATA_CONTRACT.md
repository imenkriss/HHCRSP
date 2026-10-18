# Benchmark and MTWMO Data Contract

## Fields supplied by Solomon / Gehring-Homberger files

The loader reads the first customer row as the depot and uses the remaining customer rows as patients. The standard seven numeric columns provide customer ID, x/y coordinates, benchmark capacity demand, ready time, due date, and service duration. The existing parser also reads vehicle count and vehicle capacity from the vehicle header.

- Depot: first customer row ID and coordinates (benchmark files use ID 0).
- Patient: ID, x/y coordinates, benchmark demand, ready time, due date, service duration.
- One patient window: the raw [ready_time, due_date] pair.
- Caregiver count and capacity: benchmark vehicle header values, via data.dataset_loader._parse_instance.

The domain adapter maps ready/due to one window and retains vehicle capacity as benchmark metadata and on each caregiver. Patient demand level 1 and caregiver skill level 1 are PLACEHOLDER values; Phase 1 generator must replace them with generated instance values. Laboratory and travel speed remain unset until supplied by an enrichment step.

Patient time windows are SOFT. An early arrival records waiting minutes as max(0, earliest - arrival); a late arrival records tardiness minutes as max(0, arrival - latest). These are separate cost components and are not hard constraint failures. The domain helper returns both values separately for later penalty calculation.

Caregiver rated_minutes=480 is the rated-time threshold. max_overtime_minutes=240 is the extra time permitted. The resulting hard daily work limit is 480 + 240 = 720 minutes.

## Fields that must be generated or supplied by an enrichment step

- Multiple time windows and their count/distribution (the benchmark gives one window).
- Clinical skill demand levels (1, 2, 3) and caregiver skill levels (1, 2, 3). Current level-1 fields are PLACEHOLDER values to be replaced by the Phase 1 generator.
- Patient-caregiver preference scores.
- Caregiver start times when different from the default 0.
- Travel speed or a calibrated conversion from benchmark distance to travel time.
- A laboratory node and its closing time/horizon.
- Clinical service-time adjustments beyond benchmark service duration.

## Paper parameters or semantics requiring a project decision

The paper benchmark input does not determine how benchmark demand maps to the model three-level clinical demand. A mapping policy is needed. It also does not specify preference scores for synthetic benchmark patients, the travel-speed conversion, or a laboratory location and closing time when adapting vehicle routing data to the home-care model. The benchmark supplies vehicle capacity, but the clinical interpretation and units of patient demand versus that capacity must be stated before enforcing capacity.

Robustness budgets, travel-time deviations, caregiver departure deviations, and demand perturbations require their own instance-generation policy; benchmark rows alone do not provide them. The adapter deliberately avoids inventing those values.


## Phase 1 generator provenance and decisions

The generator follows Section 5.1 and Table 1 of docs/specs/s10479-026-07090-4.pdf. Section 5.1 states: “Six datasets are generated as shown in Table 1.” The same section describes locations as generated “in a 50×50, 100×100, or 200×200 area” based on location distribution. Dataset scale tuples below are transcribed from the requested Table 1 specification; the implementation uses these literal values and does not infer counts from a ratio.

| Dataset | Patients | Caregiver counts | Location codes | Table 1 reference |
|---|---:|---|---|---|
| A | 10 | 2, 3, 4 | 1, 2 | Section 5.1, Table 1 |
| B | 30 | 5, 8 | 1, 2 | Section 5.1, Table 1 |
| C | 50 | 10, 12 | 1, 2 | Section 5.1, Table 1 |
| D | 75 | 15, 18 | 1, 2 | Section 5.1, Table 1 |
| E | 100 | 20, 25 | 1, 2 | Section 5.1, Table 1 |
| F | 200 | 45, 58 | 2, 3 | Section 5.1, Table 1 |

Location code 1 maps to 50×50, code 2 to 100×100, and code 3 to 200×200 (Section 5.1, location-generation procedure). Names use <dataset>_<caregivers>_<location>, e.g. A_2_1.

| Generated parameter | Value or procedure | Provenance |
|---|---|---|
| Patient / facility coordinates | Independent uniform coordinates in the location square | Section 5.1: locations are randomly generated within the selected area |
| Distance | Euclidean | Section 5.1, distance definition |
| Travel speed | 1 coordinate unit/minute, so time = distance | Project decision; the paper does not specify this conversion |
| Horizon end l0 | 720 minutes | Section 5.1, generation procedure |
| Caregiver start | 0 | Section 5.1, departure-time nominal setting |
| Service time θ_i | Uniform [30, 90] minutes | Section 5.1, service-time generation |
| Demand level p_i | {1, 2, 3} with probabilities {0.50, 0.35, 0.15} | Section 5.1, demand generation |
| Caregiver skill g_k | Uniform over {1, 2, 3} | Project decision, filling the generated categorical field |
| Window count | Per-patient draw: 1/2/3 with probabilities 0.50/0.30/0.20 | Section 5.1, multiple-window generation; instance shares are approximate |
| Window centers | The one-, two-, and three-window intervals use the supplied Section 5.1 center formulas with t0_i = depot-to-patient travel time | Section 5.1, time-window generation |
| Window width and clipping | Width uniform [60, 120]; centered interval clipped to [0, l0] | Project assumption; paper does not state this center-to-interval mapping. Clipping can reduce realized width |
| Preference q_ik | Uniform integer {0, 1, 2, 3} | Section 5.1, preference generation |
| Rejection z_ik | Bernoulli(0.05) | PLACEHOLDER pending paper-specific clarification |
| P | ceil(patients / caregivers) + 2 | Project decision |

The Phase 1 generator adds the following optional fields to the existing domain objects; existing field defaults remain unchanged:

- Patient.rejections: caregiver ID to rejection boolean.
- Patient.sampled_window_widths and Patient.clipped_window_widths: preserve generated widths and disclose clipping effects.
- Instance.max_patients_per_caregiver: P.
- Instance.distances and Instance.travel_times: symmetric node-pair lookup maps, including depot and laboratory.
- Instance.seed: explicit reproducibility input.

For the Section 5.1 center intervals (with l0=720): for three windows, the first interval [t0, 240] reverses when t0 > 240; the middle interval [240, 480] never reverses; and the last interval [480, 720 - t0 - θ] reverses when t0 + θ > 240. For two windows, the first interval [t0, 360] reverses when t0 > 360; the second interval [360, 720 - t0 - θ] reverses when t0 + θ > 360. These reversal conditions are described from the Section 5.1 center formulas. They only occur at large depot-to-patient distances relative to the horizon and service time. In the 200×200 location, the maximum depot-to-patient distance is about 283 units, so t0 > 240 is reachable and the first three-window interval can trigger a downgrade; t0 > 360 is not reachable. Before sampling any centers, the generator checks every interval for the requested count, then decreases the count from 3 to 2 to 1 until all intervals have lower <= upper. If even the one-window interval is empty, it raises ValueError with the instance name and patient ID. Patient.requested_window_count records the sampled count; Patient.used_window_count records the count actually generated.

The generator writes no output files. If future workflows persist generated instances, use data/generated/ and add that path to .gitignore before writing files.
