import inspect

from optimisation.nsga2 import NSGA2


def test_zero_gamma_robust_result_matches_deterministic():
    patient = {
        "id": "P1",
        "care_type": "General",
        "required_skill_level": 1,
        "service_hours": 1,
        "travel_time": 10,
        "travel_time_min": 8,
        "travel_time_max": 12,
        "time_windows": [(0, 120)],
    }
    caregiver = {
        "id": "K1",
        "skill": "General",
        "skill_level": 1,
        "available": True,
        "delay": 0,
        "current_workload": 0,
        "max_work_hours": 8,
    }
    parameters = inspect.signature(NSGA2).parameters
    gamma_names = {"gamma_t", "gamma_s", "gamma_p"}
    assert gamma_names <= parameters.keys(), (
        "NSGA2 has no robustness-budget parameters, so a Gamma=0 robust run "
        "cannot be compared with the deterministic run"
    )

    deterministic = NSGA2([patient], [caregiver], seed=7).optimize()
    robust_zero = NSGA2(
        [patient],
        [caregiver],
        seed=7,
        gamma_t=0,
        gamma_s=0,
        gamma_p=0,
    ).optimize()

    assert robust_zero == deterministic
