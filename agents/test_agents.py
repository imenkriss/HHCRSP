import unittest

from agents.agentOrchestrateur import Orchestrator
from décision.decision import DecisionEngine
from optimisation.nsga2 import NSGA2


class AgentRulesTests(unittest.TestCase):
    def setUp(self):
        self.patient = {
            "id": "P1",
            "care_type": "Cardio",
            "priority": "High",
            "preferred_caregiver": "S1",
        }
        self.incompatible_caregiver = {
            "id": "S1",
            "skill": "Diabetes",
            "max_work_hours": 8,
            "current_workload": 1,
            "available": True,
            "delay": 0,
        }
        self.compatible_caregiver = {
            "id": "S2",
            "skill": "Cardio",
            "max_work_hours": 8,
            "current_workload": 2,
            "available": True,
            "delay": 0,
        }

    def test_orchestrator_excludes_incompatible_caregiver(self):
        orchestrator = Orchestrator(
            [self.patient],
            [self.incompatible_caregiver, self.compatible_caregiver],
        )

        candidates = orchestrator.get_candidate_caregivers("Cardio")

        self.assertEqual(["S2"], [candidate["id"] for candidate in candidates])

    def test_decision_does_not_keep_incompatible_initial_caregiver(self):
        decision = DecisionEngine().choose_best_caregiver(
            self.patient,
            self.incompatible_caregiver,
            [self.compatible_caregiver],
        )

        self.assertEqual("S2", decision["assigned_caregiver"])

    def test_nsga_ii_rejects_delayed_and_incompatible_caregivers(self):
        delayed = dict(self.compatible_caregiver, id="S3", delay=21)
        optimizer = NSGA2(
            [self.patient],
            [self.incompatible_caregiver, delayed],
        )

        self.assertFalse(optimizer.is_compatible(self.patient, self.incompatible_caregiver))
        self.assertFalse(optimizer.is_compatible(self.patient, delayed))

    def test_nsga_ii_separates_coverage_satisfaction_and_workload_objectives(self):
        skilled = dict(self.compatible_caregiver, skill_level=5)
        optimizer = NSGA2([self.patient], [skilled])

        objectives, summary = optimizer.evaluate(optimizer.create_solution())

        self.assertEqual(4, len(objectives))
        self.assertEqual(1, summary["served_patients"])
        self.assertEqual(60.0, summary["satisfaction_percent"])
        self.assertEqual(5.0, summary["average_assigned_skill_level"])

    def test_nsga_ii_accepts_any_sufficient_patient_time_window(self):
        patient = dict(
            self.patient,
            service_hours=1,
            time_windows=[(0, 30), (120, 240)],
        )
        optimizer = NSGA2([patient], [self.compatible_caregiver], seed=1)

        solution = optimizer.create_solution()

        self.assertEqual(1, len(solution))
        self.assertEqual((120.0, 240.0), solution[0]["time_window"])

    def test_nsga_ii_charges_allowed_overtime(self):
        patient = dict(self.patient, service_hours=2)
        caregiver = dict(
            self.compatible_caregiver,
            current_workload=7,
            max_work_hours=8,
            max_overtime_hours=2,
            overtime_cost_rate=50,
            skill_level=5,
        )
        optimizer = NSGA2([patient], [caregiver], seed=1)

        solution = optimizer.create_solution()
        _, summary = optimizer.evaluate(solution)

        self.assertEqual(1, len(solution))
        self.assertEqual(1.0, summary["expected_overtime_hours"])
        self.assertEqual(50.0, summary["overtime_cost"])

    def test_nsga_ii_uses_reproducible_uncertainty_scenarios(self):
        patient = dict(
            self.patient,
            service_hours=1,
            service_hours_min=0.5,
            service_hours_max=1.5,
            travel_time=2,
            travel_time_min=1,
            travel_time_max=3,
        )
        first = NSGA2([patient], [self.compatible_caregiver], seed=7)
        second = NSGA2([patient], [self.compatible_caregiver], seed=7)

        self.assertEqual(first.scenarios, second.scenarios)
        self.assertEqual(32, len(first.scenarios))
        _, summary = first.evaluate(first.create_solution())
        self.assertEqual(32, summary["uncertainty_scenarios"])

    def test_nsga_ii_respects_required_skill_level(self):
        patient = dict(self.patient, required_skill_level=4)
        caregiver = dict(self.compatible_caregiver, skill_level=3)
        optimizer = NSGA2([patient], [caregiver])

        self.assertFalse(optimizer.is_compatible(patient, caregiver))


if __name__ == "__main__":
    unittest.main()
