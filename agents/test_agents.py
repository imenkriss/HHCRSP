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


if __name__ == "__main__":
    unittest.main()
