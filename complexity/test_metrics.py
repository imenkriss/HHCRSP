import unittest

from complexity.metrics import ComplexityMetrics


class ComplexityMetricsTests(unittest.TestCase):
    def test_uncertainty_reports_missing_evidence(self):
        metrics = ComplexityMetrics()

        metrics.uncertainty_indicators(
            {"patient_id": None, "urgency": False, "delayed": False},
            [],
        )

        uncertainty = metrics.get_metrics()["uncertainty"]
        self.assertEqual("low_evidence", uncertainty["status"])
        self.assertEqual(1, uncertainty["uncertainty_score"])
        self.assertEqual(1, uncertainty["missing_signals"])

    def test_uncertainty_uses_available_rag_evidence(self):
        metrics = ComplexityMetrics()

        metrics.uncertainty_indicators(
            {"patient_id": "P1", "urgency": True, "delayed": False},
            ["rule 1", "constraint 1", "constraint 2"],
        )

        uncertainty = metrics.get_metrics()["uncertainty"]
        self.assertEqual(1.0, uncertainty["confidence_score"])
        self.assertEqual(0.0, uncertainty["uncertainty_score"])


if __name__ == "__main__":
    unittest.main()