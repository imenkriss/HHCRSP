import unittest

from rag.rag import SimpleRAG
from rag.retriever import Retriever


class RagTests(unittest.TestCase):
    """Legacy tests updated to the documented R2 contract.

    The retriever returns R-series rules only: constraints C001-C004 are
    validation documents consumed by the agents, never retrieval candidates
    (docs/ASSUMPTIONS.md R1-R4; tests/test_rag_rules.py is authoritative).
    """

    def setUp(self):
        self.retriever = Retriever()
        self.rag = SimpleRAG()

    def test_retriever_matches_accents_and_case(self):
        results = self.retriever.retrieve("Retard du soignant", top_k=1)

        self.assertEqual("R002", results[0]["document"]["id"])

    def test_retriever_returns_deterministic_order(self):
        first = [result["document"]["id"] for result in self.retriever.retrieve("delay retard")]
        second = [result["document"]["id"] for result in self.retriever.retrieve("delay retard")]

        self.assertEqual(first, second)
        self.assertEqual(["R002"], first)

    def test_rag_returns_empty_context_without_match(self):
        self.assertEqual([], self.rag.process("weather forecast", top_k=3))

    def test_rag_limits_context(self):
        self.assertEqual(1, len(self.rag.process("patient priority cardio", top_k=1)))


if __name__ == "__main__":
    unittest.main()