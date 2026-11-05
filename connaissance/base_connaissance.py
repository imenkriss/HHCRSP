from __future__ import annotations

from connaissance.contraintes import CONSTRAINTS
from connaissance.regles import RULES


class KnowledgeBase:
    def __init__(self):
        self.rules = list(RULES)
        self.constraints = list(CONSTRAINTS)
        self.documents = self.rules + self.constraints

    def get_rules(self):
        return list(self.rules)

    def get_constraints(self):
        return list(self.constraints)

    def get_all_documents(self):
        return list(self.documents)

    def get_document(self, document_id: str):
        for document in self.documents:
            if document["id"] == document_id:
                return dict(document)
        raise KeyError(document_id)