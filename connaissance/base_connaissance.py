from connaissance.regles import RULES
from connaissance.contraintes import CONSTRAINTS


class KnowledgeBase:
    def __init__(self):
        self.documents = RULES + CONSTRAINTS

    def get_all_documents(self):
        return self.documents