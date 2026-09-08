from typing import Optional

from ollama import chat
from pydantic import BaseModel


SYSTEM_PROMPT = """
You are an HHCOP reasoning assistant.

Use only the scenario and RAG knowledge provided.
Do not assign a caregiver.
Do not make the final scheduling decision.
Do not invent missing information.
Return only data matching the required JSON schema.
"""


USER_PROMPT = """
HHCOP scenario:
{query}

Knowledge retrieved by RAG:
{context}
"""


class ReasoningResult(BaseModel):
    patient_id: Optional[str]
    caregiver_id: Optional[str]
    urgency: bool
    delayed: bool
    relevant_rules: list[str]
    constraints: list[str]


class LLMRaisonner:
    def __init__(self, model: str = "qwen3:4b"):
        self.model = model

    def analyze(self, query: str, retrieved_docs: list[str]) -> dict:
        context = "\n\n---\n\n".join(retrieved_docs)

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        query=query,
                        context=context,
                    ),
                },
            ],
            format=ReasoningResult.model_json_schema(),
            think=False,
            keep_alive="10m",
            options={
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 512,
            },
        )

        result = ReasoningResult.model_validate_json(response.message.content)

        return {
            "query": query,
            **result.model_dump(),
            "rag_context": retrieved_docs,
        }