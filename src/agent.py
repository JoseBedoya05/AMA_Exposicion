from __future__ import annotations

from dataclasses import dataclass

from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retriever import SemanticRetriever, format_context


@dataclass
class AgentAnswer:
    answer: str
    context: str
    retrieved_titles: list[str]


class LLMAgent:
    """Agente especializado en LLM que combina recuperación semántica y generación."""

    def __init__(
        self,
        client,
        llm_model: str,
        retriever: SemanticRetriever | None = None,
        temperature: float = 0.2,
    ):
        self.client = client
        self.llm_model = llm_model
        self.retriever = retriever
        self.temperature = temperature

    def answer(self, question: str, use_retrieval: bool = True) -> AgentAnswer:
        retrieved_results = []
        retrieved_context = ""

        if use_retrieval and self.retriever is not None:
            retrieved_results = self.retriever.search(question, top_k=3)
            retrieved_context = format_context(retrieved_results)

        user_prompt = build_user_prompt(question, retrieved_context)

        response = self.client.responses.create(
            model=self.llm_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )

        retrieved_titles = [chunk.title for chunk, _ in retrieved_results]
        return AgentAnswer(
            answer=response.output_text,
            context=retrieved_context,
            retrieved_titles=retrieved_titles,
        )
