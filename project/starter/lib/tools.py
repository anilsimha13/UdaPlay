"""UdaPlay's agent tools: internal retrieval, retrieval evaluation, and web search fallback."""

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import LLM
from .memory import LongTermMemory
from .messages import SystemMessage, UserMessage
from .tooling import Tool, tool

RETRIEVE_TOOL = "retrieve_game"
EVALUATE_TOOL = "evaluate_retrieval"
WEB_SEARCH_TOOL = "game_web_search"


class RetrievedDocument(BaseModel):
    source: Literal["local_dataset", "long_term_memory"]
    id: str
    title: str
    content: str
    similarity: float = Field(description="1 - cosine distance; higher is more similar")
    metadata: dict[str, Any] = {}


class JudgeVerdict(BaseModel):
    useful: bool = Field(description="True if the documents contain what is needed to fully answer the question")
    confidence: Literal["high", "medium", "low"]
    description: str = Field(description="Short explanation of why the documents are or are not sufficient")
    missing_information: str = Field(description="What is missing to answer the question; empty string if nothing")


class EvaluationReport(JudgeVerdict):
    best_similarity: float
    should_search_web: bool


JUDGE_PROMPT = """You are a strict evaluator for a video game question-answering system.
Given a user question and documents retrieved from an internal knowledge base, decide whether
the documents alone contain the information needed to answer the question correctly and completely.

Rules:
- Only judge what is literally in the documents; do not use your own knowledge.
- A document about a different game (even a related one) does not answer the question.
- The documents only need to answer what was asked; they do not need to be exhaustive.
- If the question is limited to "your knowledge base" / "your dataset", the retrieved documents ARE that
  knowledge base: judge whether they answer the question as scoped, not whether more games exist elsewhere.
  Example: "Compare the Halo games in your knowledge base" is fully answered by the retrieved Halo documents,
  even though the real-world series has more games - mark it useful.
- Questions about current events, news, upcoming or unreleased games, "right now", or anything
  time-sensitive are NOT answerable from the static game dataset (local_dataset documents).
- long_term_memory documents are facts previously learned from the web, with a learned_at date. They are
  useful for time-sensitive questions if they directly answer the question and were learned recently
  (within about 30 days of today).
- confidence reflects how sure you are about the usefulness verdict."""


def create_udaplay_tools(
    games_collection,
    judge_llm: LLM,
    tavily_client,
    long_term_memory: LongTermMemory | None = None,
    n_results: int = 3,
) -> list[Tool]:
    """Build the three agent tools, bound to the given collection, judge LLM, and Tavily client."""
    last_retrieval: list[RetrievedDocument] = []

    @tool
    def retrieve_game(query: str) -> list[dict]:
        """Semantic search over UdaPlay's internal knowledge: the local video game dataset plus
        facts previously learned from web searches (long-term memory). Always call this first.

        Args:
            query: A self-contained search query about a video game, e.g. "Who published Grand Theft Auto: San Andreas". Resolve pronouns using the conversation.
        """
        results = games_collection.query(query_texts=[query], n_results=n_results)
        docs = [
            RetrievedDocument(
                source="local_dataset",
                id=game_id,
                title=meta["Name"],
                content=document,
                similarity=round(1 - distance, 4),
                metadata=meta,
            )
            for game_id, document, meta, distance in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]
        if long_term_memory is not None:
            for record in long_term_memory.search(query):
                docs.append(RetrievedDocument(
                    source="long_term_memory",
                    id=record.id,
                    title=record.question,
                    content=f"{record.answer}\n(learned_at: {record.learned_at})",
                    similarity=round(1 - record.distance, 4),
                    metadata={"sources": record.sources, "learned_at": record.learned_at},
                ))
        last_retrieval[:] = docs
        return [d.model_dump() for d in docs]

    @tool
    def evaluate_retrieval(question: str, retrieved_docs: list[dict] | None = None) -> dict:
        """Judge whether the documents from retrieve_game are good enough to answer the question.
        Returns useful, confidence, a description, and should_search_web. Call this right after retrieve_game.

        Args:
            question: The user's question, rewritten to be self-contained.
            retrieved_docs: The documents returned by retrieve_game. Optional; if omitted, the most recent retrieve_game results are evaluated.
        """
        docs = [RetrievedDocument(**d) for d in retrieved_docs] if retrieved_docs else list(last_retrieval)
        best_similarity = max((d.similarity for d in docs), default=0.0)
        if not docs:
            return EvaluationReport(
                useful=False, confidence="high", description="No documents were retrieved.",
                missing_information="Everything", best_similarity=0.0, should_search_web=True,
            ).model_dump()

        formatted = "\n\n".join(
            f"[{i}] source={d.source} title={d.title!r} similarity={d.similarity}\n{d.content}"
            for i, d in enumerate(docs, 1)
        )
        verdict = judge_llm.parse(
            [
                SystemMessage(content=f"{JUDGE_PROMPT}\n\nToday's date is {date.today().isoformat()}."),
                UserMessage(content=f"Question: {question}\n\nRetrieved documents:\n{formatted}"),
            ],
            JudgeVerdict,
        )
        return EvaluationReport(
            **verdict.model_dump(),
            best_similarity=best_similarity,
            should_search_web=(not verdict.useful) or verdict.confidence == "low",
        ).model_dump()

    @tool
    def game_web_search(question: str) -> dict:
        """Search the web (Tavily) for video game information. Only use this after evaluate_retrieval
        reports should_search_web=true, i.e. internal knowledge is missing or insufficient.

        Args:
            question: A self-contained web search query about video games.
        """
        response = tavily_client.search(
            query=question, search_depth="advanced", max_results=5, include_answer=True
        )
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "answer": response.get("answer"),
            "results": [
                {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:1200]}
                for r in response.get("results", [])
            ],
        }

    return [retrieve_game, evaluate_retrieval, game_web_search]
