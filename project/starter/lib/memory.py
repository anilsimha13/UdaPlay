"""Short-term (per-session conversation) and long-term (persistent, cross-session) memory."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from .config import MEMORY_COLLECTION
from .messages import BaseMessage
from .vector_db import COLLECTION_METADATA


@dataclass
class Session:
    session_id: str
    messages: list[BaseMessage] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)  # AgentResult per turn


class ShortTermMemory:
    """Conversation state, kept in-process and keyed by session id."""

    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        return self.sessions[session_id]

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


class MemoryRecord(BaseModel):
    id: str
    question: str
    answer: str
    sources: list[str]
    learned_at: str
    distance: float | None = None


class LongTermMemory:
    """Facts learned from web searches, persisted in their own Chroma collection."""

    def __init__(self, client, embedding_fn, collection_name: str = MEMORY_COLLECTION):
        self.collection = client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn, metadata=COLLECTION_METADATA
        )

    def register(self, question: str, answer: str, sources: list[str]) -> str:
        memory_id = f"mem-{uuid.uuid4().hex[:10]}"
        self.collection.add(
            ids=[memory_id],
            documents=[f"Question: {question}\nAnswer: {answer}"],
            metadatas=[{
                "question": question,
                "answer": answer,
                "sources": "\n".join(sources),
                "learned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }],
        )
        return memory_id

    def search(self, query: str, n_results: int = 3, max_distance: float = 0.45) -> list[MemoryRecord]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(n_results, count))
        records = []
        for memory_id, meta, distance in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
            if distance <= max_distance:
                records.append(MemoryRecord(
                    id=memory_id,
                    question=meta["question"],
                    answer=meta["answer"],
                    sources=[s for s in meta.get("sources", "").split("\n") if s],
                    learned_at=meta["learned_at"],
                    distance=round(distance, 4),
                ))
        return records

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
