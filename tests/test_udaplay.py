"""Offline tests: fake embeddings, a scripted LLM, and a fake Tavily client (no API keys needed)."""

import hashlib
import json
import re
import sys
from pathlib import Path

import chromadb
import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "starter"))

from lib.agent import AgentAnswer, Citation, GameFact, UdaPlayAgent  # noqa: E402
from lib.memory import LongTermMemory  # noqa: E402
from lib.messages import AIMessage  # noqa: E402
from lib.tooling import tool  # noqa: E402
from lib.tools import JudgeVerdict, create_udaplay_tools  # noqa: E402
from lib.vector_db import build_games_collection, load_games  # noqa: E402


class HashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic bag-of-words embedding so semantic search works offline."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = np.zeros(512)
            for token in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % 512] += 1
            vectors.append(vec / (np.linalg.norm(vec) or 1))
        return vectors

    @staticmethod
    def name() -> str:
        return "hash-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


class ScriptedLLM:
    """Stands in for lib.llm.LLM: returns queued replies instead of calling OpenAI."""

    def __init__(self, replies=None, parsed=None):
        self.tools = {}
        self.total_tokens = 0
        self.replies = list(replies or [])
        self.parsed = list(parsed or [])
        self.seen_messages = []
        self.forced = []

    def invoke(self, messages, force_tool=None):
        self.seen_messages.append(list(messages))
        self.forced.append(force_tool)
        return self.replies.pop(0)

    def parse(self, messages, response_format):
        item = self.parsed.pop(0)
        assert isinstance(item, response_format)
        return item


class FakeTavily:
    def __init__(self):
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append(query)
        return {
            "answer": "Rockstar Games is developing Grand Theft Auto VI.",
            "results": [{"title": "GTA VI news", "url": "https://example.com/gta6", "content": "GTA VI is in development."}],
        }


def call(name, **args):
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def tool_reply(name, **args):
    return AIMessage(content="", tool_calls=[call(name, **args)])


def answer(text, source="local_dataset", citations=None):
    return AgentAnswer(
        answer=text, confidence="high", answered_from=source, reasoning="test",
        citations=citations or [Citation(number=1, source_type="local_dataset", title="GTA SA", reference="games/002.json")],
        games=[GameFact(name="Grand Theft Auto: San Andreas", platform="PlayStation 2", release_year=2004,
                        publisher="Rockstar Games", genre="Action-adventure")],
    )


@pytest.fixture
def chroma():
    client = chromadb.EphemeralClient()
    yield client
    for c in client.list_collections():
        client.delete_collection(c.name)


@pytest.fixture
def games_collection(chroma):
    return build_games_collection(chroma, HashEmbedding(), load_games(), reset=True)


def test_dataset_has_core_and_extra_games():
    games = load_games()
    assert len(games) == 20
    names = {g["Name"] for g in games}
    assert {"Grand Theft Auto: San Andreas", "Pokémon Gold and Silver", "Super Mario 64"} <= names
    for g in games:
        assert set(g) == {"id", "Name", "Platform", "Genre", "Publisher", "Description", "YearOfRelease"}


def test_tool_schema_from_signature_and_docstring():
    @tool
    def lookup(query: str, limit: int = 3) -> list:
        """Look something up.

        Args:
            query: What to search for.
            limit: Max results.
        """

    spec = lookup.to_api()["function"]
    assert spec["name"] == "lookup"
    assert spec["description"] == "Look something up."
    assert spec["parameters"]["required"] == ["query"]
    assert spec["parameters"]["properties"]["query"] == {"type": "string", "description": "What to search for."}


def test_semantic_search_finds_game(games_collection):
    res = games_collection.query(query_texts=["Pokémon Gold and Silver Game Boy Color"], n_results=3)
    assert "004" in res["ids"][0]


def test_agent_answers_from_local_dataset(games_collection):
    judge = ScriptedLLM(parsed=[JudgeVerdict(useful=True, confidence="high", description="ok", missing_information="")])
    tavily = FakeTavily()
    tools = create_udaplay_tools(games_collection, judge, tavily)
    q = "Who published Grand Theft Auto: San Andreas?"
    llm = ScriptedLLM(
        replies=[tool_reply("retrieve_game", query=q), tool_reply("evaluate_retrieval", question=q), AIMessage(content="done")],
        parsed=[answer("Rockstar Games published it [1].")],
    )
    result = UdaPlayAgent(llm, tools).invoke(q)

    assert [c.tool for c in result.tool_calls] == ["retrieve_game", "evaluate_retrieval"]
    assert "Grand Theft Auto: San Andreas" in [d["title"] for d in result.tool_calls[0].result]
    assert result.tool_calls[1].result["should_search_web"] is False
    assert tavily.queries == []
    assert llm.forced == ["retrieve_game", "evaluate_retrieval", None]  # retrieve + evaluate are forced
    assert result.step_path == ["__entry__", "message_prep", "llm_processor", "tool_executor", "llm_processor",
                                "tool_executor", "llm_processor", "format_answer", "persist_memory", "__termination__"]


def test_web_fallback_is_saved_to_long_term_memory(chroma, games_collection):
    memory = LongTermMemory(chroma, HashEmbedding())
    q = "What is Rockstar Games working on right now?"
    judge = ScriptedLLM(parsed=[
        JudgeVerdict(useful=False, confidence="high", description="time-sensitive", missing_information="news"),
        JudgeVerdict(useful=True, confidence="high", description="memory answers it", missing_information=""),
    ])
    tavily = FakeTavily()
    tools = create_udaplay_tools(games_collection, judge, tavily, long_term_memory=memory)
    web_answer = answer("Rockstar is developing GTA VI [1].", source="web",
                        citations=[Citation(number=1, source_type="web", title="GTA VI news", reference="https://example.com/gta6")])
    llm = ScriptedLLM(
        replies=[tool_reply("retrieve_game", query=q), tool_reply("evaluate_retrieval", question=q),
                 tool_reply("game_web_search", question=q), AIMessage(content="done")],
        parsed=[web_answer],
    )
    result = UdaPlayAgent(llm, tools, long_term_memory=memory).invoke(q)

    assert tavily.queries == [q]
    assert result.memory_id and memory.count() == 1

    # A brand-new agent (new session) now finds the learned fact through internal retrieval.
    llm2 = ScriptedLLM(
        replies=[tool_reply("retrieve_game", query=q), tool_reply("evaluate_retrieval", question=q), AIMessage(content="done")],
        parsed=[answer("Rockstar is developing GTA VI [1].", source="long_term_memory")],
    )
    result2 = UdaPlayAgent(llm2, tools, long_term_memory=memory).invoke(q, session_id="new")
    sources = [d["source"] for d in result2.tool_calls[0].result]
    assert "long_term_memory" in sources
    assert len(tavily.queries) == 1  # no second web search


def test_web_search_blocked_until_internal_knowledge_checked(games_collection):
    tavily = FakeTavily()
    tools = create_udaplay_tools(games_collection, ScriptedLLM(), tavily)
    llm = ScriptedLLM(replies=[tool_reply("game_web_search", question="x"), AIMessage(content="done")],
                      parsed=[answer("n/a", source="none", citations=[])])
    result = UdaPlayAgent(llm, tools).invoke("x")
    assert "Call evaluate_retrieval before game_web_search" in result.tool_calls[0].error
    assert tavily.queries == []


def test_session_remembers_previous_turns(games_collection):
    tools = create_udaplay_tools(games_collection, ScriptedLLM(), FakeTavily())
    llm = ScriptedLLM(replies=[AIMessage(content="hi"), AIMessage(content="again")],
                      parsed=[answer("Hello! [1]"), answer("You said hello.")])
    agent = UdaPlayAgent(llm, tools)
    agent.invoke("Hello", session_id="s1")
    agent.invoke("What did I just say?", session_id="s1")
    second_call = llm.seen_messages[1]
    assert [m.role for m in second_call] == ["system", "user", "assistant", "user"]
    assert second_call[1].content == "Hello"
    assert len(agent.get_session_history("s1")) == 5
    assert agent.get_session_history("s2") == []


def test_iteration_cap_drops_unanswered_tool_call(games_collection):
    judge = ScriptedLLM(parsed=[JudgeVerdict(useful=True, confidence="high", description="ok", missing_information="")] * 3)
    tools = create_udaplay_tools(games_collection, judge, FakeTavily())
    llm = ScriptedLLM(replies=[tool_reply("retrieve_game", query="q")] * 3, parsed=[answer("x")])
    seen = []
    parse = llm.parse
    llm.parse = lambda messages, fmt: (seen.append(messages), parse(messages, fmt))[1]
    UdaPlayAgent(llm, tools, max_tool_iterations=2).invoke("q")
    # History handed to the final structured call ends with a tool result, then the format instructions.
    assert [m.role for m in seen[0][-2:]] == ["tool", "system"]
