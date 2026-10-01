"""The UdaPlay agent: a stateful, tool-using research agent driven by a state machine.

Workflow (one run per user query):

    __entry__ -> message_prep -> llm_processor --(tool calls)--> tool_executor -> llm_processor ...
                                       |
                                       +--(no tool calls)--> format_answer -> persist_memory -> __termination__
"""

import json
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import LLM
from .memory import LongTermMemory, ShortTermMemory
from .messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, UserMessage
from .state_machine import EntryPoint, Run, StateMachine, Step, Termination
from .tooling import Tool
from .tools import EVALUATE_TOOL, RETRIEVE_TOOL, WEB_SEARCH_TOOL

# --------------------------------------------------------------------------------------
# Structured output
# --------------------------------------------------------------------------------------


class Citation(BaseModel):
    number: int = Field(description="The [n] marker used in the answer text")
    source_type: Literal["local_dataset", "long_term_memory", "web"]
    title: str
    reference: str = Field(description="games/<id>.json for the local dataset, the memory id, or the URL")


class GameFact(BaseModel):
    name: str
    platform: str | None
    release_year: int | None
    publisher: str | None
    genre: str | None


class AgentAnswer(BaseModel):
    answer: str = Field(description="Clear natural-language answer with inline [n] citation markers")
    confidence: Literal["high", "medium", "low"]
    answered_from: Literal["local_dataset", "long_term_memory", "web", "mixed", "conversation", "none"]
    reasoning: str = Field(description="One or two sentences on how the answer was found and why the sources are trusted")
    citations: list[Citation]
    games: list[GameFact] = Field(description="Structured facts for each game mentioned in the answer")


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None


class AgentResult(BaseModel):
    session_id: str
    query: str
    answer: AgentAnswer
    tool_calls: list[ToolCallRecord]
    thoughts: list[str] = Field(description="Any text the model produced alongside its tool calls")
    step_path: list[str]
    memory_id: str | None = None
    total_tokens: int = 0
    run: Any = Field(default=None, exclude=True, repr=False)


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------

DEFAULT_INSTRUCTIONS = f"""You are UdaPlay, an AI research assistant that answers questions about video games:
titles, release dates, platforms, genres, publishers, descriptions, and industry news.

Every turn follows this workflow:
1. Call `{RETRIEVE_TOOL}` with a self-contained query. Resolve references such as "it", "their" or
   "that game" using the conversation history.
2. Call `{EVALUATE_TOOL}` with the user's question, rewritten only to resolve references. Keep scope words
   such as "in your knowledge base" or "in your dataset" exactly as the user wrote them.
3. If the evaluation says should_search_web is true, call `{WEB_SEARCH_TOOL}`. Otherwise, do NOT search the web.
   Write web queries without adding a year unless the user asked about one.
4. Answer using only the information returned by the tools (or earlier turns of this conversation).
   Never invent facts. If the sources disagree or are incomplete, say so.

For greetings or questions about the conversation itself (e.g. "what did I ask before?"), answer from the
conversation after the retrieval step. Keep answers concise and precise."""

FORMAT_INSTRUCTIONS = """Now write the final answer for the user's latest question as structured output.
- Use only facts from the tool results in this turn or from earlier turns of the conversation.
- Combine all relevant facts from all tool results (including the web search summary and every web result),
  not just one source. For time-sensitive questions, prefer the most recent information and mention dates.
- Put inline citation markers like [1], [2] in the answer text, numbered in the order of the citations list.
  Cite every source you used; never cite a document that was not returned by a tool.
- Citations: for local dataset games use reference "games/<id>.json"; for long-term memory use the memory id
  (and list the original URL as a separate web citation if available); for web results use the URL.
- answered_from: where the facts came from ("mixed" if both internal knowledge and the web were used,
  "conversation" if from earlier turns, "none" if you could not find an answer).
- confidence: high if a trusted source states the fact directly, medium if inferred or partially supported,
  low if the sources were weak, conflicting, or missing."""


class UdaPlayAgent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        long_term_memory: LongTermMemory | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        max_tool_iterations: int = 6,
        force_retrieval_first: bool = True,
    ):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.llm.tools = dict(self.tools)
        self.long_term_memory = long_term_memory
        self.instructions = f"{instructions}\n\nToday's date is {date.today().isoformat()}."
        self.force_retrieval_first = force_retrieval_first and {RETRIEVE_TOOL, EVALUATE_TOOL} <= self.tools.keys()
        self.max_tool_iterations = max_tool_iterations
        self.memory = ShortTermMemory()
        # Tool -> tool that must already have run in this turn (enforces internal-first retrieval).
        self.prerequisites = {EVALUATE_TOOL: RETRIEVE_TOOL, WEB_SEARCH_TOOL: EVALUATE_TOOL}
        self.workflow = self._build_workflow()

    # ---------------------------------------------------------------- workflow steps

    def _message_prep(self, state: dict) -> dict:
        history = state["history"]
        messages = list(history) if history else [SystemMessage(content=self.instructions)]
        messages.append(UserMessage(content=state["user_query"]))
        return {"messages": messages}

    def _llm_processor(self, state: dict) -> dict:
        # Each turn must search internal knowledge and then evaluate it before anything else happens.
        ran = {r.tool for r in state["tool_calls"] if r.error is None}
        force = None
        if self.force_retrieval_first:
            force = RETRIEVE_TOOL if RETRIEVE_TOOL not in ran else EVALUATE_TOOL if EVALUATE_TOOL not in ran else None
        response = self.llm.invoke(state["messages"], force_tool=force)
        thoughts = state["thoughts"] + ([response.content] if response.content and response.tool_calls else [])
        return {"messages": state["messages"] + [response], "pending_tool_calls": response.tool_calls or [], "thoughts": thoughts}

    def _tool_executor(self, state: dict) -> dict:
        messages = list(state["messages"])
        records = list(state["tool_calls"])
        for call in state["pending_tool_calls"]:
            name = call["function"]["name"]
            record = ToolCallRecord(tool=name, arguments={})
            try:
                record.arguments = json.loads(call["function"]["arguments"] or "{}")
                already_ran = {r.tool for r in records if r.error is None}
                required = self.prerequisites.get(name)
                if name not in self.tools:
                    raise ValueError(f"Unknown tool {name!r}")
                if required and required not in already_ran:
                    raise ValueError(f"Call {required} before {name}: internal knowledge must be checked and evaluated first.")
                content = self.tools[name].run_json(call["function"]["arguments"])
                record.result = json.loads(content)
            except Exception as exc:  # report tool errors back to the model instead of crashing
                record.error = f"{type(exc).__name__}: {exc}"
                content = json.dumps({"error": record.error})
            records.append(record)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
        return {"messages": messages, "tool_calls": records, "pending_tool_calls": [], "iterations": state["iterations"] + 1}

    def _format_answer(self, state: dict) -> dict:
        messages = list(state["messages"])
        if state["pending_tool_calls"]:
            # Iteration cap hit: drop the unanswered tool-call message so the API accepts the history.
            messages.pop()
        answer = self.llm.parse(messages + [SystemMessage(content=FORMAT_INSTRUCTIONS)], AgentAnswer)
        return {"answer": answer}

    def _persist_memory(self, state: dict) -> dict:
        """Long-term memory: remember what was learned from the web so future sessions can reuse it."""
        answer: AgentAnswer = state["answer"]
        searches = [r for r in state["tool_calls"] if r.tool == WEB_SEARCH_TOOL and r.error is None]
        if self.long_term_memory is None or not searches or answer.answered_from not in ("web", "mixed"):
            return {"memory_id": None}
        urls = [c.reference for c in answer.citations if c.source_type == "web"]
        memory_id = self.long_term_memory.register(
            question=searches[-1].arguments.get("question", state["user_query"]),
            answer=answer.answer,
            sources=urls,
        )
        return {"memory_id": memory_id}

    def _route_after_llm(self, state: dict) -> str:
        if state["pending_tool_calls"] and state["iterations"] < self.max_tool_iterations:
            return "tool_executor"
        return "format_answer"

    def _build_workflow(self) -> StateMachine:
        machine = StateMachine()
        entry, termination = EntryPoint(), Termination()
        machine.add_steps([
            entry,
            Step("message_prep", self._message_prep),
            Step("llm_processor", self._llm_processor),
            Step("tool_executor", self._tool_executor),
            Step("format_answer", self._format_answer),
            Step("persist_memory", self._persist_memory),
            termination,
        ])
        machine.connect(entry.step_id, "message_prep")
        machine.connect("message_prep", "llm_processor")
        machine.connect("llm_processor", ["tool_executor", "format_answer"], self._route_after_llm)
        machine.connect("tool_executor", "llm_processor")
        machine.connect("format_answer", "persist_memory")
        machine.connect("persist_memory", termination.step_id)
        return machine

    # ---------------------------------------------------------------- public API

    def invoke(self, query: str, session_id: str = "default") -> AgentResult:
        session = self.memory.get_session(session_id)
        tokens_before = self.llm.total_tokens
        run: Run = self.workflow.run({
            "user_query": query,
            "history": list(session.messages),
            "messages": [],
            "pending_tool_calls": [],
            "tool_calls": [],
            "thoughts": [],
            "iterations": 0,
            "answer": None,
            "memory_id": None,
        })
        final = run.final_state()
        answer: AgentAnswer = final["answer"]

        # Short-term memory keeps the compact conversation (question + cited answer) for follow-ups.
        if not session.messages:
            session.messages.append(SystemMessage(content=self.instructions))
        sources = "; ".join(f"[{c.number}] {c.title} ({c.reference})" for c in answer.citations)
        session.messages += [
            UserMessage(content=query),
            AIMessage(content=answer.answer + (f"\nSources: {sources}" if sources else "")),
        ]

        result = AgentResult(
            session_id=session_id,
            query=query,
            answer=answer,
            tool_calls=final["tool_calls"],
            thoughts=final["thoughts"],
            step_path=run.path,
            memory_id=final["memory_id"],
            total_tokens=self.llm.total_tokens - tokens_before,
            run=run,
        )
        session.results.append(result)
        return result

    def get_session_history(self, session_id: str = "default") -> list[BaseMessage]:
        return list(self.memory.get_session(session_id).messages)

    def reset_session(self, session_id: str = "default") -> None:
        self.memory.reset(session_id)
