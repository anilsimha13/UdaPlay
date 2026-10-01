"""Thin wrapper around the OpenAI Chat Completions API."""

from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import chat_model
from .messages import AIMessage, BaseMessage
from .tooling import Tool

T = TypeVar("T", bound=BaseModel)

# Reasoning models only accept the default temperature.
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class LLM:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        tools: list[Tool] | None = None,
        client: OpenAI | None = None,
    ):
        self.model = model or chat_model()
        self.temperature = temperature
        self.tools = {t.name: t for t in (tools or [])}
        self.client = client or OpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL
        self.total_tokens = 0

    def _common_kwargs(self, messages: list[BaseMessage]) -> dict:
        kwargs = {"model": self.model, "messages": [m.to_api() for m in messages]}
        if not self.model.startswith(_NO_TEMPERATURE_PREFIXES):
            kwargs["temperature"] = self.temperature
        return kwargs

    def _track(self, response) -> None:
        if getattr(response, "usage", None):
            self.total_tokens += response.usage.total_tokens

    def invoke(self, messages: list[BaseMessage], force_tool: str | None = None) -> AIMessage:
        """Send the conversation (with tools, if any) and return the assistant reply.

        force_tool makes the model call that specific tool (OpenAI tool_choice).
        """
        kwargs = self._common_kwargs(messages)
        if self.tools:
            kwargs["tools"] = [t.to_api() for t in self.tools.values()]
            kwargs["parallel_tool_calls"] = False  # keep retrieve -> evaluate -> search ordering
            if force_tool:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        response = self.client.chat.completions.create(**kwargs)
        self._track(response)
        message = response.choices[0].message
        tool_calls = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (message.tool_calls or [])
        ]
        return AIMessage(content=message.content or "", tool_calls=tool_calls or None)

    def parse(self, messages: list[BaseMessage], response_format: type[T]) -> T:
        """Structured output: the reply is validated against a Pydantic model."""
        completions = self.client.chat.completions
        parse = getattr(completions, "parse", None) or self.client.beta.chat.completions.parse
        response = parse(**self._common_kwargs(messages), response_format=response_format)
        self._track(response)
        message = response.choices[0].message
        if message.parsed is None:
            raise ValueError(f"Model refused or returned no structured output: {message.refusal}")
        return message.parsed
