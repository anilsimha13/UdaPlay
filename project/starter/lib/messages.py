"""Chat message types exchanged between the agent, the LLM, and tools."""

from typing import Any, Literal

from pydantic import BaseModel


class BaseMessage(BaseModel):
    role: str
    content: str | None = ""

    def to_api(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class AIMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[dict[str, Any]] | None = None

    def to_api(self) -> dict[str, Any]:
        message = super().to_api()
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str

    def to_api(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "tool_call_id": self.tool_call_id}
