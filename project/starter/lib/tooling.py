"""A small @tool decorator that turns a typed Python function into an OpenAI function tool."""

import inspect
import json
import re
from typing import Any, Callable

from pydantic import BaseModel, Field, create_model


def _param_descriptions(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style ``Args:`` section."""
    descriptions: dict[str, str] = {}
    in_args = False
    current = None
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue
        if in_args:
            if not stripped or re.match(r"^[A-Z][a-z]+:$", stripped):
                in_args = False
                continue
            match = re.match(r"^(\w+)\s*(\(.*?\))?:\s*(.*)$", stripped)
            if match:
                current = match.group(1)
                descriptions[current] = match.group(3)
            elif current:
                descriptions[current] += " " + stripped
    return descriptions


def _summary(docstring: str) -> str:
    """Docstring text before the Args/Returns sections."""
    lines = []
    for line in docstring.strip().splitlines():
        if line.strip() in ("Args:", "Arguments:", "Parameters:", "Returns:"):
            break
        lines.append(line.strip())
    return " ".join(line for line in lines if line)


def _remove_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _remove_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_remove_titles(v) for v in schema]
    return schema


class Tool:
    """Wraps a function and exposes the JSON schema the OpenAI API expects."""

    def __init__(self, func: Callable, name: str | None = None, description: str | None = None):
        self.func = func
        self.name = name or func.__name__
        docstring = inspect.getdoc(func) or ""
        self.description = description or _summary(docstring)
        self.parameters = self._build_schema(func, _param_descriptions(docstring))

    @staticmethod
    def _build_schema(func: Callable, descriptions: dict[str, str]) -> dict[str, Any]:
        fields = {}
        for name, param in inspect.signature(func).parameters.items():
            annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[name] = (annotation, Field(default, description=descriptions.get(name)))
        model: type[BaseModel] = create_model(f"{func.__name__}_args", **fields)
        return _remove_titles(model.model_json_schema())

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)

    def run_json(self, arguments: str) -> str:
        """Run with JSON-encoded arguments (as sent by the LLM) and return a JSON string."""
        result = self(**json.loads(arguments or "{}"))
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def tool(func: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator: ``@tool`` or ``@tool(name=..., description=...)``."""

    def wrap(f: Callable) -> Tool:
        return Tool(f, name=name, description=description)

    return wrap(func) if func is not None else wrap
