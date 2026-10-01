"""A minimal, inspectable state machine used to drive the agent workflow.

Each Step reads the current state and returns a dict of updates. Transitions decide
which step runs next. Every step execution is recorded as a Snapshot inside a Run,
so the full workflow can be inspected (and visualized) afterwards.
"""

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

State = dict[str, Any]


@dataclass
class Step:
    step_id: str
    logic: Callable[[State], dict[str, Any] | None]

    def run(self, state: State) -> State:
        updates = self.logic(state) or {}
        return {**state, **updates}


class EntryPoint(Step):
    def __init__(self, step_id: str = "__entry__"):
        super().__init__(step_id, lambda state: {})


class Termination(Step):
    def __init__(self, step_id: str = "__termination__"):
        super().__init__(step_id, lambda state: {})


@dataclass
class Transition:
    source: str
    targets: list[str]
    condition: Callable[[State], str] | None = None

    def resolve(self, state: State) -> str:
        if self.condition is None:
            return self.targets[0]
        target = self.condition(state)
        if target not in self.targets:
            raise ValueError(f"Transition from {self.source!r} returned unknown target {target!r}")
        return target


@dataclass
class Snapshot:
    step_id: str
    state: State
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Run:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    snapshots: list[Snapshot] = field(default_factory=list)

    @property
    def path(self) -> list[str]:
        return [s.step_id for s in self.snapshots]

    def final_state(self) -> State:
        return self.snapshots[-1].state if self.snapshots else {}


class StateMachine:
    def __init__(self, max_steps: int = 50):
        self.steps: dict[str, Step] = {}
        self.transitions: dict[str, Transition] = {}
        self.max_steps = max_steps

    def add_steps(self, steps: list[Step]) -> None:
        for step in steps:
            self.steps[step.step_id] = step

    def connect(self, source: str, targets: str | list[str], condition: Callable[[State], str] | None = None) -> None:
        targets = [targets] if isinstance(targets, str) else targets
        for step_id in [source, *targets]:
            if step_id not in self.steps:
                raise KeyError(f"Unknown step {step_id!r}; add it with add_steps() first")
        self.transitions[source] = Transition(source, targets, condition)

    def run(self, state: State) -> Run:
        entry = next((s for s in self.steps.values() if isinstance(s, EntryPoint)), None)
        if entry is None:
            raise ValueError("State machine needs an EntryPoint step")

        run = Run()
        step: Step = entry
        for _ in range(self.max_steps):
            state = step.run(state)
            # Shallow-copy lists so later steps don't mutate earlier snapshots.
            run.snapshots.append(Snapshot(step.step_id, {k: copy.copy(v) for k, v in state.items()}))
            if isinstance(step, Termination):
                break
            transition = self.transitions.get(step.step_id)
            if transition is None:
                raise ValueError(f"No transition defined out of step {step.step_id!r}")
            step = self.steps[transition.resolve(state)]
        else:
            raise RuntimeError(f"State machine exceeded {self.max_steps} steps")
        run.ended_at = datetime.now(timezone.utc)
        return run
