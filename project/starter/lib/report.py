"""Pretty-printing of agent runs and matplotlib visualizations of the knowledge base / retrieval."""

import json
import textwrap
from collections import Counter

from .agent import AgentResult
from .tools import EVALUATE_TOOL, RETRIEVE_TOOL, WEB_SEARCH_TOOL

# Reference palette (validated: light mode, adjacent pairs pass CVD and normal-vision checks).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ROUTE_COLORS = {
    "Local dataset": "#2a78d6",          # categorical slot 1 (blue)
    "Web search fallback": "#eb6834",    # slot 2 (orange)
    "Long-term memory": "#1baf7a",       # slot 3 (aqua) - always paired with a legend + value labels
}
SINGLE_SERIES = "#2a78d6"


# --------------------------------------------------------------------------------------
# Text report
# --------------------------------------------------------------------------------------


def _summarize_tool_result(tool: str, result) -> str:
    if result is None:
        return ""
    if tool == RETRIEVE_TOOL:
        return "; ".join(f"{d['title']} ({d['source']}, sim={d['similarity']:.2f})" for d in result)
    if tool == EVALUATE_TOOL:
        return (f"useful={result['useful']}, confidence={result['confidence']}, "
                f"should_search_web={result['should_search_web']} - {result['description']}")
    if tool == WEB_SEARCH_TOOL:
        urls = ", ".join(r["url"] for r in result.get("results", [])[:3])
        return f"{len(result.get('results', []))} results ({urls})"
    return json.dumps(result)[:200]


def format_result(result: AgentResult, show_json: bool = True) -> str:
    a = result.answer
    wrap = lambda s, indent="    ": textwrap.fill(s, 100, initial_indent=indent, subsequent_indent=indent)
    lines = [f"QUESTION: {result.query}", "", f"WORKFLOW: {' -> '.join(result.step_path)}", "", "TOOL USAGE:"]
    if not result.tool_calls:
        lines.append("    (no tools needed)")
    for i, call in enumerate(result.tool_calls, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items() if k != "retrieved_docs")
        lines.append(f"  {i}. {call.tool}({args})")
        lines.append(wrap(f"-> ERROR: {call.error}" if call.error else f"-> {_summarize_tool_result(call.tool, call.result)}", "       "))
    for thought in result.thoughts:
        lines.append(wrap(f"model note: {thought}"))
    lines += ["", "REASONING:", wrap(a.reasoning), "",
              f"ANSWER  (confidence: {a.confidence}, source: {a.answered_from}):", wrap(a.answer), "", "CITATIONS:"]
    lines += [f"    [{c.number}] {c.title} - {c.source_type}: {c.reference}" for c in a.citations] or ["    (none)"]
    if result.memory_id:
        lines.append(f"\nLONG-TERM MEMORY: saved as {result.memory_id}")
    if show_json:
        lines += ["", "STRUCTURED JSON:", a.model_dump_json(indent=2)]
    return "\n".join(lines)


def print_result(result: AgentResult, show_json: bool = True) -> None:
    print(format_result(result, show_json))


def route_of(result: AgentResult) -> str:
    if any(c.tool == WEB_SEARCH_TOOL and c.error is None for c in result.tool_calls):
        return "Web search fallback"
    if result.answer.answered_from == "long_term_memory":
        return "Long-term memory"
    return "Local dataset"


def best_local_similarity(result: AgentResult) -> float | None:
    sims = [d["similarity"] for c in result.tool_calls if c.tool == RETRIEVE_TOOL and c.result
            for d in c.result if d["source"] == "local_dataset"]
    return max(sims) if sims else None


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------


def _style(ax, title: str, subtitle: str | None = None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_SECONDARY, fontsize=10)


def plot_games_per_platform(games: list[dict]):
    import matplotlib.pyplot as plt

    counts = Counter(g["Platform"] for g in games).most_common()[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(counts) + 1.2))
    labels, values = zip(*counts)
    ax.barh(labels, values, height=0.55, color=SINGLE_SERIES)
    for y, v in enumerate(values):
        ax.text(v + 0.05, y, str(v), va="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(range(0, max(values) + 2))
    _style(ax, "Games per platform", f"{len(games)} games in the UdaPlay knowledge base")
    fig.tight_layout()
    return fig


def plot_release_timeline(games: list[dict]):
    import matplotlib.pyplot as plt

    ordered = sorted(games, key=lambda g: (g["YearOfRelease"], g["Name"]))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(ordered) + 1.2))
    years = [g["YearOfRelease"] for g in ordered]
    ax.scatter(years, range(len(ordered)), s=55, color=SINGLE_SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.set_yticks(range(len(ordered)), [f"{g['Name']}  ({g['Platform']})" for g in ordered], fontsize=8.5)
    ax.invert_yaxis()
    _style(ax, "Release timeline", f"{min(years)}-{max(years)}")
    fig.tight_layout()
    return fig


def plot_embedding_map(collection):
    """2-D PCA projection of the game embeddings - similar games land close together."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = collection.get(include=["embeddings", "metadatas"])
    vectors = np.asarray(data["embeddings"], dtype=float)
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(points[:, 0], points[:, 1], s=60, color=SINGLE_SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    for (x, y), meta in zip(points, data["metadatas"]):
        ax.annotate(meta["Name"], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)
    _style(ax, "Knowledge base embedding map", "PCA of game embeddings; nearby points are semantically similar")
    ax.grid(axis="both", color=GRID, linewidth=1)
    ax.set_xticklabels([]), ax.set_yticklabels([])
    ax.margins(x=0.18, y=0.08)
    fig.tight_layout()
    return fig


def plot_retrieval(results: list[AgentResult]):
    """Best local-dataset similarity per query, colored by how the question was finally answered."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Number rows like the summary table; numeric y positions keep repeated questions on separate rows.
    numbered = [(n, r, best_local_similarity(r)) for n, r in enumerate(results, 1)]
    plotted = [(n, r, s) for n, r, s in numbered if s is not None][::-1]
    rows = [(r, s) for _, r, s in plotted]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.9))
    ys = list(range(len(rows)))
    ax.barh(ys, [s for _, s in rows], height=0.5, color=[ROUTE_COLORS[route_of(r)] for r, _ in rows])
    ax.set_yticks(ys, [f"{n}. {textwrap.shorten(r.query, 56)}" for n, r, _ in plotted])
    for y, (r, s) in enumerate(rows):
        ax.text(s + 0.01, y, f"{s:.2f}  ·  {route_of(r)}", va="center", color=TEXT_PRIMARY, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Best cosine similarity in local dataset", color=TEXT_SECONDARY)
    used = [k for k in ROUTE_COLORS if any(route_of(r) == k for r, _ in rows)]
    fig.legend(handles=[Patch(color=ROUTE_COLORS[k], label=k) for k in used], loc="lower left",
               bbox_to_anchor=(0.01, 0.01), ncol=len(used), frameon=False, labelcolor=TEXT_PRIMARY)
    _style(ax, "Retrieval quality and answer route per query",
           "The LLM judge decides the fallback; similarity shown for context")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig
