"""Generates the two project notebooks (run from anywhere: python scripts/build_notebooks.py)."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

STARTER = Path(__file__).resolve().parent.parent / "project" / "starter"

PYSQLITE = """# Only needed in the Udacity workspace (old system sqlite). On your own machine this does nothing.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass"""

ENV = """import os
from dotenv import load_dotenv, find_dotenv

# find_dotenv() finds the .env file in this folder (or a parent folder);
# override=True makes the values in .env win over anything already set.
load_dotenv(find_dotenv(usecwd=True), override=True)

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
assert os.getenv("OPENAI_BASE_URL"), "OPENAI_BASE_URL is not set"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY is not set"
print("Environment OK")"""

md = new_markdown_cell
code = new_code_cell


def notebook(cells):
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "udaplay", "display_name": "Python (UdaPlay)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ======================================================================================
# Part 1 - RAG: dataset processing + vector database
# ======================================================================================
part1 = notebook([
    md("""# UdaPlay - Part 1: Building the Game Knowledge Base (RAG)

UdaPlay is an AI research agent that answers questions about video games. This notebook prepares its
**internal knowledge**: it loads the local game dataset, turns each game into an embeddable document,
stores the documents in a **persistent ChromaDB** collection, and shows that the collection supports
semantic search.

| Step | What happens |
|---|---|
| 1 | Load and validate the game JSON files in `games/` |
| 2 | Format each game into a text document + metadata |
| 3 | Embed with OpenAI and store in ChromaDB (`./chromadb`, persisted to disk) |
| 4 | Query the vector database (semantic search + metadata filters) |
| 5 | Visualize the knowledge base |

The dataset has the 15 games from the project brief plus 5 extra games (`016`-`020`), added to support
richer questions (e.g. comparing Rockstar's or Nintendo's catalogue)."""),
    code(PYSQLITE),
    md("## Setup\nLoad the `.env` file (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `TAVILY_API_KEY`) and check that all three keys are set."),
    code(ENV),
    code("""import json
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from lib.config import embedding_model
from lib.vector_db import COLLECTION_METADATA, game_metadata, game_to_document"""),
    md("## 1. Load and validate the dataset\nEach file in `games/` holds one game. The file name (`001`, `002`, ...) becomes the document id."),
    code("""GAMES_DIR = Path("games")
REQUIRED_FIELDS = {"Name", "Platform", "Genre", "Publisher", "Description", "YearOfRelease"}

games = []
for file in sorted(GAMES_DIR.glob("*.json")):
    game = json.loads(file.read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS - game.keys()
    assert not missing, f"{file.name} is missing {missing}"
    assert isinstance(game["YearOfRelease"], int), f"{file.name}: YearOfRelease must be an int"
    game["id"] = file.stem
    games.append(game)

print(f"Loaded {len(games)} games\\n")
print(f"{'id':<4} {'Name':<42} {'Platform':<36} {'Year':<5} Publisher")
print("-" * 120)
for g in games:
    print(f"{g['id']:<4} {g['Name']:<42} {g['Platform']:<36} {g['YearOfRelease']:<5} {g['Publisher']}")"""),
    code("""print("Platforms:", dict(Counter(g["Platform"] for g in games)))
print("Genres:   ", dict(Counter(g["Genre"] for g in games)))
print("Publishers:", dict(Counter(g["Publisher"] for g in games)))"""),
    md("""## 2. Format games into documents
Every game becomes one document. Name, platform, year, genre, and publisher go into the embedded text
(so questions like *"Who published ..."* match on those words), and the same fields are stored as
**metadata** so they can be filtered on and cited exactly."""),
    code("""documents = [game_to_document(g) for g in games]
metadatas = [game_metadata(g) for g in games]
ids = [g["id"] for g in games]

print(documents[1])
print()
print(metadatas[1])"""),
    md("""## 3. Create the persistent vector database
The embedding function uses `OPENAI_API_KEY` (instead of Chroma's default `CHROMA_OPENAI_API_KEY`) and the
configured base URL. The collection uses cosine distance, so `similarity = 1 - distance`.
`upsert` makes this cell safe to re-run."""),
    code("""chroma_client = chromadb.PersistentClient(path="chromadb")

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key_env_var="OPENAI_API_KEY",
    api_base=os.getenv("OPENAI_BASE_URL"),
    model_name=embedding_model(),
)

collection = chroma_client.get_or_create_collection(
    name="udaplay",
    embedding_function=embedding_fn,
    metadata=COLLECTION_METADATA,
)
collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
print(f"Collection '{collection.name}' now holds {collection.count()} documents (persisted in ./chromadb)")"""),
    md("## 4. Semantic search\nQueries are phrased in natural language and don't need to match the stored text word for word."),
    code("""def semantic_search(query: str, n_results: int = 3, where: dict | None = None) -> None:
    results = collection.query(query_texts=[query], n_results=n_results, where=where)
    print(f"Q: {query}" + (f"   (filter: {where})" if where else ""))
    for doc_id, meta, distance in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
        print(f"   {1 - distance:.3f}  [{doc_id}] {meta['Name']} - {meta['Platform']}, "
              f"{meta['YearOfRelease']}, {meta['Publisher']}")
    print()


semantic_search("Who published Grand Theft Auto: San Andreas?")
semantic_search("When was Pokémon Gold and Silver released?")
semantic_search("What platform was Super Mario 64 launched on?")
semantic_search("realistic car racing simulator")
semantic_search("open world fantasy role-playing game with dragons")"""),
    md("Metadata filters combine with semantic search, e.g. only games on a given platform or after a given year:"),
    code("""semantic_search("adventure game", where={"Platform": "Nintendo 64"})
semantic_search("open world game", where={"YearOfRelease": {"$gte": 2015}})"""),
    md("""### Persistence check
Open a brand-new client on the same folder and query it. This is exactly how Part 2 loads the knowledge base."""),
    code("""fresh_client = chromadb.PersistentClient(path="chromadb")
reloaded = fresh_client.get_collection("udaplay", embedding_function=embedding_fn)
print(f"Reloaded collection has {reloaded.count()} documents")
print(reloaded.query(query_texts=["Halo on the original Xbox"], n_results=1)["metadatas"][0][0])"""),
    md("## 5. Visualizing the knowledge base"),
    code("""%matplotlib inline
from lib.report import plot_embedding_map, plot_games_per_platform, plot_release_timeline

plot_games_per_platform(games);"""),
    code("plot_release_timeline(games);"),
    md("A 2-D PCA projection of the stored embeddings. Games that are semantically similar (e.g. the Gran Turismo or Pokémon titles) sit close together, which is what makes retrieval work."),
    code("plot_embedding_map(collection);"),
    md("""## Summary
- 20 game records were loaded, validated, and converted into documents with metadata.
- They are embedded with OpenAI embeddings and stored in a **persistent** ChromaDB collection (`./chromadb`, collection `udaplay`).
- Semantic search returns the right game for the three brief questions, and also supports metadata filters.

Continue with **`Udaplay_02_starter_project.ipynb`** to build the agent on top of this collection."""),
])

# ======================================================================================
# Part 2 - Agent: tools, state machine, memory, demo, report
# ======================================================================================
part2 = notebook([
    md("""# UdaPlay - Part 2: The Research Agent

This notebook builds the UdaPlay agent on top of the knowledge base from Part 1 and runs it on example questions.

**Tools**
1. `retrieve_game` - semantic search over the local dataset (and over long-term memory)
2. `evaluate_retrieval` - an LLM judge that decides whether the retrieved documents can answer the question
3. `game_web_search` - Tavily web search, used only when the evaluation says internal knowledge is insufficient

**Workflow (state machine)**
```
__entry__ -> message_prep -> llm_processor --tool calls--> tool_executor --+
                                  ^                                       |
                                  +---------------------------------------+
                                  | no more tool calls
                                  v
                             format_answer -> persist_memory -> __termination__
```
- The first two LLM calls of every turn are forced (`tool_choice`) to `retrieve_game` and then `evaluate_retrieval`,
  so no answer skips internal knowledge or its evaluation. The model then decides whether to search the web.
- `tool_executor` enforces the order *retrieve -> evaluate -> web search*: a web search attempted before the
  internal knowledge was checked and evaluated is rejected and returned to the model as an error.
- `format_answer` produces a **structured answer** (natural language + JSON, with citations and confidence).
- `persist_memory` saves facts learned from the web to **long-term memory** (a separate Chroma collection),
  so future sessions can answer them without searching again.
- **Short-term memory** keeps each session's conversation, so follow-up questions work."""),
    code(PYSQLITE),
    md("## Setup"),
    code(ENV),
    code("""%matplotlib inline
import json

import chromadb
from tavily import TavilyClient

from lib.agent import UdaPlayAgent
from lib.llm import LLM
from lib.memory import LongTermMemory
from lib.report import plot_retrieval, print_result, route_of
from lib.tools import create_udaplay_tools
from lib.vector_db import get_embedding_function, get_games_collection

chroma_client = chromadb.PersistentClient(path="chromadb")
embedding_fn = get_embedding_function()   # same OpenAI embedding function as Part 1
games_collection = get_games_collection(chroma_client, embedding_fn)
long_term_memory = LongTermMemory(chroma_client, embedding_fn)

print(f"Games collection: {games_collection.count()} documents")
print(f"Long-term memory: {long_term_memory.count()} learned facts")"""),
    md("""Long-term memory persists across runs. To make this demo reproducible, it is cleared here so the first
web-search question really goes to the web. Set `RESET_LONG_TERM_MEMORY = False` to keep what was learned."""),
    code("""RESET_LONG_TERM_MEMORY = True
if RESET_LONG_TERM_MEMORY:
    long_term_memory.clear()
print(f"Long-term memory: {long_term_memory.count()} learned facts")"""),
    md("""## 1. Tools
The tools are plain Python functions (see `lib/tools.py`) wrapped with a `@tool` decorator (`lib/tooling.py`)
that derives the OpenAI function schema from the type hints and docstring."""),
    code("""judge_llm = LLM(temperature=0)           # LLM-as-judge used inside evaluate_retrieval
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

tools = create_udaplay_tools(games_collection, judge_llm, tavily_client, long_term_memory=long_term_memory)
retrieve_game, evaluate_retrieval, game_web_search = tools

for t in tools:
    print(json.dumps(t.to_api(), indent=2))"""),
    md("### Tool 1 - `retrieve_game`"),
    code("""docs = retrieve_game(query="Who published Grand Theft Auto: San Andreas?")
for d in docs:
    print(f"{d['similarity']:.3f}  {d['source']:<14} {d['title']}")"""),
    md("### Tool 2 - `evaluate_retrieval`\nA question the dataset can answer, and one it can't:"),
    code("""print(json.dumps(evaluate_retrieval(question="Who published Grand Theft Auto: San Andreas?", retrieved_docs=docs), indent=2))

news_docs = retrieve_game(query="What is Rockstar Games working on right now?")
print(json.dumps(evaluate_retrieval(question="What is Rockstar Games working on right now?", retrieved_docs=news_docs), indent=2))"""),
    md("### Tool 3 - `game_web_search`"),
    code("""web = game_web_search(question="Rockstar Games latest game in development")
print("Tavily answer:", web["answer"])
for r in web["results"]:
    print("-", r["title"], "|", r["url"])"""),
    md("## 2. The agent"),
    code("""agent = UdaPlayAgent(llm=LLM(temperature=0), tools=tools, long_term_memory=long_term_memory)

print("Steps:", list(agent.workflow.steps))
for t in agent.workflow.transitions.values():
    print(f"  {t.source:<15} -> {t.targets}{'  (conditional)' if t.condition else ''}")"""),
    md("""## 3. Example queries
All queries below run in **one session** (`session_id="demo"`), so the agent can resolve follow-ups like
*"their"* and *"that one"*. Each result shows the workflow path, every tool call with a summary of its
result, the agent's reasoning, the final cited answer, and the structured JSON."""),
    code("""results = []

def ask(question: str, session_id: str = "demo", show_json: bool = False):
    result = agent.invoke(question, session_id=session_id)
    results.append(result)
    print_result(result, show_json=show_json)
    return result"""),
    md("### Query 1 - publisher (local dataset)"),
    code('ask("Who published Grand Theft Auto: San Andreas?", show_json=True);'),
    md("### Query 2 - release date (local dataset)"),
    code('ask("When was Pokémon Gold and Silver released?");'),
    md("### Query 3 - platform (local dataset)"),
    code('ask("What platform was Super Mario 64 launched on?");'),
    md("### Query 4 - current events (web search fallback)\nThe dataset is static, so the evaluator should reject it and the agent should search the web."),
    code('ask("What is Rockstar Games working on right now?", show_json=True);'),
    md("### Queries 5-6 - follow-ups (conversation state)\n*\"their\"* refers to Rockstar Games from the previous turn; *\"that one\"* refers to the answer of query 5."),
    code('ask("Which of their games in your knowledge base was released most recently?");'),
    code('ask("And what platform was that one released on?");'),
    md("### Query 7 - combining several local documents"),
    code('ask("Compare Gran Turismo and Gran Turismo 5: platforms, release years, and what changed between them.");'),
    md("""## 4. Long-term memory across sessions
A **new agent instance** with a **new session** (no conversation history) asks the web question again.
The answer learned in query 4 was saved to long-term memory, so `retrieve_game` now finds it and the
agent shouldn't need to search the web."""),
    code("""print(f"Long-term memory now holds {long_term_memory.count()} learned fact(s)\\n")

new_agent = UdaPlayAgent(llm=LLM(temperature=0), tools=tools, long_term_memory=long_term_memory)
memory_result = new_agent.invoke("What is Rockstar Games working on right now?", session_id="next-day")
results.append(memory_result)
print_result(memory_result, show_json=False)"""),
    md("## 5. Conversation state\nThe compact history the agent keeps for the `demo` session:"),
    code("""for m in agent.get_session_history("demo"):
    print(f"[{m.role}] {m.content[:160]}{'...' if len(m.content) > 160 else ''}")"""),
    md("## 6. Visualizing the retrieval process"),
    code("plot_retrieval(results);"),
    code("""print(f"{'#':<3}{'Query':<62}{'Route':<22}{'Conf.':<8}{'Cites':<7}{'Tools':<7}Tokens")
print("-" * 115)
for i, r in enumerate(results, 1):
    print(f"{i:<3}{r.query[:60]:<62}{route_of(r):<22}{r.answer.confidence:<8}"
          f"{len(r.answer.citations):<7}{len(r.tool_calls):<7}{r.total_tokens}")"""),
    md("""## 7. Report

**What the agent does**
- Every turn starts with internal retrieval (`retrieve_game`) followed by an explicit quality check
  (`evaluate_retrieval`), both forced via `tool_choice`. Only when the evaluator says the documents are insufficient (or it has
  low confidence) does the agent call `game_web_search`. The tool executor enforces this order.
- Questions about the dataset's games (publisher, release year, platform) are answered from the local dataset
  and cite `games/<id>.json`.
- Time-sensitive questions ("right now") are recognized as not answerable from the static dataset and fall
  back to the web; those answers cite URLs.
- Follow-up questions are resolved using short-term (session) memory.
- Web findings are written to long-term memory, so a fresh session can answer the same question from memory.
- Every answer is returned as natural language **and** as validated JSON (`AgentAnswer`): answer, confidence,
  answered_from, reasoning, citations, and per-game structured facts.

**Limitations**
- The evaluator is itself an LLM, so its usefulness verdicts can occasionally be wrong; similarity scores are
  shown alongside for transparency.
- Facts in long-term memory can go stale; each one stores its `learned_at` timestamp and source URLs so the
  judge (and the user) can take its age into account.
- Retrieval returns the top 3 matches, so questions that need an exhaustive scan of the dataset
  (e.g. "compare *all* the Gran Turismo games in your knowledge base") can't be verified as complete by the
  evaluator, which may then fall back to the web. Metadata filters (Part 1) are the better tool for those.
- Web answers depend on which pages the search returns, so results for current-events questions vary between runs.
- Release years in the dataset are the first release (e.g. Pokémon Gold and Silver: 1999 in Japan)."""),
])

for name, nb in [("Udaplay_01_starter_project.ipynb", part1), ("Udaplay_02_starter_project.ipynb", part2)]:
    nbformat.validate(nb)
    nbformat.write(nb, STARTER / name)
    print("wrote", STARTER / name)
