# UdaPlay

An AI research agent that answers questions about video games. It answers from a local game dataset
using RAG and falls back to web search (Tavily) when its internal knowledge isn't enough. Every answer is
cited, has a confidence level, and comes back as both natural language and JSON.

## Layout

```
project/starter/
├── games/                              20 game records (001-015 from the brief, 016-020 extra)
├── lib/
│   ├── config.py                       paths, model names, .env loading
│   ├── messages.py                     chat message types
│   ├── tooling.py                      @tool decorator -> OpenAI function schema
│   ├── llm.py                          OpenAI chat wrapper (tool calling + structured output)
│   ├── state_machine.py                Step / Transition / Run state machine
│   ├── vector_db.py                    ChromaDB client, embeddings, games collection
│   ├── memory.py                       short-term (session) + long-term (Chroma) memory
│   ├── tools.py                        retrieve_game, evaluate_retrieval, game_web_search
│   ├── agent.py                        UdaPlayAgent + structured answer models
│   └── report.py                       printed run reports + charts
├── Udaplay_01_starter_project.ipynb    Part 1: dataset -> vector DB -> semantic search
├── Udaplay_02_starter_project.ipynb    Part 2: tools, agent, example queries, report
├── chat.py                             interactive terminal chat
└── .env.example                        copy to .env
tests/test_udaplay.py                   offline tests (no API keys needed)
scripts/build_notebooks.py              regenerates both notebooks
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name udaplay --display-name "Python (UdaPlay)"
copy project\starter\.env.example project\starter\.env   # then fill in your keys
```

`.env` needs `OPENAI_API_KEY`, `OPENAI_BASE_URL` (`https://api.openai.com/v1` for OpenAI directly, or
`https://openai.vocareum.com/v1` in the Udacity workspace) and `TAVILY_API_KEY`. Optional:
`UDAPLAY_CHAT_MODEL` (default `gpt-4o`) and `UDAPLAY_EMBEDDING_MODEL` (default `text-embedding-3-small`).

## Run

1. Open `project/starter/Udaplay_01_starter_project.ipynb` with the **Python (UdaPlay)** kernel and run all
   cells. This creates the persistent vector DB in `project/starter/chromadb/`.
2. Run `Udaplay_02_starter_project.ipynb`.
3. Optionally, chat in the terminal: `cd project/starter; ..\..\.venv\Scripts\python chat.py`.

Tests: `.\.venv\Scripts\python -m pytest tests -q`

## How the agent works

```
__entry__ -> message_prep -> llm_processor --tool calls--> tool_executor --> (back to llm_processor)
                                  | done
                                  v
                             format_answer -> persist_memory -> __termination__
```

1. **retrieve_game** does semantic search over the games collection (plus any long-term memories).
2. **evaluate_retrieval** has an LLM judge decide whether the retrieved documents answer the question.
   It returns `useful`, `confidence`, `description` and `should_search_web`.
3. **game_web_search** runs a Tavily search, only after the evaluation asks for one. The tool executor
   rejects a web search that comes before retrieval and evaluation.
4. **format_answer** returns a validated `AgentAnswer`: answer with `[n]` citations, confidence, source,
   reasoning, citations, and per-game facts.
5. **persist_memory** saves answers learned from the web to a separate Chroma collection, so later sessions
   can answer them from memory.

Short-term memory keeps each session's question-and-answer history so follow-up questions work.
