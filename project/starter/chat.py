"""Interactive UdaPlay chat in the terminal. Run Part 1 (or this script with --build) first.

    python chat.py            # chat with the agent
    python chat.py --build    # (re)build the games collection, then chat
"""

import os
import sys

import chromadb
from tavily import TavilyClient

from lib import LLM, LongTermMemory, UdaPlayAgent, create_udaplay_tools, load_env
from lib.config import CHROMA_DIR
from lib.report import print_result
from lib.vector_db import build_games_collection, get_embedding_function, get_games_collection, load_games


def main() -> None:
    load_env()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = get_embedding_function()
    if "--build" in sys.argv:
        games = build_games_collection(client, embedding_fn, load_games())
    else:
        games = get_games_collection(client, embedding_fn)
    memory = LongTermMemory(client, embedding_fn)
    tools = create_udaplay_tools(games, LLM(), TavilyClient(api_key=os.getenv("TAVILY_API_KEY")), long_term_memory=memory)
    agent = UdaPlayAgent(LLM(), tools, long_term_memory=memory)

    print("UdaPlay - ask about video games. Type 'exit' to quit.\n")
    while (question := input("you> ").strip()).lower() not in ("exit", "quit"):
        if question:
            print_result(agent.invoke(question, session_id="cli"), show_json="--json" in sys.argv)
            print()


if __name__ == "__main__":
    main()
