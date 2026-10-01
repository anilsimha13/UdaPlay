"""ChromaDB helpers: embedding function, persistent client, and the games collection."""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, GAMES_COLLECTION, GAMES_DIR, embedding_model

# Cosine distance: 0 = identical, 2 = opposite. Similarity = 1 - distance.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_embedding_function():
    """OpenAI embeddings, using OPENAI_API_KEY instead of Chroma's default CHROMA_OPENAI_API_KEY."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY",
        api_base=os.getenv("OPENAI_BASE_URL"),
        model_name=embedding_model(),
    )


def get_chroma_client(path: Path | str = CHROMA_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(path))


def load_games(games_dir: Path | str = GAMES_DIR) -> list[dict[str, Any]]:
    """Load every game JSON file; the file stem (e.g. '001') becomes the game id."""
    games = []
    for file in sorted(Path(games_dir).glob("*.json")):
        game = json.loads(file.read_text(encoding="utf-8"))
        game["id"] = file.stem
        games.append(game)
    return games


def game_to_document(game: dict[str, Any]) -> str:
    """Flatten a game record into the text that gets embedded."""
    return (
        f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']})\n"
        f"Genre: {game['Genre']}\n"
        f"Publisher: {game['Publisher']}\n"
        f"Description: {game['Description']}"
    )


def game_metadata(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "Name": game["Name"],
        "Platform": game["Platform"],
        "Genre": game["Genre"],
        "Publisher": game["Publisher"],
        "YearOfRelease": int(game["YearOfRelease"]),
    }


def build_games_collection(client, embedding_fn, games: list[dict[str, Any]], reset: bool = False):
    """Create (or refresh) the games collection. Upsert keeps re-runs idempotent."""
    if reset and GAMES_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(GAMES_COLLECTION)
    collection = client.get_or_create_collection(
        name=GAMES_COLLECTION, embedding_function=embedding_fn, metadata=COLLECTION_METADATA
    )
    collection.upsert(
        ids=[g["id"] for g in games],
        documents=[game_to_document(g) for g in games],
        metadatas=[game_metadata(g) for g in games],
    )
    return collection


def get_games_collection(client, embedding_fn):
    return client.get_collection(name=GAMES_COLLECTION, embedding_function=embedding_fn)
