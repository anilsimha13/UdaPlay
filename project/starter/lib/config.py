"""Central configuration: paths, model names, and environment loading."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

STARTER_DIR = Path(__file__).resolve().parent.parent
GAMES_DIR = STARTER_DIR / "games"
CHROMA_DIR = STARTER_DIR / "chromadb"

GAMES_COLLECTION = "udaplay"
MEMORY_COLLECTION = "udaplay_long_term_memory"

REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "TAVILY_API_KEY")


def load_env() -> None:
    """Load the .env file from the starter folder (or a parent) and check the required keys."""
    env_path = STARTER_DIR / ".env"
    load_dotenv(env_path if env_path.exists() else find_dotenv(usecwd=True), override=True)
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. "
            f"Copy .env.example to {STARTER_DIR / '.env'} and fill in your keys."
        )


def chat_model() -> str:
    return os.getenv("UDAPLAY_CHAT_MODEL", "gpt-4o")


def embedding_model() -> str:
    return os.getenv("UDAPLAY_EMBEDDING_MODEL", "text-embedding-3-small")
