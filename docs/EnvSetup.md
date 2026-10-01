For this project you'll use the following libraries:

chromadb>=1.0.4

openai>=1.73.0

pydantic>=2.11.3

python-dotenv>=1.1.0

tavily-python>=0.5.4

pdfplumber>=0.11.6

These are installed in the included workspace (Python 3.13). To run the project on your own machine instead, use Python 3.11 or later and install the same libraries:

pip install "chromadb>=1.0.4" "openai>=1.73.0" "pydantic>=2.11.3" "python-dotenv>=1.1.0" "tavily-python>=0.5.4" "pdfplumber>=0.11.6"
The pysqlite3 cell at the top of each notebook is only for the Udacity workspace. On your own machine it does nothing, so you can leave it in.

Workspace Instructions
Sign up for a free Tavily account, which will allow you to search the web and get back N pages and textual content for the pages, for free (first 1000 requests are free).

After signing up, you can get your API key here: https://app.tavily.com/home(opens in a new tab)

Create a file named .env in the project/starter folder (the folder that holds both project notebooks), with entries for OPENAI_API_KEY, OPENAI_BASE_URL and TAVILY_API_KEY. Use the OpenAI key from the Cloud Resources panel; it starts with voc-.

e.g.

OPENAI_API_KEY="voc-****\*\*****"
OPENAI_BASE_URL="https://openai.vocareum.com/v1"
TAVILY_API_KEY="tvly-**\*\*\*\***"
The file must be named exactly .env. The notebooks look for that name and won't find a file called anything else.

At the top of each notebook, load the .env file and check that all three variables are set:

import os
from dotenv import load_dotenv, find_dotenv

# find_dotenv() finds the .env file in this folder (or a parent folder);

# override=True makes the values in .env win over anything already set.

load_dotenv(find_dotenv(), override=True)

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
assert os.getenv("OPENAI_BASE_URL"), "OPENAI_BASE_URL is not set"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY is not set"
Embedding function. Unless you tell it otherwise, ChromaDB's OpenAIEmbeddingFunction looks for its key in a separate variable called CHROMA_OPENAI_API_KEY. Point it at the key you already have:

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
api_key_env_var="OPENAI_API_KEY",
api_base=os.getenv("OPENAI_BASE_URL"),
)
Create the collection with this in Part 1. Part 2 rebuilds the same function when it loads the collection, so it needs the same .env.

You may see lines like Failed to send telemetry event … capture() takes 1 positional argument but 3 were given when you use ChromaDB. They're harmless and don't affect your results.
