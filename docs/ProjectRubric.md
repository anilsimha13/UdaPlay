Use this project rubric to understand and assess the project criteria.

RAG
Criteria Submission Requirements
Prepare and process a local dataset of video game information for use in a vector database and RAG pipeline

The submission includes the notebook (Udaplay_01_starter_project.ipynb) that loads, processes, and formats the provided game JSON files.
The processed data is added to a persistent vector database (e.g., ChromaDB) with appropriate embeddings.
The notebook or script demonstrates that the vector database can be queried for semantic search.
Agent Development
Criteria Submission Requirements
Implement agent tools for internal retrieval, evaluation, and web search fallback.

The submission includes at least three tools:
A tool to retrieve game information from the vector database.
A tool to evaluate the quality of retrieved results.
A tool to perform web search using an API (e.g., Tavily).
Each tool is implemented as a function/class and is integrated into the agent workflow.
The agent:
first attempts to answer using internal knowledge,
evaluates the result,
and falls back to web search if needed.
Build a stateful agent that manages conversation and tool usage.

The agent is implemented as a class or function that maintains conversation state.
The agent can handle multiple queries in a session, remembering previous context.
The agent’s workflow is implemented as a state machine or similar abstraction.
The agent produces clear, structured, and well-cited answers.
Demonstrate and report on the agent’s performance with example queries.

The submission includes the notebook (Udaplay_02_starter_project.ipynb) that runs the agent on at least three example queries (e.g., about game release dates, platforms, or publishers).

The output for each query includes the agent’s reasoning, tool usage, and final answer.

The report includes at least the response with citation, if any

Suggestions to Make Your Project Stand Out
Personalize the Dataset: Add extra games, companies, or platforms to the dataset and demonstrate richer queries.

Advanced Memory: Implement persistent long-term memory so the agent “learns” from web searches.
Structured Output: Return answers in both natural language and structured JSON for easy integration.

Visualization: Create a dashboard or visualization of the agent’s retrieval process or knowledge base.

Custom Tools: Add extra tools, such as sentiment analysis of game reviews or trending games detection.
