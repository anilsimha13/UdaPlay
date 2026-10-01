Project Scenario
You’ve been hired as an AI Engineer at a gaming analytics company developing an assistant called UdaPlay. Executives, analysts, and gamers want to ask natural language questions like:

“Who published Grand Theft Auto: San Andreas?”
“When was Pokémon Gold and Silver released?”
“What platform was Super Mario 64 launched on?”
“What is Rockstar Games working on right now?”
The first three can be answered from the project's local dataset of 15 games. The last one can't, so your agent has to search the web for it.

Your agent should:

Attempt to answer the question from internal knowledge (a local dataset of 15 video games)
If the information is not found or confidence is low, search the web
Generate a clean, structured answer/report that cites its source
Your agent must also remember earlier questions within a session (conversation state). Keeping what it learns across sessions, in persistent long-term memory, is an optional Stand Out extension, not a requirement.

Project Specifications
In this project, you will build an AI Research Agent called UdaPlay designed to answer questions about video games. The agent will be capable of:

Answering user questions about games, including:

Game titles and their details
Release dates and platforms
Game descriptions and genres
Publisher information
Using a two-tier information retrieval system:

Primary: RAG (Retrieval Augmented Generation) over a local dataset of games
Secondary: Web search using the Tavily API when internal knowledge is insufficient
Implementing a robust evaluation system:

Assessing the quality of retrieved information
Determining when to fall back to web search
Providing confidence levels in answers
Generating clear, well-structured responses that:

Cite information sources
Combine information from multiple sources when needed
Present information in a natural, readable format