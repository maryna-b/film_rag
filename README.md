# Film RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for recommending and searching for films based on user queries. It leverages LangChain, ChromaDB, and a custom hybrid retrieval strategy.

## Description

The system allows users to interact with a film database via a command-line interface (CLI). Users can ask for film recommendations based on genre, actors, director, or other details, or search for specific films to get information like summary, year, director, actors, and genre.

The core components include:
- **Data Loading:** Loads film information from a JSON file (`data/film_database.json`).
- **Vector Store:** Uses ChromaDB (`chroma_db/`) to store film embeddings for efficient semantic search. If the store doesn't exist, it's created on the first run.
- **Embedding:** (Implicitly uses a LangChain compatible embedding model configured via environment variables or defaults).
- **Hybrid Retrieval:** Combines semantic similarity search (via ChromaDB) with traditional keyword matching on film metadata (title, director, genre, actors) using a custom `HybridRetriever` (`src/retrieval.py`).
- **LangChain Agent:** An agent (`src/agent.py`) orchestrates the interaction, deciding whether to use the search tool or the recommendation tool based on the user's query.
- **CLI:** A simple command-line interface (`src/cli.py`, run via `main.py`) for user interaction.

## Features

- **Film Recommendations:** Suggests films based on genre or other query details.
- **Film Search:** Retrieves details for specific films mentioned by the user.
- **Hybrid Search:** Combines vector similarity and keyword matching for more relevant results.
- **Conversational Interface:** Uses a LangChain agent to handle user queries naturally.
- **Persistent Vector Store:** Saves the ChromaDB index locally for faster subsequent runs.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd film_rag
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory. You'll need to add configuration for the embedding model and potentially an LLM API key if the agent uses one (e.g., OpenAI). Example:
    ```env
    # Example using OpenAI (replace with your actual keys/models if needed)
    # OPENAI_API_KEY="your_openai_api_key" 
    # EMBEDDING_MODEL_NAME="text-embedding-ada-002" 
    # Or configure for other embedding/LLM providers supported by LangChain
    
    # If using local models like SentenceTransformers, configuration might differ
    # e.g., EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2" 
    ```
    *Note: The specific variables needed depend on the LangChain components used in `src/embedding.py` and `src/agent.py`.*

## Usage

Run the main script to start the CLI:

```bash
python main.py
```

The system will initialize, loading or creating the vector store. You can then interact with the Film Bot:

```
Initializing Film RAG system...
Loaded existing vector store from C:\Users\Marina\Desktop\film_rag\chroma_db
Film RAG system initialized. You can start asking questions.
Type 'exit' or 'quit' to end the session.

You: recommend a sci-fi movie directed by Christopher Nolan
Film Bot: Based on your interest in Sci-Fi films directed by Christopher Nolan, I recommend: ...

You: tell me about The Matrix
Film Bot: The Matrix (1999), directed by Lana Wachowski and Lilly Wachowski, is an Action/Sci-Fi film...

You: exit
```

## Project Structure

```
film_rag/
├── .env                # Environment variables (API keys, model names) - Create this file
├── .venv/              # Virtual environment directory
├── chroma_db/          # Directory for ChromaDB persistent storage
│   └── chroma.sqlite3
├── data/
│   └── film_database.json # Film data source
├── src/
│   ├── __init__.py
│   ├── agent.py        # LangChain agent logic
│   ├── cli.py          # Command-line interface handling
│   ├── data_processing.py # Loading and preparing film data
│   ├── embedding.py    # Embedding generation logic (details depend on implementation)
│   ├── recommendation.py # Recommendation tool/logic
│   └── retrieval.py    # Hybrid retriever implementation
├── config.py           # Configuration constants (e.g., TOP_K_RETRIEVAL)
├── main.py             # Main script to run the application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Dependencies

Key libraries used:
- `langchain`: Core framework for building RAG applications.
- `langchain-chroma`: Integration with ChromaDB.
- `chromadb`: Vector database.
- `pydantic`: Data validation.
- (Potentially others like `openai`, `sentence-transformers` depending on LLM/embedding choices)

Refer to `requirements.txt` for the full list.
