"""
Command-line interface for the Film RAG system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_film_database, prepare_documents
from src.embedding import (
    create_vector_store, 
    load_vector_store, 
    prepare_documents_for_embedding
)
from src.retrieval import HybridRetriever
from src.agent import FilmAgent
from config import OPENAI_API_KEY, CHROMA_PERSIST_DIRECTORY

def initialize_system():
    """
    Initialize the Film RAG system.
    
    Returns:
        FilmAgent instance.
    """
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key not found.")
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Load film database
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "film_database.json")
    films = load_film_database(data_path)
    
    if not films:
        print("Error: Film database is empty or could not be loaded.")
        sys.exit(1)
    
    # Load or create vector store
    vector_store = load_vector_store()
    
    if vector_store is None:
        print("Vector store not found. Creating a new one...")
        
        # Prepare documents
        prepared_docs = prepare_documents(films)
        langchain_docs = prepare_documents_for_embedding(prepared_docs)
        
        # Create vector store
        vector_store = create_vector_store(langchain_docs)
        
        print(f"Vector store created and persisted at {CHROMA_PERSIST_DIRECTORY}")
    else:
        print(f"Loaded existing vector store from {CHROMA_PERSIST_DIRECTORY}")
    
    # Create retriever
    retriever = HybridRetriever(vector_store=vector_store, films=films)
    
    # Create agent
    agent = FilmAgent(retriever=retriever, films=films)
    
    return agent

def run_cli():
    """
    Run the command-line interface.
    """
    print("Initializing Film RAG system...")
    agent = initialize_system()
    print("Film RAG system initialized. You can start asking questions.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            query = input("\nYou: ")
            
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue
            
            # Process query
            response = agent.process_query(query)
            
            print(f"\nFilm Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    run_cli()
