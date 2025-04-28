"""
Data processing module for the Film RAG system.
"""
import json
import os
from typing import Dict, List, Any

def load_film_database(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the film database from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing film data.
        
    Returns:
        List of dictionaries containing film data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []

def prepare_documents(films: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare film data for embedding and retrieval.
    
    Args:
        films: List of dictionaries containing film data.
        
    Returns:
        List of documents ready for embedding.
    """
    documents = []
    
    for film in films:
        # Create a document for each film with all relevant information
        document = {
            "id": f"film_{film['title'].lower().replace(' ', '_')}",
            "title": film["title"],
            "content": f"Title: {film['title']}\nYear: {film['year']}\nDirector: {film['director']}\n"
                      f"Genre: {', '.join(film['genre'])}\nActors: {', '.join(film['actors'])}\n"
                      f"Summary: {film['summary']}",
            "metadata": {
                "title": film["title"],
                "year": film["year"],
                "director": film["director"],
                "genre": ", ".join(film["genre"]),  # Convert list to comma-separated string
                "actors": ", ".join(film["actors"]) # Convert list to comma-separated string
            }
        }
        documents.append(document)
    
    return documents

def get_film_by_title(films: List[Dict[str, Any]], title: str) -> Dict[str, Any]:
    """
    Get a film by its title.
    
    Args:
        films: List of dictionaries containing film data.
        title: Title of the film to retrieve.
        
    Returns:
        Dictionary containing film data or empty dict if not found.
    """
    for film in films:
        if film["title"].lower() == title.lower():
            return film
    return {}
