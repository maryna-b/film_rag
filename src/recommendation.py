"""
Recommendation module for the Film RAG system.
"""
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIMILARITY_THRESHOLD, MAX_RECOMMENDATIONS

def get_similar_films(
    vector_store: Chroma, 
    film_title: str, 
    films: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get similar films based on vector similarity.
    
    Args:
        vector_store: Chroma vector store.
        film_title: Title of the film to find similar films for.
        films: List of film dictionaries.
        
    Returns:
        List of similar film dictionaries.
    """
    # Find the film in the database
    target_film = None
    for film in films:
        if film["title"].lower() == film_title.lower():
            target_film = film
            break
    
    if not target_film:
        return []
    
    # Create a query from the film's summary
    query = target_film["summary"]
    
    # Get similar documents
    similar_docs = vector_store.similarity_search_with_score(
        query=query,
        k=len(films)  # Get all films to filter later
    )
    
    # Filter out the target film and apply similarity threshold
    recommendations = []
    for doc, score in similar_docs:
        # Convert score to similarity (Chroma returns distance, lower is better)
        similarity = 1.0 - score
        
        if (similarity >= SIMILARITY_THRESHOLD and 
            doc.metadata["title"].lower() != film_title.lower()):
            
            # Find the full film data
            for film in films:
                if film["title"] == doc.metadata["title"]:
                    recommendations.append({
                        "film": film,
                        "similarity": similarity
                    })
                    break
    
    # Sort by similarity (descending)
    recommendations.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return only the film data for the top recommendations
    return [rec["film"] for rec in recommendations[:MAX_RECOMMENDATIONS]]

def get_recommendations_by_genre(
    film_title: str,
    films: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get film recommendations based on genre similarity.
    
    Args:
        film_title: Title of the film to find similar films for.
        films: List of film dictionaries.
        
    Returns:
        List of similar film dictionaries.
    """
    # Find the film in the database
    target_film = None
    for film in films:
        if film["title"].lower() == film_title.lower():
            target_film = film
            break
    
    if not target_film:
        return []
    
    # Get the genres of the target film
    target_genres = set(target_film["genre"])
    
    # Calculate genre similarity for each film
    recommendations = []
    for film in films:
        if film["title"] != target_film["title"]:
            film_genres = set(film["genre"])
            
            # Calculate Jaccard similarity
            intersection = len(target_genres.intersection(film_genres))
            union = len(target_genres.union(film_genres))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0:
                recommendations.append({
                    "film": film,
                    "similarity": similarity
                })
    
    # Sort by similarity (descending)
    recommendations.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return only the film data for the top recommendations
    return [rec["film"] for rec in recommendations[:MAX_RECOMMENDATIONS]]

def get_recommendations_by_director(
    film_title: str,
    films: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Get film recommendations by the same director.
    
    Args:
        film_title: Title of the film to find similar films for.
        films: List of film dictionaries.
        
    Returns:
        List of films by the same director.
    """
    # Find the film in the database
    target_film = None
    for film in films:
        if film["title"].lower() == film_title.lower():
            target_film = film
            break
    
    if not target_film:
        return []
    
    # Get the director of the target film
    target_director = target_film["director"]
    
    # Find other films by the same director
    recommendations = []
    for film in films:
        if film["title"] != target_film["title"] and film["director"] == target_director:
            recommendations.append(film)
    
    return recommendations[:MAX_RECOMMENDATIONS]

def get_combined_recommendations(
    film_title: str,
    films: List[Dict[str, Any]],
    vector_store: Chroma
) -> List[Dict[str, Any]]:
    """
    Get combined film recommendations using multiple methods.
    
    Args:
        film_title: Title of the film to find similar films for.
        films: List of film dictionaries.
        vector_store: Chroma vector store.
        
    Returns:
        List of recommended films with scores.
    """
    # Get recommendations from different methods
    vector_recs = get_similar_films(vector_store, film_title, films)
    genre_recs = get_recommendations_by_genre(film_title, films)
    director_recs = get_recommendations_by_director(film_title, films)
    
    # Combine and deduplicate recommendations
    all_recs = {}
    
    # Add vector recommendations (highest weight)
    for i, film in enumerate(vector_recs):
        score = (len(vector_recs) - i) * 3  # Higher weight for vector similarity
        all_recs[film["title"]] = {
            "film": film,
            "score": score
        }
    
    # Add genre recommendations
    for i, film in enumerate(genre_recs):
        score = (len(genre_recs) - i) * 2  # Medium weight for genre similarity
        if film["title"] in all_recs:
            all_recs[film["title"]]["score"] += score
        else:
            all_recs[film["title"]] = {
                "film": film,
                "score": score
            }
    
    # Add director recommendations
    for i, film in enumerate(director_recs):
        score = (len(director_recs) - i)  # Lower weight for same director
        if film["title"] in all_recs:
            all_recs[film["title"]]["score"] += score
        else:
            all_recs[film["title"]] = {
                "film": film,
                "score": score
            }
    
    # Convert to list and sort by score
    recommendations = list(all_recs.values())
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    
    # Return only the film data
    return [rec["film"] for rec in recommendations[:MAX_RECOMMENDATIONS]]
