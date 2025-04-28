"""
Retrieval module for the Film RAG system.
"""
from typing import List, Dict, Any

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_chroma import Chroma

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_RETRIEVAL

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines semantic search with keyword search.
    """
    vector_store: Chroma
    films: List[Dict[str, Any]]
    
    def __init__(self, vector_store: Chroma, films: List[Dict[str, Any]]):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Chroma vector store for semantic search.
            films: List of film dictionaries for keyword search.
        """
        super().__init__(vector_store=vector_store, films=films)
        # The super().__init__ call now handles assignment and validation
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get relevant documents using hybrid search.
        
        Args:
            query: User query.
            run_manager: Callback manager.
            
        Returns:
            List of relevant documents.
        """
        # Semantic search using vector store
        semantic_docs = self.vector_store.similarity_search(query, k=TOP_K_RETRIEVAL)
        
        # Simple keyword search
        keyword_docs = self._keyword_search(query)
        
        # Combine results (removing duplicates)
        combined_docs = semantic_docs.copy()
        
        for doc in keyword_docs:
            if doc not in combined_docs:
                combined_docs.append(doc)
        
        return combined_docs[:TOP_K_RETRIEVAL]
    
    def _keyword_search(self, query: str) -> List[Document]:
        """
        Perform keyword search on film data.
        
        Args:
            query: User query.
            
        Returns:
            List of documents matching keywords.
        """
        query_terms = query.lower().split()
        results = []
        
        for film in self.films:
            score = 0
            
            # Check title
            for term in query_terms:
                if term in film["title"].lower():
                    score += 3
            
            # Check director
            for term in query_terms:
                if term in film["director"].lower():
                    score += 2
            
            # Check genre
            for genre in film["genre"]:
                for term in query_terms:
                    if term in genre.lower():
                        score += 1
            
            # Check actors
            for actor in film["actors"]:
                for term in query_terms:
                    if term in actor.lower():
                        score += 1
            
            if score > 0:
                # Create a document for this film
                doc = Document(
                    page_content=f"Title: {film['title']}\nYear: {film['year']}\nDirector: {film['director']}\n"
                                f"Genre: {', '.join(film['genre'])}\nActors: {', '.join(film['actors'])}\n"
                                f"Summary: {film['summary']}",
                    metadata={
                        "id": f"film_{film['title'].lower().replace(' ', '_')}",
                        "title": film["title"],
                        "year": film["year"],
                        "director": film["director"],
                        "genre": film["genre"],
                        "actors": film["actors"],
                        "score": score
                    }
                )
                results.append(doc)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.metadata["score"], reverse=True)
        
        return results[:TOP_K_RETRIEVAL]
