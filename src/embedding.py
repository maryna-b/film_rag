"""
Embedding module for the Film RAG system.
"""
import os
from typing import Dict, List, Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP

def create_embeddings_model():
    """
    Create an OpenAI embeddings model.
    
    Returns:
        OpenAIEmbeddings model.
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def create_text_splitter():
    """
    Create a text splitter for chunking documents.
    
    Returns:
        RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def prepare_documents_for_embedding(documents: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert dictionary documents to LangChain Document objects.
    
    Args:
        documents: List of document dictionaries.
        
    Returns:
        List of LangChain Document objects.
    """
    langchain_docs = []
    
    for doc in documents:
        langchain_docs.append(
            Document(
                page_content=doc["content"],
                metadata={
                    "id": doc["id"],
                    "title": doc["metadata"]["title"],
                    "year": doc["metadata"]["year"],
                    "director": doc["metadata"]["director"],
                    "genre": doc["metadata"]["genre"],
                    "actors": doc["metadata"]["actors"]
                }
            )
        )
    
    return langchain_docs

def create_vector_store(documents: List[Document]):
    """
    Create a Chroma vector store from documents.
    
    Args:
        documents: List of LangChain Document objects.
        
    Returns:
        Chroma vector store.
    """
    # Create embeddings model
    embeddings = create_embeddings_model()
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    
    # Persist the vector store
    vector_store.persist()
    
    return vector_store

def load_vector_store():
    """
    Load an existing Chroma vector store.
    
    Returns:
        Chroma vector store or None if it doesn't exist.
    """
    embeddings = create_embeddings_model()
    
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        return None
