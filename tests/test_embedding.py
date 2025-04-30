import pytest
import os
from unittest.mock import MagicMock, patch

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Adjust the import path based on your project structure
from src.embedding import (
    create_embeddings_model,
    create_text_splitter,
    prepare_documents_for_embedding,
    create_vector_store,
    load_vector_store
)
from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PERSIST_DIRECTORY
)

# --- Sample Data ---
SAMPLE_DICT_DOCUMENTS = [
    {
        "id": "film_inception",
        "content": "Title: Inception\nYear: 2010...",
        "metadata": {
            "title": "Inception", "year": 2010, "director": "Nolan",
            "genre": "Sci-Fi, Action", "actors": "DiCaprio, Page"
        }
    },
    {
        "id": "film_matrix",
        "content": "Title: The Matrix\nYear: 1999...",
        "metadata": {
            "title": "The Matrix", "year": 1999, "director": "Wachowskis",
            "genre": "Action, Sci-Fi", "actors": "Reeves, Fishburne"
        }
    }
]

# --- Tests for Factory Functions ---

def test_create_embeddings_model():
    """Test creating the embeddings model."""
    embeddings = create_embeddings_model()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == EMBEDDING_MODEL

def test_create_text_splitter():
    """Test creating the text splitter."""
    splitter = create_text_splitter()
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    # Accessing protected attributes for testing configuration
    assert splitter._chunk_size == CHUNK_SIZE
    assert splitter._chunk_overlap == CHUNK_OVERLAP

# --- Tests for prepare_documents_for_embedding ---

def test_prepare_documents_for_embedding_conversion():
    """Test converting dicts to LangChain Documents."""
    langchain_docs = prepare_documents_for_embedding(SAMPLE_DICT_DOCUMENTS)

    assert len(langchain_docs) == 2
    assert all(isinstance(doc, Document) for doc in langchain_docs)

    # Check first document
    assert langchain_docs[0].page_content == SAMPLE_DICT_DOCUMENTS[0]["content"]
    assert langchain_docs[0].metadata["id"] == SAMPLE_DICT_DOCUMENTS[0]["id"]
    assert langchain_docs[0].metadata["title"] == SAMPLE_DICT_DOCUMENTS[0]["metadata"]["title"]
    assert langchain_docs[0].metadata["year"] == SAMPLE_DICT_DOCUMENTS[0]["metadata"]["year"]
    assert langchain_docs[0].metadata["director"] == SAMPLE_DICT_DOCUMENTS[0]["metadata"]["director"]
    assert langchain_docs[0].metadata["genre"] == SAMPLE_DICT_DOCUMENTS[0]["metadata"]["genre"]
    assert langchain_docs[0].metadata["actors"] == SAMPLE_DICT_DOCUMENTS[0]["metadata"]["actors"]

    # Check second document
    assert langchain_docs[1].page_content == SAMPLE_DICT_DOCUMENTS[1]["content"]
    assert langchain_docs[1].metadata["title"] == SAMPLE_DICT_DOCUMENTS[1]["metadata"]["title"]

def test_prepare_documents_for_embedding_empty_list():
    """Test converting an empty list."""
    langchain_docs = prepare_documents_for_embedding([])
    assert langchain_docs == []

# --- Tests for create_vector_store ---

@patch('src.embedding.create_embeddings_model')
@patch('src.embedding.Chroma.from_documents')
def test_create_vector_store(mock_from_documents, mock_create_embeddings):
    """Test creating the vector store with mocks."""
    # Prepare mock objects
    mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
    mock_create_embeddings.return_value = mock_embeddings

    mock_vector_store_instance = MagicMock(spec=Chroma)
    # Add the persist method to the mock
    mock_vector_store_instance.persist = MagicMock()
    mock_from_documents.return_value = mock_vector_store_instance

    # Prepare input documents (already tested conversion)
    input_docs = prepare_documents_for_embedding(SAMPLE_DICT_DOCUMENTS)

    # Call the function
    vector_store = create_vector_store(input_docs)

    # Assertions
    mock_create_embeddings.assert_called_once()
    mock_from_documents.assert_called_once_with(
        documents=input_docs,
        embedding=mock_embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    mock_vector_store_instance.persist.assert_called_once()
    assert vector_store == mock_vector_store_instance

# --- Tests for load_vector_store ---

@patch('src.embedding.create_embeddings_model')
@patch('src.embedding.os.path.exists')
@patch('src.embedding.Chroma')
def test_load_vector_store_exists(mock_chroma_init, mock_exists, mock_create_embeddings):
    """Test loading the vector store when the directory exists."""
    # Prepare mock objects
    mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
    mock_create_embeddings.return_value = mock_embeddings
    mock_exists.return_value = True # Simulate directory exists

    mock_vector_store_instance = MagicMock(spec=Chroma)
    mock_chroma_init.return_value = mock_vector_store_instance

    # Call the function
    vector_store = load_vector_store()

    # Assertions
    mock_create_embeddings.assert_called_once()
    mock_exists.assert_called_once_with(CHROMA_PERSIST_DIRECTORY)
    mock_chroma_init.assert_called_once_with(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=mock_embeddings
    )
    assert vector_store == mock_vector_store_instance

@patch('src.embedding.create_embeddings_model')
@patch('src.embedding.os.path.exists')
@patch('src.embedding.Chroma')
def test_load_vector_store_not_exists(mock_chroma_init, mock_exists, mock_create_embeddings):
    """Test loading the vector store when the directory does not exist."""
    # Prepare mock objects
    mock_embeddings = MagicMock(spec=OpenAIEmbeddings)
    mock_create_embeddings.return_value = mock_embeddings
    mock_exists.return_value = False # Simulate directory does not exist

    # Call the function
    vector_store = load_vector_store()

    # Assertions
    mock_create_embeddings.assert_called_once() # Still need embeddings to check
    mock_exists.assert_called_once_with(CHROMA_PERSIST_DIRECTORY)
    mock_chroma_init.assert_not_called() # Chroma should not be initialized
    assert vector_store is None
