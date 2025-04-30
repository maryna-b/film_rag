import pytest
from unittest.mock import MagicMock

from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Adjust the import path based on your project structure
from src.retrieval import HybridRetriever
from config import TOP_K_RETRIEVAL # Should be 3

# --- Sample Data (can reuse or adapt from other tests) ---
SAMPLE_FILMS_DATA = [
    {
        "title": "Inception", "year": 2010, "director": "Christopher Nolan",
        "genre": ["Sci-Fi", "Action", "Thriller"], "actors": ["Leonardo DiCaprio", "Elliot Page"],
        "summary": "Dream heist."
    },
    {
        "title": "The Dark Knight", "year": 2008, "director": "Christopher Nolan",
        "genre": ["Action", "Crime", "Drama"], "actors": ["Christian Bale", "Heath Ledger"],
        "summary": "Batman vs Joker."
    },
    {
        "title": "Interstellar", "year": 2014, "director": "Christopher Nolan",
        "genre": ["Sci-Fi", "Drama", "Adventure"], "actors": ["Matthew McConaughey", "Anne Hathaway"],
        "summary": "Space travel."
    },
    {
        "title": "The Matrix", "year": 1999, "director": "Wachowskis",
        "genre": ["Action", "Sci-Fi"], "actors": ["Keanu Reeves", "Laurence Fishburne"],
        "summary": "Reality simulation."
    }
]

# --- Mock Vector Store ---
@pytest.fixture
def mock_vector_store():
    """Fixture for a mock Chroma vector store."""
    mock_store = MagicMock()
    # Setup mock similarity_search if needed for specific tests
    mock_store.similarity_search = MagicMock(return_value=[])
    return mock_store

# --- Test Retriever Initialization ---
def test_hybrid_retriever_init(mock_vector_store):
    """Test the initialization of HybridRetriever."""
    # Use model_construct to bypass Pydantic validation for mocks
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    assert retriever.vector_store == mock_vector_store
    assert retriever.films == SAMPLE_FILMS_DATA

# --- Tests for _keyword_search ---

def test_keyword_search_title_match(mock_vector_store):
    """Test keyword search matching title."""
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    docs = retriever._keyword_search("dark knight")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "The Dark Knight"
    assert docs[0].metadata["score"] > 0 # Should have a score

def test_keyword_search_director_match(mock_vector_store):
    """Test keyword search matching director."""
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    docs = retriever._keyword_search("nolan")
    # Should find Inception, Dark Knight, Interstellar
    assert len(docs) == 3
    titles = {doc.metadata["title"] for doc in docs}
    assert titles == {"Inception", "The Dark Knight", "Interstellar"}
    # Check scores are assigned
    assert all(doc.metadata["score"] > 0 for doc in docs)

def test_keyword_search_genre_match(mock_vector_store):
    """Test keyword search matching genre."""
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    docs = retriever._keyword_search("sci-fi")
    # Should find Inception, Interstellar, The Matrix
    assert len(docs) == 3
    titles = {doc.metadata["title"] for doc in docs}
    assert titles == {"Inception", "Interstellar", "The Matrix"}

def test_keyword_search_actor_match(mock_vector_store):
    """Test keyword search matching actor."""
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    docs = retriever._keyword_search("dicaprio")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Inception"

def test_keyword_search_multiple_terms(mock_vector_store):
    """Test keyword search with multiple terms matching different fields."""
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    docs = retriever._keyword_search("nolan action") # Matches director and genre
    # Expect Nolan films, potentially higher score for those with Action genre
    assert len(docs) == 3
    titles = {doc.metadata["title"] for doc in docs}
    assert titles == {"Inception", "The Dark Knight", "Interstellar"}
    # Check if Dark Knight or Inception is first (higher score expected)
    assert docs[0].metadata["title"] in ["Inception", "The Dark Knight"]

def test_keyword_search_no_match(mock_vector_store):
    """Test keyword search with no matching terms."""
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    docs = retriever._keyword_search("qwerty zxcvb")
    assert len(docs) == 0

def test_keyword_search_limit(mock_vector_store):
    """Test that keyword search results are limited by TOP_K_RETRIEVAL."""
    # Add more films to exceed limit
    many_films = SAMPLE_FILMS_DATA * 2
    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=many_films)
    # A query likely to match many films
    docs = retriever._keyword_search("the")
    assert len(docs) == TOP_K_RETRIEVAL # Should be 3

# --- Tests for _get_relevant_documents (Hybrid Logic) ---

# Mock documents for testing combination
doc_inception = Document(page_content="Inception content", metadata={"title": "Inception"})
doc_dk = Document(page_content="Dark Knight content", metadata={"title": "The Dark Knight"})
doc_interstellar = Document(page_content="Interstellar content", metadata={"title": "Interstellar"})
doc_matrix = Document(page_content="Matrix content", metadata={"title": "The Matrix"})
doc_other = Document(page_content="Other content", metadata={"title": "Other Film"})

# Mock callback manager
mock_run_manager = MagicMock(spec=CallbackManagerForRetrieverRun)

def test_hybrid_retrieval_semantic_only(mocker, mock_vector_store):
    """Test hybrid retrieval when only semantic search returns results."""
    # Mock semantic search results
    mock_vector_store.similarity_search.return_value = [doc_inception, doc_matrix]
    # Mock keyword search to return nothing
    mocker.patch.object(HybridRetriever, '_keyword_search', return_value=[])

    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    results = retriever._get_relevant_documents("query", run_manager=mock_run_manager)

    assert len(results) == 2
    assert results == [doc_inception, doc_matrix]
    mock_vector_store.similarity_search.assert_called_once_with("query", k=TOP_K_RETRIEVAL)
    HybridRetriever._keyword_search.assert_called_once_with("query")

def test_hybrid_retrieval_keyword_only(mocker, mock_vector_store):
    """Test hybrid retrieval when only keyword search returns results."""
    # Mock semantic search to return nothing
    mock_vector_store.similarity_search.return_value = []
    # Mock keyword search results
    mocker.patch.object(HybridRetriever, '_keyword_search', return_value=[doc_dk, doc_interstellar])

    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    results = retriever._get_relevant_documents("query", run_manager=mock_run_manager)

    assert len(results) == 2
    assert results == [doc_dk, doc_interstellar] # Keyword results are appended
    mock_vector_store.similarity_search.assert_called_once_with("query", k=TOP_K_RETRIEVAL)
    HybridRetriever._keyword_search.assert_called_once_with("query")

def test_hybrid_retrieval_combined_no_overlap(mocker, mock_vector_store):
    """Test combining distinct results from semantic and keyword searches."""
    mock_vector_store.similarity_search.return_value = [doc_inception, doc_matrix]
    mocker.patch.object(HybridRetriever, '_keyword_search', return_value=[doc_dk, doc_interstellar])

    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    results = retriever._get_relevant_documents("query", run_manager=mock_run_manager)

    # Combined: [Inception, Matrix, DK, Interstellar] -> Limited to TOP_K=3
    assert len(results) == TOP_K_RETRIEVAL # 3
    # The exact 3 depend on the order, but should be from the combined set
    result_titles = {doc.metadata["title"] for doc in results}
    assert result_titles.issubset({"Inception", "The Matrix", "The Dark Knight", "Interstellar"})
    assert len(result_titles) == 3

def test_hybrid_retrieval_combined_with_overlap(mocker, mock_vector_store):
    """Test combining results with overlap, ensuring deduplication."""
    mock_vector_store.similarity_search.return_value = [doc_inception, doc_matrix] # Semantic finds Inception, Matrix
    mocker.patch.object(HybridRetriever, '_keyword_search', return_value=[doc_dk, doc_inception]) # Keyword finds DK, Inception

    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    results = retriever._get_relevant_documents("query", run_manager=mock_run_manager)

    # Combined unique: [Inception, Matrix, DK]
    assert len(results) == 3
    result_titles = {doc.metadata["title"] for doc in results}
    assert result_titles == {"Inception", "The Matrix", "The Dark Knight"}
    # Check order (semantic results first)
    assert results[0] == doc_inception
    assert results[1] == doc_matrix
    assert results[2] == doc_dk

def test_hybrid_retrieval_combined_exceeds_limit(mocker, mock_vector_store):
    """Test that the combined results are correctly limited."""
    # Semantic returns TOP_K results
    semantic_results = [doc_inception, doc_dk, doc_interstellar]
    mock_vector_store.similarity_search.return_value = semantic_results
    # Keyword returns another different result
    keyword_results = [doc_matrix]
    mocker.patch.object(HybridRetriever, '_keyword_search', return_value=keyword_results)

    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    results = retriever._get_relevant_documents("query", run_manager=mock_run_manager)

    # Combined: [Inception, DK, Interstellar, Matrix] -> Limited to 3
    assert len(results) == TOP_K_RETRIEVAL # 3
    # Should contain only the semantic results as they came first
    assert results == semantic_results

def test_hybrid_retrieval_no_results(mocker, mock_vector_store):
    """Test when neither search method returns results."""
    mock_vector_store.similarity_search.return_value = []
    mocker.patch.object(HybridRetriever, '_keyword_search', return_value=[])

    retriever = HybridRetriever.model_construct(vector_store=mock_vector_store, films=SAMPLE_FILMS_DATA)
    results = retriever._get_relevant_documents("query", run_manager=mock_run_manager)

    assert len(results) == 0
    assert results == []
