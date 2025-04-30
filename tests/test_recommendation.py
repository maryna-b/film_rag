import pytest
from unittest.mock import MagicMock, patch

# Adjust the import path based on your project structure
from src.recommendation import (
    get_similar_films,
    get_recommendations_by_genre,
    get_recommendations_by_director,
    get_combined_recommendations
)
# Import config values (adjust path if necessary)
from config import SIMILARITY_THRESHOLD, MAX_RECOMMENDATIONS

# --- Sample Data ---
# Using a slightly expanded dataset for better testing scenarios
SAMPLE_FILMS_DATA = [
    {
        "title": "Inception", "year": 2010, "director": "Christopher Nolan",
        "genre": ["Sci-Fi", "Action", "Thriller"], "actors": ["Leonardo DiCaprio"],
        "summary": "Dream heist."
    },
    {
        "title": "The Dark Knight", "year": 2008, "director": "Christopher Nolan",
        "genre": ["Action", "Crime", "Drama"], "actors": ["Christian Bale"],
        "summary": "Batman vs Joker."
    },
    {
        "title": "Interstellar", "year": 2014, "director": "Christopher Nolan",
        "genre": ["Sci-Fi", "Drama", "Adventure"], "actors": ["Matthew McConaughey"],
        "summary": "Space travel."
    },
    {
        "title": "The Matrix", "year": 1999, "director": "Wachowskis",
        "genre": ["Action", "Sci-Fi"], "actors": ["Keanu Reeves"],
        "summary": "Reality simulation."
    },
    {
        "title": "Pulp Fiction", "year": 1994, "director": "Quentin Tarantino",
        "genre": ["Crime", "Drama"], "actors": ["John Travolta"],
        "summary": "Non-linear crime stories."
    },
    {
        "title": "Fight Club", "year": 1999, "director": "David Fincher",
        "genre": ["Drama", "Thriller"], "actors": ["Brad Pitt"],
        "summary": "Underground fight clubs."
    }
]

# --- Mock Document Class ---
# Need a simple mock for Langchain Document used by Chroma
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# --- Tests for get_recommendations_by_genre ---

def test_get_recommendations_by_genre_found():
    """Test finding recommendations by genre."""
    # Target: Inception (Sci-Fi, Action, Thriller)
    # Expected: The Matrix (Action, Sci-Fi) - High overlap
    #           The Dark Knight (Action, Crime, Drama) - Some overlap
    #           Interstellar (Sci-Fi, Drama, Adventure) - Some overlap
    #           Fight Club (Drama, Thriller) - Some overlap
    recs = get_recommendations_by_genre("Inception", SAMPLE_FILMS_DATA)
    rec_titles = [film["title"] for film in recs]

    assert len(recs) <= MAX_RECOMMENDATIONS # Should be 3
    assert "The Matrix" in rec_titles
    # The exact order depends on Jaccard similarity calculation, check presence
    assert "The Dark Knight" in rec_titles or "Interstellar" in rec_titles or "Fight Club" in rec_titles
    assert "Pulp Fiction" not in rec_titles # Low overlap

def test_get_recommendations_by_genre_no_overlap():
    """Test when no films share genres."""
    single_genre_films = [
        {"title": "Film A", "year": 2000, "director": "Dir A", "genre": ["UniqueA"], "actors": [], "summary": ""},
        {"title": "Film B", "year": 2001, "director": "Dir B", "genre": ["UniqueB"], "actors": [], "summary": ""},
    ]
    recs = get_recommendations_by_genre("Film A", single_genre_films)
    assert recs == []

def test_get_recommendations_by_genre_target_not_found():
    """Test when the target film title doesn't exist."""
    recs = get_recommendations_by_genre("NonExistent Film", SAMPLE_FILMS_DATA)
    assert recs == []

# --- Tests for get_recommendations_by_director ---

def test_get_recommendations_by_director_found():
    """Test finding recommendations by the same director."""
    # Target: Inception (Christopher Nolan)
    # Expected: The Dark Knight, Interstellar
    recs = get_recommendations_by_director("Inception", SAMPLE_FILMS_DATA)
    rec_titles = [film["title"] for film in recs]

    assert len(recs) == 2 # Less than MAX_RECOMMENDATIONS
    assert "The Dark Knight" in rec_titles
    assert "Interstellar" in rec_titles
    assert "Inception" not in rec_titles # Should exclude target
    assert "The Matrix" not in rec_titles

def test_get_recommendations_by_director_only_one_film():
    """Test when the director only has one film in the list."""
    recs = get_recommendations_by_director("Pulp Fiction", SAMPLE_FILMS_DATA)
    assert recs == []

def test_get_recommendations_by_director_target_not_found():
    """Test when the target film title doesn't exist."""
    recs = get_recommendations_by_director("NonExistent Film", SAMPLE_FILMS_DATA)
    assert recs == []

# --- Tests for get_similar_films ---

@pytest.fixture
def mock_vector_store(mocker):
    """Fixture to create a mock Chroma vector store."""
    mock_store = MagicMock()

    # Define mock results for similarity search based on query (film summary)
    def mock_similarity_search(query, k):
        # Simulate results for "Inception" query ("Dream heist.")
        if "Dream heist" in query:
            # Higher similarity (lower distance score) for Sci-Fi/Action
            return [
                (MockDocument("Reality simulation.", {"title": "The Matrix"}), 0.1), # High similarity
                (MockDocument("Space travel.", {"title": "Interstellar"}), 0.2),     # Medium similarity
                (MockDocument("Batman vs Joker.", {"title": "The Dark Knight"}), 0.3), # Lower similarity
                (MockDocument("Dream heist.", {"title": "Inception"}), 0.0),         # Target film itself
                (MockDocument("Non-linear crime stories.", {"title": "Pulp Fiction"}), 0.8), # Low similarity
                (MockDocument("Underground fight clubs.", {"title": "Fight Club"}), 0.7), # Low similarity
            ]
        # Add more query simulations if needed for other tests
        return []

    mock_store.similarity_search_with_score = MagicMock(side_effect=mock_similarity_search)
    return mock_store

def test_get_similar_films_found(mock_vector_store):
    """Test finding similar films using mocked vector store."""
    # Target: Inception
    # Mock search should return Matrix, Interstellar, Dark Knight above threshold
    recs = get_similar_films(mock_vector_store, "Inception", SAMPLE_FILMS_DATA)
    rec_titles = [film["title"] for film in recs]

    # Similarity = 1.0 - score. Threshold = 0.75
    # Matrix: 1.0 - 0.1 = 0.9 (Keep)
    # Interstellar: 1.0 - 0.2 = 0.8 (Keep)
    # Dark Knight: 1.0 - 0.3 = 0.7 (Discard - below threshold)
    # Pulp Fiction: 1.0 - 0.8 = 0.2 (Discard)
    # Fight Club: 1.0 - 0.7 = 0.3 (Discard)

    assert len(recs) == 2 # Matrix, Interstellar (DK below threshold)
    assert "The Matrix" in rec_titles
    assert "Interstellar" in rec_titles
    assert "The Dark Knight" not in rec_titles
    assert "Inception" not in rec_titles # Excludes target

    # Check call to mock
    mock_vector_store.similarity_search_with_score.assert_called_once_with(
        query="Dream heist.", k=len(SAMPLE_FILMS_DATA)
    )

def test_get_similar_films_target_not_found(mock_vector_store):
    """Test get_similar_films when the target film isn't in the films list."""
    recs = get_similar_films(mock_vector_store, "NonExistent Film", SAMPLE_FILMS_DATA)
    assert recs == []
    mock_vector_store.similarity_search_with_score.assert_not_called()

def test_get_similar_films_no_results_above_threshold(mocker):
    """Test when similarity search returns results but none meet the threshold."""
    mock_store = MagicMock()
    # Return scores that result in similarity below threshold (0.75)
    mock_results = [
        (MockDocument("", {"title": "The Matrix"}), 0.3), # Similarity 0.7
        (MockDocument("", {"title": "Interstellar"}), 0.4), # Similarity 0.6
    ]
    mock_store.similarity_search_with_score = MagicMock(return_value=mock_results)

    recs = get_similar_films(mock_store, "Inception", SAMPLE_FILMS_DATA)
    assert recs == []

# --- Tests for get_combined_recommendations ---

# Use patch to mock the helper functions directly for combined test
@patch('src.recommendation.get_similar_films')
@patch('src.recommendation.get_recommendations_by_genre')
@patch('src.recommendation.get_recommendations_by_director')
def test_get_combined_recommendations_all_methods(
    mock_get_director, mock_get_genre, mock_get_similar, mock_vector_store # mock_vector_store is unused here but needed by signature
):
    """Test combining recommendations from all methods."""
    # Mock return values for helper functions
    # Use film objects directly
    inception = SAMPLE_FILMS_DATA[0]
    dark_knight = SAMPLE_FILMS_DATA[1]
    interstellar = SAMPLE_FILMS_DATA[2]
    matrix = SAMPLE_FILMS_DATA[3]
    pulp_fiction = SAMPLE_FILMS_DATA[4]

    mock_get_similar.return_value = [matrix, interstellar] # Vector: Matrix (2*3=6), Interstellar (1*3=3)
    mock_get_genre.return_value = [matrix, dark_knight, interstellar] # Genre: Matrix (3*2=6), DK (2*2=4), Interstellar (1*2=2)
    mock_get_director.return_value = [dark_knight, interstellar] # Director: DK (2*1=2), Interstellar (1*1=1)

    # Expected scores:
    # Matrix: 6 + 6 = 12
    # Interstellar: 3 + 2 + 1 = 6
    # Dark Knight: 4 + 2 = 6

    # Expected order (desc score): Matrix, Interstellar/Dark Knight (tie), limited to MAX_RECOMMENDATIONS=3
    recs = get_combined_recommendations("Inception", SAMPLE_FILMS_DATA, mock_vector_store) # Pass mock store, though it won't be used by patched functions
    rec_titles = [film["title"] for film in recs]

    assert len(recs) == MAX_RECOMMENDATIONS # Should be 3
    assert rec_titles[0] == "The Matrix"
    # Order of tied scores might vary, check presence
    assert "Interstellar" in rec_titles
    assert "The Dark Knight" in rec_titles

    # Check mocks were called
    mock_get_similar.assert_called_once()
    mock_get_genre.assert_called_once()
    mock_get_director.assert_called_once()

@patch('src.recommendation.get_similar_films', return_value=[])
@patch('src.recommendation.get_recommendations_by_genre', return_value=[])
@patch('src.recommendation.get_recommendations_by_director', return_value=[])
def test_get_combined_recommendations_no_results(
    mock_get_director, mock_get_genre, mock_get_similar, mock_vector_store
):
    """Test combined recommendations when no individual method returns results."""
    recs = get_combined_recommendations("Inception", SAMPLE_FILMS_DATA, mock_vector_store)
    assert recs == []
