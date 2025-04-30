import json
import pytest
from unittest.mock import mock_open, patch

# Adjust the import path based on your project structure
# This assumes 'tests' is at the same level as 'src'
from src.data_processing import (
    load_film_database,
    prepare_documents,
    get_film_by_title
)

# --- Sample Data ---
SAMPLE_FILMS_DATA = [
    {
        "title": "Inception",
        "year": 2010,
        "director": "Christopher Nolan",
        "genre": ["Sci-Fi", "Action", "Thriller"],
        "actors": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "summary": "A thief who steals corporate secrets through the use of dream-sharing technology..."
    },
    {
        "title": "The Matrix",
        "year": 1999,
        "director": "Lana Wachowski, Lilly Wachowski",
        "genre": ["Action", "Sci-Fi"],
        "actors": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
        "summary": "A computer hacker learns from mysterious rebels about the true nature of his reality..."
    }
]

SAMPLE_FILMS_JSON = json.dumps(SAMPLE_FILMS_DATA)

# --- Tests for load_film_database ---

def test_load_film_database_success(mocker):
    """Test loading a valid JSON file."""
    mock_file = mock_open(read_data=SAMPLE_FILMS_JSON)
    mocker.patch("builtins.open", mock_file)
    mocker.patch("json.load", return_value=SAMPLE_FILMS_DATA)

    films = load_film_database("dummy/path/films.json")

    assert films == SAMPLE_FILMS_DATA
    mock_file.assert_called_once_with("dummy/path/films.json", 'r', encoding='utf-8')
    json.load.assert_called_once()

def test_load_film_database_file_not_found(mocker, capsys):
    """Test handling FileNotFoundError."""
    mocker.patch("builtins.open", side_effect=FileNotFoundError("File not found"))

    films = load_film_database("nonexistent/path.json")

    assert films == []
    captured = capsys.readouterr()
    assert "Error: File not found at nonexistent/path.json" in captured.out

def test_load_film_database_json_decode_error(mocker, capsys):
    """Test handling JSONDecodeError."""
    mock_file = mock_open(read_data="invalid json")
    mocker.patch("builtins.open", mock_file)
    mocker.patch("json.load", side_effect=json.JSONDecodeError("Expecting value", "invalid json", 0))

    films = load_film_database("dummy/invalid.json")

    assert films == []
    captured = capsys.readouterr()
    assert "Error: Invalid JSON format in dummy/invalid.json" in captured.out

# --- Tests for prepare_documents ---

def test_prepare_documents_multiple_films():
    """Test preparing documents for multiple films."""
    documents = prepare_documents(SAMPLE_FILMS_DATA)

    assert len(documents) == 2

    # Check first document
    assert documents[0]["id"] == "film_inception"
    assert documents[0]["title"] == "Inception"
    assert "Title: Inception" in documents[0]["content"]
    assert "Year: 2010" in documents[0]["content"]
    assert "Director: Christopher Nolan" in documents[0]["content"]
    assert "Genre: Sci-Fi, Action, Thriller" in documents[0]["content"]
    assert "Actors: Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page" in documents[0]["content"]
    assert "Summary: A thief who steals corporate secrets" in documents[0]["content"]
    assert documents[0]["metadata"]["title"] == "Inception"
    assert documents[0]["metadata"]["year"] == 2010
    assert documents[0]["metadata"]["director"] == "Christopher Nolan"
    assert documents[0]["metadata"]["genre"] == "Sci-Fi, Action, Thriller"
    assert documents[0]["metadata"]["actors"] == "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page"

    # Check second document
    assert documents[1]["id"] == "film_the_matrix"
    assert documents[1]["title"] == "The Matrix"
    assert "Title: The Matrix" in documents[1]["content"]
    assert documents[1]["metadata"]["title"] == "The Matrix"

def test_prepare_documents_single_film():
    """Test preparing documents for a single film."""
    single_film_list = [SAMPLE_FILMS_DATA[0]]
    documents = prepare_documents(single_film_list)

    assert len(documents) == 1
    assert documents[0]["id"] == "film_inception"
    assert documents[0]["title"] == "Inception"

def test_prepare_documents_empty_list():
    """Test preparing documents for an empty list."""
    documents = prepare_documents([])
    assert documents == []

# --- Tests for get_film_by_title ---

def test_get_film_by_title_found_exact_case():
    """Test finding a film with exact title case."""
    film = get_film_by_title(SAMPLE_FILMS_DATA, "Inception")
    assert film == SAMPLE_FILMS_DATA[0]

def test_get_film_by_title_found_different_case():
    """Test finding a film with different title case."""
    film = get_film_by_title(SAMPLE_FILMS_DATA, "the matrix")
    assert film == SAMPLE_FILMS_DATA[1]

def test_get_film_by_title_not_found():
    """Test when the film title is not found."""
    film = get_film_by_title(SAMPLE_FILMS_DATA, "NonExistent Film")
    assert film == {}

def test_get_film_by_title_empty_list():
    """Test searching in an empty list of films."""
    film = get_film_by_title([], "Inception")
    assert film == {}
