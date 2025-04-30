import pytest
from unittest.mock import MagicMock, patch

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.schema import Document, AIMessage
from langchain.memory import ConversationBufferMemory

# Adjust the import path based on your project structure
from src.agent import FilmAgent
from src.retrieval import HybridRetriever
# Import recommendation function to mock it
import src.recommendation

# --- Sample Data ---
SAMPLE_FILMS_DATA = [
    {"title": "Inception", "year": 2010, "director": "Nolan", "genre": ["Sci-Fi"], "actors": ["DiCaprio"], "summary": "Dream heist."},
    {"title": "The Matrix", "year": 1999, "director": "Wachowskis", "genre": ["Sci-Fi"], "actors": ["Reeves"], "summary": "Reality simulation."}
]

# --- Fixtures ---
@pytest.fixture
def mock_retriever(mocker):
    """Fixture for a mock HybridRetriever."""
    # Use model_construct if HybridRetriever uses Pydantic v2 validation
    # If not, MagicMock() might suffice, but model_construct is safer
    mock = MagicMock(spec=HybridRetriever)
    mock.vector_store = MagicMock() # Mock the vector_store attribute needed by get_recommendations
    mock.get_relevant_documents = MagicMock(return_value=[])
    return mock

@pytest.fixture
def mock_llm(mocker):
    """Fixture for a mock ChatOpenAI."""
    mock = MagicMock(spec=ChatOpenAI)
    mock.invoke = MagicMock(return_value=AIMessage(content="LLM Response"))
    return mock

@pytest.fixture
def mock_memory(mocker):
    """Fixture for mock ConversationBufferMemory."""
    return MagicMock(spec=ConversationBufferMemory)

@pytest.fixture
def mock_agent_executor(mocker):
    """Fixture for mock AgentExecutor."""
    mock = MagicMock(spec=AgentExecutor)
    mock.invoke = MagicMock(return_value={"output": "Agent Response"})
    return mock

# --- Test Initialization ---
# Patch dependencies used within __init__ and _create_agent
@patch('src.agent.ChatOpenAI')
@patch('src.agent.ConversationBufferMemory')
@patch('src.agent.create_openai_tools_agent')
@patch('src.agent.AgentExecutor')
@patch('src.agent.ChatPromptTemplate')
@patch('src.agent.Tool')
def test_film_agent_init(
    mock_tool, mock_prompt, mock_executor, mock_create_tools_agent,
    mock_mem_init, mock_llm_init, mock_retriever
):
    """Test FilmAgent initialization and _create_agent calls."""
    # Mock the LLM and Memory instances returned by their constructors
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_init.return_value = mock_llm_instance
    mock_memory_instance = MagicMock(spec=ConversationBufferMemory)
    mock_mem_init.return_value = mock_memory_instance
    mock_executor_instance = MagicMock(spec=AgentExecutor)
    mock_executor.return_value = mock_executor_instance

    # Initialize the agent
    agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)

    # Assertions
    assert agent.retriever == mock_retriever
    assert agent.films == SAMPLE_FILMS_DATA
    mock_llm_init.assert_called_once() # Check LLM was initialized
    assert agent.llm == mock_llm_instance
    mock_mem_init.assert_called_once_with(memory_key="chat_history", return_messages=True)
    assert agent.memory == mock_memory_instance

    # Check if _create_agent components were called (indirectly via __init__)
    assert mock_tool.call_count == 2 # Two tools defined
    mock_prompt.from_messages.assert_called_once()
    mock_create_tools_agent.assert_called_once()
    mock_executor.assert_called_once()
    assert agent.agent_executor == mock_executor_instance

# --- Test _search_film Method ---
# We need to instantiate the agent, but mock its dependencies LLM and Retriever
@patch('src.agent.ChatOpenAI') # Mock the class used in __init__
@patch('src.agent.ConversationBufferMemory') # Mock the class used in __init__
def test_search_film_found(mock_mem_init, mock_llm_init, mock_retriever):
    """Test _search_film when documents are found."""
    # Setup mocks for this specific test
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_instance.invoke.return_value = AIMessage(content="Found film info.")
    mock_llm_init.return_value = mock_llm_instance # Return mock LLM when Agent initializes

    doc1 = Document(page_content="Info about Inception.", metadata={"title": "Inception"})
    mock_retriever.get_relevant_documents.return_value = [doc1]

    # Initialize agent (most internal creations are mocked by class patches)
    # We need a real agent instance to call the method on
    with patch.object(FilmAgent, '_create_agent', return_value=MagicMock()): # Prevent _create_agent running fully
        agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)
        agent.llm = mock_llm_instance # Ensure the instance uses our mock LLM

    # Call the method
    result = agent._search_film("info on inception")

    # Assertions
    mock_retriever.get_relevant_documents.assert_called_once_with("info on inception")
    mock_llm_instance.invoke.assert_called_once()
    assert "Info about Inception." in mock_llm_instance.invoke.call_args[0][0] # Check context in prompt
    assert result == "Found film info."

@patch('src.agent.ChatOpenAI')
@patch('src.agent.ConversationBufferMemory')
def test_search_film_not_found(mock_mem_init, mock_llm_init, mock_retriever):
    """Test _search_film when no documents are found."""
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_init.return_value = mock_llm_instance
    mock_retriever.get_relevant_documents.return_value = [] # No docs found

    with patch.object(FilmAgent, '_create_agent', return_value=MagicMock()):
        agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)
        agent.llm = mock_llm_instance

    result = agent._search_film("info on unknown film")

    mock_retriever.get_relevant_documents.assert_called_once_with("info on unknown film")
    mock_llm_instance.invoke.assert_not_called() # LLM shouldn't be called if no docs
    assert result == "I couldn't find any information about that film."

# --- Test _get_recommendations Method ---
@patch('src.agent.ChatOpenAI')
@patch('src.agent.ConversationBufferMemory')
@patch('src.agent.get_combined_recommendations') # Correct patch target
def test_get_recommendations_found(mock_get_combined, mock_mem_init, mock_llm_init, mock_retriever):
    """Test _get_recommendations when recommendations are found."""
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_instance.invoke.return_value = AIMessage(content="Here are recommendations.")
    mock_llm_init.return_value = mock_llm_instance

    recommended_films = [{"title": "The Matrix", "year": 1999, "director": "Wachowskis", "genre": ["Sci-Fi"], "summary": "..."}]
    mock_get_combined.return_value = recommended_films

    with patch.object(FilmAgent, '_create_agent', return_value=MagicMock()):
        agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)
        agent.llm = mock_llm_instance

    result = agent._get_recommendations("Inception")

    mock_get_combined.assert_called_once_with(
        film_title="Inception",
        films=SAMPLE_FILMS_DATA,
        vector_store=mock_retriever.vector_store # Check vector_store is passed
    )
    mock_llm_instance.invoke.assert_called_once()
    # Check that recommendation details are in the prompt sent to LLM
    assert "Title: The Matrix (1999)" in mock_llm_instance.invoke.call_args[0][0]
    assert result == "Here are recommendations."

@patch('src.agent.ChatOpenAI')
@patch('src.agent.ConversationBufferMemory')
@patch('src.agent.get_combined_recommendations') # Correct patch target
def test_get_recommendations_target_film_not_found(mock_get_combined, mock_mem_init, mock_llm_init, mock_retriever):
    """Test _get_recommendations when the target film is not in the list."""
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_init.return_value = mock_llm_instance

    with patch.object(FilmAgent, '_create_agent', return_value=MagicMock()):
        agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)
        agent.llm = mock_llm_instance

    result = agent._get_recommendations("Unknown Film")

    mock_get_combined.assert_not_called()
    mock_llm_instance.invoke.assert_not_called()
    assert result == "I couldn't find a film titled 'Unknown Film' in my database."

@patch('src.agent.ChatOpenAI')
@patch('src.agent.ConversationBufferMemory')
@patch('src.agent.get_combined_recommendations') # Correct patch target
def test_get_recommendations_no_similar_found(mock_get_combined, mock_mem_init, mock_llm_init, mock_retriever):
    """Test _get_recommendations when get_combined_recommendations returns empty."""
    mock_llm_instance = MagicMock(spec=ChatOpenAI)
    mock_llm_init.return_value = mock_llm_instance
    mock_get_combined.return_value = [] # No recommendations found

    with patch.object(FilmAgent, '_create_agent', return_value=MagicMock()):
        agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)
        agent.llm = mock_llm_instance

    result = agent._get_recommendations("Inception")

    mock_get_combined.assert_called_once()
    mock_llm_instance.invoke.assert_not_called() # LLM not called if no recs
    assert result == "I couldn't find any similar films to 'Inception'."

# --- Test process_query Method (Optional - more integration like) ---
# This test mocks the agent executor directly
@patch('src.agent.ChatOpenAI')
@patch('src.agent.ConversationBufferMemory')
@patch('src.agent.AgentExecutor') # Patch the AgentExecutor used by the agent
def test_process_query(mock_executor_init, mock_mem_init, mock_llm_init, mock_retriever):
    """Test the main process_query method by mocking the executor."""
    mock_llm_init.return_value = MagicMock(spec=ChatOpenAI)
    mock_mem_init.return_value = MagicMock(spec=ConversationBufferMemory)

    # Mock the executor instance that _create_agent would return
    mock_executor_instance = MagicMock(spec=AgentExecutor)
    mock_executor_instance.invoke.return_value = {"output": "Final Agent Response"}
    # Patch _create_agent to return our mock executor instance
    with patch.object(FilmAgent, '_create_agent', return_value=mock_executor_instance):
        agent = FilmAgent(retriever=mock_retriever, films=SAMPLE_FILMS_DATA)

    # Call process_query
    response = agent.process_query("some query")

    # Assert that the agent executor's invoke method was called
    mock_executor_instance.invoke.assert_called_once_with({"input": "some query"})
    assert response == "Final Agent Response"
