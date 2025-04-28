"""
Agent module for the Film RAG system.
"""
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL
from src.retrieval import HybridRetriever
from src.recommendation import get_combined_recommendations

class FilmAgent:
    """
    Agent for handling film-related queries.
    """
    
    def __init__(self, retriever: HybridRetriever, films: List[Dict[str, Any]]):
        """
        Initialize the film agent.
        
        Args:
            retriever: Hybrid retriever for film information.
            films: List of film dictionaries.
        """
        self.retriever = retriever
        self.films = films
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create a LangChain agent with tools.
        
        Returns:
            AgentExecutor instance.
        """
        # Define tools
        tools = [
            Tool(
                name="film_search",
                func=self._search_film,
                description="Search for information about a specific film. Input should be the film title or a query about a film."
            ),
            Tool(
                name="film_recommendations",
                func=self._get_recommendations,
                description="Get film recommendations similar to a given film. Input should be the film title."
            )
        ]
        
        # Define system message
        system_message = """You are a helpful film information assistant. You can provide information about films and recommend similar films.
        
When asked about a film, use the film_search tool to find information about it.
When asked for recommendations, use the film_recommendations tool to suggest similar films.

Always be concise and helpful. If you don't know about a film, admit that you don't have information about it.

For film information, include:
- Title, year, and director
- Genre
- Main actors
- Plot summary

For recommendations, explain why each film is recommended (e.g., similar genre, same director, similar themes).
"""
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _search_film(self, query: str) -> str:
        """
        Search for film information.
        
        Args:
            query: User query about a film.
            
        Returns:
            String response with film information.
        """
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        if not docs:
            return "I couldn't find any information about that film."
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt for the LLM
        prompt = f"""
Based on the following information about films, answer the query: "{query}"

Film Information:
{context}

If the information doesn't contain details about the specific film being asked about, say that you don't have information about that film.
"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return response.content
    
    def _get_recommendations(self, film_title: str) -> str:
        """
        Get film recommendations.
        
        Args:
            film_title: Title of the film to get recommendations for.
            
        Returns:
            String response with film recommendations.
        """
        # Check if the film exists
        film_exists = False
        for film in self.films:
            if film["title"].lower() == film_title.lower():
                film_exists = True
                break
        
        if not film_exists:
            return f"I couldn't find a film titled '{film_title}' in my database."
        
        # Get recommendations
        recommendations = get_combined_recommendations(
            film_title=film_title,
            films=self.films,
            vector_store=self.retriever.vector_store
        )
        
        if not recommendations:
            return f"I couldn't find any similar films to '{film_title}'."
        
        # Format recommendations
        recs_text = "\n\n".join([
            f"Title: {film['title']} ({film['year']})\n"
            f"Director: {film['director']}\n"
            f"Genre: {', '.join(film['genre'])}\n"
            f"Summary: {film['summary']}"
            for film in recommendations
        ])
        
        # Create prompt for the LLM
        prompt = f"""
Based on the film "{film_title}", here are some recommended similar films:

{recs_text}

Provide these recommendations to the user, explaining why each film might be similar to "{film_title}" (e.g., similar genre, same director, similar themes).
Format your response in a clear, readable way.
"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return response.content
    
    def process_query(self, query: str) -> str:
        """
        Process a user query.
        
        Args:
            query: User query.
            
        Returns:
            Agent response.
        """
        return self.agent_executor.invoke({"input": query})["output"]
