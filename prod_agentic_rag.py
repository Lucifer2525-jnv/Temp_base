# agentic_rag.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

def create_basic_tools():
    """Create basic tools for the agent"""
    
    def safe_calculator(expression: str) -> str:
        """Safe calculator that handles basic math operations"""
        try:
            # Remove any potentially dangerous characters
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            # Use eval safely (in production, use a proper math parser)
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_current_time(query: str = "") -> str:
        """Get current time and date"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def search_knowledge(query: str) -> str:
        """Basic knowledge search (placeholder)"""
        # This is a placeholder - implement actual knowledge search here
        knowledge_base = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "fastapi": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            "database": "A database is an organized collection of structured information or data.",
        }
        
        query_lower = query.lower()
        for key, value in knowledge_base.items():
            if key in query_lower:
                return value
        
        return "I don't have specific information about that topic in my knowledge base."
    
    tools = [
        Tool(
            name="calculator",
            description="Useful for mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'",
            func=safe_calculator
        ),
        Tool(
            name="current_time",
            description="Get current time and date",
            func=get_current_time
        ),
        Tool(
            name="knowledge_search",
            description="Search for information in the knowledge base. Input should be a search query.",
            func=search_knowledge
        )
    ]
    
    return tools

def create_agent_prompt():
    """Create the agent prompt template"""
    template = """You are a helpful assistant with access to various tools. 
You can help users with calculations, provide current time, and search for information.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""
    
    return PromptTemplate.from_template(template)

def create_agent_executor(llm=None, memory=None):
    """Create an agent executor with tools and memory"""
    try:
        # Create LLM if not provided
        if llm is None:
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1000,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Create memory if not provided
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        
        # Create tools and prompt
        tools = create_basic_tools()
        prompt = create_agent_prompt()
        
        # Create agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False
        )
        
        logger.info("Successfully created agent executor")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Failed to create agent executor: {e}")
        raise

# Create default agent executor (this is what gets imported in main.py)
try:
    agent_executor = create_agent_executor()
    logger.info("Default agent executor created successfully")
except Exception as e:
    logger.error(f"Failed to create default agent executor: {e}")
    agent_executor = None

# Alternative function for creating RAG-enabled agent
def create_rag_agent(
    vector_store=None,
    retriever=None,
    llm=None,
    memory=None
):
    """
    Create a RAG-enabled agent (placeholder for future implementation)
    
    Args:
        vector_store: Vector store for document retrieval
        retriever: Document retriever
        llm: Language model
        memory: Conversation memory
        
    Returns:
        RAG-enabled agent executor
    """
    # This is a placeholder for RAG implementation
    # You can extend this to include document retrieval capabilities
    
    def rag_search(query: str) -> str:
        """RAG search function (placeholder)"""
        if vector_store and retriever:
            # Implement actual RAG search here
            docs = retriever.get_relevant_documents(query)
            if docs:
                return f"Found relevant information: {docs[0].page_content[:200]}..."
        
        return "RAG search not configured or no relevant documents found."
    
    # Create enhanced tools with RAG
    basic_tools = create_basic_tools()
    rag_tool = Tool(
        name="rag_search",
        description="Search for information in the document database using RAG",
        func=rag_search
    )
    
    tools = basic_tools + [rag_tool]
    
    # Create LLM if not provided
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
    
    # Create memory if not provided
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
    
    # Create enhanced prompt for RAG
    rag_template = """You are a helpful assistant with access to various tools including document search.
You can help users with calculations, provide current time, search for information, and retrieve relevant documents.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(rag_template)
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False
    )
    
    return agent_executor