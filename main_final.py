import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from uuid import uuid4
import os
import requests

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from typing import List, Any

import sys
sys.path.append(os.path.dirname(__file__))

from prompt_templates import *

APPINSIGHTS_CONNECTION_STRING = os.getenv("App_Insight_Conn_String")


# Configure basic logging first (before any Azure imports)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger("app_insights_logger")

# Azure App Insights - with proper error handling
if APPINSIGHTS_CONNECTION_STRING:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        from opencensus.ext.azure.trace_exporter import AzureExporter
        from opencensus.ext.fastapi.fastapi_middleware import FastAPIMiddleware
        from opencensus.trace.samplers import ProbabilitySampler
        from app_insights_callback import AppInsightsHandler

        if AppInsightsHandler is None:
            raise ImportError("AppInsightsHandler could not be imported")
        
        logger.info(f"AppInsightsHandler imported Successfully.....!!!!!!!:{ type(AppInsightsHandler)}")
        # Add Azure handler
        azure_handler = AzureLogHandler(connection_string=os.getenv("App_Insight_Conn_String"))
        logger.addHandler(azure_handler)
        logger.info("Azure Application Insights logging enabled")
        AZURE_ENABLED=True

    except Exception as e:
        logger.error(f"Failed to initialize Azure Application Insights: {e}")
        logger.info("Continuing without Azure Application Insights")
        AZURE_ENABLED = False
        # AppInsightsHandler = None
else:
    logger.warning("App_Insight_Conn_String not set. Azure Application Insights disabled.")
    AZURE_ENABLED = False
    # AppInsightsHandler = None

# Function to create callbacks list
def create_callbacks() -> List[Any]:
    """Create callbacks list for LangChain operations"""
    callbacks = []
    
    if AZURE_ENABLED and AppInsightsHandler:
        try:
            # Create a new instance of AppInsightsHandler
            app_insights_callback = AppInsightsHandler()
            callbacks.append(app_insights_callback)
            logger.info("AppInsights callback added successfully")
        except Exception as e:
            logger.error(f"Failed to create AppInsights callback: {e}")
            logger.info("Continuing without AppInsights callback")
    
    return callbacks

# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_sessions_task())
    queue_task = asyncio.create_task(process_queue_task())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    cleanup_task.cancel()
    queue_task.cancel()


app = FastAPI(
    title="GSC ARB Chatbot API",
    description="GSC ARB chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# Add Azure middleware only if enabled
if AZURE_ENABLED:
    try:
        app.add_middleware(
            FastAPIMiddleware,
            exporter=AzureExporter(connection_string=APPINSIGHTS_CONNECTION_STRING),
            sampler=ProbabilitySampler(1.0)
        )
        logger.info("Azure Application Insights middleware enabled")
    except Exception as e:
        logger.error(f"Failed to add Azure middleware: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from utils import create_chat_openai_client, load_vectorstore
    from db_utils import *
    from auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user, get_sso_auth_url, get_sso_token, get_sso_user_info, create_user_token
    from models import ChatHistory
    # from agentic_rag import agent_executor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all required modules exist: utils.py, db_utils.py, auth.py, models.py, agentic_rag.py")

# Create LLM client with error handling
try:
    llm_client = create_chat_openai_client()
    logger.info("LLM client created successfully")
except Exception as e:
    logger.error(f"Failed to create LLM client: {e}")
    # We can use other LLMs here
    # llm_client = our other llms client
    llm_client = None

# Configuration
@dataclass
class AppConfig:
    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Connection pooling
    max_concurrent_requests: int = 10
    connection_pool_size: int = 20
    
    # Session management
    session_timeout_minutes: int = 30
    max_sessions: int = 1000
    
    # LLM settings
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # Queue settings
    queue_timeout_seconds: int = 30
    max_queue_size: int = 100

config = AppConfig()

# Rate Limiting
class RateLimiter:
    def __init__(self):
        self._requests = defaultdict(lambda: {"minute": deque(), "hour": deque()})
        self._lock = threading.Lock()
    
    def is_allowed(self, session_id: str) -> bool:
        """Check if request is allowed based on rate limits"""
        with self._lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            
            # Clean old requests
            user_requests = self._requests[session_id]
            
            # Remove requests older than 1 minute
            while user_requests["minute"] and user_requests["minute"][0] < minute_ago:
                user_requests["minute"].popleft()
            
            # Remove requests older than 1 hour
            while user_requests["hour"] and user_requests["hour"][0] < hour_ago:
                user_requests["hour"].popleft()
            
            # Check limits
            if len(user_requests["minute"]) >= config.max_requests_per_minute:
                return False
            
            if len(user_requests["hour"]) >= config.max_requests_per_hour:
                return False
            
            # Add current request
            user_requests["minute"].append(now)
            user_requests["hour"].append(now)
            
            return True
    
    def get_reset_time(self, session_id: str) -> Dict[str, datetime]:
        """Get when rate limits reset"""
        with self._lock:
            user_requests = self._requests[session_id]
            reset_times = {}
            
            if user_requests["minute"]:
                reset_times["minute"] = user_requests["minute"][0] + timedelta(minutes=1)
            
            if user_requests["hour"]:
                reset_times["hour"] = user_requests["hour"][0] + timedelta(hours=1)
            
            return reset_times

# Connection Pool Manager
class ConnectionPoolManager:
    def __init__(self):
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._active_connections = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a connection from the pool"""
        await self._semaphore.acquire()
        async with self._lock:
            self._active_connections += 1
            logger.info(f"Connection acquired. Active: {self._active_connections}")
    
    async def release(self):
        """Release a connection back to the pool"""
        self._semaphore.release()
        asyncio.create_task(self._update_active_count())
    
    async def _update_active_count(self):
        async with self._lock:
            self._active_connections -= 1
            logger.info(f"Connection released. Active: {self._active_connections}")
    
    @property
    def active_connections(self) -> int:
        return self._active_connections

# Request Queue Manager
class RequestQueue:
    def __init__(self):
        self._queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._processing = False
        self._lock = asyncio.Lock()
    
    async def add_request(self, request_data: Dict[str, Any]) -> str:
        """Add request to queue and return request ID"""
        request_id = str(uuid4())
        request_data["request_id"] = request_id
        request_data["timestamp"] = datetime.now()
        
        try:
            await asyncio.wait_for(
                self._queue.put(request_data),
                timeout=config.queue_timeout_seconds
            )
            logger.info(f"Request {request_id} added to queue")
            return request_id
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Request queue is full")
    
    async def get_request(self) -> Optional[Dict[str, Any]]:
        """Get next request from queue"""
        try:
            return await self._queue.get()
        except asyncio.QueueEmpty:
            return None
    
    @property
    def size(self) -> int:
        return self._queue.qsize()

# Session Manager with updated memory handling
@dataclass
class SessionData:
    session_id: str
    chat_history: ChatMessageHistory
    agent_executor: Optional[AgentExecutor] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def _create_session_internal(self, session_id: str) -> SessionData:
        """Internal method to create session - assumes lock is already held"""
        if len(self._sessions) >= config.max_sessions:
            # Remove oldest session
            oldest_session = min(
                self._sessions.values(),
                key=lambda s: s.last_accessed
            )
            del self._sessions[oldest_session.session_id]
            logger.info(f"Removed oldest session: {oldest_session.session_id}")
        
        # Creating new session with updated memory
        chat_history = ChatMessageHistory()
        
        session_data = SessionData(
            session_id=session_id,
            chat_history=chat_history
        )
        
        # Create agent executor with isolated memory
        session_data.agent_executor = self._create_agent_executor(chat_history)
        
        self._sessions[session_id] = session_data
        logger.info(f"Created new session: {session_id}")
        
        return session_data
    
    async def create_session(self, session_id: str) -> SessionData:
        """Public method to create a new session - acquires lock"""
        async with self._lock:
            return await self._create_session_internal(session_id)
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get existing session or create new one"""
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now()
                return session
            else:
                return await self._create_session_internal(session_id)
    
    def _create_agent_executor(self, chat_history: ChatMessageHistory) -> AgentExecutor:
        """Create agent executor with tools and memory"""
        if not llm_client:
            raise Exception("LLM client not available")
            
        # LLM with connection pooling
        from utils import get_embedding_model
        from langchain.vectorstores import FAISS
        from config import get_vectorstore_backend
        from openai_services import get_embedding_model
        from langchain.schema import Document as LCDocument
        from langchain.tools import Tool


        llm = llm_client
        
        def format_azure_search_results(docs):
            """Format Azure AI Search results with metadata"""
            formatted = []
            for doc in docs:
                metadata = doc.metadata
                content = doc.page_content.strip()
                title = metadata.get("title", "Untitled Document")
                url = metadata.get("page_url", "No URL")
                search_score = metadata.get("search_score", "N/A")
                document_id = metadata.get("document_id", "Unknown")
                
                formatted_content = f"""🔹 *Title*: {title}
*URL*: {url}
*Document ID*: {document_id}
*Search Score*: {search_score}
*Content*:
{content}
"""
                formatted.append(LCDocument(page_content=formatted_content, metadata=metadata))
            return formatted

        def azure_ai_search_tool(query: str) -> str:
            """Azure AI Search tool function"""
            try:
                vectorstore = get_vectorstore_backend()
                embedding_model = get_embedding_model()
                
                # Get query embedding
                query_embedding = embedding_model.embed_query(query)
                
                # Perform hybrid search
                results = vectorstore.hybrid_search(
                    query=query,
                    embedding=query_embedding,
                    top_k=5
                )
                
                # Convert results to LangChain Documents
                retrieved_docs = []
                for result in results:
                    retrieved_docs.append(LCDocument(
                        page_content=result.get('content', ''),
                        metadata={
                            "title": result.get('title', ''),
                            "page_url": result.get('page_url', ''),
                            "document_id": result.get('document_id', ''),
                            "chunk_index": result.get('chunk_index', ''),
                            "search_score": result.get('@search.score', 0)
                        }
                    ))
                
                if not retrieved_docs:
                    return "No relevant documents found in Azure AI Search."
                
                # Format the results
                formatted_docs = format_azure_search_results(retrieved_docs)
                return "\n\n---\n\n".join([doc.page_content for doc in formatted_docs])
                
            except Exception as e:
                logger.error(f"Error in Azure AI Search tool: {str(e)}")
                return f"Error performing Azure AI Search: {str(e)}"

        def azure_ai_filtered_search_tool(query_and_filters: str) -> str:
            """Azure AI Search tool with filtering capability"""
            try:
                # Parse the input to extract query and potential filters
                # Expected format: "query|document_id:value|title_filter:value"
                parts = query_and_filters.split("|")
                query = parts[0].strip()
                
                document_id = None
                title_filter = None
                
                for part in parts[1:]:
                    if ":" in part:
                        key, value = part.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == "document_id":
                            document_id = value
                        elif key == "title_filter":
                            title_filter = value
                
                vectorstore = get_vectorstore_backend()
                embedding_model = get_embedding_model()
                
                # Build filter string
                filters = []
                if document_id:
                    filters.append(f"document_id eq '{document_id}'")
                if title_filter:
                    filters.append(f"search.ismatch('{title_filter}', 'title')")
                filter_str = " and ".join(filters) if filters else None
                
                # Get query embedding
                query_embedding = embedding_model.embed_query(query)
                
                # Perform filtered search
                results = vectorstore.hybrid_search(
                    query=query,
                    embedding=query_embedding,
                    filter=filter_str,
                    top_k=5
                )
                
                # Convert results to LangChain Documents
                retrieved_docs = []
                for result in results:
                    retrieved_docs.append(LCDocument(
                        page_content=result.get('content', ''),
                        metadata={
                            "title": result.get('title', ''),
                            "page_url": result.get('page_url', ''),
                            "document_id": result.get('document_id', ''),
                            "chunk_index": result.get('chunk_index', ''),
                            "search_score": result.get('@search.score', 0)
                        }
                    ))
                
                if not retrieved_docs:
                    return f"No relevant documents found with filters: {filter_str}"
                
                # Format the results
                formatted_docs = format_azure_search_results(retrieved_docs)
                filter_info = f"Applied filters: {filter_str}\n\n" if filter_str else ""
                return filter_info + "\n\n---\n\n".join([doc.page_content for doc in formatted_docs])
                
            except Exception as e:
                logger.error(f"Error in Azure AI filtered search tool: {str(e)}")
                return f"Error performing Azure AI filtered search: {str(e)}"

        # Create the tools
        azure_search_tool = Tool(
            name="azure_ai_search",
            func=azure_ai_search_tool,
            description="Use this to search the Azure AI Search index for relevant Confluence documentation. Input should be a search query string. Returns relevant content with titles, URLs, and search scores."
        )
        
        azure_filtered_search_tool = Tool(
            name="azure_ai_filtered_search",
            func=azure_ai_filtered_search_tool,
            description="Use this to search Azure AI Search with filters. Input format: 'query|document_id:value|title_filter:value'. Example: 'ARB process|document_id:383516798' or 'governance|title_filter:architecture'. Returns filtered results."
        )

        # Fallback FAISS tool
        def fallback_confluence_search_tool(query: str):
            """Fallback FAISS search tool"""
            try:
                from utils import get_embedding_model
                from langchain.vectorstores import FAISS
                
                DB_FAISS_PATH = "./FINAL_VECTOR_DB/"
                embedding_model = get_embedding_model()
                retriever = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 5})
                
                docs = retriever.get_relevant_documents(query)
                formatted_docs = format_azure_search_results(docs)
                return "\n\n---\n\n".join([doc.page_content for doc in formatted_docs])
            except Exception as e:
                logger.error(f"Error in fallback FAISS search: {str(e)}")
                return f"Error performing fallback search: {str(e)}"

        fallback_tool = Tool(
            name="fallback_confluence_search",
            func=fallback_confluence_search_tool,
            description="Fallback search tool using FAISS when Azure AI Search is unavailable. Use only when azure_ai_search fails."
        )

        # Define the tools list (prioritize Azure AI Search)
        tools = [
            azure_search_tool,
            azure_filtered_search_tool,
            fallback_tool
        ]

        from langchain.prompts import PromptTemplate
        template = """You are an intelligent ARB(Architecture Review Board) assistant of GSK GSC(Global Supply Chain) that answers questions by always using internal Confluence documentation through Azure AI Search.
You can think step-by-step, decide how to use tools, and always explain your reasoning.

---
Available Tools:
You have access to these search tools (use in order of preference):
{tools}

• **azure_ai_search**: Primary search tool using Azure AI Search for Confluence documentation. Provides hybrid search with relevance scores.
• **azure_ai_filtered_search**: Use when you need to filter results by document ID or title. Input format: 'query|document_id:value|title_filter:value'
• **fallback_confluence_search**: Backup search using FAISS. Use only if Azure AI Search fails.

---
Tool Usage Guidelines:
1. Always start with azure_ai_search for general queries
2. Use azure_ai_filtered_search when you need specific documents or want to filter by title
3. Only use fallback_confluence_search if Azure tools fail
4. For follow-up questions in the same conversation, consider using filtered search if you know the relevant document ID

---
When a user submits a query:

1. Carefully interpret the request — especially for ARB processes, artefacts, governance workflows, lifecycle stages, or review board interactions.
2. Never respond with any generic information.
3. If query seems unclear then politely ask user to give more keywords or details regarding query.
4. If query seems harmful or irrelevant then politely deny to answer.
5. If user greets you then greet back politely telling about yourself.

---
Your Process:
Use the following format exactly when answering questions:

Question: the input question you must answer
Thought: what you contextually think about the question
Action: the tool to use [{tool_names}]
Action Input: the input to the tool and the chat history containing previous answer if any exists
Observation: the result of the action/tool
... (you can repeat Thought → Action → Action Input → Observation if needed and also ensure to logically reason the result with available contextual information to finetune it and conclude in every iteration)
Thought: I now know the answer
Final Answer: the answer to the original question, including proper citation(s).

Always try to articulate the relevant information in detail and include it in final answer to give it as background.
Always ensure to provide any resource URL(if available) in final answer.
Ensure to provide the source document information(if available) in final answer.
Include search scores when available to indicate relevance.
If any tabular data is available then include it in tabular format within final answer.
If any artifact is available which is relevant then include it in final answer with proper formatting & appropriate dimensions(especially if images are available).

Always cite your sources in this format (if available):
Title: <document title> 
URL: <document page_url>
Document ID: <document_id>
Search Score: <search_score>

If you can't find a good answer in the documentation, say:
"I couldn't find relevant information in the Azure AI Search index."

---
Begin!
Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(llm, tools, prompt)
        callbacks = create_callbacks()
        
        # Create a function to get chat history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            return chat_history
        
        # Create agent executor with proper memory
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=120,
            return_intermediate_steps=False,
            callbacks=callbacks
        )
        
        # Wrap with message history
        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        return agent_with_chat_history

    async def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        async with self._lock:
            now = datetime.now()
            expired_sessions = [
                session_id for session_id, session in self._sessions.items()
                if now - session.last_accessed > timedelta(minutes=config.session_timeout_minutes)
            ]
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                logger.info(f"Removed expired session: {session_id}")
    
    @property
    def active_sessions_count(self) -> int:
        return len(self._sessions)


# Global Managers
rate_limiter = RateLimiter()
connection_pool = ConnectionPoolManager()
request_queue = RequestQueue()
session_manager = SessionManager()

# Background Tasks
async def cleanup_sessions_task():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            await session_manager.cleanup_expired_sessions()
            await asyncio.sleep(300)  # Clean every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

async def process_queue_task():
    """Background task to process queued requests"""
    while True:
        try:
            request_data = await request_queue.get_request()
            if request_data:
                await process_chat_request(request_data)
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
            await asyncio.sleep(1)

# Request Models
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    request_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None

class StatusResponse(BaseModel):
    active_sessions: int
    active_connections: int
    queue_size: int
    uptime: str
    
# Add these request models after your existing models
class FeedbackRequest(BaseModel):
    chat_history_id: Optional[int] = None
    response_id: Optional[str] = None  # Alternative to chat_history_id
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 star rating
    is_helpful: Optional[bool] = None  # thumbs up/down
    feedback_text: Optional[str] = None
    feedback_category: Optional[str] = None  # accuracy, helpfulness, clarity, etc.
    is_accurate: Optional[bool] = None
    is_relevant: Optional[bool] = None
    is_clear: Optional[bool] = None
    is_complete: Optional[bool] = None

class FeedbackResponse(BaseModel):
    feedback_id: int
    message: str
    timestamp: datetime

class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    helpful_feedback: int
    unhelpful_feedback: int
    helpfulness_rate: float
    average_rating: Optional[float]
    rating_distribution: dict
    feedback_categories: dict

#My token_counting
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken
class AggregatingTokenTracker(BaseCallbackHandler):
    def __init__(self, model_name="gpt-35-turbo"):
        super().__init__()
        self.model_name = model_name
        self.reset()
    def reset(self):
        """Resets the counters for a new run/session."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    def on_llm_end(self, response, **kwargs):
        """
        Called after each LLM call; aggregates tokens + estimated cost.
        Uses token_usage if present, else calculates using tiktoken.
        """
        usage = response.llm_output.get("token_usage", {})
        # Prefer usage reported by the LLM (if reliable)
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        # Fallback to local calculation if usage missing or suspiciously zero
        if total == 0 or prompt + completion == 0:
            prompt_text = kwargs.get("prompts", [""])[0]
            generations = response.generations[0]
            completion_texts = [gen.text for gen in generations]
            enc = None
            try:
                enc = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            prompt = len(enc.encode(prompt_text))
            completion = sum(len(enc.encode(text)) for text in completion_texts)
            total = prompt + completion
        # Aggregate totals
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        # Compute cost - adjust pricing for your Azure region and agreement
        cost_per_prompt_token = 0.005 / 1000  # example: $0.005 per 1k prompt tokens
        cost_per_completion_token = 0.015 / 1000  # example: $0.015 per 1k completion tokens
        cost = (prompt * cost_per_prompt_token) + (completion * cost_per_completion_token)
        self.total_cost += cost
        print(
            f"[AggregatingTokenTracker] Iteration: Prompt={prompt}, "
            f"Completion={completion}, Total={total}, "
            f"Accumulated Cost=${self.total_cost:.4f}"
        )
    def get_totals(self):
        """Returns aggregated token usage and cost as a dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
        }

# Dependencies
def get_session_id(request: ChatRequest) -> str:
    """Extract and validate session ID"""
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    return request.session_id

def check_rate_limit(session_id: str = Depends(get_session_id)) -> str:
    """Check rate limiting for session"""
    if not rate_limiter.is_allowed(session_id):
        reset_times = rate_limiter.get_reset_time(session_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Reset times: {reset_times}"
        )
    return session_id

# Core Processing Function
async def process_chat_request(request_data: Dict[str, Any]) -> ChatResponse:
    """Process a chat request with full isolation"""
    session_id = request_data["session_id"]
    message = request_data["message"]
    request_id = request_data.get("request_id")
    
    # Acquire connection
    await connection_pool.acquire()
    
    try:
        # Get session (creates if doesn't exist)
        session = await session_manager.get_session(session_id)
        session.request_count += 1
        
        # Process with agent
        logger.info(f"Processing message for session {session_id}: {message[:50]}...")
        
        token_tracker = AggregatingTokenTracker("gpt-4o")
        # Use agent executor with isolated memory
        result = await session.agent_executor.ainvoke(
            {"input": message},
            config={"callbacks":[token_tracker],"configurable": {"session_id": session_id}}
        )
        total = token_tracker.get_totals()
        print(f"************************************  {total}  **********************************************\n")
        logger.info(
            f"Token usage per session {session_id}:"
            # f"prompt={token_tracker.prompt_tokens}, Completions={token_tracker.completion_tokens}, "
            f"Total={total}"
        )
        
        response = ChatResponse(
            response=result["output"],
            session_id=session_id,
            timestamp=datetime.now(),
            request_id=request_id,
            # prompt_tokens = token_tracker.prompt_tokens,
            # completion_tokens = token_tracker.completion_tokens,
            # total_tokens = token_tracker.total_tokens,
            # total_cost = token_tracker.total_cost
        )
        
        logger.info(f"Completed processing for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        await connection_pool.release()

# API Endpoints
@app.post("/signup")
def signup(username: str, password: str, db: Session = Depends(get_db)):
    if get_user(db, username):
        raise HTTPException(400, "User already exists")
    create_user(db, username, password)
    return {"msg": "User registered."}

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@app.get("/login/sso")
def sso_login():
    """Redirect to Azure AD for SSO login"""
    auth_url = get_sso_auth_url()
    return {"auth_url": auth_url}

@app.get("/callback/sso")
def sso_callback(code: str, db: Session = Depends(get_db)):
    """Callback endpoint for Azure AD SSO"""
    try:
        token_data = get_sso_token(code)
        if "error" in token_data:
            raise HTTPException(status_code=400, detail=token_data.get("error_description"))
        
        access_token = token_data.get("access_token")
        user_info = get_sso_user_info(access_token)
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Could not fetch user info")
            
        username = user_info.get("userPrincipalName")
        user = get_user(db, username)
        
        if not user:
            # Create a new user if they don't exist
            user = create_user(db, username, password=str(uuid4())) # Generate a random password
            
        # Create a token for the user
        return create_user_token(user)

    except Exception as e:
        logger.error(f"SSO Callback Error: {e}")
        raise HTTPException(status_code=500, detail="SSO callback failed")

@app.get("/status", response_model=StatusResponse)
async def status_endpoint():
    """Get system status"""
    return StatusResponse(
        active_sessions=session_manager.active_sessions_count,
        active_connections=connection_pool.active_connections,
        queue_size=request_queue.size,
        uptime=str(datetime.now())
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    async with session_manager._lock:
        if session_id in session_manager._sessions:
            del session_manager._sessions[session_id]
            return {"message": f"Session {session_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    session = await session_manager.get_session(session_id)
    if session:
        # Get messages from ChatMessageHistory
        chat_messages = session.chat_history.messages
        messages = []
        for message in chat_messages:
            messages.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": getattr(message, 'timestamp', None)
            })
        return {"session_id": session_id, "messages": messages}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_sessions": session_manager.active_sessions_count,
        "active_connections": connection_pool.active_connections,
        "azure_enabled": AZURE_ENABLED
    }


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a chat response with enhanced fallback logic"""
    try:
        # Debug logging
        logger.info(f"Feedback submission attempt - user_id: {user.id}, "
                    f"response_id: {feedback.response_id}, "
                    f"chat_history_id: {feedback.chat_history_id}, "
                    f"use_latest_chat: {getattr(feedback, 'use_latest_chat', False)}")
        
        chat_history = None
        
        # Method 1: Try to find by chat_history_id first (most direct)
        if feedback.chat_history_id:
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.id == feedback.chat_history_id,
                ChatHistory.user_id == user.id  # Ensure user owns this chat
            ).first()
            
            if chat_history:
                logger.info(f"Found chat_history by ID: {chat_history.id}")
            else:
                logger.warning(f"Chat history not found by ID: {feedback.chat_history_id}")
        
        # Method 2: Try to find by response_id if chat_history_id didn't work
        if not chat_history and feedback.response_id:
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.response_id == feedback.response_id,
                ChatHistory.user_id == user.id  # Ensure user owns this chat
            ).first()
            
            if chat_history:
                logger.info(f"Found chat_history by response_id: {feedback.response_id}")
            else:
                logger.warning(f"Chat history not found by response_id: {feedback.response_id}")
        
        # Method 3: If still not found, try to get the latest chat for this user
        # This is the fallback method when frontend can't provide identifiers
        if not chat_history:
            logger.info("Attempting to find latest chat for user as fallback")
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.user_id == user.id
            ).order_by(ChatHistory.timestamp.desc()).first()
            
            if chat_history:
                logger.warning(f"Using latest chat as fallback: {chat_history.id}")
                # Optionally, you can add a time window check here
                # to ensure the feedback is for a recent chat (e.g., within last 30 minutes)
                from datetime import datetime, timedelta
                if datetime.utcnow() - chat_history.timestamp > timedelta(minutes=30):
                    logger.warning(f"Latest chat is older than 30 minutes, proceeding anyway")
        
        # Method 4: If still no chat found, check if user has any chats at all
        if not chat_history:
            total_user_chats = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count()
            
            # Enhanced error response with debugging info
            error_msg = "No chat response found for feedback"
            debug_info = {
                "user_id": user.id,
                "response_id": feedback.response_id,
                "chat_history_id": feedback.chat_history_id,
                "total_user_chats": total_user_chats,
                "search_methods_tried": [
                    "chat_history_id" if feedback.chat_history_id else None,
                    "response_id" if feedback.response_id else None,
                    "latest_chat_fallback"
                ]
            }
            
            logger.error(f"Chat not found - Debug info: {debug_info}")
            
            if total_user_chats == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No chat history found. Please send a message first before providing feedback."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Chat response not found. User has {total_user_chats} total chats but none match the provided identifiers."
                )
        
        # Check if feedback already exists for this chat
        existing_feedback = get_feedback_by_chat_id(db, chat_history.id)
        
        if existing_feedback:
            logger.info(f"Updating existing feedback: {existing_feedback.id}")
            # Update existing feedback
            updated_feedback = update_feedback(
                db,
                existing_feedback.id,
                rating=feedback.rating,
                is_helpful=feedback.is_helpful,
                feedback_text=feedback.feedback_text,
                feedback_category=feedback.feedback_category,
                is_accurate=feedback.is_accurate,
                is_relevant=feedback.is_relevant,
                is_clear=feedback.is_clear,
                is_complete=feedback.is_complete,
                ip_address=str(request.client.host) if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
            
            if not updated_feedback:
                raise HTTPException(status_code=500, detail="Failed to update feedback")
            
            return FeedbackResponse(
                feedback_id=updated_feedback.id,
                message="Feedback updated successfully",
                timestamp=updated_feedback.timestamp
            )
        else:
            logger.info(f"Creating new feedback for chat: {chat_history.id}")
            # Create new feedback
            new_feedback = save_feedback(
                db=db,
                chat_history_id=chat_history.id,
                user_id=user.id,
                session_id=chat_history.session_id,
                rating=feedback.rating,
                is_helpful=feedback.is_helpful,
                feedback_text=feedback.feedback_text,
                feedback_category=feedback.feedback_category,
                is_accurate=feedback.is_accurate,
                is_relevant=feedback.is_relevant,
                is_clear=feedback.is_clear,
                is_complete=feedback.is_complete,
                ip_address=str(request.client.host) if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
            
            return FeedbackResponse(
                feedback_id=new_feedback.id,
                message="Feedback submitted successfully",
                timestamp=new_feedback.timestamp
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


# Enhanced chat endpoint to ensure response_id is always returned
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(check_rate_limit),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Enhanced chat endpoint that ensures response identifiers are returned"""
    try:
        # Generate response_id if not provided
        response_id = getattr(request, 'response_id', None) or str(uuid.uuid4())
        request_data = {
            "message": request.message,
            "session_id": session_id
        }
        
        # Your existing chat logic here...
        # (Replace this with your actual chat processing logic)


        if connection_pool.active_connections < config.max_concurrent_requests:
            answer = await process_chat_request(request_data)
            
        
        # After generating the response, save to database
        try:
            chat_history = save_chat_history(
                db=db,
                session_id=f"{user.username} -{request.session_id}",
                user_id=user.id,
                question=request_data["message"],
                answer=answer.response,  # Your generated response
                response_id=response_id
            )
            db.add(chat_history)
            db.commit()
        except Exception as db_error:
                logger.error(f"Database error: {db_error}")
                db.rollback()

        
        # Return response with both identifiers
        return {
            "response": answer.response,
            "request_id": response_id,  # For backward compatibility
            "response_id": response_id,
            "chat_history_id": chat_history.id,
            "session_id": request.session_id,
            "timestamp": chat_history.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional debug endpoint for troubleshooting
@app.get("/debug/user-chats")
async def get_user_chats_debug(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check user's chat history"""
    try:
        chats = db.query(ChatHistory).filter(
            ChatHistory.user_id == user.id
        ).order_by(ChatHistory.timestamp.desc()).limit(10).all()
        
        return {
            "user_id": user.id,
            "username": user.username,
            "total_chats": db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count(),
            "recent_chats": [
                {
                    "id": chat.id,
                    "response_id": chat.response_id,
                    "session_id": chat.session_id,
                    "question": chat.question[:100] + "..." if len(chat.question) > 100 else chat.question,
                    "timestamp": chat.timestamp.isoformat(),
                    "has_feedback": db.query(Feedback).filter(Feedback.chat_history_id == chat.id).first() is not None
                }
                for chat in chats
            ]
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Updated FeedbackRequest model to include the use_latest_chat flag
from pydantic import BaseModel
from typing import Optional

class FeedbackRequest(BaseModel):
    response_id: Optional[str] = None
    chat_history_id: Optional[int] = None
    rating: Optional[int] = None
    is_helpful: Optional[bool] = None
    feedback_text: Optional[str] = None
    feedback_category: Optional[str] = None
    is_accurate: Optional[bool] = None
    is_relevant: Optional[bool] = None
    is_clear: Optional[bool] = None
    is_complete: Optional[bool] = None
    use_latest_chat: Optional[bool] = False  # New field for fallback
##########################################################################################################################


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_statistics(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get feedback statistics (admin only or user's own feedback)"""
    try:
        stats = get_feedback_stats(db)
        return FeedbackStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback statistics")



@app.get("/feedback/my-feedback")
async def get_my_feedback(
    limit: int = 50,
    offset: int = 0,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's own feedback history"""
    try:
        feedback_list = get_user_feedback_history(db, user.id, limit, offset)
        results = []
        for feedback in feedback_list:
# Get associated chat history
            chat = db.query(ChatHistory).filter(ChatHistory.id == feedback.chat_history_id).first()
            result = {
                "feedback_id": feedback.id,
                "rating": feedback.rating,
                "is_helpful": feedback.is_helpful,
                "feedback_text": feedback.feedback_text,
                "feedback_category": feedback.feedback_category,
                "timestamp": feedback.timestamp,
                "chat_question": chat.question if chat else None,
                "chat_answer": chat.answer[:200] + "..." if chat and len(chat.answer) > 200 else chat.answer if chat else None
            }
            results.append(result)
        return {"feedback_history": results, "total": len(results)}
    except Exception as e:
        logger.error(f"Error getting user feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback history")


@app.get("/feedback/recent")
async def get_recent_feedback_admin(
    limit: int = 10,
    include_text: bool = True,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recent feedback (for admin/analysis purposes)"""
    try:
# In a real app, you'd check if user has admin privileges
# For now, users can only see aggregated recent feedback
        recent_feedback = get_recent_feedback(db, limit, include_text)
# Remove sensitive information for non-admin users
        cleaned_feedback = []
        for feedback in recent_feedback:
            cleaned = {
                "rating": feedback["rating"],
                "is_helpful": feedback["is_helpful"],
                "category": feedback["category"],
                "timestamp": feedback["timestamp"]
            }
            if include_text and feedback.get("feedback_text"):
                cleaned["feedback_text"] = feedback["feedback_text"]
            cleaned_feedback.append(cleaned)
        return {"recent_feedback": cleaned_feedback}
    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recent feedback")


@app.get("/feedback/trends")
async def get_feedback_trends_endpoint(
    days: int = 30,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get feedback trends over time"""
    try:
        trends = get_feedback_trends(db, days)
        return trends
    except Exception as e:
        logger.error(f"Error getting feedback trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback trends")
        

@app.delete("/feedback/{feedback_id}")
async def delete_feedback_endpoint(
    feedback_id: int,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user's own feedback"""
    try:
# Check if feedback belongs to user
        feedback = db.query(Feedback).filter(
            Feedback.id == feedback_id,
            Feedback.user_id == user.id).first()
        
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        success = delete_feedback(db, feedback_id)
        if success:
            return {"message": "Feedback deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete feedback")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete feedback")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )