import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
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

# Initialize Azure App Insights - with proper error handling
AZURE_ENABLED = False
AppInsightsHandler = None

if APPINSIGHTS_CONNECTION_STRING:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        from opencensus.ext.azure.trace_exporter import AzureExporter
        from opencensus.ext.fastapi.fastapi_middleware import FastAPIMiddleware
        from opencensus.trace.samplers import ProbabilitySampler
        
        # Import the custom AppInsightsHandler
        from app_insights_callback import AppInsightsHandler
        
        # Verify the handler is properly imported
        if AppInsightsHandler is None:
            raise ImportError("AppInsightsHandler could not be imported")
            
        logger.info(f"AppInsightsHandler successfully imported: {type(AppInsightsHandler)}")
        
        # Add Azure handler for general logging
        azure_handler = AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION_STRING)
        logger.addHandler(azure_handler)
        logger.info("Azure Application Insights logging enabled")
        AZURE_ENABLED = True
 
    except ImportError as e:
        logger.error(f"Failed to import Azure Application Insights modules: {e}")
        logger.info("Continuing without Azure Application Insights")
        AZURE_ENABLED = False
        AppInsightsHandler = None
    except Exception as e:
        logger.error(f"Failed to initialize Azure Application Insights: {e}")
        logger.info("Continuing without Azure Application Insights")
        AZURE_ENABLED = False
        AppInsightsHandler = None
else:
    logger.warning("App_Insight_Conn_String not set. Azure Application Insights disabled.")
    AZURE_ENABLED = False
    AppInsightsHandler = None

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
    from auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user
    from models import ChatHistory
    from agentic_rag import agent_executor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all required modules exist: utils.py, db_utils.py, auth.py, models.py, agentic_rag.py")
 
# Create LLM client with error handling
try:
    llm_client = create_chat_openai_client()
    logger.info("LLM client created successfully")
except Exception as e:
    logger.error(f"Failed to create LLM client: {e}")
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
        llm = llm_client
        db, embedding_model = load_vectorstore()
 
        vstore = db
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm_client,
            chain_type="stuff",
            retriever=vstore.as_retriever(search_kwargs={"k": 7}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_kb},
        )
 
        def _run(query: str) -> str:
            out = rag_chain.invoke({"query": query})
            answer = out["result"]
            sources = out.get("source_documents", [])
 
            source_texts = "\n\n".join(
                f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(sources)
            )
 
            return f"Final Answer: \n{answer}\n\nSources:\n{source_texts}"
 
        def safe_calculator(expression: str) -> str:
            """Safe calculator that handles basic math operations"""
            try:
                # Removing potentially dangerous characters
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters in expression"
               
                # Use eval safely (we can use a proper math parser)
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
       
        tools = [
            Tool(
                name="calculator",
                description="Useful for mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'",
                func=safe_calculator
            ),
            Tool(
                name="current_time",
                description="Get current time and date",
                func=lambda x: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            Tool(
                name="ARB_QA_Local_Knowledge_Base_Tool",
                func=_run,
                description="Answer ARB -Architecture Review Board related questions using the internal knowledge base."
            )
        ]
 
        template = """You are a helpful assistant. You have access to the following tools:
 
{tools}
 
Use the available tools smartly and be contextually aware of the domain & information present in tools to understand user questions & then answer those questions.
 
When a user submits a query:
 
1. Carefully interpret the request — especially for ARB processes, artefacts, governance workflows, lifecycle stages, or review board interactions.
1. Decide whether tool use is necessary to fetch information.
1. If tool use is needed, mention only the tool name.
1. If tool use is not needed, mention & highlight that response is generated without using tool on top of final answer.
1. Always ensure to provide any resource URL(if available) in final answer.
1. Ensure to provide the source document information(if available) in final answer.
1. If any tabular data is available then include it in tabular format within final answer.
1. If any artifact is available which is relevant then include it in final answer.
 
Use the following format:
 
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
… (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
 
Previous conversation:
{chat_history}
 
Question: {input}
Thought: {agent_scratchpad}"""
       
        prompt = PromptTemplate.from_template(template)
       
        # Create agent
        agent = create_react_agent(llm, tools, prompt)
       
        # Create callbacks - this is the key fix!
        callbacks = create_callbacks()
       
        # Create a function to get chat history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            return chat_history
       
        # Create agent executor with proper memory and callbacks
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=False,
            callbacks=callbacks  # Use the properly created callbacks
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
       
        # Use agent executor with isolated memory and callbacks
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: session.agent_executor.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
        )
       
        response = ChatResponse(
            response=result["output"],
            session_id=session_id,
            timestamp=datetime.now(),
            request_id=request_id
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
       
        # Process chat request with proper callbacks
        if connection_pool.active_connections < config.max_concurrent_requests:
            answer = await process_chat_request(request_data)
           
        # After generating the response, save to database
        try:
            chat_history = save_chat_history(
                db=db,
                session_id=f"{user.username} -{request.session_id}",
                user_id=user.id,
                question=request_data["message"],
                answer=answer.response,
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
            "request_id": response_id,
            "response_id": response_id,
            "chat_history_id": chat_history.id,
            "session_id": request.session_id,
            "timestamp": chat_history.timestamp
        }
       
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
if __name__ == "__main__":
    uvicorn.run(
        "main_updated:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )