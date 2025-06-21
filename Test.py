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
 
from prompt_templates import *
 
APPINSIGHTS_CONNECTION_STRING = "InstrumentationKey=60079e68-cfcd-4d79-baae-f7addf6e3669"
 
 
# Configure basic logging first (before any Azure imports)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
 
logger = logging.getLogger(__name__)
 
# Azure App Insights - with proper error handling

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
            await asyncio.sleep(300)  # Clean every 5 minutes
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
   response_id: Optional[str] = None  # Alternative to chat_history_id
   rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 star rating
   is_helpful: Optional[bool] = None  # thumbs up/down
   feedback_text: Optional[str] = None
   feedback_category: Optional[str] = None  # accuracy, helpfulness, clarity, etc.
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
    description="GSC ARB chatbot with session isolation and rate limiting",
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
       
        # Use agent executor with isolated memory
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
                ChatHistory.user_id == user.id  # Ensure user owns this chat
            ).first()
           
            if chat_history:
                logger.info(f"Found chat_history by ID: {chat_history.id}")
            else:
                logger.warning(f"Chat history not found by ID: {feedback.chat_history_id}")
       
        # Method 2: Try to find by response_id if chat_history_id didn't work
        if not chat_history and feedback.response_id:
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.response_id == feedback.response_id,
                ChatHistory.user_id == user.id  # Ensure user owns this chat
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
@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced chat endpoint that ensures response identifiers are returned"""
    try:
        # Generate response_id if not provided
        response_id = getattr(request, 'response_id', None) or str(uuid.uuid4())
       
        # Your existing chat logic here...
        # (Replace this with your actual chat processing logic)
       
        # After generating the response, save to database
        chat_history = save_chat_history(
            db=db,
            session_id=request.session_id,
            user_id=user.id,
            question=request.message,
            answer=response_text,  # Your generated response
            response_id=response_id
        )
       
        # Return response with both identifiers
        return {
            "response": response_text,
            "request_id": response_id,  # For backward compatibility
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