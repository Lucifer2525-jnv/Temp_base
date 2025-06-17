# main.py
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

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain.memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

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
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    
    # Queue settings
    queue_timeout_seconds: int = 30
    max_queue_size: int = 100

config = AppConfig()

# =============================================================================
# Rate Limiting
# =============================================================================

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

# =============================================================================
# Connection Pool Manager
# =============================================================================

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
    
    def release(self):
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

# =============================================================================
# Request Queue Manager
# =============================================================================

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

# =============================================================================
# Session Manager
# =============================================================================

@dataclass
class SessionData:
    session_id: str
    memory: ChatMessageHistory
    agent_executor: Optional[AgentExecutor] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    request_count: int = 0

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._cleanup_task = None
    
    def create_session(self, session_id: str) -> SessionData:
        """Create a new session with isolated memory and agent"""
        with self._lock:
            if len(self._sessions) >= config.max_sessions:
                # Remove oldest session
                oldest_session = min(
                    self._sessions.values(),
                    key=lambda s: s.last_accessed
                )
                del self._sessions[oldest_session.session_id]
                logger.info(f"Removed oldest session: {oldest_session.session_id}")
            
            # Create new session
            memory = ChatMessageHistory()
            session_data = SessionData(
                session_id=session_id,
                memory=memory
            )
            
            # Create agent executor with isolated memory
            session_data.agent_executor = self._create_agent_executor(memory)
            
            self._sessions[session_id] = session_data
            logger.info(f"Created new session: {session_id}")
            
            return session_data
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get existing session or create new one"""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now()
                return session
            else:
                return self.create_session(session_id)
    
    def _create_agent_executor(self, memory: ChatMessageHistory) -> AgentExecutor:
        """Create agent executor with tools and memory"""
        # Initialize LLM with connection pooling
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.max_tokens,
            request_timeout=30,
            max_retries=3
        )
        
        # Define tools
        tools = [
            Tool(
                name="calculator",
                description="Useful for mathematical calculations",
                func=lambda x: str(eval(x))  # Note: Use safe_eval in production
            ),
            Tool(
                name="current_time",
                description="Get current time",
                func=lambda x: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        ]
        
        # Create prompt template
        template = """
        You are a helpful assistant. You have access to the following tools:
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
        Thought: {agent_scratchpad}
        """
        
        prompt = PromptTemplate.from_template(template)
        
        # Create agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        with self._lock:
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

# =============================================================================
# Global Managers
# =============================================================================

rate_limiter = RateLimiter()
connection_pool = ConnectionPoolManager()
request_queue = RequestQueue()
session_manager = SessionManager()

# =============================================================================
# Background Tasks
# =============================================================================

async def cleanup_sessions_task():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            session_manager.cleanup_expired_sessions()
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

# =============================================================================
# Request Models
# =============================================================================

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

# =============================================================================
# FastAPI App
# =============================================================================

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
    title="Agentic Chatbot API",
    description="Production-ready chatbot with session isolation and rate limiting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Dependencies
# =============================================================================

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

# =============================================================================
# Core Processing Function
# =============================================================================

async def process_chat_request(request_data: Dict[str, Any]) -> ChatResponse:
    """Process a chat request with full isolation"""
    session_id = request_data["session_id"]
    message = request_data["message"]
    request_id = request_data.get("request_id")
    
    # Acquire connection
    await connection_pool.acquire()
    
    try:
        # Get session (creates if doesn't exist)
        session = session_manager.get_session(session_id)
        session.request_count += 1
        
        # Process with agent
        logger.info(f"Processing message for session {session_id}: {message[:50]}...")
        
        # Use agent executor with isolated memory
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: session.agent_executor.invoke({"input": message})
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
        connection_pool.release()

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Depends(check_rate_limit)
):
    """Main chat endpoint with queuing and rate limiting"""
    try:
        # Add to queue
        request_data = {
            "message": request.message,
            "session_id": session_id
        }
        
        # Process immediately if possible, otherwise queue
        if connection_pool.active_connections < config.max_concurrent_requests:
            return await process_chat_request(request_data)
        else:
            # Add to queue for background processing
            request_id = await request_queue.add_request(request_data)
            return ChatResponse(
                response="Request queued for processing",
                session_id=session_id,
                timestamp=datetime.now(),
                request_id=request_id
            )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    with session_manager._lock:
        if session_id in session_manager._sessions:
            del session_manager._sessions[session_id]
            return {"message": f"Session {session_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    session = session_manager.get_session(session_id)
    if session:
        messages = []
        for message in session.memory.messages:
            messages.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": getattr(message, 'timestamp', None)
            })
        return {"session_id": session_id, "messages": messages}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_sessions": session_manager.active_sessions_count,
        "active_connections": connection_pool.active_connections
    }

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1,  # Single worker to maintain session state
        log_level="info"
    )

# =============================================================================
# Streamlit Frontend Example
# =============================================================================

"""
# streamlit_app.py

import streamlit as st
import requests
import uuid
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Agentic Chatbot")
st.write(f"Session ID: {st.session_state.session_id}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
                elif response.status_code == 429:
                    st.error("Rate limit exceeded. Please wait before sending another message.")
                    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

# Sidebar with system status
with st.sidebar:
    st.header("System Status")
    
    if st.button("Refresh Status"):
        try:
            status_response = requests.get(f"{API_BASE_URL}/status")
            if status_response.status_code == 200:
                status = status_response.json()
                st.json(status)
        except:
            st.error("Could not fetch status")
    
    if st.button("Clear Session"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
"""

# =============================================================================
# Requirements.txt
# =============================================================================

"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
langchain==0.1.0
langchain-openai==0.0.2
pydantic==2.5.0
python-multipart==0.0.6
streamlit==1.28.0
requests==2.31.0
"""