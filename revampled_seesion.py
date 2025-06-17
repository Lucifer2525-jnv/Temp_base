# =============================================================================
# Session Data Class (Must be defined before SessionManager)
# =============================================================================

@dataclass
class SessionData:
    session_id: str
    memory: ChatMessageHistory
    agent_executor: Optional[AgentExecutor] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    request_count: int = 0

# =============================================================================
# Fixed Session Manager
# =============================================================================

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()  # Use asyncio.Lock for async compatibility
        self._cleanup_task = None
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get existing session or create new one"""
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now()
                return session
            else:
                # Create session without releasing the lock
                return await self._create_session_internal(session_id)
    
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
    
    async def create_session(self, session_id: str) -> SessionData:
        """Public method to create a new session - acquires lock"""
        async with self._lock:
            return await self._create_session_internal(session_id)
    
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

# =============================================================================
# Updated Background Tasks
# =============================================================================

async def cleanup_sessions_task():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            await session_manager.cleanup_expired_sessions()  # Now async
            await asyncio.sleep(300)  # Clean every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

# =============================================================================
# Updated Core Processing Function
# =============================================================================

async def process_chat_request(request_data: Dict[str, Any]) -> ChatResponse:
    """Process a chat request with full isolation"""
    session_id = request_data["session_id"]
    message = request_data["message"]
    request_id = request_data.get("request_id")
    
    # Acquire connection
    await connection_pool.acquire()
    
    try:
        # Get session (creates if doesn't exist) - now async
        session = await session_manager.get_session(session_id)
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
# Updated API Endpoints
# =============================================================================

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