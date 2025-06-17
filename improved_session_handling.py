# Fix 1: Use asyncio.Lock instead of threading.Lock for async code
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()  # Change to asyncio.Lock
        self._cleanup_task = None
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get existing session or create new one"""
        async with self._lock:  # Use async with
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now()
                return session
            else:
                return await self._create_session_internal(session_id)
    
    async def _create_session_internal(self, session_id: str) -> SessionData:
        """Internal method to create session - assumes lock is already held"""
        if len(self._sessions) >= config.max_sessions:
            oldest_session = min(
                self._sessions.values(),
                key=lambda s: s.last_accessed
            )
            del self._sessions[oldest_session.session_id]
            logger.info(f"Removed oldest session: {oldest_session.session_id}")
        
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

# Fix 2: Update the process_chat_request function
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

# Fix 3: Update cleanup task to use async
async def cleanup_sessions_task():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            await session_manager.cleanup_expired_sessions()  # Now async
            await asyncio.sleep(300)  # Clean every 5 minutes
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)

# Fix 4: Update the delete session endpoint
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    async with session_manager._lock:  # Use async with
        if session_id in session_manager._sessions:
            del session_manager._sessions[session_id]
            return {"message": f"Session {session_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

# Fix 5: Update get session history endpoint
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    session = await session_manager.get_session(session_id)  # Now async
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