from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_buffer_string
from langchain.schema.memory import BaseMemory
import sqlite3
from typing import Dict, Any, List, Optional
import threading
import logging
import os
from contextlib import contextmanager
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple ChatMessageHistory class implementation
class ChatMessageHistory:
    """Simple chat message history implementation."""
    
    def __init__(self):
        self.messages: List[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)
    
    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

class SQLiteMemory(BaseChatMemory):
    """SQLite-based chat memory that works with LangChain agents."""
    
    def __init__(self, session_id: str, db_path: str = "memory.db", return_messages: bool = True, max_messages: Optional[int] = None, memory_key: str = "history"):
        # Set required attributes first
        self.session_id = session_id
        self.db_path = db_path
        self.max_messages = max_messages
        self.memory_key = memory_key
        self.lock = threading.RLock()  # Use RLock for re-entrant locking
        
        # Initialize chat_memory
        self.chat_memory = ChatMessageHistory()
        
        # Initialize the parent class
        super().__init__(return_messages=return_messages)
        
        # Setup database and load history
        try:
            self._setup_table()
            self._load_existing_history()
            logger.info(f"SQLiteMemory initialized for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLiteMemory: {e}")
            raise

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variables (required abstract method)."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables from storage (required abstract method)."""
        try:
            if self.return_messages:
                # Return messages as list of BaseMessage objects
                return {self.memory_key: self.chat_memory.messages}
            else:
                # Return as formatted string
                return {self.memory_key: get_buffer_string(self.chat_memory.messages)}
        except Exception as e:
            logger.error(f"Error loading memory variables: {e}")
            # Return empty memory on error
            return {self.memory_key: [] if self.return_messages else ""}

    def _setup_table(self):
        """Create the chat_memory table if it doesn't exist."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id 
                    ON chat_memory(session_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                    ON chat_memory(session_id, timestamp)
                """)
                
                conn.commit()
            logger.info(f"Database table setup complete for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error setting up database table: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with better error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False, 
                timeout=30.0,
                isolation_level=None  # Autocommit mode
            )
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=memory")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            yield conn
        except sqlite3.Error as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"SQLite error: {e}")
            raise
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def _load_existing_history(self):
        """Load existing chat history from database into memory."""
        try:
            with self.lock:
                messages = self._get_history_from_db()
                # Clear existing and add all messages
                self.chat_memory.clear()
                for message in messages:
                    self.chat_memory.add_message(message)
                logger.info(f"Loaded {len(messages)} messages for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error loading existing history: {e}")
            # Don't raise here, just log and continue with empty history

    def _get_history_from_db(self) -> List[BaseMessage]:
        """Retrieve chat history from database as LangChain messages."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                query = "SELECT role, message FROM chat_memory WHERE session_id = ? ORDER BY timestamp ASC, id ASC"
                
                if self.max_messages:
                    query += f" LIMIT {self.max_messages}"
                
                cur.execute(query, (self.session_id,))
                rows = cur.fetchall()
                
            messages = []
            for role, msg in rows:
                try:
                    if role == "user" or role == "human":
                        messages.append(HumanMessage(content=msg))
                    elif role == "ai" or role == "assistant":
                        messages.append(AIMessage(content=msg))
                    else:
                        logger.warning(f"Unknown role: {role}, treating as human message")
                        messages.append(HumanMessage(content=msg))
                except Exception as e:
                    logger.error(f"Error creating message object: {e}")
                    continue
                    
            return messages
        except Exception as e:
            logger.error(f"Error retrieving history from DB: {e}")
            return []

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save conversation context to both memory and database."""
        try:
            # Let parent class handle the memory part first
            super().save_context(inputs, outputs)
            
            # Extract messages
            user_msg = self._extract_user_message(inputs)
            ai_msg = self._extract_ai_message(outputs)

            if not user_msg and not ai_msg:
                logger.warning("No user or AI message found to save")
                return

            # Save to database
            with self.lock:
                with self._get_connection() as conn:
                    cur = conn.cursor()
                    
                    if user_msg:
                        cur.execute(
                            "INSERT INTO chat_memory (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)", 
                            (self.session_id, "user", user_msg, datetime.now().isoformat())
                        )
                    if ai_msg:
                        cur.execute(
                            "INSERT INTO chat_memory (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)", 
                            (self.session_id, "ai", ai_msg, datetime.now().isoformat())
                        )
                
                # Cleanup old messages if max_messages is set
                if self.max_messages:
                    self._cleanup_old_messages(conn)
                
            logger.info(f"Saved context to database for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving context to database: {e}")
            # Don't raise here to avoid breaking the conversation flow

    def _extract_user_message(self, inputs: Dict[str, Any]) -> str:
        """Extract user message from inputs with multiple fallbacks."""
        for key in ["input", "question", "query", "prompt", "human_input"]:
            if key in inputs and inputs[key]:
                return str(inputs[key])
        return ""

    def _extract_ai_message(self, outputs: Dict[str, Any]) -> str:
        """Extract AI message from outputs with multiple fallbacks."""
        for key in ["output", "answer", "response", "result", "ai_response"]:
            if key in outputs and outputs[key]:
                return str(outputs[key])
        return ""

    def _cleanup_old_messages(self, conn):
        """Remove old messages if we exceed max_messages limit."""
        try:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM chat_memory 
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM chat_memory 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC, id DESC 
                    LIMIT ?
                )
            """, (self.session_id, self.session_id, self.max_messages))
            
            deleted_count = cur.rowcount
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old messages for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error during message cleanup: {e}")

    def clear(self) -> None:
        """Clear chat history from both memory and database."""
        try:
            # Clear in-memory
            super().clear()
            
            # Clear database
            with self.lock:
                with self._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "DELETE FROM chat_memory WHERE session_id = ?", 
                        (self.session_id,)
                    )
                    deleted_count = cur.rowcount
                    
            logger.info(f"Cleared {deleted_count} messages for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            raise

    def get_message_count(self) -> int:
        """Get the number of messages for this session."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM chat_memory WHERE session_id = ?", 
                    (self.session_id,)
                )
                return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0

    def get_recent_messages(self, limit: int = 10) -> List[BaseMessage]:
        """Get the most recent messages for this session."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT role, message FROM chat_memory 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC, id DESC 
                    LIMIT ?
                """, (self.session_id, limit))
                rows = cur.fetchall()
                
            messages = []
            for role, msg in reversed(rows):  # Reverse to get chronological order
                if role == "user":
                    messages.append(HumanMessage(content=msg))
                elif role == "ai":
                    messages.append(AIMessage(content=msg))
                    
            return messages
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []

    def buffer_as_str(self) -> str:
        """Get the buffer as a formatted string."""
        return get_buffer_string(self.chat_memory.messages)

    def buffer_as_messages(self) -> List[BaseMessage]:
        """Get the buffer as a list of messages."""
        return self.chat_memory.messages

# Enhanced utility functions
def get_all_sessions(db_path="memory.db"):
    """Get all unique session IDs from the database with message counts."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()
        cur.execute("""
            SELECT session_id, COUNT(*) as message_count, 
                   MIN(timestamp) as first_activity,
                   MAX(timestamp) as last_activity
            FROM chat_memory 
            GROUP BY session_id 
            ORDER BY last_activity DESC
        """)
        sessions = cur.fetchall()
        conn.close()
        return sessions
    except Exception as e:
        logger.error(f"Error getting all sessions: {e}")
        return []

def get_chat_history(session_id, db_path="memory.db", limit=None):
    """Get chat history for a specific session with optional limit."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()
        
        query = "SELECT role, message, timestamp FROM chat_memory WHERE session_id = ? ORDER BY timestamp ASC, id ASC"
        params = (session_id,)
        
        if limit:
            query += " LIMIT ?"
            params = (session_id, limit)
            
        cur.execute(query, params)
        history = cur.fetchall()
        conn.close()
        return history
    except Exception as e:
        logger.error(f"Error getting chat history for session {session_id}: {e}")
        return []

def clear_session(session_id, db_path="memory.db"):
    """Clear all messages for a specific session."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM chat_memory WHERE session_id = ?", 
            (session_id,)
        )
        deleted_count = cur.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Cleared {deleted_count} messages for session: {session_id}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error clearing session {session_id}: {e}")
        return 0

def cleanup_old_sessions(db_path="memory.db", days_old=30):
    """Remove sessions older than specified days."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM chat_memory 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days_old))
        deleted_count = cur.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Cleaned up {deleted_count} old messages")
        return deleted_count
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return 0

def get_database_stats(db_path="memory.db"):
    """Get database statistics."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()
        
        stats = {}
        
        # Total messages
        cur.execute("SELECT COUNT(*) FROM chat_memory")
        stats['total_messages'] = cur.fetchone()[0]
        
        # Total sessions
        cur.execute("SELECT COUNT(DISTINCT session_id) FROM chat_memory")
        stats['total_sessions'] = cur.fetchone()[0]
        
        # Database size
        cur.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats['db_size_bytes'] = cur.fetchone()[0]
        
        # Most active session
        cur.execute("""
            SELECT session_id, COUNT(*) as msg_count 
            FROM chat_memory 
            GROUP BY session_id 
            ORDER BY msg_count DESC 
            LIMIT 1
        """)
        result = cur.fetchone()
        if result:
            stats['most_active_session'] = result[0]
            stats['most_active_session_messages'] = result[1]
        
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}

def test_memory_connection(db_path="memory.db"):
    """Test the database connection and setup."""
    try:
        test_memory = SQLiteMemory("test_session", db_path)
        test_memory.save_context(
            {"input": "Hello, this is a test"}, 
            {"output": "Test response received"}
        )
        count = test_memory.get_message_count()
        test_memory.clear()
        logger.info(f"Memory test successful. Created and cleaned up {count} messages.")
        return True
    except Exception as e:
        logger.error(f"Memory test failed: {e}")
        return False

# Usage example
if __name__ == "__main__":
    # Test the memory system
    print("Testing SQLite Memory System...")
    
    if test_memory_connection():
        print("✅ Memory system test passed!")
        
        # Show database stats
        stats = get_database_stats()
        print(f"Database stats: {stats}")
        
        # Show all sessions
        sessions = get_all_sessions()
        print(f"Active sessions: {len(sessions)}")
        
    else:
        print("❌ Memory system test failed!")