from langchain.schema import AIMessage, HumanMessage, BaseMessage
import sqlite3
from typing import Dict, Any, List
import threading
import logging
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLiteMemory:
    """Enhanced SQLite-based chat memory that works with LangChain agents."""
    
    def __init__(self, session_id: str, db_path: str = "memory.db"):
        self.session_id = session_id
        self.db_path = db_path
        self.lock = threading.Lock()
        self._setup_table()

    def _setup_table(self):
        """Create the chat_memory table if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX(session_id)
                    )
                """)
            logger.info(f"Database table setup complete for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error setting up database table: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variable names."""
        return ["chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load chat history from SQLite database."""
        try:
            return {"chat_history": self._get_history()}
        except Exception as e:
            logger.error(f"Error loading memory variables: {e}")
            return {"chat_history": []}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save conversation context to SQLite database."""
        user_msg = inputs.get("input") or inputs.get("question", "")
        ai_msg = outputs.get("output") or outputs.get("answer", "")

        if not user_msg and not ai_msg:
            logger.warning("No user or AI message found to save")
            return

        try:
            with self.lock:
                with self._get_connection() as conn:
                    if user_msg:
                        conn.execute(
                            "INSERT INTO chat_memory (session_id, role, message) VALUES (?, ?, ?)", 
                            (self.session_id, "user", user_msg)
                        )
                    if ai_msg:
                        conn.execute(
                            "INSERT INTO chat_memory (session_id, role, message) VALUES (?, ?, ?)", 
                            (self.session_id, "ai", ai_msg)
                        )
                    conn.commit()
            logger.info(f"Saved context for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error saving context: {e}")
            raise

    def clear(self) -> None:
        """Clear chat history for this session."""
        try:
            with self.lock:
                with self._get_connection() as conn:
                    conn.execute(
                        "DELETE FROM chat_memory WHERE session_id = ?", 
                        (self.session_id,)
                    )
                    conn.commit()
            logger.info(f"Cleared history for session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            raise

    def _get_history(self) -> List[BaseMessage]:
        """Retrieve chat history as LangChain messages."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT role, message FROM chat_memory WHERE session_id = ? ORDER BY id ASC", 
                    (self.session_id,)
                )
                rows = cur.fetchall()
                
            messages = []
            for role, msg in rows:
                if role == "user":
                    messages.append(HumanMessage(content=msg))
                elif role == "ai":
                    messages.append(AIMessage(content=msg))
                else:
                    logger.warning(f"Unknown role: {role}")
                    
            return messages
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []

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


# Enhanced utility functions
def get_all_sessions(db_path="memory.db"):
    """Get all unique session IDs from the database with message counts."""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur = conn.cursor()
        cur.execute("""
            SELECT session_id, COUNT(*) as message_count, MAX(timestamp) as last_activity
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
        
        query = "SELECT role, message, timestamp FROM chat_memory WHERE session_id = ? ORDER BY id ASC"
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
        with conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM chat_memory WHERE session_id = ?", 
                (session_id,)
            )
            deleted_count = cur.rowcount
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
        with conn:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM chat_memory 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_old))
            deleted_count = cur.rowcount
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
        
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}