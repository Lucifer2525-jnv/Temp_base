from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import AIMessage, HumanMessage, BaseMessage
import sqlite3
from typing import Dict, Any, List

class SQLiteMemory(BaseChatMemory):
    """SQLite-based chat memory for LangChain agents."""
    
    def __init__(self, session_id: str, db_path: str = "memory.db", **kwargs):
        # Initialize parent class first
        super().__init__(**kwargs)
        
        # Set instance attributes (not Pydantic fields)
        self._session_id = session_id
        self._db_path = db_path
        
        # Initialize SQLite connection
        self.conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._setup_table()

    @property
    def session_id(self):
        return self._session_id
    
    @property 
    def db_path(self):
        return self._db_path

    def _setup_table(self):
        """Create the chat_memory table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_memory (
                    session_id TEXT,
                    role TEXT,
                    message TEXT
                )
            """)

    @property
    def memory_variables(self) -> List[str]:
        """Return list of memory variable names."""
        return ["chat_history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load chat history from SQLite database."""
        return {"chat_history": self._get_history()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save conversation context to SQLite database."""
        user_msg = inputs.get("input") or inputs.get("question", "")
        ai_msg = outputs.get("output") or outputs.get("answer", "")

        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_memory VALUES (?, ?, ?)", 
                (self._session_id, "user", user_msg)
            )
            self.conn.execute(
                "INSERT INTO chat_memory VALUES (?, ?, ?)", 
                (self._session_id, "ai", ai_msg)
            )

    def clear(self) -> None:
        """Clear chat history for this session."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM chat_memory WHERE session_id = ?", 
                (self._session_id,)
            )

    def _get_history(self) -> List[BaseMessage]:
        """Retrieve chat history as LangChain messages."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, message FROM chat_memory WHERE session_id = ?", 
            (self._session_id,)
        )
        rows = cur.fetchall()
        messages = []

        for role, msg in rows:
            if role == "user":
                messages.append(HumanMessage(content=msg))
            elif role == "ai":
                messages.append(AIMessage(content=msg))

        return messages

    def __del__(self):
        """Close database connection when object is destroyed."""
        if hasattr(self, 'conn'):
            self.conn.close()


# Utility functions
def get_all_sessions(db_path="memory.db"):
    """Get all unique session IDs from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT session_id FROM chat_memory")
    sessions = [row[0] for row in cur.fetchall()]
    conn.close()
    return sessions

def get_chat_history(session_id, db_path="memory.db"):
    """Get chat history for a specific session."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, message FROM chat_memory WHERE session_id = ?", 
        (session_id,)
    )
    history = cur.fetchall()
    conn.close()
    return history

def clear_session(session_id, db_path="memory.db"):
    """Clear all messages for a specific session."""
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "DELETE FROM chat_memory WHERE session_id = ?", 
            (session_id,)
        )
    conn.close()