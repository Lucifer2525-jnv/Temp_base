# db_utils.py
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import ChatHistory, User, get_db
from passlib.context import CryptContext
from typing import List, Tuple, Optional
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =============================================================================
# User Management Functions
# =============================================================================

def get_user(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    try:
        return db.query(User).filter(User.username == username).first()
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    try:
        return db.query(User).filter(User.id == user_id).first()
    except Exception as e:
        print(f"Error getting user by ID: {e}")
        return None

def create_user(db: Session, username: str, password: str) -> User:
    """Create a new user"""
    try:
        # Hash the password
        hashed_password = pwd_context.hash(password)
        
        # Create user object
        user = User(
            username=username,
            hashed_password=hashed_password
        )
        
        # Add to database
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    except Exception as e:
        db.rollback()
        raise Exception(f"Error creating user: {e}")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Error verifying password: {e}")
        return False

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

# =============================================================================
# Chat History Functions
# =============================================================================

def save_chat_history(
    db: Session,
    session_id: str,
    user_id: int,
    question: str,
    answer: str
) -> ChatHistory:
    """Save chat history to database"""
    try:
        chat_history = ChatHistory(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer
        )
        
        db.add(chat_history)
        db.commit()
        db.refresh(chat_history)
        
        return chat_history
    except Exception as e:
        db.rollback()
        raise Exception(f"Error saving chat history: {e}")

def get_user_chat_history(
    db: Session,
    user_id: int,
    limit: int = 50,
    offset: int = 0
) -> List[ChatHistory]:
    """Get chat history for a specific user"""
    try:
        return (
            db.query(ChatHistory)
            .filter(ChatHistory.user_id == user_id)
            .order_by(desc(ChatHistory.timestamp))
            .offset(offset)
            .limit(limit)
            .all()
        )
    except Exception as e:
        print(f"Error getting user chat history: {e}")
        return []

def get_session_chat_history(
    db: Session,
    session_id: str,
    limit: int = 50
) -> List[ChatHistory]:
    """Get chat history for a specific session"""
    try:
        return (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.timestamp)
            .limit(limit)
            .all()
        )
    except Exception as e:
        print(f"Error getting session chat history: {e}")
        return []

def get_top_questions(db: Session, limit: int = 5) -> List[Tuple[str, int]]:
    """Get the most frequently asked questions"""
    try:
        return (
            db.query(
                ChatHistory.question,
                func.count(ChatHistory.question).label("count")
            )
            .group_by(ChatHistory.question)
            .order_by(desc(func.count(ChatHistory.question)))
            .limit(limit)
            .all()
        )
    except Exception as e:
        print(f"Error getting top questions: {e}")
        return []

def delete_user_chat_history(db: Session, user_id: int) -> bool:
    """Delete all chat history for a user"""
    try:
        db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting user chat history: {e}")
        return False

def delete_session_chat_history(db: Session, session_id: str) -> bool:
    """Delete all chat history for a session"""
    try:
        db.query(ChatHistory).filter(ChatHistory.session_id == session_id).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting session chat history: {e}")
        return False

# =============================================================================
# Statistics Functions
# =============================================================================

def get_user_stats(db: Session, user_id: int) -> dict:
    """Get statistics for a user"""
    try:
        total_chats = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).count()
        
        # Get first and last chat dates
        first_chat = (
            db.query(ChatHistory)
            .filter(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.timestamp)
            .first()
        )
        
        last_chat = (
            db.query(ChatHistory)
            .filter(ChatHistory.user_id == user_id)
            .order_by(desc(ChatHistory.timestamp))
            .first()
        )
        
        return {
            "total_chats": total_chats,
            "first_chat_date": first_chat.timestamp if first_chat else None,
            "last_chat_date": last_chat.timestamp if last_chat else None,
        }
    except Exception as e:
        print(f"Error getting user stats: {e}")
        return {
            "total_chats": 0,
            "first_chat_date": None,
            "last_chat_date": None,
        }

def get_system_stats(db: Session) -> dict:
    """Get overall system statistics"""
    try:
        total_users = db.query(User).count()
        total_chats = db.query(ChatHistory).count()
        
        return {
            "total_users": total_users,
            "total_chats": total_chats,
        }
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return {
            "total_users": 0,
            "total_chats": 0,
        }