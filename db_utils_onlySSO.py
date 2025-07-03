#db_utils.py
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import ChatHistory, User, get_db, Feedback
from passlib.context import CryptContext
from typing import List, Tuple, Optional, Dict, Any
import os
import uuid
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(__file__))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User Management Functions
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



# Chat History Functions
def save_chat_history(
    db: Session,
    session_id: str,
    user_id: int,
    question: str,
    answer: str,
    response_id: str = None
) -> ChatHistory:
    """Save chat history to database"""
    try:
        if response_id is None:
            response_id = str(uuid.uuid4())
        chat_history = ChatHistory(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            response_id=response_id
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
    


def get_chat_by_response_id(db: Session, response_id: str) -> Optional[ChatHistory]:
    """Get chat history by response ID"""
    try:
        return db.query(ChatHistory).filter(ChatHistory.response_id == response_id).first()
    except Exception as e:
        print(f"Error getting chat by response ID: {e}")
        return None
    
    
    
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
    

# Statistics Functions
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



# Feedback Functions
def save_feedback(
    db: Session,
    chat_history_id: int,
    user_id: int,
    session_id: str,
    rating: Optional[int] = None,
    is_helpful: Optional[bool] = None,
    feedback_text: Optional[str] = None,
    feedback_category: Optional[str] = None,
    is_accurate: Optional[bool] = None,
    is_relevant: Optional[bool] = None,
    is_clear: Optional[bool] = None,
    is_complete: Optional[bool] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Feedback:
    """Save user feedback to database"""
    try:
        feedback = Feedback(
            chat_history_id=chat_history_id,
            user_id=user_id,
            session_id=session_id,
            rating=rating,
            is_helpful=is_helpful,
            feedback_text=feedback_text,
            feedback_category=feedback_category,
            is_accurate=is_accurate,
            is_relevant=is_relevant,
            is_clear=is_clear,
            is_complete=is_complete,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        return feedback
    except Exception as e:
        db.rollback()
        raise Exception(f"Error saving feedback: {e}")

def get_feedback_by_chat_id(db: Session, chat_history_id: int) -> Optional[Feedback]:
    """Get feedback for a specific chat response"""
    try:
        return db.query(Feedback).filter(Feedback.chat_history_id == chat_history_id).first()
    except Exception as e:
        print(f"Error getting feedback by chat ID: {e}")
        return None

def get_user_feedback_history(
    db: Session,
    user_id: int,
    limit: int = 50,
    offset: int = 0
) -> List[Feedback]:
    """Get feedback history for a specific user"""
    try:
        return (
            db.query(Feedback)
            .filter(Feedback.user_id == user_id)
            .order_by(desc(Feedback.timestamp))
            .offset(offset)
            .limit(limit)
            .all()
        )
    except Exception as e:
        print(f"Error getting user feedback history: {e}")
        return []

def get_session_feedback(db: Session, session_id: str) -> List[Feedback]:
    """Get all feedback for a session"""
    try:
        return (
            db.query(Feedback)
            .filter(Feedback.session_id == session_id)
            .order_by(desc(Feedback.timestamp))
            .all()
        )
    except Exception as e:
        print(f"Error getting session feedback: {e}")
        return []

def update_feedback(
    db: Session,
    feedback_id: int,
    **kwargs
) -> Optional[Feedback]:
    """Update existing feedback"""
    try:
        feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if not feedback:
            return None
        
        for key, value in kwargs.items():
            if hasattr(feedback, key) and value is not None:
                setattr(feedback, key, value)
        
        db.commit()
        db.refresh(feedback)
        return feedback
    except Exception as e:
        db.rollback()
        print(f"Error updating feedback: {e}")
        return None

def delete_feedback(db: Session, feedback_id: int) -> bool:
    """Delete feedback by ID"""
    try:
        feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if feedback:
            db.delete(feedback)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"Error deleting feedback: {e}")
        return False

# Feedback Analytics Functions
def get_feedback_stats(db: Session) -> Dict[str, Any]:
    """Get overall feedback statistics"""
    try:
        total_feedback = db.query(Feedback).count()
        helpful_feedback = db.query(Feedback).filter(Feedback.is_helpful == True).count()
        unhelpful_feedback = db.query(Feedback).filter(Feedback.is_helpful == False).count()
        
        # Average rating
        avg_rating_result = db.query(func.avg(Feedback.rating)).filter(Feedback.rating.isnot(None)).scalar()
        avg_rating = round(avg_rating_result, 2) if avg_rating_result else None
        
        # Rating distribution
        rating_dist = {}
        for i in range(1, 6):
            count = db.query(Feedback).filter(Feedback.rating == i).count()
            rating_dist[f"rating_{i}"] = count
        
        # Feedback categories
        category_stats = (
            db.query(Feedback.feedback_category, func.count(Feedback.feedback_category))
            .filter(Feedback.feedback_category.isnot(None))
            .group_by(Feedback.feedback_category)
            .all()
        )
        
        return {
            "total_feedback": total_feedback,
            "helpful_feedback": helpful_feedback,
            "unhelpful_feedback": unhelpful_feedback,
            "helpfulness_rate": round((helpful_feedback / total_feedback * 100), 2) if total_feedback > 0 else 0,
            "average_rating": avg_rating,
            "rating_distribution": rating_dist,
            "feedback_categories": dict(category_stats)
        }
    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return {}

def get_recent_feedback(
    db: Session,
    limit: int = 10,
    include_text: bool = True
) -> List[Dict[str, Any]]:
    """Get recent feedback with chat context"""
    try:
        query = (
            db.query(Feedback, ChatHistory, User)
            .join(ChatHistory, Feedback.chat_history_id == ChatHistory.id)
            .join(User, Feedback.user_id == User.id)
            .order_by(desc(Feedback.timestamp))
            .limit(limit)
        )
        
        results = []
        for feedback, chat, user in query.all():
            result = {
                "feedback_id": feedback.id,
                "username": user.username,
                "rating": feedback.rating,
                "is_helpful": feedback.is_helpful,
                "category": feedback.feedback_category,
                "timestamp": feedback.timestamp,
                "question": chat.question[:100] + "..." if len(chat.question) > 100 else chat.question,
                "answer": chat.answer[:200] + "..." if len(chat.answer) > 200 else chat.answer,
            }
            
            if include_text and feedback.feedback_text:
                result["feedback_text"] = feedback.feedback_text
            
            results.append(result)
        
        return results
    except Exception as e:
        print(f"Error getting recent feedback: {e}")
        return []

def get_low_rated_responses(
    db: Session,
    rating_threshold: int = 2,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get chat responses with low ratings for improvement"""
    try:
        query = (
            db.query(Feedback, ChatHistory)
            .join(ChatHistory, Feedback.chat_history_id == ChatHistory.id)
            .filter(Feedback.rating <= rating_threshold)
            .order_by(desc(Feedback.timestamp))
            .limit(limit)
        )
        
        results = []
        for feedback, chat in query.all():
            results.append({
                "feedback_id": feedback.id,
                "rating": feedback.rating,
                "question": chat.question,
                "answer": chat.answer,
                "feedback_text": feedback.feedback_text,
                "timestamp": feedback.timestamp
            })
        
        return results
    except Exception as e:
        print(f"Error getting low rated responses: {e}")
        return []

def get_feedback_trends(
    db: Session,
    days: int = 30
) -> Dict[str, Any]:
    """Get feedback trends over time"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Daily feedback counts
        daily_feedback = (
            db.query(
                func.date(Feedback.timestamp).label('date'),
                func.count(Feedback.id).label('count'),
                func.avg(Feedback.rating).label('avg_rating')
            )
            .filter(Feedback.timestamp >= cutoff_date)
            .group_by(func.date(Feedback.timestamp))
            .order_by(func.date(Feedback.timestamp))
            .all()
        )
        
        # Helpfulness trend
        helpfulness_trend = (
            db.query(
                func.date(Feedback.timestamp).label('date'),
                func.sum(func.case([(Feedback.is_helpful == True, 1)], else_=0)).label('helpful'),
                func.sum(func.case([(Feedback.is_helpful == False, 1)], else_=0)).label('unhelpful')
            )
            .filter(Feedback.timestamp >= cutoff_date)
            .filter(Feedback.is_helpful.isnot(None))
            .group_by(func.date(Feedback.timestamp))
            .order_by(func.date(Feedback.timestamp))
            .all()
        )
        
        return {
            "daily_feedback": [
                {
                    "date": str(row.date),
                    "count": row.count,
                    "avg_rating": round(float(row.avg_rating), 2) if row.avg_rating else None
                }
                for row in daily_feedback
            ],
            "helpfulness_trend": [
                {
                    "date": str(row.date),
                    "helpful": row.helpful,
                    "unhelpful": row.unhelpful
                }
                for row in helpfulness_trend
            ]
        }
    except Exception as e:
        print(f"Error getting feedback trends: {e}")
        return {"daily_feedback": [], "helpfulness_trend": []}
    

from contextlib import contextmanager
from db_utils import save_chat_history, get_user_by_id
from models import SessionLocal, create_tables, engine

# Initialize database when module is imported
def init_database():
    """Initialize database - cached to run only once"""
    try:
        create_tables()
        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False

# Context manager for database sessions
@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()