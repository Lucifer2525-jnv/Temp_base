# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from sqlalchemy.pool import StaticPool
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Create declarative base
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chat_histories = relationship("ChatHistory", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Add a unique identifier for this chat response
    response_id = Column(String(36), unique=True, index=True, nullable=True)  # UUID for feedback linking
    
    # Relationships
    user = relationship("User", back_populates="chat_histories")
    feedbacks = relationship("Feedback", back_populates="chat_history")

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), index=True, nullable=False)
    
    # Feedback types
    rating = Column(Integer, nullable=True)  # 1-5 star rating
    is_helpful = Column(Boolean, nullable=True)  # thumbs up/down
    
    # Detailed feedback
    feedback_text = Column(Text, nullable=True)  # Optional text feedback
    feedback_category = Column(String(50), nullable=True)  # accuracy, helpfulness, clarity, etc.
    
    # Specific issues
    is_accurate = Column(Boolean, nullable=True)
    is_relevant = Column(Boolean, nullable=True)
    is_clear = Column(Boolean, nullable=True)
    is_complete = Column(Boolean, nullable=True)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String(45), nullable=True)  # For analytics
    user_agent = Column(String(500), nullable=True)  # For analytics
    
    # Relationships
    user = relationship("User", back_populates="feedbacks")
    chat_history = relationship("ChatHistory", back_populates="feedbacks")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat.db")

# Create engine with proper configuration
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    poolclass=StaticPool if "sqlite" in DATABASE_URL else None,
    echo=False,  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    """Database session dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

# Initialize tables when module is imported
create_tables()