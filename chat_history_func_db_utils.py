# Chat History Functions
# =============================================================================

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