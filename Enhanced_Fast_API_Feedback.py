@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a chat response with enhanced fallback logic"""
    try:
        # Debug logging
        logger.info(f"Feedback submission attempt - user_id: {user.id}, "
                   f"response_id: {feedback.response_id}, "
                   f"chat_history_id: {feedback.chat_history_id}, "
                   f"use_latest_chat: {getattr(feedback, 'use_latest_chat', False)}")
        
        chat_history = None
        
        # Method 1: Try to find by chat_history_id first (most direct)
        if feedback.chat_history_id:
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.id == feedback.chat_history_id,
                ChatHistory.user_id == user.id  # Ensure user owns this chat
            ).first()
            
            if chat_history:
                logger.info(f"Found chat_history by ID: {chat_history.id}")
            else:
                logger.warning(f"Chat history not found by ID: {feedback.chat_history_id}")
        
        # Method 2: Try to find by response_id if chat_history_id didn't work
        if not chat_history and feedback.response_id:
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.response_id == feedback.response_id,
                ChatHistory.user_id == user.id  # Ensure user owns this chat
            ).first()
            
            if chat_history:
                logger.info(f"Found chat_history by response_id: {feedback.response_id}")
            else:
                logger.warning(f"Chat history not found by response_id: {feedback.response_id}")
        
        # Method 3: If still not found, try to get the latest chat for this user
        # This is the fallback method when frontend can't provide identifiers
        if not chat_history:
            logger.info("Attempting to find latest chat for user as fallback")
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.user_id == user.id
            ).order_by(ChatHistory.timestamp.desc()).first()
            
            if chat_history:
                logger.warning(f"Using latest chat as fallback: {chat_history.id}")
                # Optionally, you can add a time window check here
                # to ensure the feedback is for a recent chat (e.g., within last 30 minutes)
                from datetime import datetime, timedelta
                if datetime.utcnow() - chat_history.timestamp > timedelta(minutes=30):
                    logger.warning(f"Latest chat is older than 30 minutes, proceeding anyway")
        
        # Method 4: If still no chat found, check if user has any chats at all
        if not chat_history:
            total_user_chats = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count()
            
            # Enhanced error response with debugging info
            error_msg = "No chat response found for feedback"
            debug_info = {
                "user_id": user.id,
                "response_id": feedback.response_id,
                "chat_history_id": feedback.chat_history_id,
                "total_user_chats": total_user_chats,
                "search_methods_tried": [
                    "chat_history_id" if feedback.chat_history_id else None,
                    "response_id" if feedback.response_id else None,
                    "latest_chat_fallback"
                ]
            }
            
            logger.error(f"Chat not found - Debug info: {debug_info}")
            
            if total_user_chats == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No chat history found. Please send a message first before providing feedback."
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Chat response not found. User has {total_user_chats} total chats but none match the provided identifiers."
                )
        
        # Check if feedback already exists for this chat
        existing_feedback = get_feedback_by_chat_id(db, chat_history.id)
        
        if existing_feedback:
            logger.info(f"Updating existing feedback: {existing_feedback.id}")
            # Update existing feedback
            updated_feedback = update_feedback(
                db,
                existing_feedback.id,
                rating=feedback.rating,
                is_helpful=feedback.is_helpful,
                feedback_text=feedback.feedback_text,
                feedback_category=feedback.feedback_category,
                is_accurate=feedback.is_accurate,
                is_relevant=feedback.is_relevant,
                is_clear=feedback.is_clear,
                is_complete=feedback.is_complete,
                ip_address=str(request.client.host) if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
            
            if not updated_feedback:
                raise HTTPException(status_code=500, detail="Failed to update feedback")
            
            return FeedbackResponse(
                feedback_id=updated_feedback.id,
                message="Feedback updated successfully",
                timestamp=updated_feedback.timestamp
            )
        else:
            logger.info(f"Creating new feedback for chat: {chat_history.id}")
            # Create new feedback
            new_feedback = save_feedback(
                db=db,
                chat_history_id=chat_history.id,
                user_id=user.id,
                session_id=chat_history.session_id,
                rating=feedback.rating,
                is_helpful=feedback.is_helpful,
                feedback_text=feedback.feedback_text,
                feedback_category=feedback.feedback_category,
                is_accurate=feedback.is_accurate,
                is_relevant=feedback.is_relevant,
                is_clear=feedback.is_clear,
                is_complete=feedback.is_complete,
                ip_address=str(request.client.host) if request.client else None,
                user_agent=request.headers.get("user-agent")
            )
            
            return FeedbackResponse(
                feedback_id=new_feedback.id,
                message="Feedback submitted successfully",
                timestamp=new_feedback.timestamp
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


# Enhanced chat endpoint to ensure response_id is always returned
@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced chat endpoint that ensures response identifiers are returned"""
    try:
        # Generate response_id if not provided
        response_id = getattr(request, 'response_id', None) or str(uuid.uuid4())
        
        # Your existing chat logic here...
        # (Replace this with your actual chat processing logic)
        
        # After generating the response, save to database
        chat_history = save_chat_history(
            db=db,
            session_id=request.session_id,
            user_id=user.id,
            question=request.message,
            answer=response_text,  # Your generated response
            response_id=response_id
        )
        
        # Return response with both identifiers
        return {
            "response": response_text,
            "request_id": response_id,  # For backward compatibility
            "response_id": response_id,
            "chat_history_id": chat_history.id,
            "session_id": request.session_id,
            "timestamp": chat_history.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional debug endpoint for troubleshooting
@app.get("/debug/user-chats")
async def get_user_chats_debug(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check user's chat history"""
    try:
        chats = db.query(ChatHistory).filter(
            ChatHistory.user_id == user.id
        ).order_by(ChatHistory.timestamp.desc()).limit(10).all()
        
        return {
            "user_id": user.id,
            "username": user.username,
            "total_chats": db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count(),
            "recent_chats": [
                {
                    "id": chat.id,
                    "response_id": chat.response_id,
                    "session_id": chat.session_id,
                    "question": chat.question[:100] + "..." if len(chat.question) > 100 else chat.question,
                    "timestamp": chat.timestamp.isoformat(),
                    "has_feedback": db.query(Feedback).filter(Feedback.chat_history_id == chat.id).first() is not None
                }
                for chat in chats
            ]
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Updated FeedbackRequest model to include the use_latest_chat flag
from pydantic import BaseModel
from typing import Optional

class FeedbackRequest(BaseModel):
    response_id: Optional[str] = None
    chat_history_id: Optional[int] = None
    rating: Optional[int] = None
    is_helpful: Optional[bool] = None
    feedback_text: Optional[str] = None
    feedback_category: Optional[str] = None
    is_accurate: Optional[bool] = None
    is_relevant: Optional[bool] = None
    is_clear: Optional[bool] = None
    is_complete: Optional[bool] = None
    use_latest_chat: Optional[bool] = False  # New field for fallback