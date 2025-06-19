@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a chat response"""
    try:
        # Debug logging
        logger.info(f"Feedback submission attempt - user_id: {user.id}, "
                   f"response_id: {feedback.response_id}, "
                   f"chat_history_id: {feedback.chat_history_id}")
        
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
        if not chat_history:
            logger.info("Attempting to find latest chat for user")
            chat_history = db.query(ChatHistory).filter(
                ChatHistory.user_id == user.id
            ).order_by(ChatHistory.timestamp.desc()).first()
            
            if chat_history:
                logger.warning(f"Using latest chat as fallback: {chat_history.id}")
        
        if not chat_history:
            # Enhanced error response with debugging info
            error_msg = "Chat response not found"
            debug_info = {
                "user_id": user.id,
                "response_id": feedback.response_id,
                "chat_history_id": feedback.chat_history_id,
                "total_user_chats": db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count()
            }
            logger.error(f"Chat not found - Debug info: {debug_info}")
            raise HTTPException(
                status_code=404, 
                detail=f"{error_msg}. Debug: User has {debug_info['total_user_chats']} total chats."
            )
        
        # Check if feedback already exists
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


# Additional helper endpoint for debugging
@app.get("/debug/user-chats")
async def get_user_chats_debug(
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check user's chat history"""
    try:
        chats = db.query(ChatHistory).filter(
            ChatHistory.user_id == user.id
        ).order_by(ChatHistory.timestamp.desc()).limit(5).all()
        
        return {
            "user_id": user.id,
            "total_chats": db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count(),
            "recent_chats": [
                {
                    "id": chat.id,
                    "response_id": chat.response_id,
                    "question": chat.question[:100] + "..." if len(chat.question) > 100 else chat.question,
                    "timestamp": chat.timestamp
                }
                for chat in chats
            ]
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))