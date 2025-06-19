def render_feedback_ui(message_id, response_id=None, chat_history_id=None):
    """Render feedback UI for a specific message"""
    if message_id in st.session_state.feedback_given:
        st.success("Feedback submitted! Thank you..!!")
        return
   
    st.markdown("---")
    st.markdown("**Was this response helpful?**")
   
    # Create unique keys for this message
    thumbs_key = f"thumbs_{message_id}"
    rating_key = f"rating_{message_id}"
    text_key = f"text_{message_id}"
    category_key = f"category_{message_id}"
   
    col1, col2, col3 = st.columns([1, 1, 2])
   
    with col1:
        thumbs_up = st.button("👍 Helpful", key=f"up_{message_id}")
    with col2:
        thumbs_down = st.button("👎 Not Helpful", key=f"down_{message_id}")
   
    # Detailed feedback form
    with st.expander("Provide detailed feedback (optional)"):
        rating = st.select_slider(
            "Rate this response (1-5 stars)",
            options=[1, 2, 3, 4, 5],
            value=3,
            key=rating_key
        )
       
        feedback_category = st.selectbox(
            "What aspect needs improvement?",
            ["accuracy", "helpfulness", "clarity", "completeness", "relevance", "other"],
            key=category_key
        )
       
        feedback_text = st.text_area(
            "Additional comments",
            placeholder="Tell us how we can improve...",
            key=text_key
        )
       
        # Detailed ratings
        col_acc, col_rel, col_clear, col_comp = st.columns(4)
        with col_acc:
            is_accurate = st.checkbox("Accurate", key=f"acc_{message_id}")
        with col_rel:
            is_relevant = st.checkbox("Relevant", key=f"rel_{message_id}")
        with col_clear:
            is_clear = st.checkbox("Clear", key=f"clear_{message_id}")
        with col_comp:
            is_complete = st.checkbox("Complete", key=f"comp_{message_id}")
       
        submit_detailed = st.button("Submit Detailed Feedback", key=f"submit_{message_id}")
       
        if submit_detailed:
            submit_feedback(
                response_id=response_id,
                chat_history_id=chat_history_id,
                rating=rating,
                is_helpful=None,
                feedback_text=feedback_text,
                feedback_category=feedback_category,
                is_accurate=is_accurate,
                is_relevant=is_relevant,
                is_clear=is_clear,
                is_complete=is_complete,
                message_id=message_id
            )
   
    # Handle thumbs up/down
    if thumbs_up:
        submit_feedback(
            response_id=response_id,
            chat_history_id=chat_history_id,
            is_helpful=True,
            message_id=message_id
        )
    elif thumbs_down:
        submit_feedback(
            response_id=response_id,
            chat_history_id=chat_history_id,
            is_helpful=False,
            message_id=message_id
        )


def submit_feedback(response_id=None, chat_history_id=None, message_id=None, **feedback_data):
    """Submit feedback to the API"""
    try:
        # Debug logging
        print(f"Submitting feedback - response_id: {response_id}, chat_history_id: {chat_history_id}")
        
        # If no identifiers provided, try to get them from session state
        if not response_id and not chat_history_id and st.session_state.messages:
            # Find the last assistant message with identifiers
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    response_id = msg.get("response_id")
                    chat_history_id = msg.get("chat_history_id") 
                    if response_id or chat_history_id:
                        break
        
        # Validate we have at least one identifier
        if not response_id and not chat_history_id:
            st.error("Unable to submit feedback: No chat response identifier found")
            return
       
        feedback_payload = {
            "response_id": response_id,
            "chat_history_id": chat_history_id,
            **feedback_data
        }
       
        # Remove message_id from payload as it's only for UI state
        feedback_payload.pop("message_id", None)
        
        # Remove None values to avoid sending unnecessary data
        feedback_payload = {k: v for k, v in feedback_payload.items() if v is not None}
       
        print(f"Feedback payload: {feedback_payload}")  # Debug logging
       
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json=feedback_payload,
            headers=headers,
            timeout=30
        )
       
        if response.status_code == 200:
            st.session_state.feedback_given[message_id] = True
            st.success("Thank you for your feedback!")
            st.rerun()
        else:
            error_detail = response.json().get('detail', response.text) if response.headers.get('content-type') == 'application/json' else response.text
            st.error(f"Failed to submit feedback: {error_detail}")
            print(f"Feedback submission failed: {response.status_code} - {error_detail}")
           
    except requests.exceptions.RequestException as e:
        st.error(f"Network error submitting feedback: {str(e)}")
        print(f"Network error: {e}")
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        print(f"Unexpected error: {e}")


def save_chat_message_with_feedback_support(session_id, user_id, question, answer, response_id=None):
    """
    Save chat message and return both response_id and chat_history_id for feedback
    This should be called when saving chat responses to ensure proper feedback linking
    """
    try:
        if not response_id:
            response_id = str(uuid.uuid4())
        
        # Save to database (you'll need to adapt this to your database saving logic)
        chat_history = save_chat_history(
            db=get_db_session(),  # You'll need to implement this
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            response_id=response_id
        )
        
        return {
            "response_id": response_id,
            "chat_history_id": chat_history.id
        }
    except Exception as e:
        print(f"Error saving chat with feedback support: {e}")
        return None


def add_assistant_message_with_feedback(message_content, response_id=None, chat_history_id=None):
    """
    Add assistant message to session state with proper feedback identifiers
    """
    import uuid
    
    if not response_id:
        response_id = str(uuid.uuid4())
    
    message = {
        "role": "assistant",
        "content": message_content,
        "response_id": response_id,
        "chat_history_id": chat_history_id,
        "timestamp": datetime.now().isoformat()
    }
    
    st.session_state.messages.append(message)
    return message