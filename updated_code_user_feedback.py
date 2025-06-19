# streamlit_chat_fixed.py - Complete working solution

import streamlit as st
from contextlib import contextmanager
import uuid
from datetime import datetime
import requests
from db_utils import save_chat_history, get_user_by_id
from models import SessionLocal, create_tables, engine

# Initialize database when module is imported
@st.cache_resource
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

def save_chat_with_feedback_support(session_id, user_id, question, answer, response_id=None):
    """Save chat message and return identifiers for feedback"""
    try:
        if not response_id:
            response_id = str(uuid.uuid4())
        
        with get_db_session() as db:
            chat_history = save_chat_history(
                db=db,
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=answer,
                response_id=response_id
            )
            
            return {
                "response_id": response_id,
                "chat_history_id": chat_history.id if chat_history else None,
                "success": True
            }
            
    except Exception as e:
        st.error(f"Failed to save chat: {e}")
        return {
            "response_id": response_id if response_id else str(uuid.uuid4()),
            "chat_history_id": None,
            "success": False,
            "error": str(e)
        }

def render_feedback_ui(message_id, response_id=None, chat_history_id=None):
    """Render feedback UI for a specific message"""
    
    # Debug: Show the IDs (remove this in production)
    if st.checkbox(f"Debug Info {message_id}", key=f"debug_{message_id}"):
        st.write(f"Response ID: {response_id}")
        st.write(f"Chat History ID: {chat_history_id}")
    
    # Check if feedback already given
    if st.session_state.feedback_given.get(message_id, False):
        st.success("✅ Feedback submitted")
        return
    
    # Create feedback form
    with st.expander(f"Rate this response"):
        col1, col2 = st.columns(2)
        
        with col1:
            thumbs_up = st.button("👍", key=f"up_{message_id}", help="Good response")
        with col2:
            thumbs_down = st.button("👎", key=f"down_{message_id}", help="Poor response")
        
        # Optional comment
        comment = st.text_area(
            "Additional feedback (optional):", 
            key=f"comment_{message_id}",
            height=60
        )
        
        if thumbs_up:
            submit_feedback(
                response_id=response_id,
                chat_history_id=chat_history_id,
                message_id=message_id,
                rating=1,
                comment=comment if comment else None
            )
        elif thumbs_down:
            submit_feedback(
                response_id=response_id,
                chat_history_id=chat_history_id,
                message_id=message_id,
                rating=-1,
                comment=comment if comment else None
            )

def submit_feedback(response_id=None, chat_history_id=None, message_id=None, **feedback_data):
    """Submit feedback to the API with better error handling"""
    try:
        # Debug logging
        print(f"Submitting feedback - response_id: {response_id}, chat_history_id: {chat_history_id}")
        
        # Validate we have at least one identifier
        if not response_id and not chat_history_id:
            st.error("Unable to submit feedback: No chat response identifier found")
            print("No identifiers found in submit_feedback")
            return
       
        feedback_payload = {
            "response_id": response_id,
            "chat_history_id": chat_history_id,
            **feedback_data
        }
       
        # Remove message_id and None values from payload
        feedback_payload.pop("message_id", None)
        feedback_payload = {k: v for k, v in feedback_payload.items() if v is not None}
       
        print(f"Feedback payload: {feedback_payload}")
       
        # Get API configuration
        API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        headers = {
            "Authorization": f"Bearer {st.session_state.get('access_token', '')}",
            "Content-Type": "application/json"
        }
       
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json=feedback_payload,
            headers=headers,
            timeout=30
        )
       
        if response.status_code == 200:
            if message_id is not None:
                st.session_state.feedback_given[message_id] = True
            st.success("Thank you for your feedback!")
            st.rerun()
        else:
            try:
                error_detail = response.json().get('detail', response.text)
            except:
                error_detail = response.text
            st.error(f"Failed to submit feedback: {error_detail}")
            print(f"Feedback submission failed: {response.status_code} - {error_detail}")
           
    except requests.exceptions.RequestException as e:
        st.error(f"Network error submitting feedback: {str(e)}")
        print(f"Network error: {e}")
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        print(f"Unexpected error: {e}")

def get_current_user():
    """Get current user from session state"""
    if 'user_id' not in st.session_state:
        return None
    
    try:
        with get_db_session() as db:
            user = get_user_by_id(db, st.session_state.user_id)
            return user
    except Exception as e:
        st.error(f"Error getting current user: {e}")
        return None

def handle_chat_interaction(user_message, ai_response):
    """Handle a complete chat interaction with proper database saving"""
    
    # Initialize database if not already done
    if not init_database():
        st.error("Database not available")
        return None
    
    # Get current user
    current_user = get_current_user()
    if not current_user:
        st.error("Please log in to save chat history")
        return None
    
    # Generate session ID if not exists
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Save the chat interaction
    result = save_chat_with_feedback_support(
        session_id=st.session_state.session_id,
        user_id=current_user.id,
        question=user_message,
        answer=ai_response
    )
    
    if result['success']:
        # Add user message first
        st.session_state.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add assistant message with feedback identifiers
        message_data = {
            "role": "assistant",
            "content": ai_response,
            "response_id": result["response_id"],
            "chat_history_id": result["chat_history_id"],
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.messages.append(message_data)
        return message_data
    else:
        st.error(f"Failed to save chat: {result.get('error', 'Unknown error')}")
        return None

def main_chat_app():
    """Main Streamlit chat application"""
    
    st.title("💬 Chat with AI")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = {}
    
    # Sidebar for debugging (remove in production)
    with st.sidebar:
        st.subheader("Debug Info")
        st.write(f"Total messages: {len(st.session_state.messages)}")
        st.write(f"User ID: {st.session_state.get('user_id', 'Not set')}")
        st.write(f"Session ID: {st.session_state.get('session_id', 'Not set')}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.feedback_given = {}
            st.rerun()
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Add feedback UI only for assistant messages
            if message["role"] == "assistant":
                response_id = message.get("response_id")
                chat_history_id = message.get("chat_history_id")
                
                # Debug: Check if IDs exist
                if not response_id and not chat_history_id:
                    st.warning(f"⚠️ Missing feedback IDs for message {i}")
                
                render_feedback_ui(
                    message_id=i,
                    response_id=response_id,
                    chat_history_id=chat_history_id
                )
    
    # Chat input
    if user_input := st.chat_input("What is your question?"):
        
        # Simulate AI response (replace with your actual AI logic)
        with st.spinner("Generating response..."):
            # Replace this with your actual AI logic
            ai_response = f"This is an AI response to your question: '{user_input}'. " \
                         f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        
        # Handle the complete interaction
        message_data = handle_chat_interaction(user_input, ai_response)
        
        if message_data:
            st.rerun()  # Refresh to show the new messages

# Test function to simulate user login (for testing purposes)
def simulate_login():
    """Simulate user login for testing"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1  # Replace with actual user ID
    if 'access_token' not in st.session_state:
        st.session_state.access_token = "test_token"  # Replace with actual token

if __name__ == "__main__":
    # For testing - simulate login
    simulate_login()
    
    # Run main app
    main_chat_app()