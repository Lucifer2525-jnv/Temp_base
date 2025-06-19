import streamlit as st
import requests
import uuid
import json
from datetime import datetime
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_utils import *

# Configuration
API_BASE_URL = "http://localhost:8000"
engine = create_engine("sqlite:///./chat.db")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Page config
st.set_page_config(
    page_title="GSC ARB Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Token Authentication
def auth_ui():
    st.sidebar.header("Login / Signup")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("Login"):
            res = requests.post(f"{API_BASE_URL}/token", data={"username": username, "password": password})
            if res.ok: 
                st.session_state.token = res.json()["access_token"]
                st.session_state.username = username
                st.sidebar.success("Logged in successfully!")
            else: 
                st.sidebar.error("Login failed")
    with col2:
        if st.sidebar.button("Signup"):
            res = requests.post(f"{API_BASE_URL}/signup", params={"username": username, "password": password})
            if res.ok: 
                st.sidebar.success("Registered!")
            else: 
                st.sidebar.error("Signup failed")
    if "token" not in st.session_state:
        st.stop()

auth_ui()
token = st.session_state.token
headers = {"Authorization": f"Bearer {token}"}

# FAQs
st.sidebar.header("FAQs")
for q, cnt in get_top_questions(db, limit=5):
    if st.sidebar.button(f"{q[:50]}... ({cnt}× asked)", key=f"faq_{hash(q)}"):
        st.session_state.question = q

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

# Helper function to test API connection
def test_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        return False, str(e)

# Feedback UI Component
def render_feedback_ui(message_id, response_id=None):
    """Render feedback UI for a specific message"""
    if message_id in st.session_state.feedback_given:
        st.success("✅ Feedback submitted! Thank you.")
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
            is_helpful=True,
            message_id=message_id
        )
    elif thumbs_down:
        submit_feedback(
            response_id=response_id,
            is_helpful=False,
            message_id=message_id
        )

def submit_feedback(response_id=None, message_id=None, **feedback_data):
    """Submit feedback to the API"""
    try:
        # If no response_id, try to get it from the last assistant message
        if not response_id and st.session_state.messages:
            # Find the last assistant message with a response_id
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and "response_id" in msg:
                    response_id = msg["response_id"]
                    break
        
        feedback_payload = {
            "response_id": response_id,
            **feedback_data
        }
        
        # Remove message_id from payload as it's only for UI state
        feedback_payload.pop("message_id", None)
        
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
            st.error(f"Failed to submit feedback: {response.text}")
            
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")

# Main UI
st.title("GSC ARB Chatbot")

# Connection status
col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    if "username" in st.session_state:
        st.write(f"**User:** {st.session_state.username}")

with col2:
    is_connected, health_data = test_api_connection()
    if is_connected:
        st.success("ARB Chatbot API Connected")
    else:
        st.error("API Disconnected")

# Display chat messages with feedback
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"*{message['timestamp']}*")
        
        # Add feedback UI for assistant messages
        if message["role"] == "assistant":
            render_feedback_ui(
                message_id=f"msg_{i}", 
                response_id=message.get("response_id")
            )

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Check API connection first
    if not test_api_connection()[0]:
        st.error("Cannot connect to ARB Chatbot API. Please check if FastAPI server is running.")
        st.stop()
    
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"*{user_message['timestamp']}*")
    
    # Get response from API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("ARB Chatbot is Generating Response..."):
            try:
                request_data = {
                    "message": prompt,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json=request_data,
                    timeout=60,
                    headers={"Content-Type": "application/json", **headers}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    response_id = result.get("request_id")  # This should be the response_id
                    
                    # Clear placeholder and show response
                    message_placeholder.empty()
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history with response_id
                    assistant_message = {
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "response_id": response_id
                    }
                    st.session_state.messages.append(assistant_message)
                    st.caption(f"*{assistant_message['timestamp']}*")
                    
                    # Show feedback UI for this new message
                    message_id = f"msg_{len(st.session_state.messages)-1}"
                    render_feedback_ui(message_id=message_id, response_id=response_id)
                    
                elif response.status_code == 429:
                    message_placeholder.error("Rate limit exceeded. Please wait before sending another message.")
                elif response.status_code == 503:
                    message_placeholder.warning("ARB Chatbot Server busy. Request queued for processing.")
                else:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("detail", error_detail)
                    except:
                        pass
                    message_placeholder.error(f"Error {response.status_code}: {error_detail}")
                    
            except requests.exceptions.Timeout:
                message_placeholder.error("Request timed out after 60 seconds.")
            except requests.exceptions.ConnectionError:
                message_placeholder.error("Connection error. Is the FastAPI server running on port 8000?")
            except requests.exceptions.RequestException as e:
                message_placeholder.error(f"Request error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("System Controls")
    
    # System Status
    if st.button("Refresh Status", use_container_width=True):
        try:
            status_response = requests.get(f"{API_BASE_URL}/status", timeout=10, headers=headers)
            if status_response.status_code == 200:
                status = status_response.json()
                st.success("Chatbot System Status")
                st.json(status)
            else:
                st.error(f"Status check failed: {status_response.status_code}")
        except Exception as e:
            st.error(f"Could not fetch status: {e}")
    
    st.divider()
    
    # Feedback Statistics
    st.subheader("📊 Feedback Stats")
    if st.button("View My Feedback", use_container_width=True):
        try:
            feedback_response = requests.get(f"{API_BASE_URL}/feedback/my-feedback", headers=headers, timeout=10)
            if feedback_response.status_code == 200:
                feedback_data = feedback_response.json()
                if feedback_data.get("feedback_history"):
                    st.subheader("Your Feedback History")
                    for feedback in feedback_data["feedback_history"][:5]:  # Show last 5
                        with st.expander(f"Feedback from {feedback['timestamp'][:10]}"):
                            st.write(f"**Rating:** {feedback.get('rating', 'N/A')}")
                            st.write(f"**Helpful:** {'Yes' if feedback.get('is_helpful') else 'No' if feedback.get('is_helpful') is False else 'N/A'}")
                            if feedback.get('feedback_text'):
                                st.write(f"**Comment:** {feedback['feedback_text']}")
                            if feedback.get('chat_question'):
                                st.write(f"**Question:** {feedback['chat_question'][:100]}...")
                else:
                    st.info("No feedback history found.")
            else:
                st.error("Could not load feedback history.")
        except Exception as e:
            st.error(f"Error loading feedback: {e}")
    
    if st.button("Overall Stats", use_container_width=True):
        try:
            stats_response = requests.get(f"{API_BASE_URL}/feedback/stats", headers=headers, timeout=10)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                st.metric("Total Feedback", stats.get("total_feedback", 0))
                st.metric("Helpfulness Rate", f"{stats.get('helpfulness_rate', 0):.1f}%")
                if stats.get("average_rating"):
                    st.metric("Average Rating", f"{stats['average_rating']:.1f}/5")
        except Exception as e:
            st.error(f"Could not load stats: {e}")
    
    st.divider()
    
    # Session Controls
    if st.button("Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        st.success("Session cleared!")
        st.rerun()
    
    if st.button("New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        st.success("New session started!")
        st.rerun()
    
    st.divider()
    
    # API Health Check
    st.subheader("ARB Chatbot API Health")
    health_status, health_info = test_api_connection()
    
    if health_status:
        st.success("ARB Chatbot API is healthy")
        if health_info:
            with st.expander("Health Details"):
                st.json(health_info)
    else:
        st.error("ARB Chatbot API is down")
        st.error(f"Error: {health_info}")
        
        st.subheader("Troubleshooting Steps:")
        st.markdown("""
        1. **Check with Team ARB**
        --> POC: Harshit, Yogesh, Sri (Line Manager)
        """)
    
    st.divider()
    
    # Session Info
    st.subheader("Session Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    # Download chat history
    if st.session_state.messages:
        chat_history = "\n\n".join([
            f"**{msg['role'].title()}** ({msg.get('timestamp', 'N/A')}):\n{msg['content']}"
            for msg in st.session_state.messages
        ])
        
        st.download_button(
            label="Download Chat",
            data=chat_history,
            file_name=f"chat_history_{st.session_state.session_id[:8]}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("**GSC ARB Chatbot** - Team ARB")