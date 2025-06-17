# streamlit_app.py
import streamlit as st
import requests
import uuid
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function to test API connection
def test_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException as e:
        return False, str(e)

# Main UI
st.title("🤖 Agentic Chatbot")

# Connection status
col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"**Session ID:** `{st.session_state.session_id}`")

with col2:
    is_connected, health_data = test_api_connection()
    if is_connected:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.error("**Troubleshooting:**")
        st.error("1. Make sure FastAPI server is running: `python main.py`")
        st.error("2. Set OpenAI API key: `export OPENAI_API_KEY='your-key'`")
        st.error("3. Check if port 8000 is available")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Check API connection first
    if not test_api_connection()[0]:
        st.error("❌ Cannot connect to API. Please check if FastAPI server is running.")
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
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Show request details for debugging
                request_data = {
                    "message": prompt,
                    "session_id": st.session_state.session_id
                }
                
                st.write("**Debug Info:**")
                st.code(f"POST {API_BASE_URL}/chat")
                st.code(json.dumps(request_data, indent=2))
                
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json=request_data,
                    timeout=60,  # Increased timeout
                    headers={"Content-Type": "application/json"}
                )
                
                st.write(f"**Response Status:** {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    
                    # Clear debug info and show response
                    message_placeholder.empty()
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": assistant_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.messages.append(assistant_message)
                    st.caption(f"*{assistant_message['timestamp']}*")
                    
                elif response.status_code == 429:
                    message_placeholder.error("⏰ Rate limit exceeded. Please wait before sending another message.")
                    
                elif response.status_code == 503:
                    message_placeholder.warning("⏳ Server busy. Request queued for processing.")
                    
                else:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("detail", error_detail)
                    except:
                        pass
                    
                    message_placeholder.error(f"❌ Error {response.status_code}: {error_detail}")
                    
            except requests.exceptions.Timeout:
                message_placeholder.error("⏰ Request timed out after 60 seconds. The server might be overloaded.")
                st.error("**Possible solutions:**")
                st.error("• Wait a moment and try again")
                st.error("• Check FastAPI server logs for errors")
                st.error("• Restart the FastAPI server")
                
            except requests.exceptions.ConnectionError:
                message_placeholder.error("🔌 Connection error. Is the FastAPI server running on port 8000?")
                
            except requests.exceptions.RequestException as e:
                message_placeholder.error(f"🚨 Request error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("🔧 System Controls")
    
    # System Status
    if st.button("🔄 Refresh Status", use_container_width=True):
        try:
            status_response = requests.get(f"{API_BASE_URL}/status", timeout=10)
            if status_response.status_code == 200:
                status = status_response.json()
                st.success("📊 System Status")
                st.json(status)
            else:
                st.error(f"Status check failed: {status_response.status_code}")
        except Exception as e:
            st.error(f"Could not fetch status: {e}")
    
    st.divider()
    
    # Session Controls
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.success("Session cleared!")
        st.rerun()
    
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("New session started!")
        st.rerun()
    
    st.divider()
    
    # API Health Check
    st.subheader("🏥 API Health")
    health_status, health_info = test_api_connection()
    
    if health_status:
        st.success("API is healthy ✅")
        if health_info:
            with st.expander("Health Details"):
                st.json(health_info)
    else:
        st.error("API is down ❌")
        st.error(f"Error: {health_info}")
        
        st.subheader("🔧 Troubleshooting Steps:")
        st.markdown("""
        1. **Start FastAPI server:**
           ```bash
           python main.py
           ```
        
        2. **Set OpenAI API key:**
           ```bash
           export OPENAI_API_KEY="your-key-here"
           ```
        
        3. **Check server logs** for any error messages
        
        4. **Test API manually:**
           ```bash
           curl http://localhost:8000/health
           ```
        
        5. **Verify port 8000** is not blocked by firewall
        """)
    
    st.divider()
    
    # Session Info
    st.subheader("📋 Session Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    # Download chat history
    if st.session_state.messages:
        chat_history = "\n\n".join([
            f"**{msg['role'].title()}** ({msg.get('timestamp', 'N/A')}):\n{msg['content']}"
            for msg in st.session_state.messages
        ])
        
        st.download_button(
            label="💾 Download Chat",
            data=chat_history,
            file_name=f"chat_history_{st.session_state.session_id[:8]}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("🤖 **Agentic Chatbot** | Built with Streamlit & FastAPI")