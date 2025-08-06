"""
Modern Streamlit frontend for Cligue - Visual Understanding Chat Assistant.
Generic, attractive, and well-structured UI for any type of video content.
"""
import streamlit as st
import requests
import tempfile
import os
from datetime import datetime
import time

# API Configuration
API_URL = "http://127.0.0.1:8000"

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .event-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .chat-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .suggestion-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def check_api_status():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_video(video_path):
    """Upload video to the backend for analysis."""
    with open(video_path, "rb") as f:
        files = {"file": ("video.mp4", f, "video/mp4")}
        try:
            response = requests.post(f"{API_URL}/upload_video", files=files, timeout=600)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading video: {e}")
            return None


def send_chat_message(message):
    """Send a chat message to the backend."""
    try:
        response = requests.post(f"{API_URL}/chat", json={"message": message})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {e}")
        return None


def get_analysis_data():
    """Get complete analysis data."""
    try:
        response = requests.get(f"{API_URL}/analysis")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting analysis: {e}")
        return None


# Page Configuration
st.set_page_config(
    page_title="Cligue - Visual Understanding Chat Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Cligue</h1>
    <p>Upload any video and chat with an AI assistant about what happened!</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "upload_progress" not in st.session_state:
    st.session_state.upload_progress = 0

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 System Status")
    
    # API Status
    api_status = check_api_status()
    status_color = "🟢" if api_status else "🔴"
    status_text = "Online" if api_status else "Offline"
    st.markdown(f"{status_color} API Server: **{status_text}**")
    
    if not api_status:
        st.warning("⚠️ Please ensure the API server is running on port 8000")
    
    st.markdown("---")
    
    # Video Upload Section
    st.markdown("### 📁 Video Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # File info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📄 File: {uploaded_file.name}\n💾 Size: {file_size:.1f} MB")
        
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            if not api_status:
                st.error("❌ API server is not available")
            else:
                with st.spinner("🔄 Analyzing video... This may take a few minutes."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        video_path = tmp.name
                    
                    # Simulate progress
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("📤 Uploading video...")
                        elif i < 70:
                            status_text.text("🔍 Processing frames...")
                        else:
                            status_text.text("🤖 Analyzing with AI...")
                    
                    analysis_result = upload_video(video_path)
                    os.unlink(video_path)
                    
                    if analysis_result:
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_data = analysis_result
                        st.session_state.messages = [
                            {
                                "role": "assistant",
                                "content": "✅ Video analysis complete! I can help you understand what happened in the video. Ask me anything about it!"
                            }
                        ]
                        st.success("🎉 Analysis Complete!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to analyze video. Please try again.")

# Main Content
if st.session_state.analysis_complete and st.session_state.analysis_data:
    # Analysis Results Dashboard
    st.markdown("## 📊 Analysis Dashboard")
    
    # Key Metrics
    stats = st.session_state.analysis_data.get("statistics", {})
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Total Events</h3>
                <h2>{stats.get('total_events', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Events/Min</h3>
                <h2>{stats.get('events_per_minute', 0):.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Duration</h3>
                <h2>{stats.get('duration_minutes', 0):.1f} min</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            event_types = stats.get('event_types', {})
            total_types = len(event_types)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏷️ Event Types</h3>
                <h2>{total_types}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Content Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Summary Section
        st.markdown("### 📝 Summary")
        summary = st.session_state.analysis_data.get("summary", "No summary available.")
        st.markdown(f"""
        <div class="highlight-box">
            {summary}
        </div>
        """, unsafe_allow_html=True)
        
        # Key Highlights
        highlights = st.session_state.analysis_data.get("key_highlights", [])
        if highlights:
            st.markdown("### ⭐ Key Highlights")
            for i, highlight in enumerate(highlights, 1):
                st.markdown(f"""
                <div class="event-card">
                    <strong>{i}.</strong> {highlight}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Event Types
        st.markdown("### 📋 Event Categories")
        events_by_type = st.session_state.analysis_data.get("events_by_type", {})
        if events_by_type:
            for event_type, events in events_by_type.items():
                with st.expander(f"🔹 {event_type.replace('_', ' ').title()} ({len(events)})"):
                    for event in events[:5]:  # Show first 5 events
                        st.markdown(f"""
                        <div class="event-card">
                            <strong>{event['description']}</strong><br>
                            <small>⏰ {event['timestamp']} | 🏷️ {event['severity']}</small>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No events detected in this video.")

    # Chat Interface
    st.markdown("## 💬 Chat with AI Assistant")
    st.markdown("Ask questions about the video content!")
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("🤔 Thinking..."):
            chat_response = send_chat_message(prompt)
            if chat_response:
                response_text = chat_response.get("response", "Sorry, I couldn't get a response.")
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            else:
                st.error("❌ Failed to get a response from the assistant.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Suggested Questions
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What happened in the video?",
        "Who or what was involved?",
        "What were the key moments?",
        "Describe the main activities",
        "What objects were present?",
        "Tell me about the timeline"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggest_{i}"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                with st.chat_message("user"):
                    st.markdown(suggestion)
                
                with st.spinner("🤔 Thinking..."):
                    chat_response = send_chat_message(suggestion)
                    if chat_response:
                        response_text = chat_response.get("response", "Sorry, I couldn't get a response.")
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        with st.chat_message("assistant"):
                            st.markdown(response_text)

else:
    # Welcome Screen
    st.markdown("""
    <div class="upload-area">
        <h2>🎬 Welcome to Cligue</h2>
        <p>Upload any video and chat with an AI assistant about what happened!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 How it works:")
        st.markdown("""
        1. **📁 Upload a video** (any type - cooking, sports, meetings, etc.)
        2. **🔍 AI analyzes** the video frame by frame
        3. **💬 Chat naturally** about what happened in the video
        4. **❓ Ask questions** about people, objects, activities, and events
        """)
    
    with col2:
        st.markdown("### 💡 Example videos you can try:")
        st.markdown("""
        - 🍳 Cooking videos
        - ⚽ Sports activities
        - 👥 Meeting recordings
        - ✈️ Travel videos
        - 📚 Educational content
        - 🎭 Any video with people, objects, or activities!
        """)
    
    # Features
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Universal Analysis</h3>
            <p>Works with any video type</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered</h3>
            <p>Advanced vision-language model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Natural Chat</h3>
            <p>Conversational AI interface</p>
        </div>
        """, unsafe_allow_html=True)
