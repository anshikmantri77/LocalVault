"""Streamlit frontend for the personal LLM chatbot."""

import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="Personal LLM Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}"
HEADERS = {
    "Authorization": f"Bearer {settings.API_KEY}",
    "Content-Type": "application/json"
}

class ChatbotClient:
    """Client for interacting with the MCP server."""
    
    def __init__(self):
        self.base_url = API_BASE_URL
        self.headers = HEADERS
    
    def check_health(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to connect to server: {e}")
            return None
    
    def send_chat(self, query: str, **kwargs) -> Dict[str, Any]:
        """Send chat message to server."""
        try:
            payload = {"query": query, **kwargs}
            response = requests.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"Server error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def upload_files(self, files) -> Dict[str, Any]:
        """Upload files to server."""
        try:
            files_data = []
            for file in files:
                files_data.append(("files", (file.name, file.getvalue(), file.type)))
            
            response = requests.post(
                f"{self.base_url}/upload",
                headers={"Authorization": f"Bearer {settings.API_KEY}"},
                files=files_data
            )
            
            return response.json() if response.status_code == 200 else {"success": False}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            response = requests.get(
                f"{self.base_url}/embeddings/stats",
                headers=self.headers
            )
            return response.json() if response.status_code == 200 else {}
        except:
            return {}

# Initialize client
client = ChatbotClient()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = client

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Personal LLM Chatbot</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Health check
        health = client.check_health()
        if health:
            st.success("✅ Server Online")
            st.json({
                "Model": health.get("model", "Unknown"),
                "Version": health.get("version", "Unknown"),
                "Documents": health.get("database_stats", {}).get("total_chunks", 0)
            })
        else:
            st.error("❌ Server Offline")
            st.stop()
        
        st.divider()
        
        # File upload
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'jpg', 'jpeg', 'png']
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    result = client.upload_files(uploaded_files)
                    if result.get("success"):
                        st.success(f"✅ Processed {result['files_processed']} files, "
                                 f"created {result['chunks_created']} chunks")
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {result.get('error')}")
        
        st.divider()
        
        # Chat settings
        st.header("⚙️ Chat Settings")
        include_sources = st.checkbox("Show sources", value=True)
        num_results = st.slider("Max results", min_value=1, max_value=10, value=5)
        use_memory = st.checkbox("Use conversation memory", value=True)
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}:</strong> {source['filename']}<br>
                            <strong>Similarity:</strong> {source['similarity']:.3f}<br>
                            <strong>Preview:</strong> {source['content_preview']}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.send_chat(
                    query=prompt,
                    include_sources=include_sources,
                    k=num_results,
                    use_conversation_context=use_memory
                )
                
                if response.get("success"):
                    st.write(response["response"])
                    
                    # Store assistant message
                    assistant_message = {
                        "role": "assistant", 
                        "content": response["response"]
                    }
                    
                    if include_sources and response.get("sources"):
                        assistant_message["sources"] = response["sources"]
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}:</strong> {source['filename']}<br>
                                    <strong>Similarity:</strong> {source['similarity']:.3f}<br>
                                    <strong>Preview:</strong> {source['content_preview']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append(assistant_message)
                else:
                    error_msg = f"❌ Error: {response.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
