"""
LocalVault — AI Document Assistant
Streamlit frontend backed by local Ollama.

Fixes applied:
  7. Service Instability: health check now handles HTTP 503 (initialising)
     separately from total failure so the user sees "Loading…" not "Offline".
  8. Legacy Code Clutter: sidebar now shows the real Ollama model from .env 
     and lets the user switch between locally available Ollama models.
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LocalVault",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .vault-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid #e94560;
    }
    .stat-card { background: #1e1e2e; border: 1px solid #2d2d44; border-radius: 8px;
                 padding: 1rem 1.2rem; text-align: center; }
    .stat-number { font-size: 1.8rem; font-weight: 600; color: #e94560; }
    .stat-label  { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .msg-user { background: #0f3460; border-left: 3px solid #e94560;
                padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .msg-bot  { background: #1e1e2e; border-left: 3px solid #4a9eff;
                padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .source-pill { display: inline-block; background: #2d2d44; border: 1px solid #3d3d5e;
                   border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.75rem;
                   color: #a8b2d8; margin: 0.2rem; }
    .model-badge { background: #e94560; color: white; padding: 0.15rem 0.6rem;
                   border-radius: 12px; font-size: 0.7rem; font-weight: 500; }
    .stButton > button { border-radius: 6px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
API_KEY  = os.getenv("API_KEY",  "localvault-secret-key")
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def api_health():
    """
    Returns:
      "healthy"      — backend up and ready
      "initializing" — backend up but still loading models
      None           — backend unreachable
    """
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        data = r.json()
        return data.get("status", "healthy") if r.ok else \
               ("initializing" if r.status_code == 503 else None)
    except Exception:
        return None


def api_chat(query, model_id, include_sources, k, use_memory, temperature):
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            headers=HEADERS,
            json={"query": query, "model": model_id, "include_sources": include_sources,
                  "k": k, "use_conversation_context": use_memory, "temperature": temperature},
            timeout=300,
        )
        return r.json() if r.ok else {"success": False, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def api_upload(files, clear_first: bool = False):
    try:
        files_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        r = requests.post(
            f"{API_BASE}/upload",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files=files_data,
            data={"clear_first": str(clear_first).lower()},
            timeout=300,
        )
        return r.json() if r.ok else {"success": False, "error": f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def api_stats():
    try:
        r = requests.get(f"{API_BASE}/embeddings/stats", headers=HEADERS, timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def api_clear_history():
    try:
        requests.post(f"{API_BASE}/conversation/clear", headers=HEADERS, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
            border: 1px solid #e94560;">
  <h1 style="margin:0; font-size:2.2rem; font-weight:800; color:#ffffff;">🔐 LocalVault</h1>
  <p style="margin:0; opacity:0.8; font-size:1.1rem; color:#a8b2d8;">
    Private AI Document Assistant · Powered by Local Ollama
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ System")
    health_status = api_health()

    if health_status == "healthy":
        st.success("🟢 Backend Online")
    elif health_status == "initializing":
        st.warning("🟡 Backend loading models — please wait ~30s then refresh.")
        st.stop()
    else:
        st.error("🔴 Backend Offline — run `python scripts/run_server.py`")
        st.stop()

    stats = api_stats()
    total_chunks = stats.get("total_chunks", 0)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{total_chunks}</div>'
                    f'<div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.messages)}</div>'
                    f'<div class="stat-label">Messages</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Model selector (reads real model from settings, lets user override) ---
    st.markdown("### 🤖 Local Model")
    default_model = settings.OLLAMA_MODEL
    # Common Ollama model options — user can type their own
    KNOWN_MODELS = [
        "qwen3.5:3b",
        "qwen3.5:0.5b",
        "qwen3.5:7b",
        "mistral:7b",
        "llama3.2:3b",
        "phi3:mini",
        "gemma2:2b",
    ]
    if default_model not in KNOWN_MODELS:
        KNOWN_MODELS.insert(0, default_model)

    selected_model = st.selectbox(
        "Ollama model",
        KNOWN_MODELS,
        index=KNOWN_MODELS.index(default_model),
        help="Make sure the model is pulled in Ollama before selecting it.",
    )
    st.caption(f"Running via Ollama on localhost")

    st.markdown("---")

    # --- Document upload ---
    st.markdown("### 📁 Documents")
    clear_on_upload = st.toggle(
        "Wipe existing knowledge", value=False,
        help="Clear old documents before indexing new ones"
    )
    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv", "md"],
        label_visibility="collapsed",
    )
    if uploaded_files:
        if st.button("⬆️ Process & Index", use_container_width=True):
            with st.spinner("Indexing documents…"):
                result = api_upload(uploaded_files, clear_first=clear_on_upload)
            if result.get("success"):
                st.success(
                    f"✅ {result.get('files_processed', 0)} new file(s) · "
                    f"{result.get('chunks_created', 0)} total chunks"
                )
                st.rerun()
            else:
                st.error(f"❌ {result.get('error', 'Upload failed')}")

    st.markdown("---")

    # --- Chat settings ---
    st.markdown("### ⚙️ Settings")
    include_sources = st.toggle("Show sources", value=True)
    use_memory      = st.toggle("Conversation memory", value=True)
    k_results       = st.slider("Chunks to retrieve (k)", 3, 15, 8)
    temperature     = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                                help="Lower = more factual. Keep low for document Q&A.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        api_clear_history()
        st.rerun()

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_chat, tab_docs, tab_about = st.tabs(["💬 Chat", "📚 Document Explorer", "ℹ️ About"])

with tab_chat:
    st.markdown(
        f'<span class="model-badge">🤖 {selected_model}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#555;">
          <div style="font-size:3rem">🔐</div>
          <div style="font-size:1.1rem;margin-top:0.5rem;color:#888;">
            Upload documents and start asking questions
          </div>
          <div style="font-size:0.85rem;margin-top:0.5rem;color:#555;">
            Supports PDF · DOCX · TXT · CSV · Markdown
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Render history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if include_sources and msg.get("sources"):
                    _render_sources(msg["sources"])


def _render_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)} chunks used)"):
        for src in sources:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<span class="source-pill">📄 {src["filename"]}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(src["content_preview"])
            with col_b:
                score = src["similarity"]
                color = "#2ecc71" if score > 0.6 else "#f39c12" if score > 0.4 else "#e74c3c"
                st.markdown(
                    f'<div style="text-align:center;color:{color};font-weight:600">'
                    f'{score:.2f}</div>'
                    f'<div style="text-align:center;font-size:0.7rem;color:#888">relevance</div>',
                    unsafe_allow_html=True,
                )


with tab_chat:
    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}…"):
                resp = api_chat(
                    query=prompt,
                    model_id=selected_model,
                    include_sources=include_sources,
                    k=k_results,
                    use_memory=use_memory,
                    temperature=temperature,
                )

            if resp.get("success"):
                st.write(resp["response"])
                assistant_msg = {
                    "role": "assistant",
                    "content": resp["response"],
                    "sources": resp.get("sources", []),
                }
                if include_sources and resp.get("sources"):
                    _render_sources(resp["sources"])
            else:
                err = resp.get("error", "Unknown error")
                st.error(f"❌ {err}")
                assistant_msg = {"role": "assistant", "content": f"Error: {err}", "sources": []}

            st.session_state.messages.append(assistant_msg)

with tab_docs:
    st.markdown("### 📚 Indexed Knowledge Base")
    if total_chunks == 0:
        st.info("No documents indexed yet. Upload files in the sidebar.")
    else:
        st.success(f"**{total_chunks}** document chunks indexed and ready")
        st.markdown(f"""
| Property | Value |
|---|---|
| Embedding model | `{stats.get('embedding_model', 'all-mpnet-base-v2')}` |
| Vector DB | ChromaDB |
| Total chunks | {total_chunks} |
| Collection | `{stats.get('collection_name', 'document_chunks')}` |
        """)

with tab_about:
    st.markdown("""
### 🔐 LocalVault — Private AI Document Assistant

**LocalVault** is a local RAG (Retrieval-Augmented Generation) system. All data stays
on your machine — nothing is sent to third-party servers.

#### Stack
- **LLM**: Local Ollama (Qwen 3.5, Mistral, LLaMA 3, Phi-3, Gemma 2 …)
- **Embeddings**: `all-mpnet-base-v2` (local, CPU)
- **Vector DB**: ChromaDB (persistent, local)
- **Backend**: FastAPI + uvicorn
- **Frontend**: Streamlit

#### Supported File Types
PDF · DOCX · TXT · CSV · Markdown

#### Tips for best results
- Use **qwen3.5:3b** or higher for document Q&A (0.5b is too small for complex answers).
- Set **Temperature → 0.1** for factual document questions.
- After uploading a new file, wait for indexing to complete before asking questions.
    """)
