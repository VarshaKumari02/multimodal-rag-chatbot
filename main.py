"""
=============================================================
  RAG Pipeline - Main Entry Point
  UI: Streamlit Chat Interface
  LLM: Groq (llama3 / mixtral)
  Vector DB: FAISS
  Embeddings: HuggingFace (sentence-transformers)
=============================================================
"""

import os
import streamlit as st
import re
import base64
import html
from dotenv import load_dotenv

from ingest import run_ingestion, FAISS_INDEX_DIR, EMBED_MODEL
from rag_core import build_chain, SQLiteChatMessageHistory

load_dotenv()

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(99,102,241,0.3);
    }
    .header-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    .header-banner p {
        color: rgba(255,255,255,0.8);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem 15%;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .bot-message {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 15% 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-size: 0.95rem;
        line-height: 1.6;
        backdrop-filter: blur(10px);
    }
    .message-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        opacity: 0.7;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .status-ready {
        background: rgba(16,185,129,0.2);
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.3);
    }
    .status-not-ready {
        background: rgba(239,68,68,0.2);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.3);
    }

    /* Streamlit overrides */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        padding: 0.7rem 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stFileUploader {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(99,102,241,0.4) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    .stSpinner > div { border-top-color: #8b5cf6 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  GROQ LLM SETUP
# ─────────────────────────────────────────────────────────
GROQ_MODELS = {
    "Llama 3.1 8B (Fastest)":    "llama-3.1-8b-instant",
    "Llama 3.3 70B (Best)":      "llama-3.3-70b-versatile",
    "Mixtral 8x7B (Balanced)":   "mixtral-8x7b-32768",
    "Gemma2 9B":                 "gemma2-9b-it",
}

@st.cache_resource(show_spinner=False)
def load_rag_chain(model_name: str):
    """Load FAISS index and build the conversational RAG chain using the shared builder."""
    return build_chain(model_name, session_id="streamlit_session")

# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        # Model selector
        st.markdown("**🤖 Groq Model**")
        # Use GROQ_MODEL from .env as default if set
        env_model = os.getenv("GROQ_MODEL", "")
        default_idx = 0
        for i, v in enumerate(GROQ_MODELS.values()):
            if v == env_model:
                default_idx = i
                break
        selected_label = st.selectbox(
            "Choose LLM",
            options=list(GROQ_MODELS.keys()),
            index=default_idx,
            label_visibility="collapsed"
        )
        model_name = GROQ_MODELS[selected_label]

        st.markdown("---")

        # FAISS index status
        st.markdown("**📦 Knowledge Base**")
        index_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
        if index_exists:
            st.markdown('<span class="status-badge status-ready">✓ Index Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-not-ready">✗ Not Indexed</span>', unsafe_allow_html=True)

        st.markdown("---")

        # PDF Upload
        st.markdown("**📄 Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            os.makedirs("data/pdfs", exist_ok=True)
            for f in uploaded_files:
                save_path = os.path.join("data/pdfs", f.name)
                with open(save_path, "wb") as out:
                    out.write(f.read())
            st.success(f"{len(uploaded_files)} PDF(s) saved!")

        # Build Index button
        if st.button("🔄 Build / Rebuild Index", use_container_width=True):
            with st.spinner("Ingesting PDFs... this may take a moment"):
                try:
                    run_ingestion()
                    st.cache_resource.clear()
                    st.success("Index built successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            SQLiteChatMessageHistory("streamlit_session").clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75rem; color: rgba(255,255,255,0.4); text-align:center;'>
            Built with LangChain · FAISS · Groq<br>
            Supports Text · Tables · Images
        </div>
        """, unsafe_allow_html=True)

    return model_name


# ─────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────
def main():
    # Sidebar
    model_name = render_sidebar()

    # Header
    st.markdown("""
    <div class="header-banner">
        <h1>🤖 RAG ChatBot</h1>
        <p>Ask anything from your PDF documents · Powered by Groq + FAISS</p>
    </div>
    """, unsafe_allow_html=True)

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        db_history = SQLiteChatMessageHistory("streamlit_session")
        for msg in db_history.messages:
            role = "user" if msg.type == "human" else "assistant"
            st.session_state.messages.append({
                "role": role,
                "content": msg.content,
                "sources": []
            })

    # Check FAISS index
    index_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    if not index_exists:
        st.warning("⚠️ No knowledge base found. Please upload PDF files and click **'Build / Rebuild Index'** in the sidebar.")
        st.info("📌 Steps:\n1. Upload your PDFs in the sidebar\n2. Click 'Build / Rebuild Index'\n3. Start chatting!")
        return

    # Check Groq API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY not found in your `.env` file. Please add it and restart.")
        return

    # Load RAG chain
    try:
        with st.spinner("Loading RAG chain..."):
            chain, vectorstore = load_rag_chain(model_name)
    except Exception as e:
        st.error(f"Failed to load RAG chain: {e}")
        return

    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding: 3rem 0; color: rgba(255,255,255,0.35);">
                <div style="font-size:3rem;">💬</div>
                <p style="font-size:1rem; margin-top:0.5rem;">Ask a question about your documents to get started</p>
            </div>
            """, unsafe_allow_html=True)

        # Get last user query safely before the render loop
        last_user_query = ""
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_query = msg["content"].lower()
                break

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                escaped_user_content = html.escape(msg["content"])
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-label">You</div>
                    {escaped_user_content}
                </div>""", unsafe_allow_html=True)
            else:
                clean_content = re.sub(
                    r"\[IMAGE REFERENCE:\s*page\s*(\d+),\s*image\s*(\d+)\]",
                    "the diagram below",
                    msg["content"],
                    flags=re.IGNORECASE
                )
                
                st.markdown(f"""
                <div class="bot-message">
                    <div class="message-label">🤖 Assistant</div>
                    {clean_content}
                </div>""", unsafe_allow_html=True)

                if "sources" in msg and msg["sources"]:
                    referenced_pairs = []
                    matches = re.findall(
                        r"\[IMAGE REFERENCE:\s*page\s*(\d+),\s*image\s*(\d+)\]",
                        msg["content"],
                        re.IGNORECASE
                    )
                    for page_str, idx_str in matches:
                        referenced_pairs.append((int(page_str), int(idx_str)))

                    # Filter the image sources based on references
                    if referenced_pairs:
                        image_sources = [
                            doc for doc in msg["sources"]
                            if doc.metadata.get("type") == "image"
                            and (doc.metadata.get("base64_image") or os.path.exists(doc.metadata.get("image_path", "")))
                            and (int(doc.metadata.get("page", -1)), int(doc.metadata.get("image_index", -1))) in referenced_pairs
                        ]
                    else:
                        # Fallback: if user asked for a diagram/architecture but LLM didn't generate tag, show top-1
                        image_keywords = [
                            "diagram", "figure", "image", "architecture", "illustration",
                            "picture", "visual", "chart", "graph", "plot", "screenshot",
                            "photo", "drawing", "sketch", "map", "fig"
                        ]
                        needs_image = any(kw in last_user_query for kw in image_keywords)
                        
                        all_image_sources = [
                            doc for doc in msg["sources"]
                            if doc.metadata.get("type") == "image"
                            and (doc.metadata.get("base64_image") or os.path.exists(doc.metadata.get("image_path", "")))
                        ]
                        if needs_image and all_image_sources:
                            image_sources = all_image_sources[:3]
                        else:
                            image_sources = []
                    for doc in image_sources:
                        meta       = doc.metadata
                        src        = meta.get("source", "Unknown")
                        pg         = meta.get("page", "?")
                        img_path   = meta.get("image_path", "")
                        base64_img = meta.get("base64_image")

                        # Prefer Base64 decoded bytes, fallback to file path
                        display_img = None
                        if base64_img:
                            try:
                                import base64
                                if "," in base64_img:
                                    _, base64_data = base64_img.split(",", 1)
                                else:
                                    base64_data = base64_img
                                display_img = base64.b64decode(base64_data)
                            except Exception:
                                display_img = img_path
                        else:
                            display_img = img_path

                        # Use columns to constrain image to ~40% of chat width
                        col1, col2, col3 = st.columns([2, 3, 2])
                        with col2:
                            st.image(
                                display_img,
                                caption=f"📌 Figure from '{src}' — Page {pg}",
                                width=400
                            )

                    # ── 2. Non-image sources stay in collapsible section ──
                    text_sources = [
                        doc for doc in msg["sources"]
                        if doc.metadata.get("type") != "image"
                    ]
                    if text_sources:
                        with st.expander("📎 Source References", expanded=False):
                            for i, doc in enumerate(text_sources, 1):
                                meta = doc.metadata
                                src  = meta.get("source", "Unknown")
                                pg   = meta.get("page", "?")
                                typ  = meta.get("type", "text").upper()
                                st.markdown(f"`[{i}]` **{src}** — Page {pg} · *{typ}*")
                                st.markdown(f"> {doc.page_content[:300]}...")

    # ── Chat input (fires ONCE per submit — no repeated triggers) ──
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input and user_input.strip():
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get RAG answer
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke({"question": user_input})
                answer  = result.get("answer", "Sorry, I couldn't generate a response.")
                sources = result.get("source_documents", [])

                # Enrich retrieved image documents with base64 on-demand from disk
                from ingest import enrich_with_base64
                sources = enrich_with_base64(sources)

            except Exception as e:
                answer  = f"Error generating response: {e}"
                sources = []

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.rerun()


if __name__ == "__main__":
    main()
