"""
=============================================================
  RAG Pipeline - FastAPI Backend
  Endpoints for: Chat, Ingestion status, Health check
  Run with: uvicorn api:app --reload
=============================================================
"""

import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from ingest import run_ingestion, FAISS_INDEX_DIR, EMBED_MODEL
from rag_core import build_chain

load_dotenv()

# ─────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG ChatBot API",
    description="Retrieval-Augmented Generation API using FAISS + Groq",
    version="1.0.0"
)

# Allow frontend (Streamlit / any client) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────
def get_rag_chain(model_name: str = None, session_id: str = "default_session"):
    """Build and return the RAG chain using the shared builder."""
    if not model_name:
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    chain, _ = build_chain(model_name, session_id=session_id)
    return chain

# ─────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = None
    session_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: Optional[int] = None


# ─────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    index_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    return {
        "status": "running",
        "service": "RAG ChatBot API",
        "index_ready": index_exists,
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check."""
    return {
        "api": "ok",
        "faiss_index": os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Ask a question. Returns an answer from the RAG pipeline
    along with source document references.
    """
    # Check index exists
    if not os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        raise HTTPException(
            status_code=400,
            detail="FAISS index not found. Please run /ingest first."
        )
 
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env")
 
    # Load chain dynamically per session
    try:
        session_id = request.session_id or "default_session"
        chain = get_rag_chain(request.model, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load RAG chain: {e}")
 
    # Run query
    try:
        try:
            result = chain.invoke({"question": request.question})
        except Exception as llm_err:
            err_str = str(llm_err).lower()
            # Groq sometimes returns empty output — retry once with a simpler prompt
            if "model output" in err_str or "output text" in err_str or "tool calls" in err_str:
                try:
                    simple_question = f"Briefly answer: {request.question}"
                    result = chain.invoke({"question": simple_question})
                except Exception:
                    return ChatResponse(
                        answer=(
                            "I'm sorry, I couldn't generate a response for that query. "
                            "Please try rephrasing your question."
                        ),
                        sources=[]
                    )
            else:
                raise

        answer = result.get("answer", "No answer generated.")
        source_docs = result.get("source_documents", [])

        # Enrich retrieved image documents with base64 on-demand from disk
        from ingest import enrich_with_base64
        source_docs = enrich_with_base64(source_docs)

        # Extract referenced images from answer (tuples of (page, index))
        referenced_pairs = []
        matches = re.findall(r"\[IMAGE REFERENCE:\s*page\s*(\d+),\s*image\s*(\d+)\]", answer, re.IGNORECASE)
        for page_str, idx_str in matches:
            referenced_pairs.append((int(page_str), int(idx_str)))

        # Clean answer text
        clean_answer = re.sub(
            r"\[IMAGE REFERENCE:\s*page\s*(\d+),\s*image\s*(\d+)\]",
            "the diagram below",
            answer,
            flags=re.IGNORECASE
        )

        sources = []
        for doc in source_docs:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "image":
                pg = doc.metadata.get("page")
                idx = doc.metadata.get("image_index")
                if referenced_pairs and (pg is not None and idx is not None):
                    if (int(pg), int(idx)) not in referenced_pairs:
                        # Skip this image if not referenced
                        continue
            
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "type": doc_type,
                "snippet": doc.page_content[:300],
                "base64_image": doc.metadata.get("base64_image")
            })

        # Fallback: If no explicit image referenced but query asks for a diagram,
        # find the best matching image from source_docs by title keyword matching.
        has_image = any(s["type"] == "image" for s in sources)
        if not has_image:
            query_lower = request.question.lower()
            needs_image = any(kw in query_lower for kw in [
                "diagram", "figure", "image", "architecture", "illustration",
                "picture", "visual", "chart", "graph", "plot", "screenshot",
                "photo", "drawing", "sketch", "map", "fig"
            ])
            if needs_image:
                all_image_docs = [d for d in source_docs if d.metadata.get("type") == "image"]
                query_norm = re.sub(r"[^a-zA-Z0-9\s]", " ", query_lower)

                # Score each image by how many query words appear in its title
                def title_score(doc):
                    title = (doc.metadata.get("figure_title") or "").lower()
                    if not title:
                        first = doc.page_content.splitlines()[0] if doc.page_content else ""
                        title = first.replace("[IMAGE]", "").strip().lower()
                    title_norm = re.sub(r"[^a-zA-Z0-9\s]", " ", title)
                    return sum(1 for w in query_norm.split() if len(w) > 3 and w in title_norm)

                scored = sorted(all_image_docs, key=title_score, reverse=True)
                docs_to_add = [d for d in scored if title_score(d) > 0] or all_image_docs[:1]

                for doc in docs_to_add[:2]:
                    sources.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "?"),
                        "type": "image",
                        "snippet": doc.page_content[:300],
                        "base64_image": doc.metadata.get("base64_image")
                    })

        return ChatResponse(answer=clean_answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest():
    """
    Trigger the PDF ingestion pipeline.
    Extracts text, tables, and images from all PDFs in data/pdfs/
    and builds/rebuilds the FAISS index.
    """
    global rag_chain
    try:
        vectorstore = run_ingestion()
        rag_chain = None
        if vectorstore is None:
            return IngestResponse(
                status="warning",
                message="No PDFs found in data/pdfs/. Please add PDF files.",
                chunks_indexed=0
            )
        return IngestResponse(
            status="success",
            message="Ingestion complete. FAISS index has been built.",
            chunks_indexed=vectorstore.index.ntotal
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/upload", tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to data/pdfs/.
    After uploading, call /ingest to rebuild the index.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("data/pdfs", exist_ok=True)
    save_path = os.path.join("data/pdfs", file.filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "size_kb": round(len(content) / 1024, 2),
        "message": "File saved. Call POST /ingest to rebuild the FAISS index."
    }


@app.delete("/chat/history", tags=["Chat"])
def clear_history():
    """Clear the in-memory conversation history."""
    global rag_chain
    rag_chain = None   # Reinitialise resets memory
    return {"status": "cleared", "message": "Chat history has been reset."}


@app.get("/index/status", tags=["Index"])
def index_status():
    """Check FAISS index details."""
    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return {"exists": False, "message": "No index found. Run POST /ingest."}

    size_mb = round(os.path.getsize(index_path) / (1024 * 1024), 2)
    return {
        "exists": True,
        "index_path": FAISS_INDEX_DIR,
        "index_size_mb": size_mb,
        "message": "Index is ready for querying."
    }
