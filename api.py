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

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate

from ingest import run_ingestion, load_faiss_index, FAISS_INDEX_DIR, EMBED_MODEL, EnsembleRAGRetriever

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
rag_chain = None
chat_history = []

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a helpful AI assistant that answers questions based on the provided PDF documents.

Use the following context from the documents to answer the question.
If the answer is not in the context, say "I couldn't find that information in the provided documents."
Always be concise, accurate, and helpful.

IMPORTANT RULES FOR IMAGES/DIAGRAMS:
1. In the context, you will find image entries showing: "File: <pdf_name> | Page: <page_num> | Image index: <image_idx>".
2. If the user asks for a diagram, figure, image, or architecture, or if a diagram is highly relevant to answering their question:
   - Identify the correct image from the context.
   - Pay close attention to the spatial position (e.g. left vs right side of the page) and map it to the corresponding figure caption text (e.g. "(left) ... (right) ...") to make sure you select the correct image.
   - You MUST include the exact tag `[IMAGE REFERENCE: page <page_num>, image <image_index>]` in your answer where appropriate. Do not modify the numbers inside the tag.
   - Example: "The architecture of Multi-Head Attention is shown in [IMAGE REFERENCE: page 4, image 2]. It consists of..."
3. If no images are relevant, do not output any `[IMAGE REFERENCE: ...]` tags.

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
)


# ─────────────────────────────────────────────────────────
#  HELPER: Load RAG Chain
# ─────────────────────────────────────────────────────────
def get_rag_chain(model_name: str = None):
    """Build and return the RAG chain."""
    if not model_name:
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    vectorstore = load_faiss_index()
    retriever = EnsembleRAGRetriever(vectorstore=vectorstore, k=8)
    llm = ChatGroq(
        model=model_name,
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
        verbose=False
    )
    return chain


# ─────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = None   # overrides GROQ_MODEL from .env if provided

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
    global rag_chain

    # Check index exists
    if not os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        raise HTTPException(
            status_code=400,
            detail="FAISS index not found. Please run /ingest first."
        )

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env")

    # Load chain (lazy init)
    if rag_chain is None:
        try:
            rag_chain = get_rag_chain(request.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load RAG chain: {e}")

    # Run query
    try:
        result = rag_chain.invoke({"question": request.question})
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

        # Fallback: If no explicit image referenced, but query asked for diagram,
        # find the first image from source_docs and append it
        has_image = any(s["type"] == "image" for s in sources)
        if not has_image:
            query_lower = request.question.lower()
            needs_image = any(kw in query_lower for kw in [
                "diagram", "figure", "image", "architecture", "illustration",
                "picture", "visual", "chart", "graph", "plot", "screenshot",
                "photo", "drawing", "sketch", "map", "fig"
            ])
            if needs_image:
                for doc in source_docs:
                    if doc.metadata.get("type") == "image":
                        sources.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "?"),
                            "type": "image",
                            "snippet": doc.page_content[:300],
                            "base64_image": doc.metadata.get("base64_image")
                        })
                        break

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
        rag_chain = None   # Reset chain so it reloads with new index
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
