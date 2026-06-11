"""
=============================================================
  RAG Core shared module
  Houses the unified prompt, retriever, memory and chain.
=============================================================
"""

import os
import sqlite3
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_classic.prompts import PromptTemplate

# Configuration (mirrors ingest.py)
FAISS_INDEX_DIR = "faiss_index"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"

# Cached vectorstore and embeddings
_shared_vectorstore = None

def get_shared_vectorstore() -> FAISS:
    """Lazy load and cache the FAISS vectorstore to speed up concurrent api/ui tasks."""
    global _shared_vectorstore
    if _shared_vectorstore is None:
        from ingest import load_faiss_index
        _shared_vectorstore = load_faiss_index()
    return _shared_vectorstore

# ─────────────────────────────────────────────────────────
#  RAG PROMPT TEMPLATE (Soften Rule - Issue 5)
# ─────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a helpful AI assistant that answers questions based on the provided PDF documents.

Use the following context from the documents to answer the question.
If the answer is not in the context, say "I couldn't find that information in the provided documents."
Always be accurate, informative, and helpful.

STRICT RULE — NO HALLUCINATION:
Base your answer ONLY on what is explicitly stated in the context below.
Do NOT add components, layers, or steps that are not mentioned in the context.

RULES FOR REFERENCING IMAGES/DIAGRAMS:
The context below may contain image entries. Each image entry looks like:

  [IMAGE] <figure title or name>
  PDF caption: <caption from the PDF if available>
  <vision model description and/or OCR text labels>
  File: <pdf_name> | Page: <page_num> | Image index: <image_idx>
  Position: <left / right / center of the page>

CRITICAL IMAGE MATCHING RULE:
- Read the [IMAGE] title and PDF caption of EVERY image entry in the context.
- Only reference an image if its [IMAGE] title or PDF caption DIRECTLY matches what
  the user is asking about.
- NEVER reference an image just because it appears in the context — match by name.
- For example:
    * If the user asks about "Multi-Head Attention diagram", ONLY reference images
      whose [IMAGE] title or PDF caption contains "Multi-Head Attention".
    * If the user asks about "Scaled Dot-Product Attention", ONLY reference images
      whose [IMAGE] title or PDF caption contains "Scaled Dot-Product".
    * If the user asks about the "Transformer architecture", ONLY reference images
      whose [IMAGE] title contains "Transformer" or "model architecture".
- Do NOT confuse diagrams. "Multi-Head Attention" and "Scaled Dot-Product Attention"
  are TWO DIFFERENT diagrams that appear on the same page. Use Page + Position fields
  to distinguish them.

WHEN TO EMIT AN IMAGE REFERENCE TAG:
- If the user asks about a diagram, figure, visual, or architectural component
- AND an image entry in the context has a matching title/caption

HOW TO EMIT IT:
- Include the EXACT tag `[IMAGE REFERENCE: page <page_num>, image <image_index>]`
  using the numbers from the "Page:" and "Image index:" fields of ONLY the matching image.
- Do NOT emit tags for images that don't match the question topic.
- Place the tag immediately before your explanation of that specific figure.

FORMAT WHEN MULTIPLE IMAGES ARE RELEVANT:
[IMAGE REFERENCE: page <N>, image <M>]
<explanation of this figure based only on its description>

[IMAGE REFERENCE: page <X>, image <Y>]
<explanation of this figure based only on its description>

After each tag, explain that diagram using ONLY what the description and OCR text labels state.
Do not say "likely" or "it is difficult to say" — base the answer strictly on the provided image details.

If no image in the context matches the question, do not output any [IMAGE REFERENCE: ...] tags.

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
)

# ─────────────────────────────────────────────────────────
#  SQLITE PERSISTENT CHAT HISTORY (Issue 7)
# ─────────────────────────────────────────────────────────
class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, db_path: str = "chat_history.db"):
        self.session_id = session_id
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT
                )
            """)
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,)
            )
            rows = cursor.fetchall()
        msgs = []
        for role, content in rows:
            if role == "human":
                msgs.append(HumanMessage(content=content))
            elif role == "ai":
                msgs.append(AIMessage(content=content))
        return msgs

    def _get_role(self, msg: BaseMessage) -> str:
        t = msg.type.lower()
        if "human" in t or "user" in t:
            return "human"
        elif "ai" in t or "assistant" in t:
            return "ai"
        cls_name = msg.__class__.__name__.lower()
        if "human" in cls_name or "user" in cls_name:
            return "human"
        return "ai"

    def add_message(self, message: BaseMessage) -> None:
        role = self._get_role(message)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                (self.session_id, role, message.content)
            )
            conn.commit()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for msg in messages:
                role = self._get_role(msg)
                conn.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (self.session_id, role, msg.content)
                )
            conn.commit()

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (self.session_id,))
            conn.commit()

# ─────────────────────────────────────────────────────────
#  UNIFIED RETRIEVER (Issue 2 & 3)
# ─────────────────────────────────────────────────────────
class EnsembleRAGRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 10
    OVERSAMPLE_K: int = 80

    def _search_with_score(self, query: str, doc_type: str, limit: int) -> List[Document]:
        """
        Perform semantic similarity search, filter by type, and return the
        top `limit` results ranked by similarity score (most similar first).
        Using similarity_search_with_score so we can rank properly.
        """
        candidates = self.vectorstore.similarity_search_with_score(query, k=self.OVERSAMPLE_K)
        # FAISS returns (doc, score) where LOWER score = more similar (L2 distance)
        type_hits = [
            (doc, score) for doc, score in candidates
            if doc.metadata.get("type") == doc_type
        ]
        # Sort ascending by score so most similar image comes first
        type_hits.sort(key=lambda x: x[1])
        return [doc for doc, _ in type_hits[:limit]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # Retrieve texts, images, and tables semantically — images ranked by similarity
        text_docs  = self._search_with_score(query, "text",  limit=self.k)
        image_docs = self._search_with_score(query, "image", limit=2)
        table_docs = self._search_with_score(query, "table", limit=1)

        # Merge: images and tables first (more specific), then text, deduplicated
        seen = set()
        merged = []
        for d in image_docs + table_docs + text_docs:
            uid = d.metadata.get("chunk_id") or hash(d.page_content)
            if uid not in seen:
                seen.add(uid)
                merged.append(d)

        return merged

# ─────────────────────────────────────────────────────────
#  CHAIN BUILDER (Issue 6)
# ─────────────────────────────────────────────────────────
def build_chain(model_name: str, session_id: str):
    """Unified chain builder for both FastAPI and Streamlit."""
    vectorstore = get_shared_vectorstore()
    retriever = EnsembleRAGRetriever(vectorstore=vectorstore, k=10)

    llm = ChatGroq(
        model=model_name,
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    history = SQLiteChatMessageHistory(session_id=session_id)
    
    # Retrieve max token limit from environment (default to 1000 tokens)
    limit_str = os.getenv("HISTORY_TOKEN_LIMIT", "1000")
    try:
        max_token_limit = int(limit_str)
    except ValueError:
        max_token_limit = 1000

    memory = ConversationTokenBufferMemory(
        llm=llm,
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        max_token_limit=max_token_limit
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
        verbose=False
    )
    return chain, vectorstore
