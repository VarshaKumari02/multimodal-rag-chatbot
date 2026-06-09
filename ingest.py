"""
=============================================================
  RAG Ingestion Pipeline
  Handles: Text + Tables + Images from PDFs
  Vector Store: FAISS
  Embeddings: HuggingFace (sentence-transformers)
=============================================================
"""

import os
import fitz
import pdfplumber
from PIL import Image
import io
import base64
import hashlib
import uuid
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

load_dotenv()

# ─────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────
PDF_DIR         = "data/pdfs"
IMAGES_DIR      = "data/images"
FAISS_INDEX_DIR = "faiss_index"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

MIN_IMAGE_WIDTH  = 100
MIN_IMAGE_HEIGHT = 100

# ─────────────────────────────────────────────────────────
#  BLIP MODEL CACHE
# ─────────────────────────────────────────────────────────
_blip_processor = None
_blip_model     = None
_blip_available = None


def initialize_models():
    """Pre-load BLIP model and processor at startup."""
    global _blip_processor, _blip_model, _blip_available
    if _blip_available is None:
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            print("\n  Pre-loading BLIP captioning model...")
            _blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _blip_model = _blip_model.to(_device)
            _blip_model.eval()
            _blip_available = True
            print(f"  BLIP model pre-loaded successfully on {_device.upper()}.\n")
        except Exception as e:
            print(f"  BLIP pre-loading failed ({e}) — falling back to generic captions.\n")
            _blip_available = False


# ─────────────────────────────────────────────────────────
#  IMAGE CAPTIONING HELPER
# ─────────────────────────────────────────────────────────
def generate_image_caption(image: Image.Image, source_hint: str = "") -> str:
    """
    Generate a rich text description using BLIP + OCR.
    Returns merged caption + OCR string.
    """
    global _blip_processor, _blip_model, _blip_available

    caption = ""

    if _blip_available is None:
        initialize_models()

    if _blip_available:
        try:
            import torch
            _device = next(_blip_model.parameters()).device
            inputs = _blip_processor(image, return_tensors="pt").to(_device)
            with torch.no_grad():
                out = _blip_model.generate(**inputs, max_new_tokens=60)
            caption = _blip_processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"    BLIP inference failed: {e}")
            caption = "visual content: diagram or figure"
    else:
        caption = "visual content: diagram or figure"

    # OCR — append any embedded text
    try:
        import pytesseract
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
        ocr_text = pytesseract.image_to_string(image).strip()
        if ocr_text and len(ocr_text) > 10:
            caption += f"\nOCR text: {ocr_text}"
    except Exception:
        pass

    if source_hint:
        caption += f"\nSource: {source_hint}"

    return caption


# ─────────────────────────────────────────────────────────
#  HELPER — load base64 on-demand from disk
#  Call this in your query/UI layer, NOT during ingestion.
# ─────────────────────────────────────────────────────────
def get_image_base64(image_path: str) -> str:
    """
    Load an image from disk and return a base64 data URI.
    Use this at query time rather than storing base64 inside FAISS docs.

    Args:
        image_path: Absolute or relative path to the saved image file.

    Returns:
        data:<mime>;base64,<encoded> string ready for <img src="...">.
    """
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "gif": "gif", "webp": "webp", "bmp": "bmp"}
    mime = mime_map.get(ext, "png")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


# ─────────────────────────────────────────────────────────
#  STEP 1: EXTRACT TEXT
# ─────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> list[Document]:
    """Extract plain text from each page using PyMuPDF."""
    docs = []
    pdf = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "type": "text"
                }
            ))
    pdf.close()
    return docs


# ─────────────────────────────────────────────────────────
#  STEP 2: EXTRACT TABLES
# ─────────────────────────────────────────────────────────
def extract_tables(pdf_path: str) -> list[Document]:
    """Extract tables from PDF using pdfplumber."""
    docs = []
    filename = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables, start=1):
                if not table:
                    continue
                table_text = ""
                for row in table:
                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                    table_text += " | ".join(clean_row) + "\n"

                if table_text.strip():
                    docs.append(Document(
                        page_content=f"[TABLE {table_idx} - Page {page_num}]\n{table_text}",
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "type": "table",
                            "table_index": table_idx
                        }
                    ))
    return docs


# ─────────────────────────────────────────────────────────
#  STEP 3: EXTRACT & SAVE IMAGES
# ─────────────────────────────────────────────────────────
def extract_images(pdf_path: str) -> list[Document]:
    """
    Extract embedded images from PDF pages.

    1: page_content is now keyword-dense (filename, page number, figure
            labels, BLIP caption, OCR text, size) so embedding-based retrieval
            can actually match visual queries.

    2: base64 is NOT stored in FAISS metadata. Only image_path is stored.
            Call get_image_base64(doc.metadata["image_path"]) at query time.
    """
    docs = []
    filename = os.path.basename(pdf_path)
    pdf_stem = os.path.splitext(filename)[0]

    seen_xrefs: set[int] = set()
    seen_hashes: set[str] = set()

    os.makedirs(IMAGES_DIR, exist_ok=True)

    pdf = fitz.open(pdf_path)

    for page_num, page in enumerate(pdf, start=1):
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list, start=1):
            try:
                xref = img_info[0]

                if xref in seen_xrefs:
                    print(f"    [img {img_idx} pg {page_num}] Skipped — duplicate xref {xref}")
                    continue
                seen_xrefs.add(xref)

                base_image  = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext   = base_image.get("ext", "png")

                img_hash = hashlib.md5(image_bytes).hexdigest()
                if img_hash in seen_hashes:
                    print(f"    [img {img_idx} pg {page_num}] Skipped — duplicate content hash")
                    continue
                seen_hashes.add(img_hash)

                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                width, height = image.size

                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue

                # Save image to disk
                img_filename  = f"{pdf_stem}_page{page_num}_img{img_idx}.{image_ext}"
                img_save_path = os.path.join(IMAGES_DIR, img_filename)
                image.save(img_save_path)

                # Determine page position
                position_desc = "center"
                rects = page.get_image_rects(xref)
                if rects:
                    rect = rects[0]
                    page_width = page.rect.width
                    if rect.x1 <= page_width / 2:
                        position_desc = "left side of the page"
                    elif rect.x0 >= page_width / 2:
                        position_desc = "right side of the page"
                    else:
                        position_desc = "center of the page"

                # Collect figure/table captions from surrounding text
                page_captions = []
                blocks = page.get_text("blocks")
                for b in blocks:
                    text_block = b[4].strip()
                    if text_block.startswith(("Figure ", "Fig. ", "Table ")):
                        clean_block = " ".join(text_block.split())
                        page_captions.append(clean_block)
                page_caption_text = " | ".join(page_captions)

                # BLIP caption + OCR
                source_hint  = f"{filename}, page {page_num}, image {img_idx}"
                blip_caption = generate_image_caption(image, source_hint=source_hint)

                # ── 1: keyword-dense page_content ──────────────────────
                # Include all contextual signals so the embedding is rich enough
                # for cosine similarity to match queries like "architecture diagram"
                # or "figure 3" or "network topology".
                rich_content = (
                    f"[IMAGE] {blip_caption}\n"
                    f"File: {filename} | Page: {page_num} | Image index: {img_idx}\n"
                    f"Position: {position_desc} | Size: {width}x{height} pixels\n"
                    f"Type: image diagram figure visual chart graph illustration\n"
                    + (f"Figure reference: {page_caption_text}\n" if page_caption_text else "")
                )

                print(f"    [img {img_idx}] Indexed: {rich_content[:80].strip()}...")

                # ── 2: no base64 in metadata ───────────────────────────
                # base64 is loaded on-demand via get_image_base64(image_path).
                docs.append(Document(
                    page_content=rich_content,
                    metadata={
                        "source":      filename,
                        "page":        page_num,
                        "type":        "image",
                        "image_index": img_idx,
                        "image_path":  img_save_path,   # ← only path stored
                        "image_size":  f"{width}x{height}",
                        "caption":     blip_caption,
                    }
                ))

            except Exception as e:
                print(f"    Could not process image {img_idx} on page {page_num}: {e}")

    pdf.close()
    print(f"    Images saved to '{IMAGES_DIR}/'")
    return docs


# ─────────────────────────────────────────────────────────
#  STEP 4: LOAD ALL PDFs
# ─────────────────────────────────────────────────────────
def load_all_pdfs(pdf_dir: str) -> list[Document]:
    """Extract text, tables, and images from every PDF in a directory."""
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'")
        return all_docs

    print(f"\n Found {len(pdf_files)} PDF(s) in '{pdf_dir}'\n")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"\n Processing: {pdf_file}")

        text_docs  = extract_text(pdf_path)
        table_docs = extract_tables(pdf_path)
        image_docs = extract_images(pdf_path)

        print(f"    Text chunks   : {len(text_docs)}")
        print(f"    Tables found  : {len(table_docs)}")
        print(f"    Images saved  : {len(image_docs)}")

        all_docs.extend(text_docs + table_docs + image_docs)

    print(f"\n Total raw documents extracted: {len(all_docs)}")
    return all_docs


# ─────────────────────────────────────────────────────────
#  STEP 5: CHUNK DOCUMENTS
# ─────────────────────────────────────────────────────────
def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split text/table docs into chunks.
    Image docs are NOT chunked — their metadata must stay intact.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )

    image_docs = [d for d in docs if d.metadata.get("type") == "image"]
    other_docs = [d for d in docs if d.metadata.get("type") != "image"]

    chunks = text_splitter.split_documents(other_docs)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    for img_doc in image_docs:
        img_doc.metadata["chunk_id"] = str(uuid.uuid4())

    chunks.extend(image_docs)

    print(f" Total chunks after splitting: {len(chunks)} "
          f"({len(other_docs)} text/table → {len(chunks) - len(image_docs)} chunks, "
          f"{len(image_docs)} image docs)")
    return chunks


# ─────────────────────────────────────────────────────────
#  STEP 6: EMBED & STORE IN FAISS
# ─────────────────────────────────────────────────────────
def create_faiss_index(chunks: list[Document]) -> FAISS:
    """Generate embeddings and store in FAISS."""
    print(f"\n Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("  Creating FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f" FAISS index saved to: '{FAISS_INDEX_DIR}/'")

    return vectorstore


# ─────────────────────────────────────────────────────────
#  STEP 7: LOAD EXISTING FAISS INDEX
# ─────────────────────────────────────────────────────────
def load_faiss_index() -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f" FAISS index loaded from: '{FAISS_INDEX_DIR}/'")
    return vectorstore


# ─────────────────────────────────────────────────────────
#  STEP 8: ENSEMBLE RETRIEVER  ← 3 applied
# ─────────────────────────────────────────────────────────
class EnsembleRAGRetriever(BaseRetriever):
    """
    Custom retriever with query-intent routing.

    3: FAISS filter={"type": "image"} applies post-retrieval only over the
    top-k results — if those k docs happen to all be text, the filter returns
    nothing. Instead we oversample (k=40) and filter by metadata type manually,
    guaranteeing image/table docs are surfaced when the query asks for them.
    """
    vectorstore: FAISS
    k: int = 8

    # How many candidates to fetch before manual type-filtering
    OVERSAMPLE_K: int = 40

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> list[Document]:

        query_lower = query.lower()

        needs_image = any(kw in query_lower for kw in [
            "diagram", "figure", "image", "architecture", "illustration",
            "picture", "visual", "chart", "graph", "plot", "screenshot",
            "photo", "drawing", "sketch", "map", "fig"
        ])
        needs_table = any(kw in query_lower for kw in [
            "table", "tabular", "column", "row", "cells", "spreadsheet",
            "csv", "matrix", "grid", "statistics", "numbers"
        ])

        # Standard similarity search (base results)
        docs = self.vectorstore.similarity_search(query, k=self.k)

        extra_docs: list[Document] = []

        if needs_image:
            # ── 3: oversample broadly, then filter by type in Python ──
            candidates = self.vectorstore.similarity_search(
                query, k=self.OVERSAMPLE_K
            )
            image_hits = [
                d for d in candidates
                if d.metadata.get("type") == "image"
            ]
            extra_docs.extend(image_hits[:1])   # top-1 image match (most relevant only)

        if needs_table:
            candidates = self.vectorstore.similarity_search(
                query, k=self.OVERSAMPLE_K
            )
            table_hits = [
                d for d in candidates
                if d.metadata.get("type") == "table"
            ]
            extra_docs.extend(table_hits[:1])   # top-1 table match (most relevant only)

        # Merge: type-specific hits first, then general results, deduplicated
        seen:   set   = set()
        merged: list  = []

        for d in extra_docs + docs:
            uid = d.metadata.get("chunk_id") or hash(d.page_content)
            if uid not in seen:
                seen.add(uid)
                merged.append(d)

        return merged[: self.k]


# ─────────────────────────────────────────────────────────
#  QUERY-TIME HELPER — attach base64 to retrieved image docs
# ─────────────────────────────────────────────────────────
def enrich_with_base64(docs: list[Document]) -> list[Document]:
    """
    After retrieval, call this to attach base64_image to any image doc
    whose image_path exists on disk. Safe to call on mixed doc lists.

    Usage in your QA chain / API response layer:
        retrieved = retriever.get_relevant_documents(query)
        retrieved = enrich_with_base64(retrieved)
        for doc in retrieved:
            if doc.metadata.get("type") == "image":
                b64 = doc.metadata.get("base64_image")   # ← ready for UI
    """
    for doc in docs:
        if doc.metadata.get("type") == "image":
            path = doc.metadata.get("image_path", "")
            if path and os.path.isfile(path):
                try:
                    doc.metadata["base64_image"] = get_image_base64(path)
                except Exception as e:
                    print(f"    Could not load image from {path}: {e}")
    return docs


# ─────────────────────────────────────────────────────────
#  MAIN — Run Ingestion
# ─────────────────────────────────────────────────────────
def run_ingestion():
    print("=" * 55)
    print("RAG INGESTION PIPELINE STARTED")
    print("=" * 55)

    initialize_models()

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    raw_docs = load_all_pdfs(PDF_DIR)
    if not raw_docs:
        print("\n No content extracted. Please add PDF files to 'data/pdfs/'.")
        return

    chunks     = chunk_documents(raw_docs)
    vectorstore = create_faiss_index(chunks)

    image_count = sum(1 for c in chunks if c.metadata.get("type") == "image")
    text_count  = sum(1 for c in chunks if c.metadata.get("type") == "text")
    table_count = sum(1 for c in chunks if c.metadata.get("type") == "table")

    print("\n" + "=" * 55)
    print("INGESTION COMPLETE!")
    print(f"  Text  chunks : {text_count}")
    print(f"  Table chunks : {table_count}")
    print(f"  Image docs   : {image_count}")
    print(f"  Total indexed: {len(chunks)}")
    print(f"  Images saved : '{IMAGES_DIR}/'")
    print("=" * 55)

    return vectorstore


if __name__ == "__main__":
    run_ingestion()