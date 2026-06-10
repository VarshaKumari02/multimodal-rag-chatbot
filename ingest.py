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

CHUNK_SIZE      = 2000
CHUNK_OVERLAP   = 400

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
def generate_image_caption(image, source_hint=""):
    """
    Generate image description using Groq Vision (Llama-3.2-11b-vision).
    Falls back to local BLIP + OCR if Groq key is not found or fails.
 
    KEY FIX: Groq Vision prompt now explicitly asks for the figure title/name
    FIRST in the response so the embedding is anchored to the diagram name,
    not a generic description.
 
    Returns (caption, ocr_text).
    """
    global _blip_processor, _blip_model, _blip_available
 
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from groq import Groq
            from io import BytesIO
 
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
 
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Look at this diagram carefully and respond in EXACTLY this format:\n\n"
                                    "Figure title: <exact title visible in or near the diagram, or 'Unknown' if none>\n"
                                    "Components:\n"
                                    "- <every box, node, and operation label exactly as written>\n"
                                    "Data flow: <step by step how data moves through the diagram following arrows>\n"
                                    "All text visible: <every word, symbol, variable name visible in the diagram>\n\n"
                                    "Example:\n"
                                    "Figure title: Scaled Dot-Product Attention\n"
                                    "Components:\n"
                                    "- MatMul\n"
                                    "- Scale\n"
                                    "- SoftMax\n"
                                    "- Mask (opt.)\n"
                                    "- Q, K, V\n"
                                    "Data flow: Q and K → MatMul → Scale (/√dk) → Mask (opt.) → SoftMax → MatMul with V → Output\n"
                                    "All text visible: MatMul, Scale, SoftMax, Mask, opt., Q, K, V, dk\n\n"
                                    "Do not write anything outside this format. No preamble, no explanation."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            }
                        ]
                    }
                ],
                model="llama-3.2-11b-vision-preview",
                max_tokens=400   # increased from 300 for richer captions
            )
            caption = response.choices[0].message.content.strip()
            if source_hint:
                caption += f"\nSource: {source_hint}"
            return caption, ""
        except Exception as e:
            print(f"    Groq Vision failed, falling back to BLIP+OCR: {e}")
 
    # ── Fallback: local BLIP + OCR (unchanged) ──
    caption  = ""
    ocr_text = ""
 
    if _blip_available is None:
        initialize_models()
 
    if _blip_available:
        try:
            import torch
            _device = next(_blip_model.parameters()).device
            inputs  = _blip_processor(image, return_tensors="pt").to(_device)
            with torch.no_grad():
                out = _blip_model.generate(**inputs, max_new_tokens=60)
            caption = _blip_processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"    BLIP inference failed: {e}")
            caption = "visual content: diagram or figure"
    else:
        caption = "visual content: diagram or figure"
 
    try:
        import pytesseract
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
 
        def extract_words(img):
            words = []
            try:
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                for i in range(len(ocr_data["text"])):
                    text = ocr_data["text"][i].strip()
                    conf = int(ocr_data["conf"][i])
                    if conf > 40 and len(text) >= 1:
                        clean_text = "".join([c for c in text if c.isalnum() or c in "-()"])
                        if clean_text:
                            words.append(clean_text)
            except Exception:
                pass
            for psm in [3, 11, 12]:
                try:
                    ocr_str = pytesseract.image_to_string(img, config=f"--psm {psm}").strip()
                    for line in ocr_str.splitlines():
                        for w in line.split():
                            w_clean = "".join([c for c in w if c.isalnum() or c in "-()"])
                            if len(w_clean) >= 1:
                                words.append(w_clean)
                except Exception:
                    pass
            return words
 
        words_orig = extract_words(image)
        gray       = image.convert("L")
        resized    = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
        words_prep = extract_words(resized)
 
        all_words  = words_orig + words_prep
        seen_lower = set()
        unique_words = []
        for w in all_words:
            w_lower = w.lower()
            if w_lower not in seen_lower:
                seen_lower.add(w_lower)
                unique_words.append(w)
 
        valid_words = []
        for w in unique_words:
            if len(w) >= 2:
                if any(c.isalpha() for c in w):
                    valid_words.append(w)
            elif len(w) == 1:
                if w.lower() in {'q', 'k', 'v', 'h', 'w', 'x', 'y', 'z'}:
                    valid_words.append(w)
 
        ocr_text = " | ".join(valid_words)
    except Exception as e:
        print(f"    [OCR WARNING] Local OCR failed (is Tesseract installed?): {e}")
 
    return caption, ocr_text

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
def extract_images(pdf_path):
    """
    Extract embedded images from PDF pages.
 
    1. page_content now leads with figure caption text (from PDF) so the
       FAISS embedding is anchored to the actual diagram name.
    2. Groq Vision / BLIP caption comes second.
    3. OCR is stored in metadata["ocr_text"] separately — NOT merged into
       page_content — so it doesn't dilute the embedding.
    4. page_content also includes the figure title extracted from Groq output
       if present (first line starting with "Figure title:").
    """
    docs     = []
    filename = os.path.basename(pdf_path)
    pdf_stem = os.path.splitext(filename)[0]
 
    seen_xrefs:  set = set()
    seen_hashes: set = set()
 
    os.makedirs(IMAGES_DIR, exist_ok=True)
 
    pdf = fitz.open(pdf_path)
 
    for page_num, page in enumerate(pdf, start=1):
        image_list = page.get_images(full=True)
 
        positioned_images = []
        for img_info in image_list:
            xref  = img_info[0]
            rects = page.get_image_rects(xref)
            x0, y0 = 0, 0
            if rects:
                x0 = rects[0].x0
                y0 = rects[0].y0
            positioned_images.append({"info": img_info, "x0": x0, "y0": y0})
 
        positioned_images.sort(key=lambda item: (int(item["y0"]) // 50, item["x0"]))
        sorted_image_list = [item["info"] for item in positioned_images]
 
        for img_idx, img_info in enumerate(sorted_image_list, start=1):
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
 
                image         = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                width, height = image.size
 
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue
 
                img_filename  = f"{pdf_stem}_page{page_num}_img{img_idx}.{image_ext}"
                img_save_path = os.path.join(IMAGES_DIR, img_filename)
                image.save(img_save_path)
 
                position_desc = "center of the page"
                rects = page.get_image_rects(xref)
                if rects:
                    rect       = rects[0]
                    page_width = page.rect.width
                    if rect.x1 <= page_width / 2:
                        position_desc = "left side of the page"
                    elif rect.x0 >= page_width / 2:
                        position_desc = "right side of the page"
                    else:
                        position_desc = "center of the page"
 
                # ── Spatially-scoped figure caption from PDF text ──
                page_captions = []
                blocks        = page.get_text("blocks")
 
                if rects:
                    rect = rects[0]
                    for b in blocks:
                        bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
                        text_block = b[4].strip()
                        if by0 >= rect.y1 and by0 <= rect.y1 + 80:
                            if text_block.startswith(("Figure ", "Fig. ", "Table ")):
                                page_captions.append(" ".join(text_block.split()))
                else:
                    for b in blocks:
                        text_block = b[4].strip()
                        if text_block.startswith(("Figure ", "Fig. ", "Table ")):
                            page_captions.append(" ".join(text_block.split()))
 
                page_caption_text = " | ".join(page_captions)
 
                # ── Groq Vision / BLIP caption ──
                source_hint            = f"{filename}, page {page_num}, image {img_idx}"
                vision_caption, ocr_text = generate_image_caption(image, source_hint=source_hint)
 
                # This gives us the clean diagram name to lead the embedding.
                figure_title = ""
                vision_lines = vision_caption.splitlines()
                remaining_lines = []
                for line in vision_lines:
                    if line.lower().startswith("figure title:"):
                        figure_title = line.split(":", 1)[-1].strip()
                    else:
                        remaining_lines.append(line)
                vision_body = "\n".join(remaining_lines).strip()

                # Fallback block when figure_title is empty
                if not figure_title:
                    if page_caption_text:
                        figure_title = page_caption_text
                    else:
                        # grab first non-empty line from vision output as title
                        for line in vision_body.splitlines():
                            line = line.strip().lstrip("-").strip()
                            if len(line) > 8 and not line.startswith("Source:"):
                                figure_title = line[:80]
                                break
 
                # OCR stored in metadata and also conditionally in page_content for local fallback
                clean_ocr_words = [
                    w for w in ocr_text.split(" | ")
                    if len(w.strip()) >= 3
                ]
                clean_ocr = " | ".join(clean_ocr_words)
 
                # ── Build page_content: figure name leads, vision body second ──
                leading_title = figure_title or page_caption_text or "diagram figure visual"
 
                rich_content = (
                    f"[IMAGE] {leading_title}\n"
                    + (f"PDF caption: {page_caption_text}\n" if page_caption_text and figure_title else "")
                    + f"{vision_body}\n"
                    f"File: {filename} | Page: {page_num} | Image index: {img_idx}\n"
                    f"Position: {position_desc} | Size: {width}x{height} pixels\n"
                    f"Type: image diagram figure visual chart graph illustration\n"
                    + (f"OCR_LABELS: {clean_ocr}\n" if clean_ocr else "")
                )
 
                print(f"    [img {img_idx}] Indexed: {rich_content[:100].strip()}...")
 
                docs.append(Document(
                    page_content=rich_content,
                    metadata={
                        "source":      filename,
                        "page":        page_num,
                        "type":        "image",
                        "image_index": img_idx,
                        "image_path":  img_save_path,
                        "image_size":  f"{width}x{height}",
                        "caption":     vision_caption,
                        "ocr_text":    clean_ocr,       
                        "figure_title": figure_title,
                        "position":    position_desc,
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
          f"({len(other_docs)} text/table -> {len(chunks) - len(image_docs)} chunks, "
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

    import shutil
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)

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
#  STEP 8: ENSEMBLE RETRIEVER
# ───────────────────────────────────────────────────────
class EnsembleRAGRetriever(BaseRetriever):
    """
    Custom retriever with query-intent routing.
 
    KEY FIXES:
    1. OVERSAMPLE_K raised to 80 (was 40) — image docs often rank 41-60.
    2. Keyword-based fallback search when dense retrieval returns no images:
       strips stop-words from query and retries with technical terms only.
    3. Returns top-2 image hits (was top-1) so LLM has better context.
    4. Extended image keyword list to catch more diagram-related queries.
    5. Extended table keyword list similarly.
    """
    vectorstore: FAISS
    k: int = 8
 
    OVERSAMPLE_K: int = 80 
 
    # FIX: broader keyword lists
    IMAGE_KEYWORDS: list = [
        "diagram", "figure", "image", "architecture", "illustration",
        "picture", "visual", "chart", "graph", "plot", "screenshot",
        "photo", "drawing", "sketch", "map", "fig", "shown", "show",
        "depict", "depicts", "look", "looks", "attention", "encoder",
        "decoder", "layer", "network", "model", "structure", "multi-head",
        "scaled", "dot-product", "softmax", "components", "operations",
        "inside", "output", "input", "what is shown", "what are shown"
    ]
 
    TABLE_KEYWORDS: list = [
        "table", "tabular", "column", "row", "cells", "spreadsheet",
        "csv", "matrix", "grid", "statistics", "numbers", "results",
        "performance", "bleu", "score", "comparison", "benchmark"
    ]
 
    # Stop words to strip when building keyword fallback query
    STOP_WORDS: set = {
        "what", "which", "where", "when", "how", "are", "is", "the",
        "a", "an", "in", "of", "to", "for", "and", "or", "shown",
        "show", "does", "do", "can", "could", "would", "between",
        "inside", "from", "with", "on", "at", "by"
    }
 
    def _keyword_fallback_search(self, query: str, doc_type: str) -> list:
        """
        Strip stop words from query and retry similarity search.
        Guarantees we search on the technical terms (e.g. 'Scaled Dot-Product Attention')
        rather than the full question string which dilutes the embedding.
        """
        tokens = [
            w for w in query.split()
            if len(w) > 2 and w.lower() not in self.STOP_WORDS
        ]
        if not tokens:
            return []
 
        fallback_query = " ".join(tokens)
        print(f"    [retriever] keyword fallback query: '{fallback_query}'")
        candidates = self.vectorstore.similarity_search(fallback_query, k=self.OVERSAMPLE_K)
        return [d for d in candidates if d.metadata.get("type") == doc_type]
 
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "CallbackManagerForRetrieverRun" = None,
    ) -> list:
 
        query_lower = query.lower()
 
        needs_image = any(kw in query_lower for kw in self.IMAGE_KEYWORDS)
        needs_table = any(kw in query_lower for kw in self.TABLE_KEYWORDS)
 
        # Base similarity search
        docs = self.vectorstore.similarity_search(query, k=self.k)
 
        extra_docs = []
 
        if needs_image:
            candidates = self.vectorstore.similarity_search_with_score(
                query, k=self.OVERSAMPLE_K
            )
            # similarity_search_with_score returns (doc, score) — lower score = more similar in FAISS
            image_hits = [
                d for d, score in candidates
                if d.metadata.get("type") == "image"
            ]

            if not image_hits:
                # keyword fallback
                tokens = [
                    w for w in query.split()
                    if len(w) > 2 and w.lower() not in self.STOP_WORDS
                ]
                if tokens:
                    fb_candidates = self.vectorstore.similarity_search_with_score(
                        " ".join(tokens), k=self.OVERSAMPLE_K
                    )
                    image_hits = [
                        d for d, score in fb_candidates
                        if d.metadata.get("type") == "image"
                    ]

            extra_docs.extend(image_hits[:4])

        if needs_table:
            candidates = self.vectorstore.similarity_search_with_score(
                query, k=self.OVERSAMPLE_K
            )
            table_hits = [
                d for d, score in candidates
                if d.metadata.get("type") == "table"
            ]

            if not table_hits:
                # keyword fallback
                tokens = [
                    w for w in query.split()
                    if len(w) > 2 and w.lower() not in self.STOP_WORDS
                ]
                if tokens:
                    fb_candidates = self.vectorstore.similarity_search_with_score(
                        " ".join(tokens), k=self.OVERSAMPLE_K
                    )
                    table_hits = [
                        d for d, score in fb_candidates
                        if d.metadata.get("type") == "table"
                    ]

            # FIX: top-1 only — prevents unrelated sibling tables from leaking in
            extra_docs.extend(table_hits[:1])
 
        # Merge: type-specific hits first, then general, deduplicated
        seen:   set  = set()
        merged: list = []
 
        for d in extra_docs + docs:
            uid = d.metadata.get("chunk_id") or hash(d.page_content)
            if uid not in seen:
                seen.add(uid)
                merged.append(d)
 
        return merged[: self.k + 6]

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
                b64 = doc.metadata.get("base64_image")
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