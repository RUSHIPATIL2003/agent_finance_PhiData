"""Multimodal PDF ingestion pipeline."""

import hashlib
import logging
import mimetypes
import os
import uuid
from dataclasses import dataclass
from typing import Any, Generator, Optional

import fitz  # PyMuPDF
from PIL import Image

from app.config import get_settings
from app.database import (
    create_chunks_batch,
    create_document,
    get_document_by_hash,
)
from app.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    content_type: str
    page_number: int
    chunk_index: int
    section_title: Optional[str] = None
    heading_hierarchy: Optional[list[str]] = None
    bbox: Optional[dict[str, float]] = None
    metadata: Optional[dict[str, Any]] = None
    embedding: Optional[list[float]] = None


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_text_with_structure(page: fitz.Page, page_num: int) -> list[DocumentChunk]:
    """Extract text from a page preserving structure."""
    chunks = []
    
    # Get text blocks with structure information
    blocks = page.get_text("dict")["blocks"]
    
    current_section = None
    heading_hierarchy = []
    chunk_index = 0
    
    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                line_text = ""
                line_bbox = None
                max_font_size = 0
                is_heading = False
                
                for span in line["spans"]:
                    line_text += span["text"]
                    if line_bbox is None:
                        line_bbox = list(span["bbox"])
                    else:
                        line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                        line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                        line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                        line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                    max_font_size = max(max_font_size, span["size"])
                
                line_text = line_text.strip()
                if not line_text:
                    continue
                
                # Detect headings based on font size
                if max_font_size > 14:
                    is_heading = True
                    heading_level = min(int(max_font_size / 4), 6)
                    heading_hierarchy = heading_hierarchy[:heading_level - 1] + [line_text]
                    current_section = line_text
                
                chunks.append(DocumentChunk(
                    content=line_text,
                    content_type="heading" if is_heading else "text",
                    page_number=page_num,
                    chunk_index=chunk_index,
                    section_title=current_section,
                    heading_hierarchy=heading_hierarchy.copy(),
                    bbox={"x0": line_bbox[0], "y0": line_bbox[1], "x1": line_bbox[2], "y1": line_bbox[3]} if line_bbox else None,
                    metadata={"font_size": max_font_size, "is_heading": is_heading}
                ))
                chunk_index += 1
    
    return chunks


def extract_tables(page: fitz.Page, page_num: int) -> list[DocumentChunk]:
    """Extract tables from a page."""
    chunks = []
    
    try:
        tabs = page.find_tables()
        for i, table in enumerate(tabs):
            # Convert table to markdown format
            markdown_table = table.to_markdown()
            if markdown_table.strip():
                chunks.append(DocumentChunk(
                    content=markdown_table,
                    content_type="table",
                    page_number=page_num,
                    chunk_index=i,
                    section_title=f"Table {i + 1}",
                    bbox={"x0": table.bbox[0], "y0": table.bbox[1], "x1": table.bbox[2], "y1": table.bbox[3]},
                    metadata={"table_index": i, "rows": table.row_count, "cols": table.col_count}
                ))
    except Exception as e:
        logger.warning("Failed to extract tables from page %d: %s", page_num, e)
    
    return chunks


def extract_images(page: fitz.Page, page_num: int, doc: fitz.Document) -> list[DocumentChunk]:
    """Extract images from a page with OCR if needed."""
    chunks = []
    
    try:
        images = page.get_images(full=True)
        for i, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image temporarily for OCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=f".{image_ext}", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            
            try:
                # Try OCR with pytesseract if available
                try:
                    import pytesseract
                    ocr_text = pytesseract.image_to_string(Image.open(tmp_path))
                    if ocr_text.strip():
                        chunks.append(DocumentChunk(
                            content=f"[Image OCR]: {ocr_text.strip()}",
                            content_type="image_ocr",
                            page_number=page_num,
                            chunk_index=i,
                            section_title=f"Image {i + 1}",
                            metadata={"image_index": i, "image_format": image_ext, "ocr_engine": "tesseract"}
                        ))
                except ImportError:
                    # No OCR available, just note the image
                    chunks.append(DocumentChunk(
                        content=f"[Image: {image_ext} format, {len(image_bytes)} bytes]",
                        content_type="image",
                        page_number=page_num,
                        chunk_index=i,
                        section_title=f"Image {i + 1}",
                        metadata={"image_index": i, "image_format": image_ext, "size_bytes": len(image_bytes)}
                    ))
            finally:
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.warning("Failed to extract images from page %d: %s", page_num, e)
    
    return chunks


def extract_formulas(page: fitz.Page, page_num: int) -> list[DocumentChunk]:
    """Extract mathematical formulas from a page."""
    chunks = []
    
    try:
        # Search for math-like patterns in text
        text = page.get_text()
        import re
        # Simple pattern for formulas (can be enhanced)
        formula_patterns = [
            r'\$[^$]+\$',  # LaTeX inline
            r'\$\$[^$]+\$\$',  # LaTeX display
            r'[A-Za-z]\s*[=]\s*[^=\n]+',  # Simple equations
        ]
        
        for pattern in formula_patterns:
            matches = re.finditer(pattern, text)
            for i, match in enumerate(matches):
                chunks.append(DocumentChunk(
                    content=match.group(),
                    content_type="formula",
                    page_number=page_num,
                    chunk_index=i,
                    section_title="Mathematical Formula",
                    metadata={"pattern": pattern}
                ))
    except Exception as e:
        logger.warning("Failed to extract formulas from page %d: %s", page_num, e)
    
    return chunks


def process_pdf(filepath: str) -> Generator[DocumentChunk, None, None]:
    """Process PDF and yield chunks with metadata."""
    settings = get_settings()
    
    doc = fitz.open(filepath)
    logger.info("Processing PDF: %s (%d pages)", filepath, doc.page_count)
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_chunks = []
        
        # Extract different content types
        page_chunks.extend(extract_text_with_structure(page, page_num + 1))
        page_chunks.extend(extract_tables(page, page_num + 1))
        page_chunks.extend(extract_images(page, page_num + 1, doc))
        page_chunks.extend(extract_formulas(page, page_num + 1))
        
        for chunk in page_chunks:
            yield chunk
    
    doc.close()


def semantic_chunking(
    chunks: list[DocumentChunk],
    chunk_size: int,
    chunk_overlap: int
) -> list[DocumentChunk]:
    """Perform semantic chunking on extracted content."""
    if not chunks:
        return []
    
    # Group by content type and page for better semantic coherence
    result = []
    current_chunk = ""
    current_metadata = {}
    chunk_index = 0
    
    for chunk in chunks:
        # If adding this chunk would exceed chunk_size, finalize current chunk
        if len(current_chunk) + len(chunk.content) > chunk_size and current_chunk:
            result.append(DocumentChunk(
                content=current_chunk.strip(),
                content_type=current_metadata.get("content_type", "text"),
                page_number=current_metadata.get("page_number", 1),
                chunk_index=chunk_index,
                section_title=current_metadata.get("section_title"),
                heading_hierarchy=current_metadata.get("heading_hierarchy"),
                bbox=current_metadata.get("bbox"),
                metadata=current_metadata.get("metadata", {})
            ))
            chunk_index += 1
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text + " " + chunk.content
            current_metadata = {
                "content_type": chunk.content_type,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "heading_hierarchy": chunk.heading_hierarchy,
                "bbox": chunk.bbox,
                "metadata": chunk.metadata,
            }
        else:
            current_chunk += " " + chunk.content
            # Update metadata with latest chunk's info
            current_metadata = {
                "content_type": chunk.content_type,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "heading_hierarchy": chunk.heading_hierarchy,
                "bbox": chunk.bbox,
                "metadata": chunk.metadata,
            }
    
    # Don't forget the last chunk
    if current_chunk.strip():
        result.append(DocumentChunk(
            content=current_chunk.strip(),
            content_type=current_metadata.get("content_type", "text"),
            page_number=current_metadata.get("page_number", 1),
            chunk_index=chunk_index,
            section_title=current_metadata.get("section_title"),
            heading_hierarchy=current_metadata.get("heading_hierarchy"),
            bbox=current_metadata.get("bbox"),
            metadata=current_metadata.get("metadata", {})
        ))
    
    return result


def ingest_document(filepath: Optional[str] = None) -> dict[str, Any]:
    """Main ingestion pipeline for a document."""
    settings = get_settings()
    filepath = filepath or settings.document_path
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Document not found: {filepath}")
    
    logger.info("Starting ingestion for: %s", filepath)
    
    # Compute file hash for deduplication
    file_hash = compute_file_hash(filepath)
    file_size = os.path.getsize(filepath)
    mime_type, _ = mimetypes.guess_type(filepath)
    filename = os.path.basename(filepath)
    
    # Check if already processed
    existing_doc = get_document_by_hash(file_hash)
    if existing_doc:
        logger.info("Document already ingested (hash: %s), skipping", file_hash[:16])
        return {
            "status": "skipped",
            "document_id": existing_doc["id"],
            "message": "Document already exists in database"
        }
    
    # Extract chunks from PDF
    raw_chunks = list(process_pdf(filepath))
    logger.info("Extracted %d raw chunks from PDF", len(raw_chunks))
    
    # Perform semantic chunking
    chunks = semantic_chunking(raw_chunks, settings.chunk_size, settings.chunk_overlap)
    logger.info("Created %d semantic chunks", len(chunks))
    
    # Generate embeddings
    embedding_model = get_embedding_model()
    texts = [c.content for c in chunks]
    embeddings = embedding_model.embed_documents(texts)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    
    # Create document record
    document = create_document(
        filename=filename,
        file_hash=file_hash,
        file_size=file_size,
        mime_type=mime_type or "application/pdf",
        page_count=max(c.page_number for c in chunks) if chunks else 0,
        metadata={"total_chunks": len(chunks), "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}
    )
    
    document_id = document["id"]
    logger.info("Created document record: %s", document_id)
    
    # Prepare chunks for batch insert
    chunk_data = []
    for chunk in chunks:
        chunk_data.append({
            "document_id": document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "content_type": chunk.content_type,
            "page_number": chunk.page_number,
            "section_title": chunk.section_title,
            "heading_hierarchy": chunk.heading_hierarchy,
            "bbox": chunk.bbox,
            "metadata": chunk.metadata or {},
            "embedding": chunk.embedding,
        })
    
    # Batch insert chunks
    inserted = create_chunks_batch(chunk_data)
    logger.info("Inserted %d chunks into database", inserted)
    
    return {
        "status": "success",
        "document_id": document_id,
        "chunks_processed": len(chunks),
        "chunks_inserted": inserted,
        "file_hash": file_hash
    }


def reindex_document(filepath: Optional[str] = None) -> dict[str, Any]:
    """Force re-index a document (delete existing and re-ingest)."""
    settings = get_settings()
    filepath = filepath or settings.document_path
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Document not found: {filepath}")
    
    file_hash = compute_file_hash(filepath)
    existing_doc = get_document_by_hash(file_hash)
    
    if existing_doc:
        # Delete existing document (cascades to chunks)
        from app.database import get_db_pool
        pool = get_db_pool()
        pool.execute("DELETE FROM rag.documents WHERE id = %s", (existing_doc["id"],))
        logger.info("Deleted existing document: %s", existing_doc["id"])
    
    return ingest_document(filepath)