from mcp.server.fastmcp import FastMCP
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import fitz  
import faiss
import numpy as np
import uuid
mcp = FastMCP("tool")

@dataclass
class PageContent:
    page_num: int
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    char_start: int
    char_end: int

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    page_num: int
    chunk_index: int
    embedding: Optional[List[float]] = None
    char_positions: Optional[Tuple[int, int]] = None

@mcp.tool()
class PDFCitationTool:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def process_pdf(self, pdf_path: str, doc_id: str = None) -> str:
        """Process PDF and extract text with page mapping"""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        doc = fitz.open(pdf_path)
        pages_content = []
        full_text = ""
        char_offset = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()

            # Store page content with character positions
            page_content = PageContent(
                page_num=page_num + 1,  # 1-indexed
                text=page_text,
                bbox=page.rect,
                char_start=char_offset,
                char_end=char_offset + len(page_text)
            )
            pages_content.append(page_content)
            full_text += page_text + "\n"
            char_offset += len(page_text) + 1

        # Create chunks with embeddings
        chunks = self._create_chunks(full_text, pages_content, doc_id)

        # Build FAISS index
        embeddings = [chunk.embedding for chunk in chunks]
        index = faiss.IndexFlatIP(len(embeddings[0]))
        index.add(np.array(embeddings).astype('float32'))

        self.documents[doc_id] = {
            'pages': [asdict(page) for page in pages_content],
            'chunks': [asdict(chunk) for chunk in chunks],
            'index': index,
            'full_text': full_text,
            'metadata': {
                'filename': pdf_path.split('/')[-1],
                'total_pages': len(doc),
                'total_chars': len(full_text)
            }
        }

        doc.close()
        return doc_id

    def _create_chunks(self, full_text: str, pages_content: List[PageContent], doc_id: str) -> List[DocumentChunk]:
        """Create overlapping chunks with page tracking"""
        chunks = []
        chunk_size = 500
        overlap = 100

        for i in range(0, len(full_text), chunk_size - overlap):
            chunk_text = full_text[i:i + chunk_size]
            if not chunk_text.strip():
                continue

            # Find which page(s) this chunk belongs to
            chunk_start = i
            chunk_end = i + len(chunk_text)

            # Find the primary page (where most of the chunk is)
            primary_page = 1
            for page in pages_content:
                if (chunk_start >= page.char_start and
                    chunk_start < page.char_end):
                    primary_page = page.page_num
                    break

            # Create embedding
            embedding = self.embedder.encode(chunk_text).tolist()

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_{len(chunks)}",
                text=chunk_text,
                page_num=primary_page,
                chunk_index=len(chunks),
                embedding=embedding,
                char_positions=(chunk_start, chunk_end)
            )
            chunks.append(chunk)

        return chunks

    def query_document(self, doc_id: str, query: str, top_k: int = 3) -> Dict:
        """Query document and return results with precise page citations"""
        if doc_id not in self.documents:
            return {"error": "Document not found"}

        doc = self.documents[doc_id]

        # Encode query
        query_embedding = self.embedder.encode(query).astype('float32').reshape(1, -1)

        # Search
        scores, indices = doc['index'].search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = doc['chunks'][idx]

            # Get more precise page information
            page_info = self._get_precise_page_info(
                chunk['text'],
                chunk['char_positions'],
                doc['pages']
            )

            results.append({
                'text': chunk['text'],
                'page_number': chunk['page_num'],
                'confidence': float(score),
                'precise_location': page_info,
                'chunk_id': chunk['chunk_id']
            })

        return {
            'query': query,
            'results': results,
            'document_info': doc['metadata']
        }

    def _get_precise_page_info(self, chunk_text: str, char_positions: Tuple[int, int], pages: List[Dict]) -> Dict:
        """Get more precise location information within the page"""
        start_char, end_char = char_positions

        # Find which page(s) the chunk spans
        spanning_pages = []
        for page in pages:
            if (start_char < page['char_end'] and end_char > page['char_start']):
                spanning_pages.append(page['page_num'])

        # Find position within the page (rough estimate)
        primary_page = spanning_pages[0] if spanning_pages else 1

        # Estimate paragraph/section within page
        page_text = None
        for page in pages:
            if page['page_num'] == primary_page:
                page_text = page['text']
                break

        paragraph_num = 1
        if page_text:
            # Count paragraphs before our chunk
            chunk_start_in_page = max(0, start_char - next(p['char_start'] for p in pages if p['page_num'] == primary_page))
            paragraphs_before = page_text[:chunk_start_in_page].count('\n\n')
            paragraph_num = max(1, paragraphs_before + 1)

        return {
            'primary_page': primary_page,
            'spanning_pages': spanning_pages,
            'estimated_paragraph': paragraph_num,
            'char_range': char_positions
        }