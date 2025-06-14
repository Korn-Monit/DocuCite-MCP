import pathlib
import hashlib
import tempfile
import pdfplumber
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_ROOT = pathlib.Path.home() / ".rag_vectors"
VECTOR_ROOT.mkdir(exist_ok=True)

def load_or_create_chroma(pdf_bytes, filename, EMBEDDER, Chroma):
    print(f"\n[INFO] Checking vectorstore for file: {filename}")
    h = hashlib.md5(pdf_bytes).hexdigest()
    vect_dir = VECTOR_ROOT / h
    if (vect_dir / "chroma.sqlite3").exists():
        print(f"[INFO] Found existing vectorstore: {vect_dir}")
        return Chroma(persist_directory=str(vect_dir), embedding_function=EMBEDDER)

    print(f"[INFO] No vectorstore found, embedding file: {filename}")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    docs = []
    BAD_PHRASES = {
        "Abstracting with credit is permitted",
        "Permission to make digital or hard copies",
        "arXiv:",
        "©",
    }

    def clean_page(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines()
            if not any(b in line for b in BAD_PHRASES)
        )

    with pdfplumber.open(tmp_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = clean_page(page.extract_text() or "")
            if not text.strip():
                continue
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1200, chunk_overlap=200
            )
            para_chunks = splitter.split_text(text)
            for para_num, chunk in enumerate(para_chunks, start=1):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"page_number": page_num, "paragraph_number": para_num}
                    )
                )
    print(f"[INFO] Extracted {len(docs)} chunks from PDF for embedding.")
    vectordb = Chroma.from_documents(
        docs, EMBEDDER, persist_directory=str(vect_dir)
    )
    vectordb.persist()
    return vectordb
