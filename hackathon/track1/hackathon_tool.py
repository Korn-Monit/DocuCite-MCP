from __future__ import annotations

import os
import re
import uuid
import json
import queue
import threading
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any

import gradio as gr
import pdfplumber
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from pathlib import Path
from langchain.schema import Document
import pdfplumber
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
import traceback
import torch
from openai import OpenAI
import os
import tempfile
from pathlib import Path
# import pdfplumber
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import re


def get_file_bytes_and_name(pdf_file):
    print("DEBUG: pdf_file type:", type(pdf_file))
    print("DEBUG: pdf_file dir:", dir(pdf_file))
    print("DEBUG: pdf_file repr:", repr(pdf_file))
    # Standard file-like object (e.g. Python's open, script mode)
    if hasattr(pdf_file, "read"):
         return pdf_file.read(), Path(pdf_file.name).name
    if isinstance(pdf_file, str):
        file_path = Path(pdf_file)
        with open(file_path, "rb") as f:
            return f.read(), file_path.name
    # else:
    #     raise ValueError("Could not extract file bytes from uploaded file.")
    raise ValueError("Could not extract file bytes from uploaded file.")

def pdf_to_documents(pdf_path: str) -> list[Document]:
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # simple paragraph split – tweak as you like
            for para_no, para in enumerate(text.split("\n\n"), start=1):
                cleaned = para.strip()
                if cleaned:           # skip blank chunks
                    docs.append(
                        Document(
                            page_content=cleaned,
                            metadata={"page": page_no, "paragraph": para_no}
                        )
                    )
    return docs

def format_sources(retrieved_docs):
    numbered, exposed = [], []
    for i, doc in enumerate(retrieved_docs, start=1):
        p, n = doc.metadata.get("page"), doc.metadata.get("paragraph")
        txt   = doc.page_content
        numbered.append(f"({i}) [page {p} ¶{n}] {txt}")
        exposed.append(f"[page {p} ¶{n}] {txt}")
    return "\n".join(numbered), exposed

def build_vectorstore_from_pdf(pdf_file, embedding_model, persist_directory=None):


    # --- read bytes & filename ---
    file_bytes, file_name = get_file_bytes_and_name(pdf_file)

    # --- write to a temp file ---
    temp_path = Path(tempfile.gettempdir()) / file_name
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

        # --- NEW: create one-paragraph Documents with location metadata ---
    docs_list = pdf_to_documents(str(temp_path))      # ← replaces the old pdfplumber loop

    # (optional) keep your existing splitter so long paragraphs still get trimmed
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700, chunk_overlap=100
    )
    doc_splits = splitter.split_documents(docs_list)  # metadata is copied to every chunk

    # --- build the Chroma vector store exactly as before ---
    vectordb = Chroma.from_documents(
        doc_splits,
        embedding_model,
        persist_directory=persist_directory,
    )
    return vectordb

def build_retriever_tool(vectorstore, name="search_user_documents"):
    retriever = vectorstore.as_retriever()
    from langchain.tools.retriever import create_retriever_tool
    retriever_tool = create_retriever_tool(
        retriever,
        name=name,
        description="Searches uploaded documents and returns relevant passages."
    )
    return retriever_tool

os.environ["NEBIUS_API_KEY"] = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMTc0OTg1MDIyNTg1OTk4NjQ4MCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNjYyODk1NywidXVpZCI6IjdlY2M0ZjZmLTM1N2YtNDUxZC05ZjNhLWNjYzNlNDIxZGVkYiIsIm5hbWUiOiJIYWNrYXRob24iLCJleHBpcmVzX2F0IjoiMjAzMC0wNi0wMlQxMTowOToxNyswMDAwIn0.XAqOc-I9MTAOnXgR94ii1ZYV7f4nJIcEUMsXroKUjnE"   # add to ~/.bashrc or HF Space secrets



# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True, "device": device} # Specify device
)

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)



def pdf_qa_pipeline(pdf_file, query_text, k=3):
    try:
        vectorstore  = build_vectorstore_from_pdf(pdf_file, embedding_model)

        #  Get real Document objects (with metadata) straight from Chroma
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print("Retriever:", retriever)
        retrieved_docs = retriever.get_relevant_documents(query_text)
        # └── each item here is a Document, so .metadata works
        print("DEBUG: Retrieved documents:", retrieved_docs)
        if not retrieved_docs:
            return "No relevant information found in the uploaded document."

        numbered_block, exposed_sources = format_sources(retrieved_docs)

        prompt = f"""You are a helpful assistant. Using only the information from the numbered EXCERPTS below, write a clear and well-structured answer to the QUESTION. Rephrase and summarize as needed. Cite the excerpt number(s) in brackets as appropriate."

EXCERPTS:
{numbered_block}

QUESTION: {query_text}
ANSWER:"""

        # --- SDK call replaces .invoke() ---
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324-fast",
            temperature=0.5,
            max_tokens=512,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ]
        )
        answer_text = completion.choices[0].message.content.strip()
        return f"{answer_text}\n\n**Paragraphs used**\n" + "\n".join(exposed_sources)


#         answer = response_model.invoke(prompt).strip()
#         return f"{answer}\n\n**Paragraphs used**\n" + "\n".join(exposed_sources)

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {str(e)}"
    
demo = gr.Interface(
    fn=pdf_qa_pipeline,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Question", placeholder="Ask anything about the PDF..."),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Q&A - All In One",
    description="Upload a PDF and ask a question—get an LLM answer grounded only in your document."
)

if __name__ == "__main__":
    demo.launch(
        mcp_server=True,
        show_error=True,
        show_api=True
        )