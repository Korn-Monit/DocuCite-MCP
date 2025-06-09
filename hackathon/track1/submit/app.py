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
from langchain_openai import ChatOpenAI
import gradio as gr
import pdfplumber
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

import os, hashlib, tempfile, pathlib, torch, re, traceback
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber


device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDER = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True, "device": device},
)


LLM = ChatOpenAI(
    openai_api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMTYxMjA0MzQ0ODU0NTI5MTczNCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNzA0Mjc0OCwidXVpZCI6ImY4ZWEzOGUyLTllNjktNDM3NS05YjkzLWE3Y2EzMThiMjZjZCIsIm5hbWUiOiJoYWNrYXRob24iLCJleHBpcmVzX2F0IjoiMjAzMC0wNi0wN1QwNjowNTo0OCswMDAwIn0.DH7JrezDuqrl2SPMdWdWWnWgBPrvBbe9yucG29-3YpQ",
    openai_api_base="https://api.studio.nebius.com/v1",
    model="Qwen/Qwen2.5-72B-Instruct"
)

VECTOR_ROOT = pathlib.Path.home() / ".rag_vectors"
VECTOR_ROOT.mkdir(exist_ok=True)

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
    raise ValueError("Could not extract file bytes from uploaded file.")
# import requests
# from urllib.parse import urlparse
# def get_file_bytes_and_name(pdf_input):
#     """
#     Accept either:
#       • A local file‐like (pdf_input.read(), pdf_input.name)
#       • A URL string (requests.get)
#     """
#     if isinstance(pdf_input, str) and urlparse(pdf_input).scheme in ("http", "https"):
#         resp = requests.get(pdf_input)
#         resp.raise_for_status()
#         filename = pdf_input.split("/")[-1] or "uploaded.pdf"
#         return resp.content, filename

#     data = pdf_input.read()
#     filename = getattr(pdf_input, "name", "uploaded.pdf")
#     return data, filename

def load_or_create_chroma(pdf_bytes: bytes, filename: str) -> Chroma:
    """
    Loads persistent Chroma vectorstore for this PDF, or creates it if not found.
    Each chunk carries page and paragraph info.
    """
    print(f"\n[INFO] Checking vectorstore for file: {filename}")
    h = hashlib.md5(pdf_bytes).hexdigest()
    vect_dir = VECTOR_ROOT / h
    if (vect_dir / "chroma.sqlite3").exists():
        print(f"[INFO] Found existing vectorstore: {vect_dir}")
        return Chroma(persist_directory=str(vect_dir), embedding_function=EMBEDDER)

    # Otherwise, embed and persist it
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
            # Split into small chunks for embedding
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


def build_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever()
    from langchain.tools.retriever import create_retriever_tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="document_search",    
        description="Searches uploaded documents and returns relevant passages."
    )
    return retriever_tool

def make_generate_query_or_respond(retriever_tool):
    def generate_query_or_respond(state):
        if "thoughts" not in state:
            state["thoughts"] = []
            print(f"[DEBUG] Thoughts: {state['thoughts']}")
        response = (
            LLM
            .bind_tools([retriever_tool]).invoke(state["messages"])
        )
        state["thoughts"].append({
            "step": len(state["thoughts"]) + 1,
            "node": "generate_query_or_respond",
            "type": "thought",
            "content": f"LLM considered response or tool call. Output: {response}",
        })
        state["messages"].append(response)
        print(f"[DEBUG] State: {state['messages']}")
        return {"messages": [response]}
    return generate_query_or_respond


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question with reference and page number."
    "attention to the context, and only use it to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    if "thoughts" in state:
        print(f"[DEBUG] Thoughts: {state['thoughts']}")
    print(f"[DEBUG] Answer node, messages so far: {state['messages']}")
    question = state["messages"][0].content
    print(f"[DEBUG] Question: {question}")
    context = state["messages"][-1].content
    print(f"[DEBUG] Context: {context}")
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = LLM.invoke([{"role": "user", "content": prompt}])
    print(f"[DEBUG] LLM final answer: {response}")
    return {"messages": [response]}

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

def build_agentic_graph(retriever_tool):
    workflow = StateGraph(MessagesState)
    # Add nodes
    # def agent_node(state: MessagesState):
    #     # LLM decides to answer or call a tool
    #     return LLM.bind_tools([retriever_tool]).invoke(state["messages"])
    # def generate_query_or_respond(state):
    #     response = ( 
    #     LLM
    #     .bind_tools([retriever_tool]).invoke(state["messages"])
    #     )
    #     return {"messages": [response]}
    workflow.add_node("generate_query_or_respond", make_generate_query_or_respond(retriever_tool))
    # ToolNode handles retrieval
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(generate_answer)
    # Edges
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)
    # workflow.add_edge("retrieve", "agent")  # cycle back for multiple tool use if needed
    return workflow.compile()


def gradio_agentic_rag(pdf_file, question, history=None):
    pdf_bytes, filename = get_file_bytes_and_name(pdf_file)
    vectordb = load_or_create_chroma(pdf_bytes, filename)
    # retriever_tool = build_retriever_tool(vectordb)
    retriever_tool = build_retriever_tool(vectordb)
    graph = build_agentic_graph(retriever_tool)
    state_messages = []
    if history:
        for turn in history:
            if isinstance(turn, list) or isinstance(turn, tuple):
                if turn[0]:
                    state_messages.append({"role": "user", "content": turn[0]})
                if len(turn) > 1 and turn[1]:
                    state_messages.append({"role": "assistant", "content": turn[1]})
    # Add the current question
    state_messages.append({"role": "user", "content": question})
    state = {"messages": state_messages}

    # Run through the agentic graph workflow (streaming or just final)
    result = None
    for chunk in graph.stream(state):
        print(f"Chunk: {chunk}")
        for node, update in chunk.items():
            print(f"Node: {node}, Update: {update}")
            # If the LLM answered directly and did NOT trigger the tool
            last_msg = update["messages"][-1]
            if node == "generate_answer" or (
                node == "generate_query_or_respond" and not update["messages"][-1].tool_calls
            ):
                result = last_msg.content
        # Fallback (if for some reason nothing returned)
    # if result is None:
    #     result = "No answer generated."
    if history is None:
        history = []
    history.append([question, result])

    return result, history
    # return result or "No answer generated."


iface = gr.Interface(
    fn=gradio_agentic_rag,
    inputs=[
        gr.File(label="Upload your PDF", file_types=[".pdf"]),
        gr.Textbox(label="Ask a question about your PDF"),
        gr.State()
    ],
    outputs=[gr.Textbox(label="Answer from RAG Agent"),
              gr.State()],
    title="Agentic RAG PDF Q&A",
    description="Upload a PDF and ask any question about its contents. The AI will read and answer using only the information from your file."

)
if __name__ == "__main__":
    iface.launch(
        mcp_server=True,
        show_error=True,
        show_api=True
        )
