import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDER = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True, "device": device},
)

LLM = ChatOpenAI(
    openai_api_key="",  
    openai_api_base="https://api.studio.nebius.com/v1",
    model="Qwen/Qwen2.5-72B-Instruct"
)
