import gradio as gr
from pdf_utils import get_file_bytes_and_name
from vector_utils import load_or_create_chroma
from retriever_tool import build_retriever_tool
from llm import EMBEDDER, LLM
from agentic_graph import build_agentic_graph
from langchain_community.vectorstores import Chroma
def gradio_agentic_rag(pdf_file, question, history=None):
    pdf_bytes, filename = get_file_bytes_and_name(pdf_file)
    vectordb = load_or_create_chroma(pdf_bytes, filename, EMBEDDER, Chroma) 
    retriever_tool = build_retriever_tool(vectordb)
    graph = build_agentic_graph(LLM, retriever_tool)
    state_messages = []
    if history:
        for turn in history:
            if isinstance(turn, (list, tuple)):
                if turn[0]:
                    state_messages.append({"role": "user", "content": turn[0]})
                if len(turn) > 1 and turn[1]:
                    state_messages.append({"role": "assistant", "content": turn[1]})
    state_messages.append({"role": "user", "content": question})
    state = {"messages": state_messages}

    result = None
    for chunk in graph.stream(state):
        print(f"Chunk: {chunk}")
        for node, update in chunk.items():
            print(f"Node: {node}, Update: {update}")
            last_msg = update["messages"][-1]
            if node == "generate_answer" or (
                node == "generate_query_or_respond" and not update["messages"][-1].tool_calls
            ):
                result = last_msg.content

    if history is None:
        history = []
    history.append([question, result])
    return result, history

iface = gr.Interface(
    fn=gradio_agentic_rag,
    inputs=[
        gr.File(label="Upload your PDF"),
        gr.Textbox(label="Ask a question about your PDF"),
        gr.State()
    ],
    outputs=[
        gr.Textbox(label="Answer from RAG Agent"),
        gr.State()
    ],
    title="DocuCite Agent",
    description="An agentic RAG (Retrieval-Augmented Generation) system that can answer questions about the contents of a PDF document with references to the page and paragraph number.",
    examples=[
        ["paper.pdf", "What is LoRA? please use the tool"],
    ],
)

if __name__ == "__main__":
    iface.launch(
        mcp_server=True,
        show_error=True,
        show_api=True
    )
