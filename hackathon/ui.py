import gradio as gr
from langchain_ollama import ChatOllama
from tools import PDFCitationTool
import uuid
# def letter_counter(word, letter):
#     """Count the occurrences of a specific letter in a word.
    
#     Args:
#         word: The word or phrase to analyze
#         letter: The letter to count occurrences of
#
#     Returns:
#         The number of times the letter appears in the word
#     """
#     return word.lower().count(letter.lower())

def remove_think_section(content: str) -> str:
    """Remove the section between <think> and </think> tags from the content."""
    closing_tag = '</think>'
    position = content.find(closing_tag)
    if position != -1:
        return content[position + len(closing_tag):].strip()
    return content.strip()

response_model = ChatOllama(model="qwen3:4b", temperature=0.2)

pdf_tool = PDFCitationTool()

def generate_response(pdf_file, query_text):
    # doc_id = str(uuid.uuid4())
    doc_id = pdf_tool.process_pdf(pdf_file)
    
    # Step 2: Retrieve relevant chunks
    query_results = pdf_tool.query_document(doc_id, query_text, top_k=3)
    
    # Step 3: Construct the prompt
    excerpts = "\n\n".join([result['text'] for result in query_results['results']])
    prompt = f"Based on the following excerpts from the document, please summarize the information relevant to the query: '{query_text}'\n\nExcerpts:\n{excerpts}\n\nPlease provide a concise summary and include page numbers where the information is found."
    response = response_model.invoke(prompt)
    return remove_think_section(response.content)



demo = gr.Interface(
    fn=generate_response,
    # inputs=["text", "text"],
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Query", placeholder="What would you like to find in the document?")
    ],
    # outputs="number",
    outputs=gr.Textbox(label="Results with Citations", lines=20),
    title="🎯 PDF Citation Tool - MCP Server",
    # title="Letter Counter",
    description="Count how many times a letter appears in a word"
)

demo.launch(mcp_server=True)