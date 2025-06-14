DocuCite-AI

Demo:
https://github.com/user-attachments/assets/24056190-9413-4bb3-9918-fcd739b690a5

A collaborative project by Monit KORN and Setthika SUN
Overview
DocuCite-AI is a collaborative, citation-aware document question-answering system built with open-source AI tools. It provides an intuitive Gradio web UI and leverages LangChain and LangGraph for agentic retrieval-augmented generation (RAG). Users can upload a PDF and ask questions; the app generates answers with references to the exact page and paragraph from the document.
Features

PDF Upload & Embedding: Upload any PDF—its contents are chunked, cleaned, and embedded into a Chroma vector store for fast, context-rich retrieval.
Agentic Retrieval: LangGraph orchestrates an agentic workflow, deciding when to use the retrieval tool or generate an answer.
Citation-Aware Tool: The custom document_search tool returns results with [Page X, Paragraph Y] references.
Interactive UI: Built with Gradio for easy, web-based interaction.

Installation

Clone the repository
git clone https://github.com/your-username/DocuCite-AI.git
cd DocuCite-AI


Install dependencies
pip install -r requirements.txt


Run the application
python main.py



Project Structure
DocuCite-AI/
├── main.py                # Entry point for Gradio app
├── pdf_utils.py           # PDF processing helpers
├── vectorstore_utils.py   # Embedding & vector store logic
├── retriever_tool.py      # Citation-aware retrieval tool
├── llm_setup.py           # LLM & embedding model configuration
├── agentic_graph.py       # Agentic workflow with LangGraph
├── requirements.txt       # Python dependencies
└── README.md              # This documentation

Usage

Launch the application using python main.py.
Open the provided Gradio URL in your browser.
Upload a PDF document.
Ask questions about the document, and receive answers with precise [Page X, Paragraph Y] citations.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
License
This project is licensed under the MIT License. See the LICENSE file for details.
