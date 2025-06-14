from langchain.tools import Tool

def build_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    def custom_search(query: str) -> str:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant passages found."
        out = []
        for d in docs:
            page = d.metadata.get("page_number", "?")
            para = d.metadata.get("paragraph_number", "?")
            txt  = d.page_content.replace("\n", " ").strip()
            out.append(f"[Page {page}, Paragraph {para}]: {txt}")
        return "\n\n".join(out)
    return Tool(
        name="document_search",
        func=custom_search,
        description=(
            "Searches the uploaded PDF for a query and returns each matching "
            "passage prefixed with its page and paragraph number."
        ),
    )
