from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question with reference and page number."
    "attention to the context, and only use it to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Question: {question} \n"
    "Context: {context}"
)

def make_generate_query_or_respond(LLM, retriever_tool):
    def generate_query_or_respond(state):
        response = (
            LLM
            .bind_tools([retriever_tool]).invoke(state["messages"])
        )
        return {"messages": [response]}
    return generate_query_or_respond

def generate_answer(state, LLM):
    print(f"[DEBUG] Answer node, messages so far: {state['messages']}")
    question = state["messages"][0].content
    print(f"[DEBUG] Question: {question}")
    context = state["messages"][-1].content
    print(f"[DEBUG] Context: {context}")
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = LLM.invoke([{"role": "user", "content": prompt}])
    print(f"[DEBUG] LLM final answer: {response}")
    return {"messages": [response]}

def build_agentic_graph(LLM, retriever_tool):
    workflow = StateGraph(MessagesState)
    workflow.add_node("generate_query_or_respond", make_generate_query_or_respond(LLM, retriever_tool))
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, LLM))
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()
