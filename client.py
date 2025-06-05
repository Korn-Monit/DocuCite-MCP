# Create server parameters for stdio connection
from hackathon.mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["server.py"],
)

# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         # Initialize the connection
#         await session.initialize()

#         # Get tools
#         tools = await load_mcp_tools(session)

#         # Create and run the agent
#         agent = create_react_agent("openai:gpt-4.1", tools)
#         agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

import asyncio
def remove_think_section(content: str) -> str:
    """Remove the section between <think> and </think> tags from the content."""
    closing_tag = '</think>'
    position = content.find(closing_tag)
    if position != -1:
        return content[position + len(closing_tag):].strip()
    return content.strip()

response_model = ChatOllama(model="qwen3:4b", temperature=0.2)
# response_model = remove_think_section(response_model.content)
# response_model.content = response_model

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(response_model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            # context = next(
            #     msg.content 
            #     for msg in reversed(agent_response.messages) 
            #     if isinstance(msg, AIMessage)  # Direct type check
            # )
            print("Agent Response:", agent_response)

asyncio.run(main())