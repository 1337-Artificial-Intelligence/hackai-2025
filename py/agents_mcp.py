# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---
# %% [markdown]

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/agents_mcp.ipynb)

# %% [markdown]
# # Building AI Agents with Model Context Protocol (MCP)
# 
# In this notebook, we'll learn how to build AI agents that can interact with external tools and data sources using MCP. This is a crucial skill for building powerful AI applications!
# 
# ## What is MCP?
# 
# Model Context Protocol (MCP) is like a universal translator for AI applications. Just like how REST APIs help web applications talk to servers, MCP helps AI applications talk to external tools and data sources.
# 
# ### Why do we need MCP?
# - AI models need context (information) to work well
# - Without MCP, connecting AI to tools is complicated and messy
# - With MCP, it's like having a standard plug-and-play system for AI tools
# 
# TODO: Add image showing comparison between with/without MCP
# 
# ## MCP Architecture
# 
# MCP works like a client-server system:
# 
# 1. **Host**: Your AI application (like a chatbot or assistant)
# 2. **MCP Server**: Programs that provide specific capabilities
# 
# TODO: Add architecture diagram
# 
# ### Types of Capabilities:
# 
# | Capability | What it does | Example |
# |------------|-------------|---------|
# | **Tools** | Functions that AI can use to do things | Weather checker, calculator |
# | **Resources** | Read-only data sources | Database of scientific papers |
# | **Prompts** | Pre-made templates for AI interactions | Summarization templates |
# 
# ## Let's Build Our First MCP Server!
# 
# We'll create a simple calculator server that AI can use. We'll use `FastMCP` to make it easy.

# %%
# Install required packages
!pip install fastmcp

# %%
# Create our MCP server
from fastmcp import FastMCP

# Initialize our server
mcp = FastMCP("HackAI Calculator")

# Create a simple calculator tool
@mcp.tool()
def calculator(equation: str) -> str:
    """
    A simple calculator that solves math problems.
    Args:
        equation: A math problem as text (like "2 + 2")
    Returns:
        The answer as text
    """
    try:
        result = eval(equation)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Run the server
if __name__ == "__main__":
    mcp.run()

# %% [markdown]
# ## Using MCP with AI Agents
# 
# Now let's create an AI agent that can use our calculator and other tools!

# %%
# Install smolagents
!pip install smolagents 'smolagents[mcp]'

# %%
from mcp import StdioServerParameters
from smolagents import MCPClient, CodeAgent, OpenAIServerModel

# Set up our tools
parameters = [
    StdioServerParameters(
        command="python",
        args=["calculator_server.py"]
    ),
    # Add more tools here if needed
]

# Create our AI agent
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta",
    api_key="YOUR_API_KEY"  # Replace with your API key
)

# Connect everything together
with MCPClient(parameters) as tools:
    agent = CodeAgent(
        model=model,
        tools=tools
    )
    
    # Try it out!
    while True:
        prompt = input("Ask me anything (type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        result = agent.run(prompt)
        print(result)

# %% [markdown]
# ## What's Next?
# 
# You've learned how to:
# 1. Create an MCP server with tools
# 2. Connect AI agents to MCP tools
# 3. Build a simple calculator agent
# 
# Try these challenges:
# - Add more tools to your server
# - Create an agent that can use multiple tools
# - Build a tool that interacts with a website or database
# 
# ## Resources
# - [MCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
# - [FastMCP Examples](https://github.com/modelcontextprotocol/fastmcp)
# - [SmolAgents Guide](https://huggingface.co/docs/smolagents)
