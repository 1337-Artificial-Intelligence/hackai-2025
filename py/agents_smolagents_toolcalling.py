# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .env
#     language: python
#     name: python3
# ---
# %% [markdown]

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/agents_smolagents_toolcalling.ipynb)

# %% [markdown] id="5wpEov0iftUj"
# # Building Your First AI Agent 🤖
# 
# In this notebook, you'll learn how to create an AI agent that can:
# - Search the internet
# - Do math calculations
# - Check weather
# - Get information about research papers
# 
# Time to complete: ~45 minutes

# %% [markdown] id="uTj-FCUdgtyv"
# ## What is an AI Agent?
# 
# An AI agent is like a smart assistant that can:
# 1. **Think**: Understand what you want
# 2. **Act**: Use tools to help you
# 3. **Learn**: Remember what it did before
# 
# Think of it as a robot that can use different tools to help you solve problems!

# %% [markdown] id="MPVdf3mPftUl"
# ## Why Do We Need AI Agents?
# 
# Regular AI models have some limitations:
# 1. **They can make mistakes** (hallucination)
# 2. **They only know what they were trained on** (knowledge cutoff)
# 3. **They can't access private information** (data privacy)
# 
# AI agents help solve these problems by:
# - Using tools to check information
# - Accessing current data
# - Remembering past conversations

# %% [markdown] id="rik1E2nLftUn"
# ## Let's Build Our Agent! 🚀
# 
# First, let's install the tools we need:

# %% id="Hr-QK749ftUn"
# ! pip install -U smolagents

# %% [markdown] id="owKtqwFXftUo"
# ## Setting Up Our AI Model
# 
# We'll use Google's Gemini model. To get started:
# 1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
# 2. Create an account
# 3. Get your API key
# 4. Replace the API key below with yours

# %% id="JjEBXbG8ftUp"
from smolagents import OpenAIServerModel

# Load the model
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta",
    api_key="YOUR_API_KEY_HERE",  # Replace with your API key
)

# %% [markdown] id="4UDZytYpftUq"
# ## Creating Our Tools 🛠️
# 
# Let's create some tools our agent can use:

# %% id="LcX-PePqftUq"
from smolagents import Tool

class CalculatorTool(Tool):
    name = "calculator"
    description = "A calculator that can do math (+, -, *, /)"
    inputs = {
        "expression": {
            "type": "string",
            "description": "The math problem to solve",
        }
    }
    output_type = "string"
    
    def forward(self, expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

# %% id="mqa_PC-7ftUq"
from smolagents import tool

@tool
def city_weather(city: str) -> str:
    """
    Get the weather for a city.
    Args:
        city: The name of the city
    """
    # TODO: Add real weather API integration
    return f"The weather in {city} is sunny with a high of 25°C."

# %% id="dGZMTTltftUr"
from smolagents import DuckDuckGoSearchTool

# Create a search tool
search_tool = DuckDuckGoSearchTool()

# %% id="0W1GGpcNftUr"
from langchain.agents import load_tools
from smolagents import Tool

# Create a tool to get information about research papers
arxiv_tool = Tool.from_langchain(load_tools(["arxiv"])[0])

# %% [markdown] id="zOy5MOGkftUs"
# ## Putting It All Together! 🎯
# 
# Now let's create our agent with all these tools:

# %% id="a4T4q43LftUs"
from smolagents import CodeAgent

# Create our agent with all tools
tools = [city_weather, search_tool, calculator_tool, arxiv_tool]
agent = CodeAgent(
    model=model,
    tools=tools
)

# %% [markdown] id="DDk4n1qhftUs"
# ## Let's Test Our Agent! 🚀
# 
# Try asking it different questions:

# %% id="68dpjxkAftUs"
query = """What's the paper 1706.03762 about,
and who is Noam Shazeer?
Also, what is the weather in Benguerir?
And what is 2 + 2 * 3?"""
response = agent.run(query)

# %% id="KVP2Bt7cftUt"
from pprint import pprint
pprint(response)

# %% [markdown] id="jDwRzdL2lHrV"
# ## Your Turn! 🎯
# 
# Try these challenges:
# 1. Ask the agent about a different research paper
# 2. Get the weather for another city
# 3. Try a more complex math problem
# 4. Ask it to search for something interesting
# 
# Remember: The more specific your questions, the better the answers!

# %% [markdown] id="7IAXinMslKfm"
# ## What's Next? 🚀
# 
# You've built your first AI agent! Here's what you can explore next:
# - Add more tools to your agent
# - Try different AI models
# - Build agents for specific tasks
# - Learn about prompt engineering
# 
# Keep experimenting and have fun! 🎉
