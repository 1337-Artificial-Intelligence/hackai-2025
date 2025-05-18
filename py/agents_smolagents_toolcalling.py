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

# %% [markdown] id="5wpEov0iftUj"
# # Before You Start

# %% [markdown] id="uTj-FCUdgtyv"
# ## What is an LLM Agent?

# %% [markdown] id="MPVdf3mPftUl"
# > ### **Common LLM Limitations:**
# 1. **Hallucination**
#    - LLMs can generate incorrect information with high confidence
#    <center>
#
#    </center>
#
# >2. **Knowledge Cutoff**
#    - LLMs are limited to information from their training data
#    - They cannot access or learn from information after their training period
#    <center>
#
#    </center>
#
# >3. **Data Privacy**
#    - LLMs can only access public training data
#    - They cannot access proprietary or private information
#
#    <center>
#
#    </center>
#
# > ### **The Solution: LLM Agents**
# An LLM Agent enhances a basic LLM by combining three key components:
# - **LLM**: The core language model
# - **Tools**: External capabilities and functions
# - **Memory**: Ability to store and recall information
# <center>
#
# </center>
#
# > This combination helps overcome the limitations by:
# - Using tools to verify information and access external data
# - Maintaining memory of past interactions and information
# - Enabling access to current data through external tools

# %% [markdown] id="Dd-xeqQ3ghxE"
# ## How Does it Work?

# %% [markdown] id="AGrkHcQlftUm"
# > The LLM agent follows a structured decision-making process when responding to instructions. This process consists of three main steps that repeat until the task is complete:
# >1. **Think/Planning:** The agent analyzes the user's request and formulates a step-by-step plan to accomplish the task.
# >2. **Action:** The agent executes specific actions by calling appropriate tools with the necessary parameters.
# >3. **Observation:** The agent evaluates the results of its actions and determines the next steps based on the outcomes.
#
# <center>
#
# </center>
#
# >This cycle continues until the agent successfully completes the requested task or determines it cannot proceed further.

# %% [markdown] id="rmoROrrGftUm"
# # Build Your First Agent Using Smolagent 🤗

# %% [markdown] id="uJheZeegftUn"
# Read more about smolagent:
# * [**smolagents github**](https://github.com/huggingface/smolagents)
# * [**smolagen course**](https://huggingface.co/learn/agents-course/en/unit2/smolagents/introduction)

# %% [markdown] id="TnNEfLV2ftUn"
# > Let's build a sample agent that can:
# * **Search the internet for information**
# * **Do math calculations (+,-,*,/)**
# * **Check the weather of a city**
# * **arxiv paper info**
#
# <center>
#
# </center>

# %% [markdown] id="rik1E2nLftUn"
# #### 1. install smolagent library

# %% id="Hr-QK749ftUn"
# ! uv pip install -U smolagents

# %% [markdown] id="owKtqwFXftUo"
# #### 2. Select Your LLM Model
#
# <center>
#
# </center>
#
# >`smolagent` supports multiple LLM providers including OpenAI, Google, and HuggingFace. For this tutorial, we'll use Google's Gemini `gemini-2.0-flash` model through their free API.
#
# >To get started:
# 1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
# 2. Create an account if you don't have one
# 3. Generate an API key
# 4. Keep your API key secure - we'll use it in the next step.
#
# >You can check **other providers** [here](https://github.com/huggingface/smolagents).

# %% id="JjEBXbG8ftUp"
from smolagents import OpenAIServerModel

# Load the model
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta",
    api_key="AIzaSyCmyEfQ2hMSy9MLUmrelkWsc_f0-msi-ro",
)

# %% id="CA9V7QKsftUp" outputId="9187debf-b9c7-4b63-b3f1-5a8aa107d59c"
# test the model
response=model.client.chat.completions.create(
    model=model.model_id,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Morocco?",
        }
    ]
).choices[0].message.content
print(response)

# %% [markdown] id="4UDZytYpftUq"
# #### 3. Define Tools
#
# >smolagents supports multiple tool types:
# - LangChain tools
# - MCP (Model Control Protocol) tools
# - HuggingFace Space as tool
#
# >In this example, we'll demonstrate how to:
# 1. Create custom tools (using `@tool` or inherit from `Tool`)
# 2. Use predefined tools from `smolagents` and `langchain`
#
# >**Find out more langchain** [tools](https://python.langchain.com/docs/integrations/tools/).
#
# <center>
#
# </center>

# %% id="LcX-PePqftUq" outputId="4fcf9c4e-cbf1-4755-c63b-a7675b178cac"
from smolagents import Tool

class CalculatorTool(Tool):
    # the name of the tool should be clear.
    name="calculator"
    # the tool should have a description clearly stating what it does.
    description="A calculator that can perform basic arithmetic operations."
    # the tool should have a list of inputs and their types.
    inputs={
        "expression": {
            "type": "string",
            "description": "The arithmetic expression to calculate.",
        }
    }
    # the tool should have a of outputs and their types.
    output_type="string"
    # the tool should have forward method consisting of the logic of the tool.
    def forward(self, expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

# test CalculatorTool
calculator_tool = CalculatorTool()
response = calculator_tool("2 + 2 * 3")
print(response)

# %% id="mqa_PC-7ftUq" outputId="8aeb389c-3a6e-439c-c834-b5cb9bbc4ac3"
from smolagents import tool

@tool
def city_weather(city: str) -> str:
    # the tool should have a clear name. and a string documnetation.
    # string documentation has description of the tool and its inputs.
    """
    Get the weather for a given city.
    Args:
        city: The name of the city.
    """
    # In a real-world scenario, this function would call a weather API.
    # For this example, we'll just return a dummy response.
    return f"The weather in {city} is sunny with a high of 25°C."

# test city_weather
response = city_weather("Benguerir")
print(response)

# %% id="dGZMTTltftUr" outputId="8e53fb63-0cf1-4a86-86b0-486593c34560"
from smolagents import DuckDuckGoSearchTool

# Create a DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()
# Test Tool
response = search_tool("What is the capital of Morocco?")
print(response)

# %% id="0W1GGpcNftUr"
from langchain.agents import load_tools
from smolagents import Tool

arxiv_tool=Tool.from_langchain(load_tools(
    ["arxiv"]
)[0])

# %% id="j-2NMgb9ftUr" outputId="f37a11eb-5f66-48ed-d878-468d3524d80c"
result=arxiv_tool("1706.03762")
print(result)

# %% [markdown] id="zOy5MOGkftUs"
# #### 4. Integrate Tools: Create Your Agent

# %% id="a4T4q43LftUs"
from smolagents import CodeAgent

tools=[city_weather,search_tool,calculator_tool,arxiv_tool]
agent=CodeAgent(
    model=model,
    tools=tools
)

# %% [markdown] id="DDk4n1qhftUs"
# #### 5. Run Your Agent

# %% [markdown] id="IpWbHZvWftUs"
# - If you want your agent to use memory you should to add `reset=False` when you run it.
#
# <center>
#
# </center>

# %% id="68dpjxkAftUs" outputId="70eedbaa-9846-44c7-ff1b-34230206b739"
query="""What's the paper 1706.03762 about,
and who is Noam Shazeer?
Also, what is the weather in Benguerir?
And what is 2 + 2 * 3?"""
response=agent.run(query)

# %% id="KVP2Bt7cftUt" outputId="a9e2d557-1d2f-4631-bc29-9402165e2227"
from pprint import pprint
pprint(response)

# %% [markdown] id="jDwRzdL2lHrV"
# # Exercice

# %% id="7IAXinMslKfm"
