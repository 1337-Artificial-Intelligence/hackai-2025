# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/agents_prompt_engineering.ipynb)

# %% [markdown]
# # Prompt Engineering for Beginners 🚀
# 
# > In this notebook, you'll learn how to effectively communicate with AI models through prompts. Think of prompts as instructions you give to AI - the better your instructions, the better the results!
# 
# ## What is Prompt Engineering? 🤔
# 
# Prompt engineering is the art of writing clear instructions (prompts) that help AI models understand exactly what you want them to do. It's like giving directions to a friend - the clearer your directions, the easier it is for them to help you!
# 
# <center>
#     
# ![image](https://i.postimg.cc/NjSKsg9h/pe1.png)
# 
# A simple prompt without instructions - the AI just tries to complete the sentence
#     
# ![image](https://i.postimg.cc/fbdJk6mV/pe2.png)
# 
# A better prompt with clear instructions and data
# 
# ![image](https://i.postimg.cc/Sspj9DfD/pe3.png)
#     
# An even better prompt that specifies exactly what we want in the output
# </center>
# 
# ## The Building Blocks of a Good Prompt 🏗️
# 
# A good prompt usually includes these key parts:
# 
# * **Role**: Tell the AI what role to play (e.g., "You are a friendly math tutor")
# * **Task**: What exactly do you want the AI to do?
# * **Context**: Any important background information
# * **Format**: How should the AI structure its response?
# * **Examples**: Show the AI what you want (optional but helpful!)
# 
# <center>
# ![image](https://i.postimg.cc/C5cKWYTQ/pe6.png)
# </center>
# 
# ## Different Ways to Give Examples 📝
# 
# There are three main ways to show the AI what you want:
# 
# * **Zero-shot**: No examples, just instructions
# * **One-shot**: One example to show what you want
# * **Few-shot**: Multiple examples to make it crystal clear
# 
# ![image](https://i.postimg.cc/9Xbz4PRY/pe7.png)
# 
# ## Understanding AI Chat Roles 👥
# 
# When chatting with AI, there are three main roles:
# 
# * **System**: Sets up the AI's personality and rules
# * **User**: That's you! Your questions and instructions
# * **Assistant**: The AI's responses
# 
# ![image](https://i.postimg.cc/vHmT515F/pe8.png)
# 
# Here's a simple example:
# ```python
# [
#   {
#     "role": "system",
#     "content": "You are a friendly math tutor who explains things simply."
#   },
#   {
#     "role": "user",
#     "content": "What is 2+2?"
#   },
#   {
#     "role": "assistant",
#     "content": "2+2 equals 4! Think of it like having 2 apples and getting 2 more - now you have 4 apples!"
#   }
# ]
# ```
# 
# ## Let's Try It Out! 🎯
# 
# We'll use Groq's free API to practice prompt engineering. First, let's set up our tools:

# %% [markdown]
# ### Setup
# 
# We'll use Groq's free API. Get your API key from [here](https://console.groq.com/keys).

# %%
from openai import OpenAI
from google.colab import userdata

# %%
base_url = "https://api.groq.com/openai/v1"
api_key = userdata.get("groq")
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# %%
def get_response(system_prompt, user_prompt):
    model_id = "llama-3.3-70b-versatile"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.0
    ).choices[0].message.content
    return response

# %% [markdown]
# ### Let's Build Some Simple Tools
# 
# We'll create two simple tools to practice with:
# 1. A calculator
# 2. A weather reporter

# %%
def calculator(num1: int, num2: int, operation: str):
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        if num2 != 0:
            return num1 / num2
        else:
            return "Cannot divide by zero"
    else:
        return "Invalid operation"

def weather(city: str):
    return f"The weather in {city} is sunny and 25°C"

# %% [markdown]
# ### Practice Time! 🎮
# 
# Let's try using our tools with different prompts. We'll create a system prompt that tells the AI how to use our tools:

# %%
system_prompt = """
You are a helpful assistant that can use tools to answer questions. You have access to these tools:

1. calculator: Does basic math (addition, subtraction, multiplication, division)
2. weather: Tells you the weather in any city

To use a tool, respond in this format:
Action:
{
  "action": "tool_name",
  "action_input": {"param1": value1, "param2": value2}
}

For example:
Action:
{
  "action": "calculator",
  "action_input": {"num1": 5, "num2": 3, "operation": "+"}
}
"""

# %% [markdown]
# ### Let's Try Some Examples! 🚀

# %%
# Example 1: Simple calculation
result = get_response(system_prompt, "What is 5 plus 3?")
print("AI's response:", result)

# %%
# Example 2: Weather check
result = get_response(system_prompt, "What's the weather like in Casablanca?")
print("AI's response:", result)

# %% [markdown]
# ## Your Turn! 🎯
# 
# Try these challenges:
# 1. Ask the AI to multiply two numbers
# 2. Ask about the weather in your city
# 3. Try combining both tools in one question!
# 
# ## What We Learned 📚
# * How to write clear prompts
# * Different ways to give examples
# * How to use tools with AI
# * The importance of being specific
# 
# ## Next Steps 🚀
# * Try creating your own tools
# * Experiment with different prompt styles
# * Check out more examples in the [system prompts repository](https://github.com/asgeirtj/system_prompts_leaks)

