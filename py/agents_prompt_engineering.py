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

# %% [markdown] id="pS-6tQ9-zqRQ"
# # Prompt Engineer
#
# > An essential part of working with text-generative LLMs is **prompt engineering**. By carefully designing our prompts we can guide the LLM to generate **desired** responses.
# Whether the prompts are questions, statements, or instructions, the main goal of prompt engineering is to elicit a **useful response** from the model.
# * Prompt engineering is more than designing effective prompts.
# <center>
#     
# ![image](https://i.postimg.cc/NjSKsg9h/pe1.png
# )
#
# A basic example of a prompt. No instruction is given so the LLM will simply try to complete the sentence
#     
# ![image](https://i.postimg.cc/fbdJk6mV/pe2.png
# )
#
# Two components of a basic instruction prompt: the instruction itself and the data it refers to.
#
# ![image](https://i.postimg.cc/Sspj9DfD/pe3.png
# )
#     
# Extending the prompt with an output indicator that allows for a specific
# output.
# </center>
#
# >We can continue adding or updating the elements of a prompt until we elicit the response we are looking for. We could add additional examples, describe the use case in more detail, provide additional context, etc.
#
# ## Use cases for instruction-based prompting:
# * the instruction-based prompting can used in number of task for example (function calling, classification as in above images,...).
#
# <center>
#
# ![image](https://i.postimg.cc/j5Q24P8K/pe4.png)
# </center>
#
# * Each of these tasks requires different prompting format as in image bellow.
#
# <center>
#
# ![image](https://i.postimg.cc/fyzL2yFd/pe5.png)
# </center>
#
# ## How to write an accurate prompt:
# > It's sample, ask a specific question, be accurate, add some examples, and you are done!
# > The right prompt will contain several components, and dome common ones are:
# > * **Persona:** Describe what role the LLM should take on. For example, use “You are an expert in math” if you want to ask a question about math.
# > * **Instruction:** The task itself. Make sure this is as specific as possible. We do not want to leave much room for interpretation.
# > * **Context:** Additional information describing the context of the problem or task. It answers questions like “What is the reason for the instruction?”
# > * **Format:** The format the LLM should use to output the generated text. Without it, the LLM will come up with a format itself, which is troublesome in automated
# systems.
# > * **Audience:** The target of the generated text. This also describes the level of the generated
# output. For education purposes, it is often helpful to use ELI5 (“Explain it like I’m 5”).
# > * **Tone**: The tone of voice the LLM should use in the generated text. If you are writing a formal email to your boss, you might not want to use an informal tone of voice.
# > * **Data:** The main data related to the task itself.
# <center>
#
# ![image](https://i.postimg.cc/C5cKWYTQ/pe6.png)
# </center>
#
# ## Zero/One/Few shot prompt:
# One of the most accurate methode to get the wright answer from your LLM is by providing the LLM with examples of exactly the thing that we want to achieve instead of describing the task only.
# this comes in a number of forms depending on how many examples you show the LLM.
# * **Zero-shot** prompting does not leverage examples,
# * **One-shot** prompts use a single example.
# * **Few-shot** prompts use two or more examples.
# ![image](https://i.postimg.cc/9Xbz4PRY/pe7.png)
#
# # Prompt: System, User, Assistant role:
# When interacting with a large language model (LLM), we should always define the roles to differentiate between the system's context and guidelines (system), our input (user), and the model's responses (assistant).
#
# ![image](https://i.postimg.cc/vHmT515F/pe8.png)
#
# ```
#
# [
#   {
#     "role": "system",
#     "content": "You are a helpful assistant that answers questions about machine learning in a clear and concise manner."
#   },
#   {
#     "role": "user",
#     "content": "Can you explain the difference between supervised and unsupervised learning?"
#   },
#   {
#     "role": "assistant",
#     "content": "Sure! In supervised learning, the model is trained on labeled data, meaning each training example includes an input and a known output. In contrast, unsupervised learning uses data without labeled responses, and the goal is often to find hidden patterns or structures in the data."
#   },
#   {
#     "role": "user",
#     "content": "Can you give me an example of each type?"
#   }
# ]
# ```

# %% [markdown] id="9CMPoEV71iFX"
# > Discover more prompt used by big providers (OpenAI, Anthropic, XAI,...) in this [repository](https://github.com/asgeirtj/system_prompts_leaks).

# %% [markdown] id="ISowEjVwsmb9"
# # Tool Calling

# %% [markdown] id="YQZWTWyft8z7"
# To enable tool use in any model, we start with the system prompt. In this special tool use system prompt, wet tell the LLM:
#
# * The basic premise of tool use and what it entails
# * How LLM can call and use the tools it's been given
# * A detailed list of tools it has access to in this specific scenario

# %% [markdown] id="WBelMYMTuPBo"
# ## Setup LLM Model

# %% id="YVq43C2bBUkB"
from openai import OpenAI
from google.colab import userdata

# %% [markdown] id="Edqsrcsu0hvl"
# We will use free tier apis from **Groq**, you can get your api [here](https://console.groq.com/keys).

# %% id="OwUyHPBnB1_G"
base_url="https://api.groq.com/openai/v1"
api_key=userdata.get("groq")
client=OpenAI(
    base_url=base_url,
  api_key=api_key,
)


# %% id="go60oNAavB-t"
def get_response(system_prompt,user_prompt):
  model_id="llama-3.3-70b-versatile"
  messages=[
      {"role":"system","content":system_prompt},
      {"role":"user","content":user_prompt}
  ]
  response=client.chat.completions.create(
    model=model_id,
    messages=messages,
    temperature=0.0
  ).choices[0].message.content
  return response


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="2ZTuX2d2wAEZ" outputId="6cfad8f0-4b9d-4160-8ed3-fee9c6884c83"
# test
get_response("","hello")


# %% [markdown] id="1KQDjVnWuanG"
# ## Build Basic Tools

# %% id="6zTxhsu2yuWI"
def calculator(num1:int, num2:int, operation:str):
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


# %% id="ezryec20zLTJ"
def weather(city:str):
  return f"the weather in {city} is cloudy"


# %% colab={"base_uri": "https://localhost:8080/"} id="E5penlGVyx8_" outputId="84c9ae3d-80f5-4883-a13e-85c81bafd487"
# test the tools
print(calculator(2,2,'+'))
print(weather("Agadir"))

# %% [markdown] id="o7iIPTvRyLRR"
# ## Prepare System Prompt

# %% [markdown] id="g1Osn68AyRDC"
# To enable tool use in any model, we start with the system prompt. In this special tool use system prompt, wet tell the LLM:
#
# * The basic premise of tool use and what it entails
# * How LLM can call and use the tools it's been given
# * A detailed list of tools it has access to in this specific scenario

# %% id="rjayC4TEDjKN"
system_prompt_tc= """
Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location.
calculator: Calculator function for doing basic arithmetic. Supports addition, subtraction, multiplication

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
calculator: Calculator function for doing basic arithmetic. args {"num1":{"type":"int"},"num2":{"type":"int"},"operation":{"type":"str"}}

example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

{{
  "action": "calculator",
  "action_input": {"num1":12,"num2":22,"operation":"+"}
}}

ALWAYS use the following format:

Action:
{
  "action": "tool_name",
  "action_input": {action_input}
}
"""

# %% colab={"base_uri": "https://localhost:8080/"} id="89Iot2uAwQrT" outputId="ebb4278a-d10d-4715-f90f-b3b4559d7e3e"
result_1=get_response(system_prompt=system_prompt_tc,user_prompt="Multiply 1,984,135 by 9,343,116")
result_2=get_response(system_prompt=system_prompt_tc,user_prompt="what is the weather in agadir?")
print(result_1)
print(result_2)

# %% colab={"base_uri": "https://localhost:8080/"} id="j_oH-ilixyzT" outputId="c5db7691-7de1-48f9-bb2f-c60548357788"
import json
function1_json_call=json.loads(result_1[result_1.index("Action:")+len("Action:"):].strip())
function2_json_call=json.loads(result_2[result_2.index("Action:")+len("Action:"):].strip())
print(function1_json_call)
print(function2_json_call)

# %% colab={"base_uri": "https://localhost:8080/"} id="h-4kWPgJL3xz" outputId="4a0706ac-298c-484d-e136-77a6c8bdd418"
# function calling
num1,num2,operation=function1_json_call["action_input"].values()
calculator(num1,num2,operation)

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="1SqKd2GI1Dfz" outputId="34acf797-c204-46f3-95e1-6c7a0da6d07f"
operation

# %% [markdown] id="0sxeNwzO30AZ"
# # Exercice

# %% [markdown] id="T6QNlUV4ynnK"
# In this exercise, you'll be writing a tool use prompt for querying and writing to the world's smallest "database". Here's the initialized database, which is really just a dictionary.

# %% id="BjyMUeX72UgZ"
db = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ],
    "products": [
        {"id": 1, "name": "Widget", "price": 9.99},
        {"id": 2, "name": "Gadget", "price": 14.99},
        {"id": 3, "name": "Doohickey", "price": 19.99}
    ]
}


# %% [markdown] id="fveG_wIbytld"
# And here is the code for the functions that write to and from the database.

# %% id="2bhWtu7i3rhy"
def get_user(user_id):
    for user in db["users"]:
        if user["id"] == user_id:
            return user
    return None

def get_product(product_id):
    for product in db["products"]:
        if product["id"] == product_id:
            return product
    return None

def add_user(name, email):
    user_id = len(db["users"]) + 1
    user = {"id": user_id, "name": name, "email": email}
    db["users"].append(user)
    return user

def add_product(name, price):
    product_id = len(db["products"]) + 1
    product = {"id": product_id, "name": name, "price": price}
    db["products"].append(product)
    return product


# %% [markdown] id="AXif4y9iy0yR"
# To solve the exercise, start by defining a system prompt like `system_prompt` above. Make sure to include the name and description of each tool, along with the name and type and description of each parameter for each function. We've given you some starting scaffolding below.

# %% id="JkGhRjJV3v7F"
system_prompt_tc= """
Answer the following questions as best you can. You have access to the following tools:

get_user: Get user from database using user id.
get_product: Get produt from database using product id.
add_user: Add user to database using name and email.
add_product: Add product to database using name and price.

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_user: Get user from database using user id, args: {"user_id": {"type": "int"}}
get_product: Get produt from database using product id, args: {"product_id":{"type":"int"}}
add_user: Add user to database using name and email, args: {"name":{"type":"str"},"email":{"type":"str"}}
add_product: Add product to database using name and price. args:{"name":{"type":"str"},"price":{"type":"float"}}

example use :

{{
  "action": "get_user",
  "action_input": {"user_id": 1}
}}

ALWAYS use the following format:

Action:
{
  "action": "tool_name",
  "action_input": {action_input}
}
"""

# %% [markdown] id="XzRPt_GAy_fi"
# When you're ready, you can try out your tool definition system prompt on the examples below. Just run the below cell!
#

# %% colab={"base_uri": "https://localhost:8080/"} id="8aZ84N7o6tMQ" outputId="4eb8760c-cdce-4267-b104-f7fe222f7a5c"
examples = [
    "Add a user to the database named Deborah.",
    "Add a product to the database named Thingo",
    "Tell me the name of User 2",
    "Tell me the name of Product 3"
]

for example in examples:
    function_calling_response = get_response(system_prompt=system_prompt_tc,user_prompt=example)
    print(example, "\n----------\n\n", function_calling_response, "\n*********\n*********\n*********\n\n")

