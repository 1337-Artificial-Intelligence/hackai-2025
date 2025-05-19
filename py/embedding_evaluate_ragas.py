# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/embedding_evaluate_ragas.ipynb)

# %% [markdown]
# # Evaluating RAG Systems with RAGAS
# 
# In this notebook, you'll learn how to evaluate the quality of Retrieval Augmented Generation (RAG) systems using RAGAS, a popular evaluation framework.
# 
# ## Learning Objectives
# - Understand what RAG evaluation means and why it's important
# - Learn about different metrics used to evaluate RAG systems
# - Practice evaluating a simple RAG system using RAGAS
# 
# ## What is RAG Evaluation?
# When we build RAG systems, we need to know if they're working well. RAGAS helps us measure:
# - How relevant the retrieved information is
# - How accurate the generated answers are
# - How well the system uses the provided context
# 
# Let's start by installing our required packages:

# %%
!pip install -U -q langchain openai ragas arxiv pymupdf chromadb wandb tiktoken

# %% [markdown]
# ## Setting Up OpenAI
# We'll need an OpenAI API key to use their models. You can get one from [OpenAI's website](https://platform.openai.com/api-keys).

# %%
import os
import openai
from getpass import getpass

openai.api_key = getpass("Please provide your OpenAI Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key

# %% [markdown]
# ## Loading Sample Data
# We'll use some academic papers about RAG as our test data. This will help us evaluate our system with real-world content.

# %%
from langchain.document_loaders import ArxivLoader

# Load 3 papers about RAG
base_docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=3).load()
print(f"Loaded {len(base_docs)} documents")

# %% [markdown]
# ## Creating a Simple RAG System
# Let's build a basic RAG system that we can evaluate:
# 1. Split documents into smaller chunks
# 2. Create embeddings for these chunks
# 3. Store them in a vector database
# 4. Set up a retriever to find relevant chunks

# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)
docs = text_splitter.split_documents(base_docs)

# Create vector store with embeddings
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# Create retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# %% [markdown]
# ## Setting Up the QA Chain
# Now we'll create a simple question-answering chain that:
# 1. Takes a question
# 2. Retrieves relevant context
# 3. Generates an answer

# %%
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough

# Create prompt template
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Create QA chain
primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

# %% [markdown]
# ## Creating Test Questions
# We'll create some test questions to evaluate our RAG system. In a real scenario, you'd want more questions, but we'll keep it simple for this demo.

# %%
test_questions = [
    "What is Retrieval Augmented Generation?",
    "How does RAG improve language models?",
    "What are the main components of a RAG system?"
]

# %% [markdown]
# ## Evaluating with RAGAS
# Now we'll use RAGAS to evaluate our system. RAGAS provides several metrics:
# - Answer Relevancy: How relevant is the answer to the question?
# - Faithfulness: Does the answer stay true to the retrieved context?
# - Context Relevancy: How relevant is the retrieved context to the question?

# %%
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy
)
from ragas import evaluate
from tqdm import tqdm

# Create evaluation dataset
eval_dataset = []
for question in tqdm(test_questions):
    result = retrieval_augmented_qa_chain.invoke({"question": question})
    eval_dataset.append({
        "question": question,
        "answer": result["response"].content,
        "contexts": [context.page_content for context in result["context"]],
        "ground_truths": ["TODO: Add ground truth answers"]  # In a real scenario, you'd have human-verified answers
    })

# Evaluate
result = evaluate(
    eval_dataset,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_relevancy
    ],
)

# %% [markdown]
# ## Understanding the Results
# Let's look at what our evaluation tells us about our RAG system:

# %%
print("Evaluation Results:")
for metric, score in result.items():
    print(f"{metric}: {score:.2f}")

# %% [markdown]
# ## What Do These Scores Mean?
# - Answer Relevancy (0-1): Higher is better. Shows how well the answer matches the question.
# - Faithfulness (0-1): Higher is better. Shows if the answer is based on the retrieved context.
# - Context Relevancy (0-1): Higher is better. Shows if we retrieved the right information.
# 
# ## Next Steps
# To improve your RAG system, you could:
# 1. Try different chunk sizes
# 2. Use different embedding models
# 3. Adjust the number of retrieved documents
# 4. Improve the prompt template
# 
# ## Additional Resources
# - [RAGAS Documentation](https://docs.ragas.io/)
# - [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
# - [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
