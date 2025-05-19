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
# %% [markdown]

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/embedding_sbert_allnli.ipynb)

# %% [markdown]
# # Text Embeddings with Sentence-BERT
# 
# In this notebook, we'll learn how to create and use text embeddings - a fundamental concept in AI that helps computers understand text. We'll use Sentence-BERT, a powerful model that can convert sentences into numbers (vectors) that capture their meaning.

# %% [markdown]
# ## What are Text Embeddings?
# 
# Text embeddings are like a special language that computers use to understand text:
# - They convert words and sentences into numbers (vectors)
# - Similar words/sentences get similar numbers
# - These numbers help computers understand meaning and relationships between text
# 
# TODO: Add image showing how words are converted to vectors in a 2D space

# %% [markdown]
# ## Why Sentence-BERT?
# 
# Sentence-BERT is special because:
# - It understands the full context of sentences (not just individual words)
# - It's fast and efficient
# - It's great for tasks like finding similar sentences or comparing text

# %% [markdown]
# ## Let's Get Started!

# %% [markdown]
# First, let's install the required packages:

# %%
!pip install -U sentence-transformers datasets

# %% [markdown]
# ## 1. Load a Pre-trained Model
# 
# We'll use a model that's already trained to understand text. This saves us time and computing power.

# %%
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a smaller, faster model perfect for learning

# %% [markdown]
# ## 2. Try it Out!
# 
# Let's see how the model converts sentences into numbers:

# %%
# Example sentences
sentences = [
    "I love learning about AI",
    "Artificial Intelligence is fascinating",
    "The weather is nice today"
]

# Get embeddings (convert to numbers)
embeddings = model.encode(sentences)

# Print the shape of our embeddings
print(f"Each sentence is converted into a vector of size: {embeddings.shape[1]}")

# %% [markdown]
# ## 3. Find Similar Sentences
# 
# Let's see how well our model can find similar sentences:

# %%
from sentence_transformers import util

# Calculate similarity between sentences
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(f"Similarity between first two sentences: {similarity.item():.2f}")

# %% [markdown]
# ## 4. Real-world Example: Finding Similar Questions
# 
# Let's use our model to find similar questions from a small dataset:

# %%
questions = [
    "What is machine learning?",
    "How does AI work?",
    "What's the weather like?",
    "Can you explain deep learning?",
    "Is it going to rain today?"
]

# Get embeddings for all questions
question_embeddings = model.encode(questions)

# Find most similar question to "What is machine learning?"
query = "What is machine learning?"
query_embedding = model.encode(query)

# Calculate similarities
similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

# Print results
print("Most similar questions to 'What is machine learning?':")
for idx, score in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:3]:
    print(f"- {questions[idx]} (Score: {score:.2f})")

# %% [markdown]
# ## 5. Try Your Own Examples!
# 
# Now it's your turn! Try these exercises:
# 1. Create your own list of sentences
# 2. Find the most similar pairs
# 3. Try sentences in different languages (the model works with many languages!)

# %%
# Your code here!

# %% [markdown]
# ## What's Next?
# 
# You've learned the basics of text embeddings! Here's what you can explore next:
# - Fine-tuning the model for specific tasks
# - Using embeddings for search engines
# - Building recommendation systems
# - Creating chatbots that understand context

# %% [markdown]
# ## Additional Resources
# - [Sentence-BERT Documentation](https://www.sbert.net/)
# - [Hugging Face Models](https://huggingface.co/models)
# - [Text Embedding Guide](https://www.sbert.net/examples/applications/computing-embeddings/README.html)
