# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/embedding_process_pdf.ipynb)

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
# # Building a Simple RAG System with PDFs
# 
# In this notebook, you'll learn how to build a Retrieval Augmented Generation (RAG) system that can answer questions about PDF documents. This is a fundamental skill in GenAI that combines document processing, information retrieval, and language models.
# 
# ## Learning Objectives
# - Understand what RAG is and why it's useful
# - Learn how to process PDF documents for AI
# - Build a simple question-answering system using RAG
# 
# ## What is RAG?
# RAG (Retrieval Augmented Generation) is a technique that combines:
# 1. **Retrieval**: Finding relevant information from documents
# 2. **Generation**: Using an AI model to generate answers based on that information
# 
# This helps AI models provide more accurate and up-to-date answers by using specific information from your documents.
# 
# Let's start by installing our required packages:

# %%
!pip install -q docling rapidocr_onnxruntime ollama scikit-learn python-dotenv

# %% [markdown]
# ## Setting Up Our Environment
# We'll need to set up our environment variables and import necessary libraries.

# %%
import base64
import re
import textwrap
import time
from io import BytesIO
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import ollama  # For running our language model locally

# --- Docling Imports ---
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Scikit-learn Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Configuration
# Let's set up our basic configuration. We'll use a simple PDF file for demonstration.

# %%
# Basic configuration
PDF_PATH = Path("sample.pdf")  # You'll need to upload your PDF
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# LLM Configuration
MODEL = "gemma:2b"  # A smaller model that works well for our needs
TEMPERATURE = 0.0  # Lower temperature for more focused answers
TOP_K = 64
TOP_P = 0.95

# %% [markdown]
# ## Helper Functions
# Let's create some helper functions that we'll use in our RAG system.

# %%
def call_model(prompt: str) -> str:
    """Calls the Ollama model with the specified prompt."""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        return f"Error: Could not get response from model. {e}"

# %% [markdown]
# ## Building Our RAG System
# Now let's create our simple RAG system. We'll break this into steps:
# 1. Process the PDF
# 2. Split it into chunks
# 3. Create a retriever
# 4. Build our question-answering system

# %%
class SimpleRetriever:
    """A simple TF-IDF based retriever for finding relevant text chunks."""
    def __init__(self, texts: list[str]):
        if not texts:
            raise ValueError("Cannot initialize retriever with empty text list.")
        self.texts = texts
        print(f"Initializing retriever with {len(texts)} text chunks.")
        self.vectorizer = TfidfVectorizer()
        try:
            self.text_vectors = self.vectorizer.fit_transform(self.texts)
            print("TF-IDF vectors created successfully.")
        except ValueError as e:
            print(f"Error during vectorization: {e}")
            self.text_vectors = None

    def retrieve(self, query: str, k: int = 3) -> tuple[list[str], list[float]]:
        """Finds the most relevant text chunks for a query."""
        if self.text_vectors is None:
            return [], []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.text_vectors)[0]
        
        # Get top k most similar chunks
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.texts[i] for i in top_k_indices], [similarities[i] for i in top_k_indices]

# %% [markdown]
# ## Processing the PDF
# Let's set up our PDF processing pipeline.

# %%
# Configure PDF processing
pipeline_options = PdfPipelineOptions(
    generate_page_images=False,
    do_ocr=True,
    do_picture_description=False,  # Simplified for this example
    ocr_options=RapidOcrOptions(),
)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# %% [markdown]
# ## Main RAG Pipeline
# Now let's put everything together to create our RAG system.

# %%
def process_pdf_and_setup_rag(pdf_path: Path) -> SimpleRetriever:
    """Processes a PDF and sets up a RAG system."""
    print(f"Processing PDF: {pdf_path}...")
    
    # Convert PDF to text
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Export to markdown
    markdown_text = doc.export_to_markdown()
    
    # Simple chunking by paragraphs
    chunks = [chunk.strip() for chunk in markdown_text.split('\n\n') if chunk.strip()]
    
    # Create retriever
    retriever = SimpleRetriever(chunks)
    return retriever

def ask_question(query: str, retriever: SimpleRetriever, k: int = 3) -> str:
    """Asks a question using our RAG system."""
    print(f"\nQuestion: {query}")
    
    # Retrieve relevant chunks
    retrieved_texts, scores = retriever.retrieve(query, k=k)
    
    if not retrieved_texts:
        return "Could not find relevant information."
    
    # Create prompt with context
    context = "\n\n".join([f"Context {i+1}:\n{text}" for i, text in enumerate(retrieved_texts)])
    prompt = f"""Use the following context to answer the question. If you can't answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Get answer from model
    answer = call_model(prompt)
    print("\nAnswer:")
    print(textwrap.fill(answer, width=80))
    return answer

# %% [markdown]
# ## Let's Try It Out!
# Now we can use our RAG system to answer questions about the PDF.

# %%
# TODO: Upload your PDF file first
# retriever = process_pdf_and_setup_rag(PDF_PATH)

# Example questions (uncomment after uploading PDF):
# ask_question("What is the main topic of this document?", retriever)
# ask_question("What are the key points discussed?", retriever)

# %% [markdown]
# ## Next Steps
# - Try different chunking strategies
# - Experiment with different retrieval methods
# - Add more sophisticated context processing
# - Evaluate the system's performance
