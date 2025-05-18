# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: mlw
#     language: python
#     name: python3
# ---

# %%

"""
Recreation of the document-processing-for-ai.ipynb notebook logic.
Processes a PDF, chunks it using LLM suggestions, enriches chunks with context,
and demonstrates a simple RAG setup.
"""

import base64
import re
import textwrap
import time
from io import BytesIO
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import ollama # Make sure ollama server is running

# --- Docling Imports ---
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    # smolvlm_picture_description, # Assuming this exists or define a placeholder
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# --- Scikit-learn Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%

# --- Configuration ---

# Placeholder for smolvlm if not directly available or substitute
# For demonstration, we'll use a simple placeholder function/value if needed.
# If smolvlm_picture_description is available in your docling setup, use it.
# Otherwise, you might need to disable picture description or use a fallback.
# Let's assume for now picture description is enabled but might yield basic results
# without the exact smolvlm setup.
try:
    # Attempt to import if it's part of your docling install structure
    from docling.datamodel.pipeline_options import smolvlm_picture_description
    PICTURE_DESCRIPTION_OPTIONS = smolvlm_picture_description
    print("Using smolvlm_picture_description.")
except ImportError:
    print("Warning: smolvlm_picture_description not found. Picture description might be basic.")
    # Define a fallback or disable picture description if necessary
    # Disabling for simplicity if not found:
    # PICTURE_DESCRIPTION_OPTIONS = None
    # Or use a placeholder that might exist:
    PICTURE_DESCRIPTION_OPTIONS = None # Set to None or a valid option


PDF_PATH = Path("API_FR.pdf")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True) # Create output dir if needed

# LLM Configuration
MODEL = "hf.co/google/gemma-3-12b-it-qat-q4_0-gguf" # Make sure this model is pulled in Ollama
TEMPERATURE = 0.0
MIN_P = 0.01
REPEAT_PENALTY = 1.0
TOP_K = 64
TOP_P = 0.95
OLLAMA_KEEP_ALIVE = "-1h" # Keep model loaded for a while

# Placeholders
IMAGE_PLACEHOLDER = "<!__ image_placeholder __>"
PAGE_BREAK_PLACEHOLDER = "<!__ page_break __>"

# Chunking Pattern (Initial basic split before LLM refinement)
# Using Markdown H2 headers as split points initially
INITIAL_SPLIT_PATTERN = "\n## "

# --- Prompts ---

CHUNKING_PROMPT = """
You are an assistant specialized in splitting text into semantically consistent sections.

<instructions>
<instruction>The text has been divided into initial chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.</instruction>
<instruction>Identify points where splits should occur, such that consecutive chunks of similar themes stay together.</instruction>
<instruction>Each final combined section should ideally be between 200 and 1000 words (this is a guideline, semantic coherence is more important).</instruction>
<instruction>If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.</instruction>
<instruction>The split points must be listed in ascending order.</instruction>
<instruction>Provide your response ONLY in the form: 'split_after: 3, 5' (use the number of the chunk AFTER which the split should occur).</instruction>
<instruction>If no splits are suitable other than the initial ones, you might return just the last chunk number, e.g., 'split_after: 15'.</instruction>
</instructions>

This is the document text with initial chunk markers:
<document>
{document_text}
</document>

Respond ONLY with the IDs of the chunks AFTER which a split should occur, in the specified format 'split_after: X, Y, Z'.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT POINT suggestion (even if it's just the last chunk).
""".strip()

CONTEXTUALIZER_PROMPT = """
You are an assistant specialized in analyzing document chunks and providing relevant context for search retrieval.

<instructions>
<instruction>You will be given a full document and a specific chunk from that document.</instruction>
<instruction>Provide 2-3 concise sentences that situate this chunk within the broader document, focusing on information useful for retrieval.</instruction>
<instruction>Identify the main topic or concept discussed in the chunk.</instruction>
<instruction>Include relevant information or comparisons from the broader document context if they help clarify the chunk's meaning or significance (e.g., overall trends, specific product lines mentioned).</instruction>
<instruction>Note how this information relates to the overall theme or purpose of the document (e.g., financial results, product announcements).</instruction>
<instruction>Include key figures, dates, or percentages from the chunk or surrounding context if they provide important context for search.</instruction>
<instruction>Avoid phrases like "This chunk discusses..." or "In this chunk...". Instead, directly state the context.</instruction>
<instruction>Keep your response brief (target 50-100 tokens) and focused on improving search retrieval for this specific chunk.</instruction>
</instructions>

Here is the full document:
<document>
{document}
</document>

Here is the specific chunk to contextualize:
<chunk>
{chunk}
</chunk>

Respond ONLY with the succinct context for this chunk. Do not add any explanations or conversational text.
""".strip()

RAG_PROMPT = """
Use the following context pieces to answer the question. Each context piece contains a chunk from a larger document and its generated context summary.

<contexts>
{contexts}
</contexts>

Based *only* on the provided contexts, answer the following question:
<question>
{question}
</question>

Answer:
""".strip()


# --- Helper Functions ---

def replace_occurrences(text: str, target: str, replacements: list[str]) -> str:
    """Replaces sequential occurrences of a target string with replacements."""
    for replacement in replacements:
        if target in text:
            # Replace only the first occurrence found in each iteration
            text = text.replace(target, replacement, 1)
        else:
            print(f"Warning: No more occurrences of '{target}' found for replacement: '{replacement[:50]}...'")
            # Decide how to handle: break, continue, raise error?
            # For robustness, let's just break and leave remaining placeholders if any
            break
            # Alternative: raise ValueError(f"No more occurrences of {target} found in the text for replacement {replacement}")
    # Check if any placeholders remain
    if target in text:
        print(f"Warning: Some occurrences of '{target}' remained unreplaced.")
    return text

def call_model(prompt: str) -> str:
    """Calls the Ollama model with the specified prompt and parameters."""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={
                "temperature": TEMPERATURE,
                "min_p": MIN_P,
                "repeat_penalty": REPEAT_PENALTY,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "num_ctx": 16384 # Set context window if needed, depends on model
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error calling Ollama model: {e}")
        return f"Error: Could not get response from model. {e}"

def split_text_by_llm_suggestions(tagged_text: str, llm_response: str) -> list[str]:
    """Splits the initially tagged text based on LLM's split_after suggestions."""
    split_after_indices = set()
    if "split_after:" in llm_response.lower():
        try:
            split_points_str = llm_response.lower().split("split_after:")[1].strip()
            if split_points_str: # Ensure there are numbers after 'split_after:'
                 split_after_indices = {int(x.strip()) for x in split_points_str.split(",") if x.strip().isdigit()}
            else:
                 print("Warning: 'split_after:' found but no numbers followed.")
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse LLM split suggestions '{llm_response}'. Error: {e}. Returning text as single chunk.")
            # Fallback: Find all chunk content and return as one big chunk
            chunk_pattern = r"<\|start_chunk_\d+\|>(.*?)<\|end_chunk_\d+\|>"
            all_content = re.findall(chunk_pattern, tagged_text, re.DOTALL)
            return ["\n".join(all_content).strip()] if all_content else []

    print(f"LLM suggested splitting after chunk indices: {sorted(list(split_after_indices))}")

    # Find all initial chunks using the tags
    chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
    # Use re.DOTALL to make '.' match newlines within chunks
    initial_chunks = re.findall(chunk_pattern, tagged_text, re.DOTALL)

    if not initial_chunks:
        print("Error: Could not find any initial chunks in the tagged text.")
        return []

    final_sections = []
    current_section_content = []

    for chunk_id_str, chunk_content in initial_chunks:
        chunk_id = int(chunk_id_str)
        current_section_content.append(chunk_content.strip())

        if chunk_id in split_after_indices:
            final_sections.append("\n".join(current_section_content).strip())
            current_section_content = [] # Start a new section

    # Add the last section if it has content
    if current_section_content:
        final_sections.append("\n".join(current_section_content).strip())

    # Filter out empty sections just in case
    final_sections = [section for section in final_sections if section]

    print(f"Split into {len(final_sections)} sections based on LLM suggestions.")
    return final_sections


# --- RAG Classes/Functions ---

class SimpleRetriever:
    """A simple TF-IDF based retriever."""
    def __init__(self, texts: list[str]):
        if not texts:
            raise ValueError("Cannot initialize SimpleRetriever with empty text list.")
        self.texts = texts
        print(f"Initializing SimpleRetriever with {len(texts)} text chunks.")
        self.vectorizer = TfidfVectorizer()
        try:
            self.text_vectors = self.vectorizer.fit_transform(self.texts)
            print("TF-IDF vectors created successfully.")
        except ValueError as e:
            print(f"Error during TF-IDF vectorization: {e}")
            print("This might happen if the input text is empty or contains only stop words.")
            # Handle error, maybe by setting vectors to None or raising exception
            self.text_vectors = None


    def retrieve(self, query: str, k: int = 3) -> tuple[list[str], list[float]]:
        """Retrieves top k relevant texts for a given query."""
        if self.text_vectors is None:
            print("Error: TF-IDF vectors not available.")
            return [], []
        if k <= 0:
            return [], []

        query_vector = self.vectorizer.transform([query])
        # similarities will be a 2D array, get the first row
        similarities = cosine_similarity(query_vector, self.text_vectors)[0]

        # Get indices of top k similarities (descending order)
        # Handle case where k is larger than number of documents
        num_docs = len(self.texts)
        actual_k = min(k, num_docs)
        if actual_k == 0:
            return [], []

        # Argsort gives indices of smallest to largest, so we take the last 'actual_k' and reverse them
        top_k_indices = np.argsort(similarities)[-actual_k:][::-1]

        retrieved_texts = [self.texts[i] for i in top_k_indices]
        retrieved_scores = [similarities[i] for i in top_k_indices]

        return retrieved_texts, retrieved_scores

def ask_question(query: str, retriever: SimpleRetriever, k: int = 3) -> str:
    """Retrieves context and asks the LLM to answer a question based on it."""
    print(f"\n--- Answering Question (k={k}): {query} ---")
    retrieved_texts, retrieved_scores = retriever.retrieve(query, k=k)

    if not retrieved_texts:
        return "Could not retrieve any relevant context for the question."

    # Format context for the prompt
    contexts_for_prompt = []
    for i, (text, score) in enumerate(zip(retrieved_texts, retrieved_scores)):
         contexts_for_prompt.append(f"<context index=\"{i}\" score=\"{score:.4f}\">\n{text}\n</context>")

    context_string = "\n\n".join(contexts_for_prompt)

    prompt = RAG_PROMPT.format(contexts=context_string, question=query)

    # print("\n--- RAG Prompt ---")
    # print(textwrap.fill(prompt, width=120)) # Optional: print the prompt sent to LLM
    # print("--- End RAG Prompt ---")

    answer = call_model(prompt)
    print("--- LLM Answer ---")
    print(textwrap.fill(answer, width=120))
    print("------\n")
    return answer


# --- Main Processing Pipeline --

# %%
# --- 1. Load Environment Variables ---
load_dotenv()
print("Environment variables loaded.")

# %%
# --- 2. Configure Docling ---
pipeline_options = PdfPipelineOptions(
    generate_page_images=False, # Don't need images for script processing
    # images_scale=1.0, # Not needed if generate_page_images is False
    do_ocr=True,
    do_picture_description=True if PICTURE_DESCRIPTION_OPTIONS else False,
    ocr_options=RapidOcrOptions(),
    picture_description_options=PICTURE_DESCRIPTION_OPTIONS,
)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
print("DocumentConverter configured.")

# %%
# --- 3. Convert PDF to Docling Document ---
if not PDF_PATH.is_file():
    print(f"Error: PDF file not found at {PDF_PATH}")
    # return

print(f"Processing PDF: {PDF_PATH}...")
convert_start = time.time()
try:
    result = converter.convert(PDF_PATH)
    doc = result.document
except Exception as e:
    print(f"Error during PDF conversion: {e}")
    # return
print(f"PDF converted in {time.time() - convert_start:.2f} seconds.")

# %%
# pip install rapidocr_onnxruntime

# %%

# --- 4. Extract Image Annotations (if generated) ---
image_annotations = []
if pipeline_options.do_picture_description:
    for picture in doc.pictures:
        if picture.annotations:
            # Take the first annotation if multiple exist
            image_annotations.append(picture.annotations[0].text)
        else:
            # Add a default placeholder if no annotation was generated
            image_annotations.append("Image detected, no description generated.")
    print(f"Extracted {len(image_annotations)} image annotations.")
else:
    print("Image description was disabled.")
    # Need to know how many images were potentially detected to replace placeholders
    # This info might be in doc.pictures even if description is off.
    num_image_placeholders = len(doc.pictures) # Assuming this count is still valid
    image_annotations = ["Image detected."] * num_image_placeholders


# %%
# --- 5. Export to Markdown with Placeholders ---
print("Exporting document to Markdown...")
markdown_text = doc.export_to_markdown(
    page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
    image_placeholder=IMAGE_PLACEHOLDER,
)
# Basic cleaning - remove potential extra newlines
markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text).strip()



# %%
# --- 6. Replace Image Placeholders with Annotations ---
print("Replacing image placeholders with annotations...")
processed_text = replace_occurrences(markdown_text, IMAGE_PLACEHOLDER, image_annotations)
# Optional: Write intermediate text to file
# with open(OUTPUT_DIR / "processed_text_intermediate.md", "w", encoding="utf-8") as f:
#     f.write(processed_text)

word_count = len(processed_text.split())
print(f"Initial processed text word count: {word_count}")

# %%
# --- 7. LLM-based Chunking ---
print("\n--- Starting LLM-based Chunking ---")
# 7a. Initial Split (using simple pattern)
initial_split_chunks = re.split(f"({INITIAL_SPLIT_PATTERN})", processed_text)
# The split includes the delimiter, need to recombine
combined_initial_chunks = []
if initial_split_chunks:
    # Add the first part if it doesn't start with the delimiter
    if not initial_split_chunks[0].strip().startswith("##") and initial_split_chunks[0].strip():
            combined_initial_chunks.append(initial_split_chunks[0].strip())
    # Combine delimiter with the following text
    for i in range(1, len(initial_split_chunks), 2):
        if i + 1 < len(initial_split_chunks):
                combined_initial_chunks.append(
                    (initial_split_chunks[i] + initial_split_chunks[i+1]).strip()
                )
        else: # Handle potential last delimiter
                combined_initial_chunks.append(initial_split_chunks[i].strip())

print(f"Initially split into {len(combined_initial_chunks)} chunks based on '{INITIAL_SPLIT_PATTERN}'.")


# %%
# 7b. Add Start/End Tags for LLM
tagged_text_parts = []
for i, chunk in enumerate(combined_initial_chunks):
    tagged_text_parts.append(f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>")
tagged_text = "\n\n".join(tagged_text_parts)
# Optional: Write tagged text
# with open(OUTPUT_DIR / "tagged_text_for_chunking.md", "w", encoding="utf-8") as f:
#      f.write(tagged_text)

# 7c. Call LLM for Chunking Suggestions
print("Asking LLM for chunking suggestions...")
chunking_prompt_filled = CHUNKING_PROMPT.format(document_text=tagged_text)
llm_chunking_response = call_model(chunking_prompt_filled)
print(f"LLM chunking suggestion response: '{llm_chunking_response}'")

# 7d. Apply LLM Suggestions to Create Final Chunks
llm_chunks = split_text_by_llm_suggestions(tagged_text, llm_chunking_response)

if not llm_chunks:
    print("Error: LLM-based chunking resulted in no chunks. Exiting.")
    # return

# Optional: Print chunk examples
# print("\n--- Example LLM Chunks ---")
# for i, chunk in enumerate(llm_chunks[:2]):
#     print(f"--- Chunk {i} ---")
#     print(textwrap.fill(chunk, width=100))
#     print("-" * 20)


# %%
# --- 8. Contextual Enrichment ---
print("\n--- Starting Contextual Enrichment ---")
enriched_chunks = []
contexts_generated = []
enrich_start = time.time()
for i, chunk in enumerate(llm_chunks):
    print(f"Generating context for chunk {i+1}/{len(llm_chunks)}...")
    context_prompt = CONTEXTUALIZER_PROMPT.format(document=processed_text, chunk=chunk)
    context = call_model(context_prompt)

    if context.startswith("Error:"):
        print(f"Warning: Failed to generate context for chunk {i}. Using original chunk only.")
        context = "Context generation failed." # Placeholder context

    contexts_generated.append(context)
    enriched_chunks.append(f"<chunk_context>\n{context}\n</chunk_context>\n\n<chunk>\n{chunk}\n</chunk>")
    # Optional: Add a small delay if hitting API rate limits
    # time.sleep(0.5)

print(f"Contextual enrichment finished in {time.time() - enrich_start:.2f} seconds.")

# Optional: Print example enriched chunk
# if enriched_chunks:
#     print("\n--- Example Enriched Chunk (Chunk 0) ---")
#     print(textwrap.fill(enriched_chunks[0], width=120))
#     print("-" * 20)

# Save enriched chunks to file
with open(OUTPUT_DIR / "enriched_chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(enriched_chunks):
        f.write(f"--- Chunk {i} ---\n")
        f.write(chunk)
        f.write("\n\n" + "="*80 + "\n\n")
print(f"Enriched chunks saved to {OUTPUT_DIR / 'enriched_chunks.txt'}")



# %%
# --- 9. Setup RAG ---
print("\n--- Setting up RAG ---")
if not enriched_chunks:
    print("Error: No enriched chunks available for RAG setup.")
    # return

try:
    retriever = SimpleRetriever(enriched_chunks)
except ValueError as e:
    print(f"Error initializing retriever: {e}")
    # return



# %%

# --- 10. Ask Questions using RAG ---
print("\n--- Asking Questions via RAG ---")
ask_question("How do you activate the public REST APIs in CCH Tagetik? Are they enabled by default?", retriever, k=3)
# ask_question("What is the gaming revenue for the fourth quarter?", retriever, k=3)
# ask_question("Summarize the financial highlights for Fiscal 2025.", retriever, k=4)
# ask_question("What are the key points about the Data Center business?", retriever, k=3)

# total_time = time.time() - start_time
# print(f"\n--- Pipeline Finished ---")
# print(f"Total execution time: {total_time:.2f} seconds.")
