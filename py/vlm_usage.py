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

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/vlm_usage.ipynb)

# %% [markdown]
# ## 🏆 Vision Language Models (VLMs)
# ### 📌 Description
#
# In this challenge, you'll explore Vision-Language Models (VLMs) - AI models that can understand both images and text! You'll learn how to:
# - Ask questions about images (Visual Question Answering)
# - Generate descriptions of images (Image Captioning)
# - Extract text from images (OCR)
#
# We'll use pre-trained models so you can focus on understanding how they work. Feel free to experiment with your own images and try prompts in **Darija** or **Arabic**!
#
# **Time**: ~45 minutes

# %% [markdown]
# ## 🔧 Setup
# First, let's install the required packages and set up our environment.

# %%
!pip install transformers -q
!pip install pyav yt-dlp qwen-vl-utils -q

# %%
import gc
import time
import torch
from PIL import Image
import requests
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    BlipProcessor, 
    BlipForConditionalGeneration
)

def clear_memory():
    """Free up GPU memory after using each model"""
    if "inputs" in globals(): del globals()["inputs"]
    if "model" in globals(): del globals()["model"]
    if "processor" in globals(): del globals()["processor"]
    
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory cleared: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")

# %% [markdown]
# ## 1. Visual Question Answering (VQA)
# VQA lets you ask questions about images and get answers in natural language.
#
# For example:
# - "What food is shown in this image?"
# - "How many people are in the photo?"
# - "What color is the car?"
#
# We'll use Qwen2-VL, a powerful multilingual VLM that can understand both English and Arabic!

# %%
# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# %%
# Let's try it with a Moroccan tajine image!
url = "https://legarconboucher.com/img/cms/Recette/tajine-maroc.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Try these questions in English or Darija:
# - "What food is this?"
# - "What ingredients can you see?"
# - "Is this a traditional Moroccan dish?"
text_query = "What food is this?"

# Prepare the model input
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text_query},
        ],
    }
]

# Get the model's response
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
output_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Model's response:", output_text)
image

# %% [markdown]
# ## 2. Image Captioning
# Image captioning generates a natural language description of an image. Unlike VQA, it doesn't need a specific question - it just describes what it sees!
#
# We'll use BLIP, a model specifically trained for image captioning.

# %%
# Load the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    torch_dtype=torch.float16
).to("cuda")

# %%
# Let's try it with a sample image
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Generate a caption
inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Image caption:", caption)
raw_image

# %% [markdown]
# ## 3. Optical Character Recognition (OCR)
# OCR helps extract text from images. This is useful for:
# - Reading text from photos
# - Digitizing documents
# - Extracting information from receipts or ID cards
#
# We'll use Qwen2-VL-OCR, which is great at reading text in images.

# %%
# Load the model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

# %%
# Let's try it with a sample invoice
url = "https://trulysmall.com/wp-content/uploads/2023/04/Simple-Invoice-Template.png"
image = Image.open(requests.get(url, stream=True).raw)

# Try these questions:
# - "What is the invoice number?"
# - "What is the total amount?"
# - "What is the date?"
text_query = "What is the invoice number?"

# Get the model's response
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_query},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
output_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Model's response:", output_text)
image

# %% [markdown]
# ## 🎉 Congratulations!
# You've learned how to use Vision-Language Models for three important tasks:
# 1. Visual Question Answering (VQA)
# 2. Image Captioning
# 3. Optical Character Recognition (OCR)
#
# ### 🤔 What's Next?
# - Try the models with your own images
# - Experiment with prompts in Darija or Arabic
# - Think about how these models could help solve real-world problems in Morocco
#
# Share your results and ideas with the mentors!

# %%
# Clean up GPU memory
clear_memory()
