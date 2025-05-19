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
# # Fine-tuning Al-Atlas with LoRA for Moroccan Darija
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/train_sft_lora_alatlas_darijasftdataset.ipynb)
# 
# In this notebook, we'll learn how to fine-tune Al-Atlas, a Moroccan Darija language model, using LoRA (Low-Rank Adaptation). This is a memory-efficient way to adapt large language models to specific tasks.
# 
# ## What you'll learn:
# - What is LoRA and why we use it
# - How to fine-tune a language model on Moroccan Darija conversations
# - How to test the model before and after fine-tuning
# 
# ## Quick Concepts:
# - **LoRA**: A technique that makes fine-tuning large models more efficient by only updating a small number of parameters
# - **Fine-tuning**: Adapting a pre-trained model to a specific task or style
# - **Moroccan Darija**: The Moroccan Arabic dialect we're working with

# %% [markdown]
# ## What is LoRA?
# 
# When fine-tuning large language models, we need to update many parameters (weights). LoRA makes this more efficient by:
# 
# 1. Breaking down the weight updates into smaller matrices
# 2. Only updating these smaller matrices during training
# 3. This saves memory and makes training faster
# 
# ![image](https://i.postimg.cc/7LtmYJ1H/lora1.png)
# 
# Instead of updating all weights (left), we only update a small number of parameters (right).

# %% [markdown]
# ## Setup
# 
# First, let's install the required packages:

# %%
! pip install -q datasets trl transformers peft

# %% [markdown]
# ## Load Dataset
# 
# We'll use the Darija SFT Dataset, which contains conversations in Moroccan Darija:

# %%
from datasets import load_dataset
from huggingface_hub import login

# Login to Hugging Face (you'll need to get your token from huggingface.co)
login()  # You'll be prompted to enter your token

# Load the dataset
dataset = load_dataset("HackAI-2025/Darija_SFT_Dataset", split="train")
print("Dataset loaded:", dataset)

# %% [markdown]
# Let's look at an example conversation:

# %%
from pprint import pprint
pprint(dataset["conversation"][0])

# %% [markdown]
# ## Load Model
# 
# We'll use Al-Atlas, a 0.5B parameter model trained on Moroccan Darija:

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Select GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "atlasia/Al-Atlas-0.5B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# %% [markdown]
# ## Test Model Before Fine-tuning
# 
# Let's see how the model responds before we fine-tune it:

# %%
prompt = "السلام لباس؟"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
ids = tokenizer(formatted_prompt, return_tensors="pt").to(device)
output_ids = model.generate(**ids, max_new_tokens=120)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)

# %% [markdown]
# ## Setup LoRA Configuration
# 
# We'll configure LoRA to only update the key and value projection layers:

# %%
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,  # Rank of the update matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.05,  # Dropout probability
    target_modules=["q_proj", "v_proj"],  # Only update these layers
    bias="none",
)

# %% [markdown]
# ## Setup Training Arguments
# 
# We'll use gradient accumulation to handle larger effective batch sizes:

# %%
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="alatlas_instruct_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    num_train_epochs=4,
    bf16=True,
    save_total_limit=2,
    save_steps=100,
    logging_steps=10,
)

# %% [markdown]
# ## Start Training
# 
# Now we'll start the fine-tuning process. This might take a while, so we've provided a pre-trained checkpoint you can use instead.
# 
# To use the pre-trained checkpoint, skip the training cell and load the model from:
# ```
# model_id = "abdeljalilELmajjodi/alatlas-sft-lora-gra"
# ```

# %%
from trl import SFTTrainer

# Prepare dataset
dataset = dataset.select_columns("conversation").rename_column("conversation", "messages")

# Initialize trainer
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=args
)

# Start training
sft_trainer.train()

# %% [markdown]
# ## Test Model After Fine-tuning
# 
# Let's see how the model responds after fine-tuning:

# %%
prompt = "السلام لباس"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
ids = tokenizer(formatted_prompt, return_tensors="pt").to(device)
output_ids = model.generate(**ids, max_new_tokens=100, repetition_penalty=1.2)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)

# %% [markdown]
# ## Save and Share Your Model
# 
# You can save your fine-tuned model to Hugging Face Hub:

# %%
# Uncomment to save your model
# sft_trainer.push_to_hub("your-username/your-model-name")
