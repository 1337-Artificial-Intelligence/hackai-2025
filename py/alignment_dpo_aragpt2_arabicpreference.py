# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: hackai
#     language: python
#     name: python3
# ---
# %% [markdown]

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/alignment_dpo_aragpt2_arabicpreference.ipynb)

# %% [markdown]
# # **Direct Preference Optimization (DPO) Using Hugging Face**
#
# Estimated time needed: **1** hour
#
# ## What is DPO?
#
# Direct Preference Optimization (DPO) is a technique that helps make AI language models better at giving helpful and safe responses. It's like teaching a student by showing them examples of good and bad answers.
#
# ### How DPO Works (Simple Explanation)
#
# 1. We show the model two answers for the same question:
#    - A good answer (chosen by humans)
#    - A less good answer (rejected by humans)
# 2. The model learns to prefer the good answers over time
# 3. No complex reward system needed - it learns directly from examples!
#
# Think of it like training a dog:
# - Show it two actions (sit nicely vs jump on people)
# - Reward it for the good action
# - Repeat until it consistently chooses the good action
#
# ## DPO vs Traditional Methods
#
# | Method | How it Works | Complexity |
# |:------|:------------|:-----------|
# | DPO | Learns directly from good/bad examples | Simple |
# | Traditional RLHF | Needs a separate reward model first | Complex |
#
# ![image](https://cdn.labellerr.com/1%201%201%20DPO/dpo-ppo-diagram.webp)
#
# ## Lab Objective
#
# In this lab, you will:
# 1. Prepare a dataset of good and bad answers
# 2. Fine-tune an Arabic language model using DPO
# 3. See how the model improves after training
#
# %% [markdown]
# ### Setup and Installation
#
# First, let's install the required libraries:

# %%
!pip install --q torch==2.3.1 trl==0.11.4 peft==0.14.0 pandas numpy==1.26.0 datasets==3.2.0 transformers==4.45.2

# %% [markdown]
# Now, let's import the necessary libraries:

# %%
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    GenerationConfig
)
from trl import DPOConfig, DPOTrainer

# %% [markdown]
# ### Model Setup
#
# We'll use AraGPT2, an Arabic language model based on GPT-2. This model is smaller and faster to train, perfect for our 1-hour lab.

# %%
# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model selection
MODEL_NAME = "aubmindlab/aragpt2-base"
FINETUNED_MODEL_NAME = "aragpt2-base-dpo"

# %% [markdown]
# ### Data Preparation
#
# We'll use a dataset of Arabic text pairs, where each pair contains:
# - A question
# - A good answer (chosen by humans)
# - A less good answer (rejected by humans)

# %%
# Load the dataset (we use only 10% to keep training time reasonable)
print("Loading preference dataset...")
ds = load_dataset("FreedomIntelligence/Arabic-preference-data-RLHF", split="train[:10%]")

# Look at an example
print("\nExample from dataset:")
print(ds[0])

# %% [markdown]
# ### Prepare Data for Training
#
# We need to format our data for DPO training:

# %%
# Format data for DPO
print("\nPreparing dataset for DPO training...")
ds = ds.rename_column("instruction", "prompt").remove_columns(["id"])

# Split into train and test sets
ds = ds.train_test_split(0.1, shuffle=True, seed=42)
train_dataset, eval_dataset = ds["train"], ds["test"]
print(f"Training set size: {len(train_dataset)}, Evaluation set size: {len(eval_dataset)}")

# %% [markdown]
# ### Training Setup
#
# We'll use LoRA (Low-Rank Adaptation) to make training faster and more efficient:

# %%
# LoRA configuration
peft_config = LoraConfig(
    r=4,                    # Rank of the low-rank decomposition
    target_modules=[        # Which parts of the model to train
        'c_proj',           # Projection layers
        'c_attn'            # Attention layers
    ],
    task_type="CAUSAL_LM",  # Type of task
    lora_alpha=8,           # Scaling factor
    lora_dropout=0.1,       # Dropout for regularization
    bias="none",           # Don't train bias parameters
)

# DPO training configuration
training_args = DPOConfig(
    beta=0.1,                      # How strongly to prefer good answers
    output_dir="dpo",              # Where to save the model
    num_train_epochs=5,            # Number of training passes
    per_device_train_batch_size=2, # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    remove_unused_columns=False,   # Keep all columns
    logging_steps=10,              # Log progress every 10 steps
    gradient_accumulation_steps=4, # Accumulate gradients
    learning_rate=1e-4,            # Learning rate
    evaluation_strategy="epoch",   # Evaluate after each epoch
    warmup_steps=2,                # Warmup steps
    save_steps=500,                # Save checkpoint every 500 steps
    report_to='none'              # Don't report to external services
)

# %% [markdown]
# ### Training Process
#
# **Note**: Training can take a while. You can skip to the next section to use a pre-trained model.

# %%
# Create trainer
print("Setting up DPO trainer...")
trainer = DPOTrainer(
    model=model,              # Model to train
    ref_model=None,           # Reference model (handled automatically with LoRA)
    args=training_args,       # Training arguments
    train_dataset=train_dataset,  # Training data
    eval_dataset=eval_dataset,    # Evaluation data
    tokenizer=tokenizer,          # Tokenizer
    peft_config=peft_config,      # LoRA configuration
    max_length=512,               # Maximum sequence length
)

# Start training
print("Starting DPO training...")
trainer.train()

# %% [markdown]
# ### Using a Pre-trained Model
#
# If you skipped training, you can load a pre-trained model:

# %%
# Load pre-trained model
print("Loading pre-trained DPO model...")
dpo_model = AutoModelForCausalLM.from_pretrained(f"HackAI-2025/{FINETUNED_MODEL_NAME}").to(device)
tokenizer = AutoTokenizer.from_pretrained(f"HackAI-2025/{FINETUNED_MODEL_NAME}")

# Load baseline model for comparison
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# %% [markdown]
# ### Testing the Model
#
# Let's see how our model performs compared to the baseline:

# %%
# Set up generation parameters
generation_config = GenerationConfig(
    max_new_tokens=70,         # Maximum length of response
    do_sample=True,            # Use sampling
    top_k=50,                  # Consider top 50 tokens
    top_p=0.8,                 # Consider tokens with 80% probability mass
    temperature=0.8,           # Control randomness
    repetition_penalty=1.2,    # Avoid repetition
    pad_token_id=tokenizer.eos_token_id
)

# Test prompt
PROMPT = "كيف يمكنني التغلب على القلق والتوتر؟"

# Generate responses
inputs = tokenizer(PROMPT, return_tensors='pt').to(device)

print("Generating response with DPO model...")
outputs = dpo_model.generate(**inputs, generation_config=generation_config)
print("DPO response:\t", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\nGenerating response with baseline model...")
outputs = model_ref.generate(**inputs, generation_config=generation_config)
print("Baseline response:\t", tokenizer.decode(outputs[0], skip_special_tokens=True))

# %% [markdown]
# ## Exercises
#
# Try these exercises to better understand DPO:
#
# 1. **Experiment with Generation Parameters**
#    - Try different values for temperature, top_p, and top_k
#    - How do they affect the responses?
#
# 2. **Test Different Prompts**
#    - Try these Arabic prompts:
#    ```python
#    test_questions = [
#        "ما هي فوائد الغذاء الصحي؟",
#        "كيف يمكنني التغلب على القلق والتوتر؟",
#        "اشرح لي كيفية استخدام الذكاء الاصطناعي في التعليم.",
#        "ما هي أفضل طريقة لتعلم لغة جديدة؟",
#        "هل يجب علي الاستثمار في العملات المشفرة؟"
#    ]
#    ```
#
# 3. **Compare Responses**
#    - How do the DPO model's responses differ from the baseline?
#    - What makes the DPO responses better or worse?
