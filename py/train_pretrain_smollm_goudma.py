# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/train_pretrain_smollm_goudma.ipynb)

# %% [markdown]
# # Training Your First Language Model 🚀
# 
# In this notebook, you'll learn how to train a small language model from scratch! We'll use the SmolLM2 model, which is perfect for learning because it's small but powerful.

# %% [markdown]
# ## What is Language Model Training? 🤔
# 
# > A language model is like a student learning to read and write. It learns by:
# > 1. Reading lots of text
# > 2. Trying to predict the next word in a sentence
# > 3. Learning from its mistakes
# 
# TODO: Add image showing how a language model predicts next words
# 
# ```
# Example:
# Input: "The cat sat on the"
# Model predicts: "mat" (or "chair", "table", etc.)
# ```

# %% [markdown]
# ## Let's Get Started! 🛠️
# 
# First, we need to install some tools:

# %%
! pip install -U torch datasets transformers wandb -q

# %% [markdown]
# ## Check Your GPU 🎮
# 
# We need a GPU to train our model faster. Let's check if you have one:

# %%
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Model:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## Load Your Dataset 📚
# 
# We'll use a dataset from Hugging Face. Think of it as a big book for our model to learn from.

# %%
from huggingface_hub import login
login("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")

# %%
from datasets import load_dataset
dataset_name_id = "atlasia/good25"
ds = load_dataset(dataset_name_id, split="train")
ds = ds.select_columns(["content"])  # We only need the text content

# %% [markdown]
# ## Prepare the Text 📝
# 
# Before training, we need to:
# 1. Split our text into small pieces (tokens)
# 2. Make all pieces the same length
# 
# TODO: Add image showing tokenization process

# %%
from transformers import AutoTokenizer
model_id = "HuggingFaceTB/SmolLM2-138M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set how long each piece of text should be
context_length = 128

def tokenize(examples):
    results = tokenizer(
        examples["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True
    )
    input_batch = []
    for l, in_ids in zip(results["length"], results["input_ids"]):
        if l == context_length:
            input_batch.append(in_ids)
    return {"input_ids": input_batch}

# Split data into train and test
ds_spliter = ds.train_test_split(test_size=0.2, seed=42)
tokenized_ds = ds_spliter.map(tokenize, batched=True, remove_columns=ds_spliter["train"].column_names)

# %% [markdown]
# ## Set Up Training 🏋️‍♂️
# 
# Now we'll:
# 1. Create our model
# 2. Set up how it should learn
# 3. Start training!

# %%
from transformers import AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Create model
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_config(config)

# Set up training settings
args = TrainingArguments(
    output_dir="test_dir",
    num_train_epochs=2,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    learning_rate=5e-4,
    warmup_steps=100,
    lr_scheduler_type="linear",
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    logging_steps=2,
    push_to_hub=False,
    report_to="wandb",
)

# Set up data preparation
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
)

# %% [markdown]
# ## Start Training! 🚀
# 
# This might take a while. While it's training, you can:
# 1. Watch the loss go down (that's good!)
# 2. Learn about what's happening in the background
# 3. Think about how you could use this model

# %%
trainer.train()

# %% [markdown]
# ## What Did We Learn? 📚
# 
# In this notebook, you learned:
# 1. How to prepare text for a language model
# 2. How to set up and train a small language model
# 3. How to monitor the training process
# 
# TODO: Add image showing the training process and results
# 
# ## Next Steps 🎯
# 
# Want to learn more? Check out:
# 1. How to make your model follow instructions
# 2. How to make your model smaller and faster
# 3. How to use your trained model for cool projects
