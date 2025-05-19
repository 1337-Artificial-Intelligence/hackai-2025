# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/embedding_mlm_train_bert_goudma.ipynb)

# %% [markdown]
# # Training a BERT Model for Moroccan Arabic (Darija)
# 
# In this notebook, we'll learn how to train a BERT model specifically for Moroccan Arabic (Darija). This is important because:
# - Pre-trained models often don't work well with Darija
# - We can create a model that better understands our local language
# - It's a great way to learn about how language models work

# %% [markdown]
# ## What is Masked Language Modeling (MLM)?
# 
# MLM is like a fill-in-the-blank game for computers:
# - We hide some words in a sentence
# - The model tries to guess what those hidden words are
# - This helps the model learn how words relate to each other
# 
# Example:
# - Original: "I love eating couscous on Fridays"
# - Masked: "I [MASK] eating couscous on [MASK]"
# - Model predicts: "love" and "Fridays"

# %% [markdown]
# ## Why BERT?
# 
# BERT is a powerful language model that:
# - Can understand context from both directions (left and right)
# - Works well for many languages
# - Can be trained on our own data
# 
# TODO: Add image showing BERT's bidirectional attention

# %% [markdown]
# ## Setup
# First, let's install the required packages

# %%
!pip install -q datasets transformers huggingface_hub wandb

# %% [markdown]
# ## 1. Load Our Dataset
# We'll use a dataset of Moroccan Arabic text

# %%
from datasets import load_dataset
from huggingface_hub import login

# Login to Hugging Face (you'll need to create an account)
login()

# Load the dataset
dataset = load_dataset("atlasia/good25")
dataset = dataset["train"]

# %% [markdown]
# ## 2. Prepare Our Data
# We need to:
# 1. Select only the text content
# 2. Split into train/test sets
# 3. Tokenize the text

# %%
# Select only the content column
dataset = dataset.select_columns(["content"])

# Split into train/test
dataset_splited = dataset.train_test_split(test_size=0.1)

# %% [markdown]
# ## 3. Load Our Model
# We'll use a pre-trained model and continue training it on our data

# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the tokenizer and model
model_id = "atlasia/XLM-RoBERTa-Morocco"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

# %% [markdown]
# ## 4. Process Our Data
# We need to:
# 1. Tokenize our text
# 2. Prepare it for training

# %%
def ds_tokenizer(examples):
    return tokenizer(examples["content"])

# Tokenize our datasets
train_tokenized = dataset_splited["train"].map(ds_tokenizer).remove_columns(dataset_splited["train"].column_names)
eval_tokenized = dataset_splited["test"].map(ds_tokenizer).remove_columns(dataset_splited["test"].column_names)

# %% [markdown]
# ## 5. Prepare for Training
# We need to:
# 1. Split text into chunks
# 2. Set up our training configuration

# %%
# Split text into chunks
context_length = 256
def concatenate_splite(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    if total_length >= context_length:
        total_length = (total_length // context_length) * context_length
    result = {"input_ids": [], "attention_mask": []}
    for k, v in concatenated_examples.items():
        for i in range(0, len(v), context_length):
            result[k].append(v[i:i+context_length])
    return result

# Apply the splitting
train_ds = train_tokenized.map(concatenate_splite, batched=True)
eval_ds = eval_tokenized.map(concatenate_splite, batched=True)

# Set up data collator for MLM
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.2  # Mask 20% of words
)

# %% [markdown]
# ## 6. Train Our Model
# Now we'll set up and start training

# %%
from transformers import TrainingArguments, Trainer

# Set up training arguments
args = TrainingArguments(
    "model",
    learning_rate=2e-5,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_total_limit=2,
    weight_decay=0.01,
    eval_steps=1000,
    logging_steps=1000,
    warmup_ratio=0.03,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    report_to="wandb",
    run_name="Bert CPT"
)

# Create trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args
)

# %% [markdown]
# ## 7. Start Training
# This might take a while. For the workshop, we'll use a pre-trained checkpoint.

# %%
# Uncomment to train (takes about 1 hour)
# trainer.train()

# %% [markdown]
# ## 8. Save Our Model
# Once training is complete, we can save our model to Hugging Face Hub

# %%
# Uncomment to save (requires Hugging Face account)
# trainer.push_to_hub("your-username/your-model-name")

# %% [markdown]
# ## What's Next?
# - Try using your model for different tasks
# - Experiment with different training parameters
# - Share your model with the community
# 
# TODO: Add image showing model usage examples
