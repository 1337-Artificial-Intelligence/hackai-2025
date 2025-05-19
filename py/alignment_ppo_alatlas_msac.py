# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/alignment_ppo_alatlas_msac.ipynb)

# %% [markdown]
# # Training a Happy/Positive LLM with PPO
# Estimated time needed: **1** hour on a free T4 (Google Colab)
#
# ## Learning Objectives
# By the end of this notebook, you will be able to:
# - Understand the basics of Reinforcement Learning (RL) and how it applies to language models
# - Learn about Proximal Policy Optimization (PPO) and its role in training LLMs
# - Fine-tune a language model to generate more positive responses
# - Evaluate the impact of RL training on model outputs
#
# ## What is Reinforcement Learning (RL)?
# RL is like teaching a model through trial and error. Instead of giving it exact instructions, we let it learn from feedback:
# - The model tries something (generates text)
# - It gets feedback (positive/negative score)
# - It learns to do better next time
#
# In our case:
# - **Model** = Al-Atlas (a Moroccan Darija language model)
# - **Action** = Generating text
# - **Reward** = How positive the text is
#
# <img src='https://superagi.com/wp-content/uploads/2024/03/Untitled-2.png.webp' width='600'>
#
# ## What is PPO?
# PPO (Proximal Policy Optimization) is a way to train models that:
# - Makes small, careful updates
# - Prevents the model from changing too much at once
# - Helps maintain stable learning
#
# ## How We'll Train Our Model
# 1. Start with Al-Atlas (a Moroccan Darija model)
# 2. Use a sentiment classifier to score responses
# 3. Train the model to generate more positive text
# 4. Compare before/after results
#
# <img src='https://superagi.com/wp-content/uploads/2024/03/Untitled-3.png.webp' width='600'>

# %% [markdown]
# ## Setup
# First, let's install the required packages:

# %%
!pip install --quiet transformers trl==0.11 wandb

# %% [markdown]
# ## Import Libraries
# We'll use these libraries to:
# - `transformers`: Load and work with language models
# - `trl`: Train models with reinforcement learning
# - `torch`: Deep learning framework
# - `datasets`: Handle our training data

# %%
import os
import torch
from tqdm import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# %% [markdown]
# ## Load Models
# We'll use:
# - Al-Atlas: A Moroccan Darija language model
# - A sentiment classifier to score responses

# %%
# Model configuration
MODEL = "atlasia/Al-Atlas-0.5B"  # Our base model
DATASET_NAME = "AbderrahmanSkiredj1/MSAC_darija_sentiment_analysis"  # Training data
REWARD_MODEL = "Davlan/afrisenti-twitter-sentiment-afroxlmr-large"  # For scoring responses

# Setup device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Load models
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL, torch_dtype=dtype)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL, torch_dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load sentiment classifier
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
sentiment_pipe = pipeline(
    "sentiment-analysis", 
    model=REWARD_MODEL, 
    device=device,
    torch_dtype=dtype,
    **sent_kwargs
)
print("Sentiment classes:", sentiment_pipe.model.config.id2label)

# %% [markdown]
# ## Prepare Training Data
# We'll use the Moroccan Sentiment Analysis Corpus (MSAC) dataset, which contains tweets in Moroccan Darija with sentiment labels.

# %%
def build_dataset(
    dataset_name=DATASET_NAME,
    input_min_text_length=4,
    input_max_text_length=12,
    tokenizer=tokenizer
):
    """Prepare dataset for training"""
    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(lambda x: {"label": 1 if x["label"] == "pos" else 0})
    ds = ds.rename_columns({"text": "review"})
    ds = ds.shuffle(seed=42)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# Build dataset
dataset = build_dataset()

# %% [markdown]
# ## Initialize PPO Trainer
# This will handle our reinforcement learning training:

# %%
config = PPOConfig(
    model_name=MODEL,
    learning_rate=1.41e-5,
    log_with="wandb",
    batch_size=32,
    mini_batch_size=32,
)

ppo_trainer = PPOTrainer(
    config, 
    model, 
    ref_model, 
    tokenizer, 
    dataset=dataset,
    data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0])
)

# %% [markdown]
# ## Training Loop
# Now we'll train our model to generate more positive responses. This will take about 20 minutes.

# %%
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Generate responses
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze().to(device)
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Score responses
    pipe_outputs = sentiment_pipe(batch["response"])
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "positive"
    ]
    rewards = [torch.tensor(score) for score in positive_scores]

    # Update model
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# %% [markdown]
# ## Evaluate Results
# Let's compare the model's responses before and after training:

# %%
# Test the model
test_prompts = [
    "كيف داير الجو اليوم؟",                      # How's the weather today?
    "شنو رأيك فالاثنين مع الصباح؟",             # What do you think about Monday mornings?
    "شرح ليا شنو هي قاعدة البيانات.",           # Explain what a database is.
    "شنو الدور ديال المعلم فالمدرسة؟",           # What is the role of a teacher?
    "كيفاش كتكون خدمة ديال المكتب؟",            # What is a typical office job like?
]

generation_kwargs = {
    "min_length": 10,
    "max_length": 20,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "temperature": 0.8,
    "pad_token_id": tokenizer.eos_token_id,
}

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    text_tokenized = tokenizer.encode(prompt, return_tensors="pt").to(device)
    response = tokenizer.decode(model.generate(text_tokenized, **generation_kwargs).squeeze())
    print(f"Response: {response}")
    
    # Get sentiment score
    sentiment = sentiment_pipe(response)
    print(f"Sentiment: {sentiment}")

# %% [markdown]
# ## Exercise: Can You Spot the Positivity Bias?
# 🎯 **Your Task:**
# 1. Try different prompts in Moroccan Darija
# 2. Compare the responses with the original model
# 3. Notice how the trained model tends to be more positive
#
# 💡 **Tips:**
# - Try neutral topics
# - Ask about everyday situations
# - Compare the emotional tone of responses
#
# 🏆 **Challenge:**
# Can you find a prompt where the model's positivity might be inappropriate or excessive?

# %% [markdown]
# ## Next Steps
# - Try different reward models
# - Experiment with different training parameters
# - Explore other alignment techniques
#
# Remember: The goal is to make AI helpful and positive, but not at the expense of accuracy or appropriateness!