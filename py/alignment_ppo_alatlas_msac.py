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

# %% [markdown] id="5vt97KjXg6p-"
# ### Training a Happy/Positive LLM 
# Estimated time needed: **1** hour on a free T4 (Google Colab)
#
#
# Imagine you're an AI engineer building LLM that is super cheerful ("Happy LLM") 
#
# You don't tell them exactly what to say. Instead, you let them **learn by trial and error** — this is **Reinforcement Learning (RL)**.  
# The LLM acts (outputs text), a **reward model** scores it (positive/negative sentiment), and the LLM improves over time.
#
# #### What is Reinforcement Learning (RL)?
#
# Reinforcement Learning is a branch of machine learning where agents learn by interacting with an environment and receiving feedback in the form of rewards or penalties.  
# Unlike supervised learning (labeled examples), RL relies on **exploration** and **learning from consequences**.
#
# In this setup:  
# - **Agent** = the LLM (Large Language Model)  
# - **Environment** = the text generation task  
# - **Action** = the generated text  
# - **Reward** = score from a sentiment classifier
#
# <img src='https://superagi.com/wp-content/uploads/2024/03/Untitled-2.png.webp' width='600'>
#
#
# #### What is PPO?
#
# **Proximal Policy Optimization (PPO)** is an RL algorithm created by OpenAI that allows stable, efficient policy updates.  
# It keeps updates **gentle** (no big jumps) to avoid breaking the learning process.
#
# #### How the Reward Model Works?
#
# You use a **sentiment classifier** (trained on the IMDb movie review dataset) to score generated text:  
# - Positive text → big reward for Happy LLM!  
#
# In other words, the classifier **judges** the LLM outputs and converts sentiment into a **numerical reward**.
#
#
#
# #### PPO Training Steps
#
# 1. **Collect Rollouts:**  
#    Let the model generate text, record states, actions, rewards.
#
# 2. **Compute Advantages:**  
#    How much better was an action compared to expected?
#
# 3. **Policy Update:**  
#    Use loss to gently improve policy.
#
# 4. **Value Update:**  
#    Improve the model's predictions of expected rewards.
#
# 5. **Entropy Regularization:**  
#    Encourage exploration by rewarding randomness.
#
# 6. **Repeat:**  
#    Across mini-batches and epochs.
#    
# <img src='https://superagi.com/wp-content/uploads/2024/03/Untitled-3.png.webp' width='600'>
#
#
# #### In This Lab
#
# You will fine-tune  Al-Atlas-0.5B to generate **positive things** using PPO, following the Hugging Face example
#

# %% [markdown] id="nVK2hzJVg6qB"
# ### Setup experiment

# %% [markdown] id="pR1kA-rEg6qB"
# - Intall dependencies

# %% id="yVvcFP-Pg6qB"
# %load_ext autoreload
# %autoreload 2

# %% colab={"base_uri": "https://localhost:8080/"} id="H4GEjiXmg6qC" outputId="c9e1be12-1c88-4feb-c4a2-ee30ee4e4226"
# %pip install --q transformers trl==0.11 wandb

# %% [markdown]
# - Packages

# %% id="gedtPRveg6qC"
import os
import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# %% [markdown] id="ETgRzUPXg6qC"
# - Configuration

# %% colab={"base_uri": "https://localhost:8080/"} id="h24cn78rg6qD" outputId="91bd9b22-7762-4929-9f67-0c1549ec0511"
MODEL = "atlasia/Al-Atlas-0.5B" # Model to finetune and also its own reference and tokenizer
DATASET_NAME = "AbderrahmanSkiredj1/MSAC_darija_sentiment_analysis" # Dataset to finetune on
REWARD_MODEL = "Davlan/afrisenti-twitter-sentiment-afroxlmr-large" # Reward model to use

# %%
device = "cuda" if torch.cuda.is_available() else "cpu" # set device to cuda if available
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32 # set dtype to fp16 if cuda is available

# Set the huggingface token
os.environ["HF_TOKEN"] = "YOUR_API_KEY" #

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown] id="zo8YjlAFg6qD"
# ### Load data and models

# %% [markdown] id="RS48mTNGiVRX"
# - Load pre-trained [Atlas AI 0.5 B model](https://huggingface.co/atlasia/Al-Atlas-0.5B)

# %% [markdown] id="kgkgPe1piVRY"
# **Al-Atlas** is a 0.5B parameter language model specifically trained on **Moroccan Darija**, making it the first dedicated foundation model for Morocco's primary spoken dialect. The model was finetuned from **Qwen-2.5** and trained on a carefully curated dataset of **155M tokens**, focusing exclusively on authentic Moroccan Darija content.
#
# We load the model with a value head and the tokenizer. 
# We load the model twice; the first model is optimized while the second model serves as **a reference** to calculate the KL-divergence from the starting point. This serves as an additional reward signal in the PPO training to make sure the optimized model does not deviate too much from the original language model.

# %% colab={"base_uri": "https://localhost:8080/"} id="mYVlDxdKD4pz" outputId="6164c417-6672-44a3-f5d1-7824adffcd98"
# Model/Reference Model
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL,torch_dtype=dtype)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL,torch_dtype=dtype)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# - Load pre-trained Reward Model afrisenti-twitter-sentiment-afroxlmr-large

# %% [markdown]
# afrisenti-twitter-sentiment-afroxlmr-large is a multilingual twitter sentiment classification model for twelve  languages including Moroccan Darija based on a fine-tuned castorini/afriberta_large large model.
# The model has been trained to classify tweets into 3 sentiment classes: negative, neutral and positive Specifically, this model is a Davlan/afro-xlmr-large model that was fine-tuned on an aggregation of 12 African language datasets obtained from AfriSenti dataset.

# %%
# Load reward model in sentiment analysis pipeline
# This configures your sentiment pipeline run in batches of 16, return raw logits for all sentiment classes and Skip applying softmax
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size":16 }
sentiment_pipe = pipeline(
    "sentiment-analysis", model=REWARD_MODEL, device=device,torch_dtype=dtype,
      **sent_kwargs
)
print("classes labels: ",sentiment_pipe.model.config.id2label)


# %% [markdown] id="qXxMP27hg6qD"
# ### Load [MSAC](https://huggingface.co/datasets/AbderrahmanSkiredj1/MSAC_darija_sentiment_analysis) dataset

# %% [markdown]
# The Moroccan Sentiment Analysis Corpus is a dataset composed of 2,000 tweets written in Maghrebi Arabic (Darija), specifically Moroccan dialect, collected from Twitter. Each entry in the corpus is typically annotated with a sentiment label (e.g., pos(for positive), neg(for negative), neu (neutral)), making it suitable for training and evaluating sentiment analysis models tailored to the unique linguistic characteristics of Moroccan Arabic.

# %% [markdown]
# - Dataset

# %% id="ATvufSKAg6qD"
def map_labels(sample):
    """ map the labels to 0 and 1 """
    label = sample["label"]
    sample["label"] = 1 if label == "pos" else 0
    return sample


def build_dataset(
    dataset_name=DATASET_NAME,
    input_min_text_length=4,
    input_max_text_length=12,
    tokenizer = tokenizer
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.
    """
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.map(map_labels)
    ds = ds.rename_columns({"text": "review"})
    ds = ds.shuffle(seed=42)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# %% [markdown]
# Using a ```LengthSampler``` to sample different text lengths during data processing introduces variability, making the model more robust and capable of handling varying input lengths in real-world scenarios. This approach prevents overfitting by exposing the model to diverse input sizes, improving generalization to new data. It also ensures efficient training by managing the length of text inputs, maintaining practicality and performance.

# %% id="RnFdHDsEg6qE"
# build the dataset
dataset = build_dataset()


# %% [markdown]
# - Collator
#
# The collator function is crucial for preparing data batches in a format suitable for the PPOTrainer. It ensures that each feature from the data samples is grouped together
#

# %% colab={"base_uri": "https://localhost:8080/"} id="xKeSxCvtACBb" outputId="d8be5ebb-e575-4b58-a7ba-a6e0f24e8be4"
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# %% [markdown]
# ##### Test the reward model performance

# %%
# positive text
text = "طابعان راه مكتاءب!"
sentiment_pipe(text)

# %%
# negative text
text = "طابعان راه فرحان!"
sentiment_pipe(text)

# %% [markdown] id="8Jy5i3P7g6qE"
# ### Initialize PPOTrainer
# The `PPOTrainer` takes care of device placement and optimization later on:
#
# - ```config``` : Configuration settings for PPO training, such as learning rate and model name
# - ```model``` : The primary model to be fine-tuned using PPO
# - ```ref_model``` : The reference model to compare with model
# - ```tokenizer```:Tokenizer corresponding to the model, used for processing input text
# - ```dataset```:  Dataset to be used for training, providing the input data for the model
# - ```data_collator```: Data collator to handle batching and formatting of the input data
#

# %%
config = PPOConfig(
    model_name=MODEL, # the model name to be trained
    learning_rate=1.41e-5, # the learning rate for the optimizer
    log_with="wandb",   # the logging method to be used
    batch_size=32,  # the batch size for training
    mini_batch_size=32,    # the mini batch size for PPO

)

# %% colab={"base_uri": "https://localhost:8080/", "height": 228} id="sXG7BLg1g6qE" outputId="44e587d2-6cbd-4d59-d418-65ab37d073e3"
ppo_trainer = PPOTrainer(
    config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
)

# %% [markdown] id="6O7Mj1OCg6qF"
# ### Generation settings
# ```generation_kwargs``` defines generation parameters used when calling a language model (like a LLM) for text generation. The c configuration below generates fully sampled, unconstrained output — no top-k or top-p restrictions, and with maximum diversity/randomness. It's good for creative generation, but can produce less coherent or less controlled results. (https://huggingface.co/docs/transformers/main_classes/text_generation)

# %% id="dvGbm7skg6qF"
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# %% [markdown] id="6NCBOuhrg6qF"
# ### Optimize model

# %% [markdown] id="WxKrNG7Kg6qF"
# ### Training loop

# %% [markdown] id="lZSwdAr5g6qF"
# The training loop consists of the following main steps:
# 1. Get the query responses from the policy network (Al-Atlas-0.5B)
# 2. Get sentiments for query/responses from afrisenti-twitter-sentiment-afroxlmr-large
# 3. Optimize policy with PPO using the (query, response, reward) triplet
#
# **Training time**
#
# This step takes **~20mins** on a RTX 3070 i with the above specified settings.

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} id="AUp9BvDsg6qG" outputId="b621ff97-64fa-4e7c-f097-9829083e85f7"
output_min_length = 4
output_max_length = 16
# same objective as the input length 
output_length_sampler = LengthSampler(output_min_length, output_max_length)


for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from gpt2
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze().to(device)
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    pipe_outputs = sentiment_pipe(batch["response"])
    positive_scores = [
        item["score"]
        for output in pipe_outputs
        for item in output
        if item["label"] == "positive"
    ]
    rewards = [torch.tensor(score) for score in positive_scores]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

# %% [markdown] id="IC4mbfbig6qG"
# ## Model inspection
# Let's inspect some examples from the IMDB dataset. We can use `ref_model` to compare the tuned model `model` against the model before optimisation.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="A3l1LqL5g6qG" outputId="55935569-be9d-40b6-d0b7-d2ae15d87c39"
#### get a batch from the dataset
bs = 20

output_min_length = 10
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data["query"] = df_batch["query"].tolist()
game_data["label"] = df_batch["label"].tolist()

game_data["review"] = df_batch["review"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors_ref, response_tensors = [], []

#### get response from gpt2 and gpt2_ref
for i in range(bs):
    query = torch.tensor(query_tensors[i]).to(device)

    gen_len = output_length_sampler()
    query_response = ref_model.generate(
        query.unsqueeze(0), **generation_kwargs
    ).squeeze()
    response_len = len(query_response) - len(query)
    response_tensors_ref.append(query_response[-response_len:])

    query_response = model.generate(
        query.unsqueeze(0), max_new_tokens=gen_len, **generation_kwargs
    ).squeeze()
    response_len = len(query_response) - len(query)
    response_tensors.append(query_response[-response_len:])

#### decode responses
game_data["response (before)"] = [
    tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
]
game_data["response (after)"] = [
    tokenizer.decode(response_tensors[i]) for i in range(bs)
]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
pipe_outputs = sentiment_pipe(texts)
positive_scores = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "positive"
]
game_data["rewards (before)"] = positive_scores

texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
pipe_outputs = sentiment_pipe(texts)
positive_scores = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "positive"
]
game_data["rewards (after)"] = positive_scores

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results

# %% [markdown] id="Jd4NcLTAg6qG"
# Looking at the reward mean/median of the generated sequences we observe a significant difference.

# %% colab={"base_uri": "https://localhost:8080/", "height": 328} id="R_HtA0-Lg6qG" outputId="7a6d3cfa-ffbd-4d7b-f3df-1434e45484f3"
print("mean:")
display(df_results[["rewards (before)", "rewards (after)"]].mean())
print()
print("median:")
display(df_results[["rewards (before)", "rewards (after)"]].median())

# %%
generation_kwargs = {
    "min_length": 10,                  # Ensures a minimum number of generated tokens (e.g., 10)
    "max_length": 20,                # Sets a maximum length for generation to avoid endless outputs
    "top_k": 50,                      # Limits sampling to top 50 tokens (standard value for diversity)
    "top_p": 0.95,                    # Nucleus sampling, picks from top tokens whose cumulative prob ≥ 0.95
    "do_sample": True,               # Enables sampling (needed when using top_k/top_p)
    "temperature": 0.8,              # Controls randomness; <1 = more deterministic, >1 = more random
    "pad_token_id": tokenizer.eos_token_id,  # Ensures correct padding
}


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="Vaqd1J8U63Ri" outputId="2e85f8cc-2b37-4c2d-c0e8-54355e605452"
text="أنا"
text_tokenized = tokenizer.encode(text,return_tensors="pt").to(device)
tokenizer.decode(model.generate(text_tokenized, **generation_kwargs).squeeze())

# %%
text_tokenized = tokenizer.encode(text,return_tensors="pt").to(device)
tokenizer.decode(model.generate(text_tokenized, **generation_kwargs).squeeze())

# %% [markdown] id="Iizk05g9g6qG"
# ### Save model
# Finally, we save the model and push it to the Hugging Face for later usage.

# %% id="Aaz1xnTrg6qG"
NEW_MODEL_NAME = "Al-Atlas-0.5B-ppo-positive-thinking" # the model is available HackAI-2025/Al-Atlas-0.5B-ppo-positive-thinking for inference purposes
model.push_to_hubd(NEW_MODEL_NAME)
tokenizer.push_to_hub(NEW_MODEL_NAME)

# %% [markdown]
# ### Exercise Time : Can You Spot the Positivity Bias?

# %% [markdown]
# 🧠 Objective:
# You’ve fine-tuned an LLM using PPO with a reward model that favors positive and happy responses. In this exercise, your task is to probe the model’s bias using a fixed evaluation set of neutral prompts.
#
#

# %% [markdown]
#
# 📝Instructions:
#
# The fine-tuned LLM has been trained to prefer positive, cheerful, and helpful responses.
#
# Your mission:
#
# - Evaluate the model on a fixed list of neutral prompts.
#
# - Score each response using the provided reward model.
#
# - Submit your average positivity score to the leaderboard with the notebook (Please keep the cell outputs)
#
# ⚠️ No model training is required—just analyze and run the evaluation!
#
# 🏆 The leaderboard ranks submissions by average positivity score.

# %%
test_prompts = [
    "كيف داير الجو اليوم؟",                      # How's the weather today?
    "شنو رأيك فالاثنين مع الصباح؟",             # What do you think about Monday mornings?
    "شرح ليا شنو هي قاعدة البيانات.",           # Explain what a database is.
    "شنو الدور ديال المعلم فالمدرسة؟",           # What is the role of a teacher?
    "كيفاش كتكون خدمة ديال المكتب؟",            # What is a typical office job like?
   "كيفاش تقضي وقتك الفراغ؟",                   # How do you spend your free time?
    "شنو كيدير الإنسان ملي كيتزوج؟",         # What does someone do when they get married?
    "كيفاش تحب تقضي عطلتك؟" ,                    # How do you like to spend your holidays?
    "شنو كيدير الإنسان ملي كيتقاعد؟",         # What does someone do when they retire?
    "وصف ليا نهار ديال الشتاء."                 # Describe a rainy day.
]
