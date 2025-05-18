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

# %% [markdown] id="2de0d41d-1eae-480c-94c1-8502848519b5"
# # **Direct Preference Optimization (DPO) Using Hugging Face**
#

# %% [markdown] id="d137eb01-2450-4bae-bbc5-dc0dab787418"
# Estimated time needed: **1** hour
#

# %% [markdown] id="b5b8a0c8-3941-4722-a66b-f34d97e0a2c1"
# # Direct Preference Optimization (DPO) for Language Model Alignment
#
# This notebook demonstrates how to align language models with human preferences using Direct Preference Optimization (DPO), a powerful technique that improves upon traditional Reinforcement Learning from Human Feedback (RLHF) methods.
#
# ## Lab Objective
#
# The goal of this lab is to give you practical experience with:
# - Preparing a dataset specially formatted for DPO,
# - Fine-tuning a model using the DPO method,
# - Evaluating how much the model's behavior improves after training
#
#
# ### How DPO Works (Simple Explanation)
#
# - You show the model two answers for the same question: one that humans prefer (the **chosen** one) and one that's less preferred (the **rejected** one).
# - The model learns **directly** from this comparison by adjusting itself to favor the "chosen" answers over the "rejected" ones.
# - It does this **without** needing a complex reward model like in traditional reinforcement learning.
#
# Think of it like training a dog: you show it two actions (e.g., sit nicely vs jump on people) and **reward** it for the one you like better, over and over, until it consistently chooses the good one.
#
#
# ## DPO vs PPO: What's the Difference?
#
# | Aspect | DPO (Direct Preference Optimization) | PPO (Proximal Policy Optimization) |
# |:------|:--------------------------------------|:----------------------------------|
# | How it works | Directly trains the model from comparisons (chosen vs rejected) | Needs a reward model first, then trains the model using rewards |
# | Complexity | Simpler (no reward model needed) | More complex (2 steps: train reward model + policy optimization) |
# | Stability | Very stable and efficient | Stable but more sensitive to hyperparameters |
# | Training Type | Preference-based fine-tuning | Reinforcement learning fine-tuning |
#
# ![image](https://cdn.labellerr.com/1%201%201%20DPO/dpo-ppo-diagram.webp)
#
# **In short:**
# - **DPO** is **easier and faster** because it skips the "build a reward model" step.
# - **PPO** is a **full reinforcement learning method**, needing more setup but offering more flexibility when rewards are tricky.
#

# %% [markdown]
#

# %% [markdown]
# ![texte](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*adNPXsn8v1qXiy98.png)
#

# %% [markdown]
# In the DPO’s paper, the authors apply the Bradley and Terry model, which is a preference model in the loss function. Through some algebraic wor, they demonstrate that the second step can be skipped because language models inherently act as reward models themselves. Surprisingly, once the second step is removed, the problem is significantly simplified to an optimization problem with a cross-entropy objective, as shown in Figure below

# %% [markdown]
# ![image](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*zE6I3BBUDMN9lfwV.png)

# %% [markdown]
# <img href="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*adNPXsn8v1qXiy98.png)">

# %% [markdown] id="a94a5cf3-64a6-4d2f-8285-aa7f3adc753a"
# #### Setup and Installation
#

# %% [markdown] id="e4d02ed1-d359-424a-969c-aa3a6abaa55e"
# - Installing required libraries
#
# **Note**: These versions are specifically selected for compatibility

# %% colab={"base_uri": "https://localhost:8080/"} id="526e16af-5b85-4f16-bba1-c97eff3feaa1" outputId="ccb00252-94ed-430e-fdfa-24a52a26febd"
# !pip install --q torch==2.3.1 trl==0.11.4 peft==0.14.0 pandas numpy==1.26.0 datasets==3.2.0 transformers==4.45.2

# %% [markdown] id="e220e00c-a0de-484d-8b44-13df3b0131b9"
# - Importing required libraries
#
#

# %% id="7e8499e9-7755-497d-8c11-22ca2364964f"
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

# %% [markdown] id="a8e0ce5b-07cb-4839-b6ca-d77910210b23"
# #### Model and Tokenizer Setup
#
# For this workshop, we'll use the OPT model, a decoder-only language model from Meta AI.
#

# %%
# Check for GPU availability and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model selection:  We're using AraGPT2, an Arabic language model based on the GPT-2 architecture
MODEL_NAME = "aubmindlab/aragpt2-base" # "unsloth/Qwen2.5-1.5B" 

# The model name for the fine-tuned version
FINETUNED_MODEL_NAME = "aragpt2-base-dpo"

# %% [markdown]
# - Set the Hugging face token found [here](https://huggingface.co/settings/tokens)
# In order to interact and use the hugging face hub

# %%
# Set Hugging Face token for accessing models 
os.environ["HF_TOKEN"] = "YOUR_HF_API_TOKEN" 

# %% [markdown]
# - Get your wandb API Key found [here](https://wandb.ai/authorize) and set it as an environment variable

# %%
os.environ["WANDB_API_KEY"] = "YOUR_API_TOKEN" 

# %% colab={"base_uri": "https://localhost:8080/", "height": 407, "referenced_widgets": ["9739f95abc684fb99066d33acd5cca03", "8ee43741711940c69f51391c5d74c8ab", "4dfaca4072c74f7cad99926d5c7d57d3", "a2896216d8fb42a09aae0d279a21124d", "4e4b62744ad64850b7c3d5b7d7a60b31", "ea44866031c940049fefecb8222efa5e", "b4c414097e0b4b199dc87d8810110d6c", "93637ff6c0014d18ab338c127ef065c4", "8f527219140a4646932c875c7a9ec7c1", "e7efe121f20c4640ac4987c2daec3ef6", "e48e096fbd8a4d8bb29512f2e4590cc9", "a588cb3652bd4506aaa2897db4c6c1f4", "39bc600bba0643b1acb7b1be4b833475", "be3d678dd3de4f29a8e16d12d92b5e59", "b1083690eb604a52b0ce985343545f5c", "97715a7076014bf182c882eefc4d7919", "039336c83542451aa22d99b8fedb28e5", "6012aaa5f5464517bc3dfbddaec48ed6", "da6e41d1a5fc427fb7a7eca697466e5d", "7e237a65ae9149e39e8a966ecdc92273", "a94390cfb0b14130ae82a4a5c67dbc88", "3a0785c4d6be49c1998a392e091ca3c0", "404c0c6e3f11430491e938ea364a0cef", "b5a8d1c1649e4036b5743098cdb6df5a", "392ee88ab0b5407ba5255501800ceef9", "fe3b2c8dc1d14545847dc97484fd7350", "306491b6625a4e58835f5e6078be338b", "f3ac65528e1d4f7c97e5635f19fde6dc", "da27d912abbb401ab7d89ae08cd4e844", "9e9d0cbef735482d9b22a5cba2453760", "af28fef6a8e5484cb6a89f58f764d9be", "5ce3466376cb4dcd8b45ce3136c6888a", "12bf20f6ce8d438aa8cd4f75fbb14370", "d1f51968e082414b8a46bfd3a7064b3f", "29642eb947934870b12972a4258a6222", "c9e6c4834ac844cfaf2e911bdfed6d1d", "4a3defa52d424ead82070cdc628f28aa", "2dcb8d3fcfa6499a8ce0e1673b970d6b", "69e6e842a7c248eb8d2ee73dfc96bbf9", "fcfd402b6d89405aac58306ba707739b", "bb78045767f94b98b1220601516af3df", "e183f8a7408d463e831decbe4fa903ed", "0339157659dd44cba1401ac2dda2f2ee", "8d477cc57a55492d94e2177a7fd9687a", "36569e83e26c43feba92864dcdcfd9f2", "02c81af079624be6a75b788ac8963924", "0b4f9ad5818647f5aab5e7c962274c54", "9269fae70c1f440ca8bb57e881ad96bc", "62ee6e420b6b49d5a23f8d08e84bf52e", "35864393d5a845659939b5a2d86a04d1", "f70b2c2a3606475184cf54e7c3b32f26", "3089e31c3b734fc1bc05b0c43d45fb82", "a3bab896d8fa469d8f1373156a53e44d", "e9decb3f6fba42bf9a4c8ec7f31e6250", "60b562cc84a94b5caad1d7a166ae2168", "ab5aa38d7e274b2d8fbd2984d0f09861", "41e3c4395e4846c9844dc1ed281a224e", "9ea88c82c8bc4db18bb1da7a45f5014a", "0bd0fed1bb33499eabd4c8205225c208", "12bf857e6a3a44eda50328351078fc37", "e06e7cc754104cf7b2222f4ea92a44aa", "241ce4620adf43e195f412a100a318d7", "68bb1295535047deb8bd047a4d3c8610", "21c6612a884e43c0acc98b0b46f30386", "91b6de4c12ce401881eb40cb5112e5f3", "1d97d0e071bf4c8abd5f7da4e792edb8", "bf265388616149f697b82f095701bc1a", "e2bbcc0168ad467ba40441eb340d0494", "290f97551f3e476ba4b0581be0b6dc44", "7362462038064e759c091bae0cebc0ed", "c8ac90585bd142fdb4f5f4947845598d", "7ff80e3c6f27422ead51da14c2fd9f94", "8f5e51c8f0194b5790098a0780b2c579", "997605d58ee94f1fb2da166759b863dd", "85785faf3e9b475da5433151ecc3f8df", "7a945c5425cb4fd98d28cc34aef7415c", "6fdf110ee4d243549fa5a78dfd4e3505"]} id="21e349e7-4856-4ad5-8117-fbda95d31997" outputId="130e9c3f-5d31-4b4f-d6f4-3bda45be5d79"
# Load model for training
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Load reference model (used during training)
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Configure padding token and padding side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Padding on the right side preserves the beginning of sequences

# Disable cache during forward pass to save memory
model.config.use_cache = False

# %% [markdown] id="iTWX1J5iSljD"
# #### Data Preparation
#
#

# %% [markdown]
# -  Load the Arabic preference [dataset](https://huggingface.co/datasets/FreedomIntelligence/Arabic-preference-data-RLHF) for RLHF

# %% id="PFO1WaJ_Sp0s"
# This dataset contains pairs of responses where one is preferred over the other
# We use only 10% for this demo to keep training time reasonable
print("Loading preference dataset...")
ds = load_dataset("FreedomIntelligence/Arabic-preference-data-RLHF", split="train[:10%]")


# %% [markdown] id="zTpN6LfISwdI"
# Examine the dataset structure to understand its format

# %% id="LXq_iaVXSp3L"
print("\nDataset sample features:", ds[0].keys())
print("\nExample entry from dataset:")
print(ds[0])

# %% [markdown] id="15_S212KS1vq"
# - Transform the dataset into the format required by DPO:
#     - `prompt`: The input query
#     - `chosen`: The preferred response
#     - `rejected`: The less preferred response
#
#

# %% id="h7NFdm6eSp6J"
print("\nPreparing dataset for DPO training...")
ds = ds.rename_column("instruction", "prompt").remove_columns(["id"])

# Split the dataset into training and evaluation sets
# We use a 90/10 split with a fixed random seed for reproducibility
ds = ds.train_test_split(0.1, shuffle=True, seed=42)
train_dataset, eval_dataset = ds["train"], ds["test"]
print(f"Training set size: {len(train_dataset)}, Evaluation set size: {len(eval_dataset)}")

# %% [markdown]
# | Chosen | Rejected | Prompt |
# | --- | --- | --- |
#  | نعم، يمكن تحقيق أرباح من تطبيق أو لعبة من خلال الإعلانات داخل التطبيق. هذا يتم عن طريق استخدام شبكات الإعلانات مثل Google AdMob أو Facebook Audience Network، حيث تُظهر الإعلانات في التطبيق أو اللعبة ويتم تحقيق العائد بناءً على عدد الأشخاص الذين ينقرون على تلك الإعلانات. العائد يمكن أن يتراوح من صفر إلى عشرات الألاف من الدولارات بناءً على شعبية التطبيق أو اللعبة وكيفية تفاعل المستخدمين مع الإعلانات.|نعم، يمكن تحقيق أرباح من تطبيق أو لعبة من خلال الإعلانات الموجودة داخل التطبيق. يعتمد هذا على مجموعة من العوامل بما في ذلك شعبية التطبيق وعدد المستخدمين، ونوع ومحتوى الإعلانات، والاستراتيجيات التسويقية المستخدمة. يمكن للشركات أن تحقق أرباح أيضًا من الإعلانات التابعة أو الإعلانات المدفوعة المستندة إلى النقر أو الإعلانات التي تظهر عند توقف المستخدمين عن استخدام التطبيق.| هل يمكنك تحقيق ارباح من تطبيق او لعبة فقط من خلال الاعلان داخل التطبيق
#
#

# %% [markdown] id="f6e743dd-7d5b-4e65-8746-bb448d3f17d0"
# ### Optional: Quantized Model Configuration (for GPUs)
# For r environments with GPU support, you can use quantization to reduce memory usage: Uncomment the following blocks if working with limited GPU memory
#

# %% [markdown]
# ![lora](https://pytorch.org/torchtune/0.4/_images/lora_diagram.png)

# %% id="01b01eb4-e7b1-4ea1-89ad-d6fed409cc66"
# # !pip install -U bitsandbytes # this package is required for quantization

# %% [markdown] id="bc8a1066-09b8-4201-b76f-e53729347d9a"
# **_Note:_**  _You can run the installed package by restarting a Kernel._
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 109} id="2a7547e9-9b38-49ff-93fc-3656152fb8c5" outputId="80593104-e3dc-4dee-c855-3ec37c13594f"
# # !pip install -U bitsandbytes  # Required for quantization

# from transformers import BitsAndBytesConfig

# # Configure quantization parameters
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,                    # Load model in 4-bit precision instead of 16/32-bit
#     bnb_4bit_use_double_quant=True,       # Use double quantization for better accuracy
#     bnb_4bit_quant_type="nf4",            # Use normalized float 4-bit quantization
#     bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for calculations
# )

# # Load models with quantization config
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, 
#     quantization_config=quantization_config,
#     device_map="auto"
# )

# model_ref = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME, 
#     quantization_config=quantization_config,
#     device_map="auto"
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="auto")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
# model.config.use_cache = False

# %% [markdown] id="410df653-d933-46e9-8e0e-499f5e6b82bd"
# #### LoRA Configuration for Efficient Fine-tuning
# #LoRA allows us to train only a small number of adapter parameters instead of the full model

# %% id="9a92494b-1e8c-4d8e-b357-ce1d6fd0213e"
# PEFT (Parameter-Efficient Finetuning) configuration
print("Setting up LoRA configuration...")
peft_config = LoraConfig(
    r=4,                    # Rank of the low-rank decomposition matrices
    target_modules=[        # Which modules to apply LoRA to
        'c_proj',           # Projection layers in the transformer
        'c_attn'            # Attention layers in the transformer
    ],
    task_type="CAUSAL_LM",  # The type of task we're performing
    lora_alpha=8,           # Scaling factor for the LoRA parameters (typically 2x rank)
    lora_dropout=0.1,       # Dropout probability for LoRA layers
    bias="none",           # Whether to train bias parameters
)

# %% [markdown] id="5968cd9e-0160-4691-8586-3320a56e1f41"
# ####  DPO Training Configuration
#

# %% colab={"base_uri": "https://localhost:8080/"} id="8c054ee8-4656-42f2-bd5c-56da25871eca" outputId="3facfb61-ea6c-46a0-d45a-db0189a310de"
# Configure DPO training parameters
print("Setting up DPO training configuration...")
training_args = DPOConfig(
    beta=0.1,                      # Temperature parameter for the DPO loss (typically 0.1-0.5)
                                   # Higher values make the model more conservative
    output_dir="dpo",              # Directory to save model checkpoints
    num_train_epochs=5,            # Number of training passes through the data
    per_device_train_batch_size=2, # Batch size for training (adjust based on GPU memory)
    per_device_eval_batch_size=2,  # Batch size for evaluation
    remove_unused_columns=False,   # Keep all columns in the dataset
    logging_steps=10,              # Log training progress every 10 steps
    gradient_accumulation_steps=4, # Accumulate gradients over multiple batches
                                   # Effectively increases batch size to 2 * 4 = 8
    learning_rate=1e-4,            # Learning rate for the optimizer
    evaluation_strategy="epoch",   # Evaluate after each epoch
    warmup_steps=2,                # Number of warmup steps for learning rate scheduler
    save_steps=500,                # Save checkpoint every 500 steps
    report_to='wandb'              # Report training metrics to Weights & Biases
                                   # Use 'none' to disable reporting
)

# %% [markdown] id="77edf717-f35a-46cf-b9ae-69cd1fa1ef8c"
# ####  DPO Trainer Setup
#
# Next step is creating the actual trainer using DPOTrainer class.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["3bd712f445c54a2d80c12307d432fdc1", "f2d2f236d09347ddbd4ef35251cb892c", "f2fa7fd7adfd4ba3be3e8902ca6a5278", "455d956aa8f445cb94605923c9e1bff2", "db72b67a59814cbcadcb86d63e125a66", "b446e780c0a44a4090e04aa98a1c2d65", "e864090947dd4843864e099824e9408f", "4d426c7b4f274db581a07e534e5e0bbb", "ed0d9f274b3b47089f802a4efdd0cf15", "7a3b334663524eec920d6023713ecd89", "57beadc0d7854ca39713e0331990e481"]} id="334f52f1-00e5-4c16-bcb5-bc0a91c6f3dd" outputId="f4ed2dde-007a-48d1-f110-bc5e2fe5210c"
# Create the DPO trainer that will handle the training process
print("Setting up DPO trainer...")
trainer = DPOTrainer(
    model=model,              # The model to be fine-tuned
    ref_model=None,           # Reference model (None because we're using LoRA)
                              # When using LoRA, DPOTrainer will automatically handle the reference model
    args=training_args,       # Training arguments defined above
    train_dataset=train_dataset,  # Training data
    eval_dataset=eval_dataset,    # Evaluation data
    tokenizer=tokenizer,          # Tokenizer
    peft_config=peft_config,      # LoRA configuration
    max_length=512,               # Maximum sequence length for inputs and outputs
    
)

# %% [markdown] id="f6685ddb-2a23-4e2d-9b84-acc0b45e2c0f"
# Please note that when using LoRA for the base model, it's efficient to leave the model_ref param null, in which case the DPOTrainer will unload the adapter for reference inference.
#
#
# Now, you're all set for training the model.
#

# %% [markdown] id="4b2658b6-ed1e-41d7-952b-87f23569c406"
# #### Training Process
#
#

# %% [markdown] id="30af9935-f1fd-4c37-9024-a8217e51c25d"
# **Training can be time-consuming on CPU and may cause memory issues, If you encounter problems, skip to the next section to load a pre-trained model**

# %% colab={"base_uri": "https://localhost:8080/", "height": 380} id="9088c26f-dee1-4e46-9aac-304d11409c63" outputId="22b70404-99e5-4123-cc63-7ab1e3c4e602"
# Start the training process
print("Starting DPO training...")
trainer.train()

# %% [markdown]
# !!!!You can skip the training !!!!

# %%
# Save the trained model to Hugging Face Hub
# print("Pushing model to Hugging Face Hub...")
# trainer.push_to_hub(FINETUNED_MODEL_NAME, commit_message="DPO finetuning with LoRA")

# %% id="c05a7bef-4cb6-4372-8d85-41a57086cc88"
# Load the trained model from the local checkpoint
# print("Loading trained model from checkpoint...")
# dpo_model = AutoModelForCausalLM.from_pretrained('./dpo/checkpoint-3895').to(device)
# dpo_tokenizer = AutoTokenizer.from_pretrained('./dpo/checkpoint-3895')

# %% [markdown] id="9a7f533d-e4fc-45bb-a142-ba2af70eec57"
# #### Loading Pre-trained Model (Alternative)
#

# %% [markdown] id="ad6250e2-f8f2-43a6-876e-f0e58ff1ef3b"
# If training is too resource-intensive, you can load a pre-trained model
#
# This section loads a model that's already been fine-tuned with DPO

# %% id="aa9098a6-c6f6-4bb4-b36e-06b9ce212b7b"
# Load the DPO-fine-tuned model from Hugging Face Hub
print("Loading pre-trained DPO model from Hub...")
dpo_model = AutoModelForCausalLM.from_pretrained(f"HackAI-2025/{FINETUNED_MODEL_NAME}").to(device)
tokenizer = AutoTokenizer.from_pretrained(f"HackAI-2025/{FINETUNED_MODEL_NAME}")

# %%
# Load reference (baseline) model for comparison
model_ref = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# %% [markdown] id="72e5ec7e-1ae5-48a9-9e63-b1180b0e042b"
# ####  Text Generation and Comparison
#

# %% id="4e92681c-7ea8-4103-937d-6cf8e45448b5"
# Set random seed for reproducible generation
set_seed(40)

# Configure generation parameters
print("Setting up text generation configuration...")
generation_config = GenerationConfig(
    max_new_tokens=70,         # Maximum number of tokens to generate
    do_sample=True,            # Use sampling instead of greedy decoding
    top_k=50,                  # Consider top 50 tokens at each step
    top_p=0.8,                 # Consider tokens with cumulative probability of 0.8
    temperature=0.8,           # Controls randomness (higher = more random)
    repetition_penalty=1.2,    # Penalize repetition of tokens
    pad_token_id=tokenizer.eos_token_id  # Use EOS token for padding
)

# Define a test prompt in Arabic
PROMPT = "كيف يمكنني التغلب على القلق والتوتر؟" # "What are the benefits of healthy food?"

# Tokenize the prompt and move to the appropriate device
inputs = tokenizer(PROMPT, return_tensors='pt').to(device)

# Generate text with the DPO-fine-tuned model
print("Generating response with DPO model...")
outputs = dpo_model.generate(**inputs, generation_config=generation_config).to(device)
print("DPO response:\t", tokenizer.decode(outputs[0], skip_special_tokens=True))

# Generate text with the baseline model for comparison
print("\nGenerating response with baseline model...")
outputs = model_ref.generate(**inputs, generation_config=generation_config).to(device)
print("Baseline response:\t", tokenizer.decode(outputs[0], skip_special_tokens=True))


# %% [markdown] id="eadacd93-bc91-4dca-a849-4a589aa08ab3"
# Althought the model is trained on a small data for 5 epochs only, it can be seen that the response generated by the DPO-tuned model is more concise and straightforward.
#

# %% [markdown] id="5bf59ca9-20fa-4f6b-acdd-425d69b8d9ae"
# # Exercises
#
#

# %%
test_questions = ["ما هي فوائد الغذاء الصحي؟",
"كيف يمكنني التغلب على القلق والتوتر؟",
"اشرح لي كيفية استخدام الذكاء الاصطناعي في التعليم.",
"ما هي أفضل طريقة لتعلم لغة جديدة؟",
"هل يجب علي الاستثمار في العملات المشفرة؟",
"ما هي أخطر تهديدات البيئة في العالم اليوم؟",
"كيف يمكنني تحسين مهارات التواصل لدي؟",
"اقترح برنامجاً لتمارين رياضية لشخص مبتدئ.",
"ما هي الخطوات اللازمة لبدء مشروع تجاري ناجح؟",
"كيف يمكن للتكنولوجيا أن تساعد في حل مشكلة تغير المناخ؟"]

# %% [markdown]
# ## Exercise 1: Experiment with Generation Parameters
# Try different generation parameters (temperature, top_p, top_k) and compare their effects on:
# 1. The quality of the generated text
# 2. The diversity of responses
# 3. How closely they align with human preferences
#
# ## Exercise 2: Test with Different Prompts
# Create 3-5 different prompts and compare the responses from:
# 1. The base model (model_ref)
# 2. The DPO fine-tuned model
# Analyze the differences and explain how the DPO training has affected the outputs.
#
# ## Exercise 3: Error Analysis
# Identify cases where the DPO model still produces suboptimal responses and suggest:
# 1. Possible reasons for these failures
# 2. How you might improve the training data to address these issues
# 3. Alternative training strategies that might help
#
