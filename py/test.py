# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %% [markdown]
#  # Before You Start [Learn About LORA]

# %% [markdown]
#  > Large language models are large, and it can be expensive to update all model weights during training due to GPU memory limitations.
#
#  * **Problem**
#
#  >For example, suppose we have an LLM with 7B parameters represented in a weight matrix `W`. (In reality, the model parameters are, of course, distributed across different matrices in many layers, but for simplicity, we refer to a single weight matrix here).** During backpropagation**, we learn a `ΔW` matrix, which contains information on how much we want to update the original weights to minimize the loss function during training.
#  * The weight update is then as follows:
#  `W_updated = W + ΔW`
#  * If the weight matrix `W` contains **7B parameters**, then the weight update matrix `ΔW` also contains **7B parameters**, and computing the matrix `ΔW``
#   can be very compute and memory intensive.
#
#  * **Solution: Low Rank Adaptation (LORA)**
#
#  >To make understanding LoRA easier, let’s take a sample example:
#  1. suppose we have model parameters represented by a `W (10x10)` matrix.
#  ![image](https://i.postimg.cc/7LtmYJ1H/lora1.png)
#
#  >2. We can come up with two smaller matrices, which when multiplied, reconstruct a 10×10 matrix for example `W(10x10)=A(10,r)*B(r,10)`.
#  ![image](https://i.postimg.cc/3Ry9yr9g/lora2.png)
#
#  > 3.This is a major efficiency win because instead of using **100 weights (10x10)** we now only have **2*(10*r) weights**.
#
#  >LORA method proposed replaces to decompose the weight changes,`ΔW=A*B`, into a lower-rank representation and make W frozen.
#  ![image](https://i.postimg.cc/YqNRLsNy/lora3.png)
#
#  > the image bellow show the difference between full ft and ft+LORA.
#  ![image](https://i.postimg.cc/QtRmcLnv/lora4.png)
#
#  **How much memory does this save?**
#
#  >It depends on the rank `r`, which is a **hyperparameter**. For example, if `ΔW` has 10,000 rows and 20,000 columns, it stores `200,000,000` parameters. If we choose A and B with r=8, then A has 10,000 rows and 8 columns, and B has 8 rows and 20,000 columns, that's 10,000×8 + 8×20,000 = `240,000` parameters, which is about **830× less than 200,000,000**.
#
#  **Are A and B will capture all the information that ΔW could capture?**
#
#  > Of course, **`A` and `B` can't capture all the information that `ΔW` could capture**, but this is by design. When using LoRA, we hypothesize that the model requires `W` to be a large matrix with full rank to capture all the knowledge in the pretraining dataset. However, when we finetune an LLM, we don't need to update all the weights and capture the core information for the adaptation in a smaller number of weights than ΔW would; hence, we have the low-rank updates via `AB`.
#
#  **Which parameters we will target with LORA?**
#
#  >You can target all model architecture layers, in our use case, we will target only the Key and Value weight matrices in each transformers layer to reduce memory requirements.
#
#  **Scaling Coefficient**
#
#
#  >```
#  scaling = alpha / r
#  weight += (lora_B @ lora_A) * scaling
#  ```
#  * Choosing **alpha as two times r** is a common rule of thumb when using LoRA for LLM
#

# %% [markdown]
#  # Install dependencies 📚
#
#  We need multiple librairies:
#
#  - `peft`for LoRA adapters
#  - `Transformers`for loading the model
#  - `datasets`for loading and using the fine-tuning dataset
#  - `trl`for the trainer class

# %%
# ! uv pip install -U datasets trl -q


# %% [markdown]
#  # load dataset

# %% [markdown]
#  > We will use [`HackAI-2025/Darija_SFT_Dataset`](https://huggingface.co/datasets/HackAI-2025/Darija_SFT_Dataset) dataset to fine tune [`atlasia/Al-Atlas-0.5B`](https://huggingface.co/atlasia/Al-Atlas-0.5B) or any other model in your choice.
#
#  * **SFT Dataset Example**
#
#  1. Instructions
#
#  <center>
#
#  ![image](https://i.postimg.cc/hvTrrCkv/lora5.png)
#
#  </center>
#
#  2. Conversations
#
#  <center>
#
#  ![image](https://i.postimg.cc/tRx2pTsb/lora6.png
#  )
#  </center>

# %%
# hf login
from huggingface_hub import login
login()


# %%
from datasets import load_dataset


# %%
dataset=load_dataset("HackAI-2025/Darija_SFT_Dataset",split="train")
dataset


# %%
# show some examples from ds
dataset.to_pandas().head()


# %%
# remove other columns and rename conversation to messages
dataset=dataset.select_columns("conversation").rename_column("conversation","messages")
dataset


# %%
# show the first example of messages
from pprint import pprint # pprint for pretty print
pprint(dataset["messages"][0])


# %% [markdown]
#  # Load Model/Tokenizer

# %%
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
# select gpu if available
device="cuda" if torch.cuda.is_available() else "cpu"
model_id="atlasia/Al-Atlas-0.5B"

tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id).to(device)


# %% [markdown]
#  ## Model Chat Template [Test]

# %% [markdown]
#  Instruction fine-tuning involves training a model on a dataset where the input-output pairs, like those we extracted from the JSON file, are explicitly provided. There are various methods to format these entries for LLMs.
#
#  <center>
#
#  ![image](https://i.postimg.cc/J4CKnLXk/lora7.png)
#  </center>
#
#  * Comparison of prompt styles for instruction fine-tuning in LLMs. The Alpaca style (left) uses a structured format with defined sections for instruction, input, and response, while the Phi-3 style (right) employs
#  a simpler format with designated <|user|> and <|assistant|> tokens.

# %%
# test tokenizer chat template
result=tokenizer.apply_chat_template(dataset["messages"][0],tokenize=False)
pprint(result)


# %% [markdown]
#  ## Test Model Before SFT

# %%
# Generate with before ft
prompt="السلام لباس؟"
messages=[{"role":"user","content":prompt}]
formatted_prompt=tokenizer.apply_chat_template(messages,tokenize=False)
ids=tokenizer(formatted_prompt,return_tensors="pt").to(device)
output_ids=model.generate(**ids,max_new_tokens=120)
output=tokenizer.decode(output_ids[0],skip_special_tokens=True)
print(output)


# %% [markdown]
#  # SFT + LORA

# %% [markdown]
#  ## Show Model Architecture

# %%
# show model architecture to select which layer we will apply lora
model


# %% [markdown]
#  ## set LORA Configs

# %% [markdown]
#  * `r`:  This is the rank of the compressed matrices, Increasing this value will also increase the sizes of compressed matrices leading to less compression and thereby improved representative power. Values typically range between 4 and 64.
#
#  * `lora_alpha`: Controls the amount of change that is added to the original weights. In essence, it balances the knowledge of the original model with that of the new task. A rule of thumb is to choose a value twice the size of r.
#
#  * `target_modules`: Controls which layers to target. The LoRA procedure can choose to ignore specific layers, like specific projection layers. This can speed up training but reduce performance and vice versa.
#

# %%
from peft import LoraConfig
lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], # apply lora only on q_proj and v_proj
    bias="none",
)
lora_config


# %% [markdown]
#  # TODO @nouamane: add qlora

# %% [markdown]
#  ## Set Training Args

# %% [markdown]
#  * **What is gradient accumulation?**
#
#  > Gradient accumulation is a way to virtually increase the batch size during training, which is very useful when the available GPU memory is insufficient to accommodate the desired batch size. In gradient accumulation, gradients are computed for smaller batches and accumulated (usually summed or averaged) over multiple iterations instead of updating the model weights after every batch. Once the accumulated gradients reach the target “virtual” batch size, the model weights are updated with the accumulated gradients.
#
#  <center>
#
#  ![image](https://i.postimg.cc/x10Rv3tj/lora10.png
#  )
#  </center>
#

# %%
from transformers import TrainingArguments


# %%
args=TrainingArguments(
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
    report_to="wandb",
    hub_token="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"
)


# %% [markdown]
#  ## SFT Trainer

# %%
from trl import SFTConfig,SFTTrainer
sft_trainer=SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=args
)


# %%
sft_trainer.get_num_trainable_parameters()


# %% [markdown]
#  ## Start Training

# %%
sft_trainer.train()


# %% [markdown]
#  ## Test Model After SFT

# %%
# Generate with after ft
prompt="السلام لباس"
messages=[{"role":"user","content":prompt}]
formatted_prompt=tokenizer.apply_chat_template(messages,tokenize=False)
ids=tokenizer(formatted_prompt,return_tensors="pt").to(device)
output_ids=model.generate(**ids,max_new_tokens=100,
                          repetition_penalty=1.2)
output=tokenizer.decode(output_ids[0],skip_special_tokens=True)
print(output)


# %% [markdown]
#  ## Push To THe HUB

# %%
sft_trainer.push_to_hub("abdeljalilELmajjodi/alatlas-sft-lora-gra")



