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

# %% [markdown] id="kYiFCaX4bm-i"
#

# %% [markdown] id="Uah0ia3h9gFD"
# # Before You Star

# %% [markdown] id="bUCmvBPWyfeo"
# * **What is Regular Pretraining (RPT)?**
#
# > The first step in creating a high-quality LLM is to pretrain it on one or more massive text datasets. During training, it attempts to **predict the next token** to accurately learn **linguistic** and **semantic** representations found in the text.
# this is called language modeling and is a **self-supervised** method.
#
# <center>
#
#
# </center>
#
# > The purpose of pretraining a model on large datasets is that it is **able to reproduce language and its meaning**. During this process, the model learns to complete input phrases.
#
# > The pretraining stage produce to us **base model** and also called **foundation model**.
#
#
#
# ```
# 🤗 huggingface Base models examples:
#
# - HuggingFaceTB/SmolLM2-135M
# - Qwen/Qwen2.5-0.5B
# - meta-llama/Llama-3.3-70B
# - mistralai/Mistral-7B-v0.3
# ```
#
#

# %% [markdown] id="lYIl_oMD9LD3"
# * **Continued Pretraining (CPT)**
#
# >Continued or continual pretraining (CPT) is necessary to “steer” the language model to **understand new domains of knowledge**, or **out of distribution domains**. Base models like Llama-3 or Qwen are **first pretrained on gigantic datasets of trillions of tokens** (Llama-3 for eg is 15 trillion). But sometimes these models have **not been well trained on other languages (eg arabic, darija)**, or text specific domains, like **law, medicine or other areas**.
#
# <center>
#
#
# </center>
#
#
# > Continued pretraining (CPT) is necessary to make the language model learn new tokens or datasets.

# %% [markdown] id="vV1FIJZ7zsaF"
# * **What is Finetuning (SFT)?**
#
# > In this stage after we got our base model that's understand the languge, with **supervised finetuning (SFT**) we can adapt the base model ***to follow instructions***.
#
# > During SFT process the model parameters are **updated** to be more in line with out target task (eg Q/A.)
#
#
# <center>
#
# </center>
#
# > SFT training is like pretraining (trained using next-token prediction); the only difference is that SFT is **based on user input**.
#
# > After SFT of our base model we will get new model that's called **Instruction Model**
#
# ```
# 🤗 huggingface Instruct models examples:
#
# - HuggingFaceTB/SmolLM2-135M-Instruct
# - Qwen/Qwen2.5-0.5B-Instruct
# - meta-llama/Llama-3.3-70B-Instruct
# - mistralai/Mistral-7B-Instruct-v0.3
#
# ```

# %% [markdown] id="uTY-PPPf7Zgs"
# <center>
#
#
# </center>
#
#
# > In the stage of pretraining your model you need a huge amount of unlabeled data (text)
#
# > In the finetuning stage you need labeled data (also called instruction dataset )

# %% [markdown] id="zWNP81srBZSc"
# # Install Requirements

# %% colab={"base_uri": "https://localhost:8080/"} id="l6pEQW4KnFyU" outputId="ee5c5bc2-b30a-4356-ce34-67be14ccd580"
# ! pip install -U torch datasets transformers wandb -q

# %% [markdown] id="pymBYMAIUNSZ"
# # Check GPU Memory

# %% colab={"base_uri": "https://localhost:8080/"} id="A1ROC2vhUo6_" outputId="b311d00a-fb26-4616-a04a-a59e36186ed7"
import torch
torch.cuda.is_available() # True means you're using nvidia gpu

# %% colab={"base_uri": "https://localhost:8080/"} id="DHxwB93FURRw" outputId="c5828031-4e16-4c3b-97d0-89118addcd2b"
# ! nvidia-smi # you can also use nvitop "pip install nvitop" to see real time gpu consumption in your terminal

# %% [markdown] id="8StLuVFpBuL8"
# # Load Your Dataset

# %% [markdown] id="hFqm0Ua847U7"
# > In this section we will learn how to load our dataset from huggingface 🤗, select the column we want to train our model on, finallu split it to train/test.
#

# %% [markdown] id="EZT5K3MH5z3_"
# 1. huggingface loging

# %% id="cq0ZJw6Q3Nty"
from huggingface_hub import login
login("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")

# %% [markdown] id="XlJynrdj547l"
# 2. load dataset using `load_dataset`and the id of dataset from huggingface `username\datasetname` example `atlasia\atlaset`

# %% id="CUYz97bXBsqw"
from datasets import load_dataset, Dataset
import pandas as pd

# %% colab={"base_uri": "https://localhost:8080/", "height": 306, "referenced_widgets": ["e3baf9ccc52f4bbca882aab8b4b586df", "bbfc8e1200d54210a380f4e894d6b77b", "55eb328626f24c2a943edba9ba8d02c1", "769be9f2705e44c2b0ee26df7909951e", "c190a621f4d8412393c6d35bc9bef281", "cc5b577b16604dc49572103d4c428f21", "0a05edb52ae44d2fb6cdf23192c4f18a", "0355e49bcb744dcd90bbc4439126eded", "9866048b113b4bb2a3e6086f6e9abfd5", "a21666f30e6141cabed255471b1ab207", "34cc4bab21164a5d85bdfabf07a1e024", "780c9c809b67461ebca963402852e53e", "86f56512cac44d1fbf0bba88519c1f1c", "e105dc5bbbc741c8b663c5e1042cbabc", "d1775de5c3cd4693834b1d91ad1908b7", "b51e5f194a36408280866a2b7ea8f1d3", "9f49ea714f7a4f5eb71ae64905a72568", "f78e25044c1c4d4cbd39bfd877a7c9f6", "58c1c19455f7490198829410c93c9a41", "cf950c8faded4fcabab02f488d717784", "1b724db15285483a9ae0329152e8b232", "5767491525624228b850fb9df9b470db", "8f2b136d5fad482b954d924f6426143f", "dcb5b759583140b9959b5c6be6276109", "e8dcf814ba3643a1a6d14ebb70372f8c", "e9b8a929a60945f1947c5922cb73c8fd", "75ae751d296348bba7d1573ed76bb6ee", "0c76e1e70fa34a13b56aae546a833069", "41f0b0d8fe1a472f8b894affa09d3caf", "755f6808c3414a18a6e9b44d26459dcf", "9dc8c080521e45c29880be67afcc2628", "d158759648234326951bc6f97531a736", "e2402a2fe23c4c889e0de00c82cfd37c"]} id="ic1aWv1EB3m2" outputId="e0ee0c0d-885e-43f2-be7e-7ea69d4f150c"
dataset_name_id="atlasia/good25"
ds=load_dataset(dataset_name_id,split="train")
ds

# %% colab={"base_uri": "https://localhost:8080/"} id="whLlkgGj5VQf" outputId="2f73815b-b58e-4285-ea27-d8f6202f2221"
# select only the text columns (eg content) that's our model want to train on.
ds=ds.select_columns(["content"])
ds

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="P5idU4nW45QN" outputId="a0aa27e4-d002-40a1-8bcb-c35aa053be85"
# take a look in our dataset using pandas
ds.to_pandas().head()

# %% [markdown] id="wkKnspWY7RpZ"
# 3. split our dataset into train and test

# %% colab={"base_uri": "https://localhost:8080/"} id="wv40ZG-B613Y" outputId="8af5c3bf-8203-4216-b5ec-ccba2e346854"
ds_spliter=ds.train_test_split(test_size=0.2,seed=42)
ds_spliter

# %% [markdown] id="WRlhxqiI84nb"
# # Preprocessing For Pretraining LM

# %% [markdown] id="xNOv1XJI9Sav"
# > In this step we will go through the necessary preprocessing steps befor starting train.

# %% [markdown] id="rVzWDcO2_AmY"
# 1. select the base model you want to train from hf 🤗
#

# %% id="N3sDFEAv_Tr_"
model_id="HuggingFaceTB/SmolLM2-135M-Instruct" # example HuggingFaceTB/SmolLM2-135M-Instruct

# %% [markdown] id="-8kIBoh9-1-H"
# 2. load model tokenizer

# %% colab={"base_uri": "https://localhost:8080/", "height": 177, "referenced_widgets": ["afe42e6011664aeb9dabf69138471c59", "1310ca3eb3cb4222bdc86a838649379d", "4ccdc8fb35864c38aeea6a419cd64dab", "2d90bea4fbd749009461753cb0bfef97", "877918b819bd4280a574460123fdf9aa", "36cc8bdbf9a34be3b2c902cd49d2191c", "2af96a735c504f6fa09572b2a5741538", "f798ae352eb74436ac73d9a29eb04d56", "6f645582fc4e430c8409a10e8b40c256", "6b580cb2ae6a4f1f8943e7f1c5d93d39", "295447566c5241349b6b611d3afe815e", "840f881140184b1f80192680c0d9e434", "8a853bc14a664af38b35f3a7c19afcd8", "27a7c70ad8bb41428778522b9aaeefc2", "757bba4fd53745e09ef35023050eb627", "5bbac73706a34e90bc366e663af4b056", "93a13d205ae5431ca98cfa6418f5a574", "87914a41cb174d3e9d535da3b834341a", "f40ac1ab0441483c952b7a895dac1570", "1d6499d929d54f10ac9b05f59e1fcd37", "ba91a8abce0546ebaa3d58cf3719d0b4", "e6c6ec2b68df4c4981c529d578467508", "ab3f5aa64584453ea01a3e0938618396", "1c7c0a06afbb424cb5faaf61eaf11d34", "35e993c57f1a46a282c316bf49a00103", "5c7bd06bbb9542ed9ddbec6363859a1e", "fae04d9f94b64835a46f04db7b0e34ec", "b7f26c1b7d614102a9eadc75061dd83e", "bb72452d79264f4a8e5f1dab67e4ae92", "75c9d8fd4a174dfdbc9c10b1a761eba5", "d453ce9cd6ea4fb1beafb890aebe4202", "989ce170840b42babf7c3e7eb1e7b89a", "bb40dcba093e4ee88a834c7f2feb2f65", "5328a7f8b6a94d33a305b931691a0fc1", "15a85235b1ba428f8d7d71f936c3dd0b", "55bf670566e2477f80b1349046c6ead3", "f242b786ffd6404382c674a9b9f8f8fe", "a74fe26514c94e7eadbc215603d38f17", "cc7f3536491f4c3a85d43bf0aae58979", "08f39ca989784d62b3a4c2f7de5a3c06", "cf5c3b60239b42acb0437bbd7887da9d", "63fb07f12e354c319a164d9feda1439b", "9a6e9202072344b59b860fdbf6532fed", "2e50e37497444c85a3b93f235f10ea07", "dddd45e907bd478a801d431b95ab1278", "ef7f0353c25e4befa123d8b517f6e9eb", "337b313326ed424a9756849eb17eab26", "6904cfcb868d4be197ad10728e146f44", "ea7e5b5efe9d4034933f45e69070a061", "5ad1dadd510c4d1c9b1efc680e58812d", "42c9c50df32d4444a4770b23b157d44a", "775e159cc9ad4573bf6581e24120a3ea", "6c7648014c1445899ff1bc7c256308ae", "25631f67eb3546929f87b78ec414de3f", "dfa43f35440d4d108801865cc01473ea"]} id="Hqlt8UWG7yyl" outputId="b8939416-937c-477c-b09f-da0013626797"
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(model_id)

# %% [markdown] id="PN9CUwujA_IQ"
# 3. check some tokenizer configs

# %% colab={"base_uri": "https://localhost:8080/"} id="sUUSNwdH7m6H" outputId="d65f44b1-2ceb-4c4e-ba0f-147fdf029efb"
# check the vocabe size of our tokenizer
print(f"{len(tokenizer)}")
# model max length (means the max len of the input)
print(f"{tokenizer.model_max_length}")
# tokenizer special tokens
print(tokenizer.special_tokens_map)

# %% [markdown] id="7Kk63WlcBNit"
# 4. test tokenizer
#
# <center>
#
#
# </center>

# %% colab={"base_uri": "https://localhost:8080/"} id="BI6Q8QZOBPlw" outputId="030aab35-5f77-4076-cb7d-40ec16fb4255"
example="Hello my name x I'm ThinkAI participant"
ids=tokenizer.encode(example)
print(ids)
tokens=tokenizer.convert_ids_to_tokens(ids)
print(tokens)
decode_=tokenizer.decode(ids)
print(decode_)

# %% [markdown] id="HF8_RLMYDSwn"
# > After learned about Tokenizer, now we will go through encode our input text. the only things we need to fix is the `context length`and it should be less than or equal the `model_max_length`.
# * Big `context length` means more context
# * More `context length` more GPU memory
# * `contenxt length` Recommended to be in **%64** `[64,128,256,...max_length]` to train you model faster try to use 128.
# * all input ids will **turncated** to be in selected `context length`
#
# <center>
#
# </center>

# %% [markdown] id="DvMn0LtNHlcP"
# 4. set context length

# %% id="629bxw_lHtqp"
context_length=128


# %% [markdown] id="R7cV3a3IHB3R"
# 5. tokenize function
#
# > with tokenize function we will tokenize all the text input and truncate it to be in the same `context_length`, and finally we will keep only the input ids with `length == context_lenght`

# %% id="FI_F9WKf761A"
def tokenize(examples):
  results=tokenizer(
      examples["content"],
      truncation=True,
      max_length=context_length,
      return_overflowing_tokens=True, # with this you will get also the input ids with length less than context_length
      return_length=True
  )
  input_batch=[]
  for l,in_ids in zip(results["length"],results["input_ids"]):
    if l==context_length:
      input_batch.append(in_ids)
  return {"input_ids":input_batch}


# %% [markdown] id="s2SyYeE4rRgZ"
# > After creating the tokenize function now we will tokenize all text using `map()` method by datasets. after tokenized you text dataset the number of rows will increase cause of each example will have `context_size`.
# * we need only the input_ids column to train our model, that's why we need to remove the others with `remove_columns` attribute.

# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["c8800125de7848a59015fded5cc072b7", "b5a55150d8484309b6c8b6507d12eccc", "df0f75a4448443fb9fda86469b3547eb", "b12937c833484a749ccbb90b77fa92c5", "c4a36c16d54c4aada0db39715a7695a8", "28ddc8b3a449496c8cbb5fc74bdd843d", "a6de6335f2ae4473af7f495d954c49f5", "1eaeffef632d48439ace1a707c4dda92", "5d65602198614dfab7e93489f6f8cbb0", "fbef100184964f84b80418dc07bef209", "bdceb7afbd144b5fa585b4836c9a0787", "4541e689586f4bbdace1d9f4775fe32c", "c722bb01a90147aeb664001b775c472c", "0ab0941363e64ac8a52e4b39624d0b61", "8e610a1f6002436198d56cb102aa6edc", "0e4375c732234fd49719391a94c2ff5a", "9163b8ac05eb4e618d7efaed04185aca", "635ffb55948a4344aa867fd1552c344b", "8bcbbf5306254bbc97e536a1ae4488b6", "bfecd39b3df248da9eabdb23dba2a51f", "47fe9851b8234312bcc4b2751130b6e0", "aa53f7bdc324487eb5b4943bf37988e3"]} id="uhzD-6APrh-S" outputId="dd4dfe5a-20bc-4bcc-811b-b33659d29990"
tokenized_ds=ds_spliter.map(tokenize,batched=True,remove_columns=ds_spliter["train"].column_names)

# %% [markdown] id="UuuxqLgOxJ_t"
# > now the last step in data preprocessing is to generate batches and labels using `DataCollatorForLanguageModeling`, the main roles of this function is:
#
# <center>
#
# </center>
#
# * Create batches
# * Add Padding to batches (Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.)
# * Creating Labels
#
# <center>
#
#
# </center>
#
# * Masked % from text when we train Masked Language Model (MLM) models. (in our case `mlm=False`)
#

# %% id="QFGsqhxTSMa_"
from transformers import DataCollatorForLanguageModeling
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

# %% [markdown] id="6S_KxpejuBfX"
# # Load & Train Your Selected Model

# %% [markdown] id="JyhTYTOKnYtv"
# ### Load Model

# %% [markdown] id="DwGN4ZMThStV"
# > After preprocessing stage, now we are ready to train our model, but first of all we need to select which method we want to use as explained above [first section].
#
# * training from scratch (RPT)
#
# <center>
#
# </center>
#
# ```
#   *  will load model and init weights with random values
#   1. load model configs using AutoConfig.form_pretrained
#   2. load model architect using AutoModelForCausalLM.from_config
# ```
#
# * continuos training (CPT)
#
# <center>
#
# </center>
#
# ```
#   * will load the model with trained weights
#   1. load model  and weights using AutoModelForCausalLM.form_pretrained
# ```
#
# the difference between those 2 methods is only on loading model code.

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["495e8b1543ea46fca964554554944780", "f6f24601db8d42e7a91d64eb39dbac6d", "4f7c6a0475b64d62a4eabd9ae92608db", "c7c04a0fa07c49889ee2fa74f38f6c7b", "91137d9815cd4b65981a88c0976a29d0", "ab2668a9743b4241b6a171d754d1034c", "0a864af16b86437cadafe9c91ba4aa54", "9a533fe1a3e242e7bd7b9c432a8f3365", "9a0f4ac175b24f8fb07a251ce7d55e8e", "c1487a3706224c4baf48075fb9de395f", "7a13ba2293444fe187f420c11635799b"]} id="nIwbmsjJi5OT" outputId="a710d15f-313e-4e7e-8d70-568be07a4096"
from transformers import AutoModelForCausalLM,AutoConfig
config=AutoConfig.from_pretrained(model_id)
model=AutoModelForCausalLM.from_config(config)

# %% [markdown] id="1LnWf0zYncQi"
# ### Train Args

# %% [markdown] id="QYPQp51Hozuj"
# > Now in the last step before training, we should set the training arguments, and the important one is:
# * `output_dir` [dir to save the model checkpoints]
# * `num_train_epochs`
# * `learning_rate`
# * `lr_scheduler_type` [linear/cosine]
# * `batch_size` [train/eval]
# * `warmup_steps` [increase lr from low val to target value during the begining x steps]
# * `save_steps` [save every x steps]
# * `save_total_limit` [save only x checkpoints to avoid memory space :( ]
# * `fp16` [mixed-precision training]
# * `push_to_hub` [push the trained model to the hub after finishing]
# * `logging_steps` [show logs (loss/accuracy) after x steps]
# * `report_to` [we will use wand]
# * `...`

# %% id="-HlbxgBznhWY"
from transformers import TrainingArguments
args=TrainingArguments(
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

# %% [markdown] id="J3nr-tgg9FLV"
# ### Trainer

# %% [markdown] id="2OXoIy3F5d1T"
# > After setting Training args now we will connect everthing using `Trainer` and then initialize the training using `trainer.train()`

# %% colab={"base_uri": "https://localhost:8080/"} id="vK2ewxKV5aOA" outputId="f7fd7d17-0d63-4272-e467-6bdf10607c27"
from transformers import Trainer
trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 533} id="faLCB-ho6avL" outputId="e5158f45-3e34-408d-a26f-d952ae96b2ab"
trainer.train()

# %% [markdown] id="VUvzpA9lRiXV"
# # Evaluation
#
# > Here are the most important and common metrics that you will likely need before launching your LLM system into production:
# * **Answer Relevancy:** Determines whether an LLM output is able to address the given input in an informative and concise manner.
# * **Correctness:** Determines whether an LLM output is factually correct based on some ground truth.
# * **Hallucination:** Determines whether an LLM output contains fake or made-up information.

# %% [markdown] id="dDbkPZR7Opy9"
# * challenge evaluation
#
# - quiz
# - training / valid loss
# - example prompt

# %% id="NNcPmQusaL-S"
