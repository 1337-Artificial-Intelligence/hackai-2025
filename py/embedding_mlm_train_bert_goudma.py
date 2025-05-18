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

# %% [markdown] id="0So2Q4B1b13a"
# # Before You Start

# %% [markdown] id="WxxfSYjdb5wf"
# * **What is Masked Language Modeling ?**
# > **Masked language modeling** predicts a masked token in a sequence, and the model can attend to tokens bidirectionally. This means the model has full access to the tokens on the left and right. Masked language modeling is great for tasks that require a good contextual understanding of an entire sequence. BERT is an example of a masked language model.
#
# <center>
#
# </center>

# %% [markdown] id="BROajBry9C65"
#  * **How can Masked Language Modeling be used in transfer learning for downstream NLP tasks?**
#
#
# > Once pretrained, these models can be fine-tuned on specific downstream NLP tasks, even with relatively small labeled datasets. The idea is that the model already understands language patterns, so it can adapt quickly to new tasks.
#

# %% [markdown] id="efnr5RusywjH"
# * ***BERT (Bidirectional encoder representations from transformers)***
#
# > BERT inspired from the Transformer architecture introduced in "Attention is all you need", to become an encoder-only transformer that can produce meaningful representations and understand language.
#
# > In the pretraining phase, BERT is trained to learn:
# * **Masked Language Modeling**: is to predict masked words in a sentence (I [MASKED] this book before -> read)
#

# %% [markdown] id="gCGDFFwyB8se"
# * **Steps To Get Your BERT Model**
#   - **RPT – Regular Pretraining (i.e., From Scratch)**
#
#     * **Train a BERT model from scratch**, starting with **randomly initialized weights**.
#     * Use **your own corpus** and perform MLM from the beginning.
#     * **Goal**: Create a completely custom model tailored to a specific domain or language.
#     * **Advantage**: Maximum control and domain alignment.
#     * **Downside**: Requires huge amounts of data and computational resources.  
#
#   - **CPT – Continued Pretraining (also called Domain-Adaptive Pretraining)**
#
#     * **Start from** a general-purpose pretrained BERT model.
#     * **Continue training** on your **own unlabeled data** using the same **Masked Language Modeling (MLM)** objective.
#     * **Goal**: Adapt the model to your domain's vocabulary and style (e.g., medical, legal, scientific text).
#     * **Advantage**: Fast and resource-efficient since you're not starting from scratch.
#
# <center>
#
# </center>
#     
#
#
#

# %% [markdown] id="GIDFjfKf9rS2"
# # Pretraining Bert Model (RPT/CPT) [Optional]

# %% [markdown] id="Z5DFyJWBWDfF"
# * As we said before in the pre-train phase, we train our model on masked lm task.

# %% [markdown] id="bf1-RUOBhjKB"
# ## Load Dataset from HF 🤗
#

# %% colab={"base_uri": "https://localhost:8080/"} id="M_M_ki-EiDyy" outputId="35a05152-de74-4791-d7f1-eab55f25ff83"
# install datasets
# ! pip install -U datasets -q

# %% id="s2n26recU_rF" colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["787f0830f9c047d785fdcd47b3081681", "dbbba007b61849a1904162fd1982c2ab", "d5b3693bd943406990237a418732aacc", "2bddd785317040628833f2b4ca9f04bc", "489010ce4d704458b183f33acf10aaa3", "d7ce0a0996ad4401a67d9c67c5abff2a", "d938afe6bcc94702a9f42d839e36b132", "2081745ff7d94f559039c1f3cff60e37", "c47cb10cc4c44412a242e1f31afe9cfb", "f91d337f466d4d7ebba80844178333d1", "a8fff571490f48f9859f4a8dbc4f8144", "b002d8e3379c4dbca4085db93866ac9a", "993179db5f7c49af9ebe3360e946ec19", "227a881e72a145d1a2ed07c86f9758c4", "4d5750e74278466a9da295e887607034", "fb6a1b11154e4f1daeac6919d379c815", "9efb2e745f76462f9d32c2f557e2049e", "c8064733f11741dda9be0a7116210ef2", "490175a68c7340d286f66730b9690496", "4dfa3d8dda6e4f64a8413fb9bca204e5"]} outputId="5a1a5507-6e7b-453f-b078-64786e728a71"
from datasets import load_dataset
from huggingface_hub import login
login()

# %% id="BCuihKtSU7uH" colab={"base_uri": "https://localhost:8080/", "height": 306, "referenced_widgets": ["544e56172c4d402cb694bab8e7cd10fe", "614aae6b84934a589a88324925df665f", "21897a1c14c1497da6cec37da41e8dd8", "b6d7bb930afd4785a84ddd0feed0e75b", "deced5f7ed2f41d781e1a672b2ad754c", "372e20eac5b1434991fa7d757faae59b", "d8ce6669278047ceaaa726983a31b634", "846585d11f6f4818a2ed283a02cde2b5", "23287583c9de4767820d3eb960a21999", "134cbd8ca1e04f9a942f539e1df89a43", "b92e1edc04d64769a6fea0b962323cbe", "6383b50bdf6c4d01ae9529d953e24123", "2cde84ab9f264fcfb266de5961e074b0", "56061495a9b64b828572ba2e244cf0e9", "04b2e0101def4eb7ab49832dc7f4eb2c", "5ef98e6ebddf46db896ebded7355a959", "c12edc11258c4db1b86025be428980ff", "767ca7ef62c442078a6ca85fb309dbe8", "34b37a3d165c45dc8a1afbb65c8f108c", "d76258868f6949948478ed118cc0210f", "a59aebd35e414ceb9b65ccad8b5ba78a", "58bb72a6ac814df28a3ad4c8579a0db8", "3908cfa5133144b19274061a11282ec8", "b76d601281994a35a89b735961cf47bd", "bd851dcca89a402e98e4f16fb525c9d1", "191ae516680e417dafce2e08a028049a", "b95334df3f354c0f9979ca35c564bf96", "4aa55a947f3244c7bbef80dd7546c740", "31e60593d28f43ec9a4ac51a4e124431", "1ca4793595f8444ab0271167129c919c", "dd17f742852844feb04eb7c11ada9ae8", "a9c85d12a0be454dbf35257bf86efcbf", "fb125274b3964174bbbade6434db9c9d"]} outputId="26cb1940-e837-432a-fca4-47c8eb05ffca"
ds=load_dataset("atlasia/good25")
ds=ds["train"]
ds

# %% colab={"base_uri": "https://localhost:8080/", "height": 237} id="4W0Q0TNdkO6v" outputId="9d5f8ee4-5f1d-463f-81fd-b3fef0fd758a"
ds.to_pandas()

# %% colab={"base_uri": "https://localhost:8080/"} id="CMuYfEjUkWQR" outputId="fd994568-542a-42a5-c608-10d4c75d56c2"
# we will take only the content column
dataset=ds.select_columns(["content"])
dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="dmHEsrw7kvvI" outputId="e1936437-fddf-408b-c405-fc395e277e3c"
# split dataset train/test
dataset_splited=dataset.train_test_split(test_size=0.1)
dataset_splited

# %% [markdown] id="3KR7pD9PvctB"
# ## Select Your BERT Model

# %% [markdown] id="7rFs6CtZvZji"
# - before data processing you need to select the bert model you want to train in huggingface, to do this follow the next steps:

# %% [markdown] id="ehpFxOsblvuK"
#
#

# %% [markdown] id="nFJZXNWg5RyI"
# *

# %% id="73TY6wv85XMD"
from transformers import AutoConfig,AutoTokenizer,AutoModelForMaskedLM

# %% colab={"base_uri": "https://localhost:8080/", "height": 209, "referenced_widgets": ["bd2a695ec137439d96120d99b1f07d33", "0b0f0a9bd9a04b21b769e32dccbacdae", "01c5ad80f31241f4853952574c46a912", "85618b98aeb0440195305f3a0aad1ec0", "4e754d61e60d42d1b2146e521463b2f1", "48a4510893d2450e9e30658a178048a9", "250b0f74fce04f38861111c8293fe466", "b606a3fe30e744e6859457d4f73caabf", "b370410a00814ad0ab59c293897ea0cb", "577df1ca491c4f89af9d85ee3521189d", "e1d12fa9d3014ced9396fd293cdc1b30", "16200b2e5481404e949f5fb02e20213c", "5788d134e7064de3b92b909057c5e65d", "d938f1cd3ea84c0485db1df2b1d4f8aa", "e9af088cfa0c4c12894d7fe8e0b5c38b", "77f9588c34d64895b0813a45f35fb082", "fc5a6d81b978493ba27142b362378c39", "a4610826e43c4b969f11bb01d6e3709a", "65eb01fa167b48d2aa6e548a88f55830", "cba9cf5c7dfd4702ad6331f103cae9d2", "e607b0b1ddea4cf9afaea0048f753697", "767c3b13901c4b0788c5f27e695f9a9d", "e5ddef8dac6341aea25ffeb9738b9cad", "e84c987204184e67b9259782b9ca532e", "487836906edd45589d6ff55833429337", "7e5a7a59acf34d99b5a308cae0aaddb6", "8ed0474cac3449dc9a387fec46f6c5a7", "afb6b0048eb449ef93b57c8705091a6b", "3b0aff75833e45ee8365cad1f84b0e72", "e1e5daf6f3ea4e34846f1bd92e0f390d", "519d57517e88477ba47cd50b2540d357", "f3b4dfd4ee5d49de89f8cedd9f358b51", "b599baa778cb45549d7b44c46e1df18c", "9acd805f64d940e2a71a0074c76b884f", "100deab44a734bb5b57146aaff3abcb4", "9c258ab75eca488e90d80c50e1937040", "df54e19f41f64578a7a6c7d9d006d97d", "78ad59548f424fc6a1a2d2d15b264431", "209cbd07f6fb42efb438502dcc8f91eb", "9ef09d5f027b44a7bd1e1c7da202697b", "fa001015de7b4e03a01f61f6ccde270d", "d8c160c3ea8543aa9af0b529ccda802a", "9d8cb3089b8845d8acc375c3625930ac", "2757e5db33d44dc98f82278e2ba3e444", "f5681eef8b184ba69f4b1912d5035a0e", "d3d7a16a4276424d86394a774177a379", "caabe3530b5543efa9012ebd360ad531", "80fce97b6fc348edab5913dd681b306c", "026f40ce4efd4a7b93b9b18590276356", "f03decc11e5643f6bddf7c9541fecad2", "7b6172fd86d84c0c97971105048457f1", "b86b73d1a6d443bb9ade96f667170d6e", "e2541356c22a470187776c5a795fe30c", "e4c3fbd5ff7f4ca1a96a2840d0943224", "38133574bb7c436792ea36ced2cc5970", "537c284e18fd4598b33fb53578ac2765", "31f5de68bde349d7847bac7c6f3bb80e", "b445d16aacd74e5985a1baf84c1370ad", "0085220a4bca4b6089f84fb5959a32d7", "db24a41815894bf0a828c1083672e01f", "f436c45d1301481eb00bd8683dc105ad", "e990fb11d2094cc2b118bf3bd15665ed", "d12311e99a144a7a992086d458e5bc20", "0ae64f5e41b444cb91e03ff478fdc670", "ae596e116a064a758a536559d3dbbe33", "15be9400d1794229ac8637f2e14572f2"]} id="aVM0LgV45mx3" outputId="0b44affa-ef9c-44eb-ed4e-198fa495a410"
model_id="atlasia/XLM-RoBERTa-Morocco"
tokenizer=AutoTokenizer.from_pretrained(model_id)
#model_config=AutoConfig.from_pretrained(model_id) for RPT
model=AutoModelForMaskedLM.from_pretrained(model_id) # .from_config(model_config) for RPT


# %% [markdown] id="1cGAN7djlSSK"
# ## Data Processing

# %% [markdown] id="tMFggpRulaJk"
# * Before we begin training our model, we need to prepare the data for our task (MLM).
#

# %% [markdown] id="c4ObEaEXDuGe"
# #### Data Tokenization

# %% id="INTpgBZE7w1x"
def ds_tokenizer(examples):
  return tokenizer(examples["content"])


# %% colab={"base_uri": "https://localhost:8080/", "height": 257, "referenced_widgets": ["052b1b32db604baa9473d06604849165", "b75497bc6b934e36ac72b2cf74bb64da", "63a2223896ce43bda457ef34f3ed77bb", "c13a8702bc1f46918f8fccaeee1000cc", "5f91c3140fdc4453b6b1e1c05a0a3c7d", "29b59cd1bcb2462da4a872fdb94ee02d", "31e16c2ea6ce44cb9d3ce513e7ae1cf4", "4acfadca63d84dea95dd54049072c0a1", "eb3dce7af3664badb2018abb5ef7c2ed", "ca5b2b19baa44863aceb882112cc264b", "8d0dd0a8664948409829f702364b2190", "80709a3755bf46dd958c3bd4c640edf3", "3ed59b95072f41d481d1a3d3d4136b16", "21543f2a84be474c9f6c414d808d2bb1", "a5ae6d7916644a268893f82bd8507c0f", "6b40e5449f7e4aa2aabc57571809f473", "8ca7c745c9404155839066393ffce583", "64554616d6a24665b8e09b015189f567", "480254c0fbf94688b891d07544d43ace", "1e9c7ac9d5a5407ca6b607c95cc35322", "029695e819f142c2b56338946eb5ce93", "a0ac06f9634942f4a768ef405cdab84c"]} id="7q15Rg7Q8IEE" outputId="802b2b02-dda4-45af-fd89-29302796eebc"
train_tokenized=dataset_splited["train"].map(ds_tokenizer).remove_columns(dataset_splited["train"].column_names)
eval_tokenized=dataset_splited["test"].map(ds_tokenizer).remove_columns(dataset_splited["test"].column_names)
train_tokenized,eval_tokenized

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="yH4R2yfv8m7C" outputId="0df87994-fb79-4a36-9e97-c019e0626a93"
train_tokenized.to_pandas().head()

# %% [markdown] id="We3uSCRHEeti"
# #### Conctenate/Splite

# %% id="If6WoWkQO77g"
context_length = 256
def concatenate_splite(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    if total_length >= context_length:
        total_length = (total_length // context_length) * context_length
    # Split by chunks of context_length.
    result={"input_ids":[],"attention_mask":[]}
    for k,v in concatenated_examples.items():
      for i in range(0,len(v),context_length):
        result[k].append(v[i:i+context_length])
    return result


# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["043599a1dfdf4eb4a3892b21fc31a412", "11b47886babc4b44a8adc1907836cfa8", "ab03e9cb6d5e4f078360ecceb4a26922", "7f56cbe1219a4094a1948b175fd8dda6", "51d34b6fb5ab48668c4a4f6f92a8862e", "c55d41ed6c2847fb972a63ddc30d2add", "d564a99e6ee54b67a048cabf08e301b7", "d975c42c76da4afeb416a2c0c6cdf8ce", "efdda3b0d7a549779f7b6ee7d49aa5c7", "c52fae507cca4b0b9acce2a2ce1239fa", "3ce954ba52aa455f8b98be5376eaa6db", "aab45541c1c643d49f64738d44bfed71", "533b0adcb1bc4d9392fc34e2e8ef0708", "e25ab66eea39442c94ff166830d2bfac", "801a7f81f3bd4fb09b1aa708131b44a7", "1365217c68a4483295f9ada065fdb19d", "ccfa44c5517a407380445dbaa1b83c8a", "de847b6c14ba42a299a7205b1b29ba94", "bb3b5403487242c09c635499f5097f7e", "e7d44237e33543eb8658830fa5a58814", "ef26b21f974e49bfaf8d49fb0361bf42", "82a00770c3ee409c8d5e93ed420561cb"]} id="w_0ppdshLRL2" outputId="86094c5a-556b-4030-cd94-df8960592d00"
train_ds=train_tokenized.map(concatenate_splite,batched=True)
eval_ds=eval_tokenized.map(concatenate_splite,batched=True)

# %% [markdown] id="GRLPAjiaRJNP"
# #### DataCollator

# %% id="ZE0obumoRVB3"
from transformers import DataCollatorForLanguageModeling

# %% id="XkrUP0tKUXcH"
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True,mlm_probability=0.2) # mlm: how much i will mask in sentence (e.g 20%)

# %% [markdown] id="C4hEW-hU6_pf"
# ## Training

# %% [markdown] id="F8iQ8x6K-fdL"
# > After preparing our model and the data for training, the **training arguments** component remained to run our training

# %% [markdown] id="GqgzkYrcAXfU"

# %% [markdown] id="ReKfFZS8AoTs"
# #### Training Arguments

# %% id="40nJqsBdb5CZ"
# trainer
from transformers import TrainingArguments,Trainer
args=TrainingArguments(
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

# %% [markdown] id="rWt-sgdPAuNR"
# #### Trainer

# %% id="dxkYB3Ula5XU" colab={"base_uri": "https://localhost:8080/"} outputId="133ce45b-d4f2-4566-da4a-0d141bf5e6bc"
trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=args
)

# %% [markdown] id="aPZewU0XA8i4"
# #### Check GPU & Start Training

# %% id="ON3wWzj_1Qzd"
# ! nvidia-smi # if the result is not found => there's no nvidia gpus

# %% id="o8IQPBOpV2sN"
trainer.train()

# %% [markdown] id="SO4kOIDfBAMe"
# #### Push Model to the hub

# %% colab={"base_uri": "https://localhost:8080/", "height": 197, "referenced_widgets": ["2fcb9de5192c4dae99835127a9f351a2", "e46d034102144187880383fa70fadc2c", "5e935080fe914cc18122b3daa9687b39", "d389e20e2c4c4ed0acf250c1e990cac3", "28a3af4f16a848b59ad065bcd5c9b162", "48dded9af6de4710acae20413e2f1ff5", "8b0b1f4f4bf74492ad6aff455d6cf4af", "5e38639c358b4487b89f8503494f2c7e", "c505e1f9075b43099b04213c4a1c8e25", "c4298223a25f485ea61b340f8255b397", "94cbf2e0a5d04aba89143b287b52d7a9", "7f91045974824490a31e16ce4caa9fae", "70764317fb3e4afb91920a69bcafcc9d", "dbf6d2b4600f4e6295deba60a84a6914", "86bb4a0f777e4b77a2b57a6e789101e2", "6b80c055c9d74873a184e7ab5ed0eecd", "4a34ada19aca4d76a2f14431db93ae1f", "adb16031128b4b4689e74c63bd78d744", "08d6e08bf8814e60abe9e3ad9ec1dd42", "2cefa353fc974226ac13339cfb182253", "331be4e8f2a543edb0f9023dceda9f7c", "d545bbd3c4a84eef9d66460af02c0e58", "7c8c192a1ab0400ea0175e1583bbf561", "00d0a457ba6146e7bfd463e6870adec4", "1950dca5caf246e8903efa8784818ae6", "ecb2ed2af9c5491fb419219aa44b86bb", "84ebce27547f4cfcb102947d758729dc", "ff5e48a3070e44d5984d96569476ad14", "12917e591ed045d98f77dd6f408e8a19", "387f4a5a5f064c84b354ed7ebd377622", "fd1528cac73e42f0b96e5d8cd45a4c59", "488cfd28668b47b5b965853a88660874", "bad710575ced4124aef7d9af584b1c2c", "fc3407f332f24552a9d7c6455582eb1a", "0aa9fe1d3ffc4d27b6a7367a73cab228", "065d450a098a4a3c906bd645f9c9dec2", "fc46f136ee6d4d3baa89dfdc71706978", "acfffe3376d34253b52109020ec60ea3", "2597536ddbaa4b4bb3f6382a0b4186a6", "e75fefd5cd76453b916d451b4ebd1ade", "9650d29f6c134a7bbe39cfcb2c85eb60", "8a4333e3367843ecb5031cdda99c3a35", "df9840566efa42a0b03414a419bf7270", "f6daa05b547a4c68af6dbee57a5cb3f0"]} id="KXvOyI-l2VfB" outputId="6c24311f-1f97-4574-c35d-866b270c2599"
trainer.push_to_hub("abdeljalilELmajjodi/test-bert")

# %% [markdown] id="DXxs6pSuEhyp"
# ## Show Your Work (Model Space) at Hugging Face

# %% [markdown] id="tg_36UkxGjh8"
# **In this final step, you will create a Gradio Space on Hugging Face to showcase your work. Follow these steps:**
#
# 1. **Clone the repository** from this [space](https://huggingface.co/spaces/atlasia/Masked-LM-Moroccan-Darija).
#
# <cneter>
#
# </center>
#
# 2. **Open the code files** and **replace the model ID** with the ID of your own trained MLM model.
#
# <center>
#
# </center>
#
# 3. **Update the example inputs** to be compatible with your model’s expected format (e.g., appropriate masked sentences).
# 4. **Save and commit your changes**, then save changes.
#
