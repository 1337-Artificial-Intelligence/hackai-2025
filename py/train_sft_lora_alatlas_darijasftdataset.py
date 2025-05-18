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

# %% [markdown] id="DtT-y1Yk3C_g"
# # Before You Start [Learn About LORA]

# %% [markdown] id="SL3hmar73OmL"
# > Large language models are large, and it can be expensive to update all model weights during training due to GPU memory limitations.
#
# * **Problem**
#
# >For example, suppose we have an LLM with 7B parameters represented in a weight matrix `W`. (In reality, the model parameters are, of course, distributed across different matrices in many layers, but for simplicity, we refer to a single weight matrix here).** During backpropagation**, we learn a `ΔW` matrix, which contains information on how much we want to update the original weights to minimize the loss function during training.
# * The weight update is then as follows:
# `W_updated = W + ΔW`
# * If the weight matrix `W` contains **7B parameters**, then the weight update matrix `ΔW` also contains **7B parameters**, and computing the matrix `ΔW``
#  can be very compute and memory intensive.
#
# * **Solution: Low Rank Adaptation (LORA)**
#
# >To make understanding LoRA easier, let’s take a sample example:
# 1. suppose we have model parameters represented by a `W (10x10)` matrix.
# ![image](https://i.postimg.cc/7LtmYJ1H/lora1.png)
#
# >2. We can come up with two smaller matrices, which when multiplied, reconstruct a 10×10 matrix for example `W(10x10)=A(10,r)*B(r,10)`.
# ![image](https://i.postimg.cc/3Ry9yr9g/lora2.png)
#
# > 3.This is a major efficiency win because instead of using **100 weights (10x10)** we now only have **2*(10*r) weights**.
#
# >LORA method proposed replaces to decompose the weight changes,`ΔW=A*B`, into a lower-rank representation and make W frozen.
# ![image](https://i.postimg.cc/YqNRLsNy/lora3.png)
#
# > the image bellow show the difference between full ft and ft+LORA.
# ![image](https://i.postimg.cc/QtRmcLnv/lora4.png)
#
# **How much memory does this save?**
#
# >It depends on the rank `r`, which is a **hyperparameter**. For example, if `ΔW` has 10,000 rows and 20,000 columns, it stores `200,000,000` parameters. If we choose A and B with r=8, then A has 10,000 rows and 8 columns, and B has 8 rows and 20,000 columns, that's 10,000×8 + 8×20,000 = `240,000` parameters, which is about **830× less than 200,000,000**.
#
# **Are A and B will capture all the information that ΔW could capture?**
#
# > Of course, **`A` and `B` can't capture all the information that `ΔW` could capture**, but this is by design. When using LoRA, we hypothesize that the model requires `W` to be a large matrix with full rank to capture all the knowledge in the pretraining dataset. However, when we finetune an LLM, we don't need to update all the weights and capture the core information for the adaptation in a smaller number of weights than ΔW would; hence, we have the low-rank updates via `AB`.
#
# **Which parameters we will target with LORA?**
#
# >You can target all model architecture layers, in our use case, we will target only the Key and Value weight matrices in each transformers layer to reduce memory requirements.
#
# **Scaling Coefficient**
#
#
# >```
# scaling = alpha / r
# weight += (lora_B @ lora_A) * scaling
# ```
# * Choosing **alpha as two times r** is a common rule of thumb when using LoRA for LLM
#

# %% [markdown] id="5Thjsc9fj6Ej"
# # Install dependencies 📚
#
# We need multiple librairies:
#
# - `peft`for LoRA adapters
# - `Transformers`for loading the model
# - `datasets`for loading and using the fine-tuning dataset
# - `trl`for the trainer class

# %% id="l0e4WG9Du7CC"
# ! uv pip install -U datasets trl -q

# %% [markdown] id="rfDzoGTXlfl5"
# # load dataset

# %% [markdown] id="ktQ2EHbnD4u9"
# > We will use [`HackAI-2025/Darija_SFT_Dataset`](https://huggingface.co/datasets/HackAI-2025/Darija_SFT_Dataset) dataset to fine tune [`atlasia/Al-Atlas-0.5B`](https://huggingface.co/atlasia/Al-Atlas-0.5B) or any other model in your choice.
#
# * **SFT Dataset Example**
#
# 1. Instructions
#
# <center>
#
# ![image](https://i.postimg.cc/hvTrrCkv/lora5.png)
#
# </center>
#
# 2. Conversations
#
# <center>
#
# ![image](https://i.postimg.cc/tRx2pTsb/lora6.png
# )
# </center>

# %% colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["d5eb45ddc65b4ffc82c630d125350118", "a0f31bdb5ea748b6b0b598f3b70e16da", "01f5a04af1dc4ef2a76c90682fb7244b", "c5d831afcffa4b7a9c5f2f66124dff57", "c29df7c316404c0a961e484e69194e6c", "d3c51b1ba4934c07a3506d169e8d9cf2", "a2bb53dd7eda43189578f103fd8ce1a5", "410a26b975b24dccb0fcf4eb9e47b35e", "31615ae659394aca8588b5357521ff01", "3b1f21b29856438cbbcd4e8019e81363", "e33ee1426b8b4e979f769f8614474dab", "8ec52e20cf2b405c8fccb552564dabd9", "bb610a51e1b64208962dc209a0ed5424", "2dd6f5716a914e059623e06028d795d7", "ca9394a223dd4c989e68068992b82b66", "ec469d73d9434e00ba72e8c13a0890bd", "e954552365414ce3bf6407e4f110e245", "74c0560119fe40e6bd8ebdaecb30a116", "2aeb7690853e4ec78788d5cca99558c4", "5c55edf3e2ce4dcdbfe0652311954070"]} id="Qqia5lcJch_n" outputId="6163163e-53ea-4bc2-80f0-8d3c983cc0a5"
# hf login
from huggingface_hub import login
login()

# %% id="-FuaIbfPlfRy"
from datasets import load_dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 150, "referenced_widgets": ["0570fe4e49524488a5cb192108b44334", "5e7c331e92044f7bbed3b4130ff113d1", "071c1ab76ac547aea98f254515f272ed", "cef5231a81534289813bc5dd06610fc3", "9d2d2c89950c422c8e6b451e332195c9", "3eb2f635deb94de2a7242874fd7dc839", "97fd339a18344a1986fd878d1ca980cb", "a20fed23398f4616af14a2f5aa8e7999", "78b07bc7f2054c9ebf044e5f44339cf5", "00a8f4cc32f04bb497c51bab14f544d1", "d54ef59090a44648bbbce208d4f49af5", "d4b2069e6b394dfd900e3285517b4be1", "37f1efd63f254380972b57cd667f137b", "896992722bcf41169e2eb6b6a4dad2b0", "2276ca2f27ea450b9907e86c5ac63845", "0c8d210035f145d4a61c067ae8912e28", "9bcf8d1f29474143a7ed90aaec9b1bd8", "8cfc80d59f16413585e4906c5529ee7d", "a8a854375bd34c869286afd867915fa5", "ab38e468c48947ea8508459a89f5fae0", "163948760b094011ae348c14eac572c9", "30627e72c9484fae949ee1b6b882168a"]} id="1_EmaKhclkRw" outputId="fc035361-756d-4105-b644-4d41d50b036d"
dataset=load_dataset("HackAI-2025/Darija_SFT_Dataset",split="train")
dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="Po3RNGMqqZmf" outputId="4a1fdce9-c43c-436c-f6ac-0f905ce51b8f"
# show some examples from ds
dataset.to_pandas().head()

# %% colab={"base_uri": "https://localhost:8080/"} id="XEidJOMqc-rP" outputId="11059795-c066-4f39-ce01-475664800b02"
# remove other columns and rename conversation to messages
dataset=dataset.select_columns("conversation").rename_column("conversation","messages")
dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="VSmdhSJHtath" outputId="00cdd071-bf5a-41b3-8ac4-0d67916d5ed2"
# show the first example of messages
from pprint import pprint # pprint for pretty print
pprint(dataset["messages"][0])

# %% [markdown] id="UE4bU4swrKlr"
# # Load Model/Tokenizer

# %% colab={"base_uri": "https://localhost:8080/", "height": 273, "referenced_widgets": ["4d8f2c42d2ca4d93be9b762225357896", "6d5c7898d08346b5b32558435c0d1e8f", "860f433818e849d58af50ec5dfc0c18d", "d8018804fe1642c6b1b730b506bac837", "063b593faff44817af382426550c8ba6", "acadfe136281426b87305849af54d657", "8e17bbfec1df463199e106d773a2f2e2", "f7687cfbd1f44bfe98b2a0fdf1279e5a", "4bf4becbee364fe8bdb19c585152a6c7", "555fcaf52319446693c28b2ba23614af", "8e431f6fe3324a0a99a664a906cde296", "c7e983c8781c4cebb55995299c658f5e", "6e8d4e5106144a92a33b7675a33d7639", "f01cb2fd25324253b86386aecc9f40d1", "24fc541f8fa340038436e6ebcc1e4da6", "83a3d35763ff44e49ca399bc8e100bda", "a66374514704469db70c7a7bb9cd656f", "159940d72a814bb5a9742ce64695c816", "5fa4c2148bdf401f900f97569276365f", "219972ad7f7e49d199632776b1fc40ed", "fcfadd25f09c423f99913616160254b1", "cb0a3d34fee74bba8d9d766ebf282f4d", "329a89b1a20547cebbeee8b0ef151c99", "44614437ce84435da42aa8b4e9777811", "4553ac8daef04060882f226abb43bb84", "9719016c8cc04805b947ae31962afb19", "76605f8b735d4a8ebd80432848b4196b", "569220b9398b438aa05f9de91838ac0e", "763f2208f9404eb9808a0e33bef2dca6", "1ffc5d0d2b654346b44d7401961b39bb", "8c275d683a704ba28e839131294fa026", "e07c164f5755427dae398ba2448fedcf", "cff41ba7dd0646a19bc6428f3a4b0886", "a94f179e31354844a778954b7366007b", "5891bd770a20486aab2b68efdd5cc53a", "d356e2b8baed43ae9d835ecd5ecdf1ec", "686bb048d4b84bc6bdd3b663f64b7533", "f883f52303334ef4a7c195b130229934", "f0d4a3644399468a858acf0a09392329", "91f5fdf24648471195411ab25cab3c32", "abbb755913c3498b864fa38be6cf6ebf", "ca413683dc624e54af62ba1c9864865d", "91741e059aeb4e97bbcdb7f21c87ab0f", "b7e429c6ca9540b0abe302df31cde562", "c0a65fb5616c4addbb6ed377d863c683", "3c1eda51eb1449c68bb6bf5a59b839e1", "ba7536cdd893400a9e63f07ce3339855", "58a67a356a384d99a6d75bdfb8804e9c", "c4aec965030943788cc44a70b3fa44e2", "8e7be34cee594488a1a255711e493a7d", "56ac0f169131412fbf9cd26882b4b88d", "57c55871c4764b4192a68c161c236dd8", "97c6484f324944919a7daffbba96f474", "bc6d814177fd47188185589f94e96e04", "e5ec05717d314da29a50a73cdb2a1250", "3d974c784a374963bd32c796cca35f2e", "2b36dcc0068249639f681024cbc7b85e", "bdf578523b2b4715ab7289df1612f4d0", "a951f71324c34a058110c130cbfc8e14", "f2bd64fac4364bec8ee98ca6047201e3", "ce0f45660cd549078d4928668f3a9f2b", "9c3e18afa8f549e9bc819680f6f8cd7d", "cdeadbe2b9cf43edad96ad0896e355e0", "55bb956e52c34b779f914610663e8b09", "2d97f3430f514fe08ad34f3f2588d09d", "eaac8524ee18483d995203dbd8bacf46", "e350484cf4dd45ffa1011988d1806b6c", "28f02ab6706b47269a1caf6f38bd4456", "05fac7603e65490098988a032fb39551", "c4183000ee1740ca807c4c06eff3ca23", "0421a28901814c94ad1803a70b921728", "0b796676ed55411e9695bc24bccd9477", "604382afb6cf4dab988d56f09390d795", "5591fba2d31a481bb08162f4c13b549c", "a46743c01638479cba5ce0fa7fb19c5a", "e14bfa612ed4472aa7c13c270575371f", "da8c3eec28ef47ea92d6c9567e0f54cd", "cbbc5acf600c440c857978e4cb0726a3", "830a6a9b9cbf4c94ae90178963f4d9e2", "92e9158759a84edeb154bd870faac78e", "898b1fd980c64683ac9d47ec26d2e0be", "c5e3b6f3870a43a087a1e0c6b74231f3", "a417c354bbdc494ea69f95fd338d053f", "9f0d24a234be442a8c8e4b50d4475c9b", "e76ad573fb174c66ade6f08773e47b34", "c7a94372afef45f9ac6a36f136e0f819", "2a0f3f77d2994f0aa196644cac715ba8", "6222a6998f5a45b2be6db86fe778bead"]} id="O2L9E_NNt2Fc" outputId="709a4f24-98d5-43ca-d2c9-6641325bcbef"
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
# select gpu if available
device="cuda" if torch.cuda.is_available() else "cpu"
model_id="atlasia/Al-Atlas-0.5B"

tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id).to(device)

# %% [markdown] id="joPsnUPPrO36"
# ## Model Chat Template [Test]

# %% [markdown] id="oNX9-z11HYfO"
# Instruction fine-tuning involves training a model on a dataset where the input-output pairs, like those we extracted from the JSON file, are explicitly provided. There are various methods to format these entries for LLMs.
#
# <center>
#
# ![image](https://i.postimg.cc/J4CKnLXk/lora7.png)
# </center>
#
# * Comparison of prompt styles for instruction fine-tuning in LLMs. The Alpaca style (left) uses a structured format with defined sections for instruction, input, and response, while the Phi-3 style (right) employs
# a simpler format with designated <|user|> and <|assistant|> tokens.

# %% colab={"base_uri": "https://localhost:8080/"} id="aT7bF_Z6e4ib" outputId="0f6e578a-b9da-49d7-aee3-ec0d4dce334b"
# test tokenizer chat template
result=tokenizer.apply_chat_template(dataset["messages"][0],tokenize=False)
pprint(result)

# %% [markdown] id="mP9C08ayrcuS"
# ## Test Model Before SFT

# %% colab={"base_uri": "https://localhost:8080/"} id="L0nwKqrRf41_" outputId="62ed7279-25d6-41fb-a03d-eb6e23f123d0"
# Generate with before ft
prompt="السلام لباس؟"
messages=[{"role":"user","content":prompt}]
formatted_prompt=tokenizer.apply_chat_template(messages,tokenize=False)
ids=tokenizer(formatted_prompt,return_tensors="pt").to(device)
output_ids=model.generate(**ids,max_new_tokens=120)
output=tokenizer.decode(output_ids[0],skip_special_tokens=True)
print(output)

# %% [markdown] id="DtQ9xQPkC83J"
# # SFT + LORA

# %% [markdown] id="wxp4XDBWrkL9"
# ## Show Model Architecture

# %% colab={"base_uri": "https://localhost:8080/"} id="9l3krCoOfZnS" outputId="7b264b93-e65c-4dfb-c2b0-b2189258f144"
# show model architecture to select which layer we will apply lora
model

# %% [markdown] id="P-0NoByBryZZ"
# ## set LORA Configs

# %% [markdown] id="dJJKDvqoB4ZM"
# * `r`:  This is the rank of the compressed matrices, Increasing this value will also increase the sizes of compressed matrices leading to less compression and thereby improved representative power. Values typically range between 4 and 64.
#
# * `lora_alpha`: Controls the amount of change that is added to the original weights. In essence, it balances the knowledge of the original model with that of the new task. A rule of thumb is to choose a value twice the size of r.
#
# * `target_modules`: Controls which layers to target. The LoRA procedure can choose to ignore specific layers, like specific projection layers. This can speed up training but reduce performance and vice versa.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="sLOm6yXj8F88" outputId="d87cbb5d-2695-430e-fd7c-b1fc17656893"
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
# # TODO @nouamane: add qlora

# %% [markdown] id="dSMN78e3DNjO"
# ## Set Training Args

# %% [markdown] id="hB6wq-T5I3MP"
# * **What is gradient accumulation?**
#
# > Gradient accumulation is a way to virtually increase the batch size during training, which is very useful when the available GPU memory is insufficient to accommodate the desired batch size. In gradient accumulation, gradients are computed for smaller batches and accumulated (usually summed or averaged) over multiple iterations instead of updating the model weights after every batch. Once the accumulated gradients reach the target “virtual” batch size, the model weights are updated with the accumulated gradients.
#
# <center>
#
# ![image](https://i.postimg.cc/x10Rv3tj/lora10.png
# )
# </center>
#

# %% id="hF51h6lH8gRd"
from transformers import TrainingArguments

# %% id="NfalQe-2-WnW"
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

# %% [markdown] id="jfD88KB9DZ_i"
# ## SFT Trainer

# %% colab={"base_uri": "https://localhost:8080/", "height": 182, "referenced_widgets": ["e09cea7c456044539e280175458e3c12", "6f828333c3d74ce8a90885cfc52a2b23", "2e18862b849a484386b9dc204afcdd0c", "05ce31886bfc4c7c8083d804717c88ef", "9ec0893b581d4e62a1dab7fdda160aec", "50ec517b44d54ecca62a66e2191a9e42", "0f51563da5be4ad39f3252a868ef1f64", "c5d6c229e355453cae01f7f2b70f057e", "0515a61e54b24f35b65bc551aa13f831", "fe46304623d64a938be479fc656bd0f1", "cdd69db7241047819631d6a9bc4759c8", "455eb1e1fba3480589fadfbe5c0401de", "c11135b80f204195812a5a42cb501ee4", "66c8153ea8224329a172828d12a6aa28", "80dce1831fa1492e90cb0a2390deba66", "291300ec8a794b3d932903f9bff6238d", "5bccf7a87d6247b6b956fae0a7301421", "1c83f65e81f14471a4237c224e832087", "249b636ce769494e81eb9969a08ef0d2", "c52d1fe9f9944731aa02ab1b59161ebd", "bf15cc21751f4a76a4b5c030dec45658", "ed6626f3013f462fbe8a532a847a9930", "d08535290a1a4387b887510535059f56", "1e4c1ea498e3498991b397c80f08e62f", "e9c24950bbff4e259db0b14679e77a1f", "6afb48105f844377a7ccf4d8d6032f84", "2810cd62894b4de9aa93be6274891817", "2a3a94cbc43f429598516be72f0a1437", "1a998224d6d848b3ae2ee896d1c26770", "d7ae76e956e04bd18de6d21bbf584d42", "152c0347e8fe435fa52021fd970a113f", "f8db48a4ae104b10a943cced2598a6af", "af97b5e15e4b437493b48e776590cf0e", "0f03f847460f4ff6a904e0a4c44550cc", "f6e9469d2c0942dd93cc1b1856925529", "a23e7ec899914ebaadb41627dddd5a81", "77bc694d5a2c478ab19108bc511404ee", "b5b8e4d8f64f449088325f715cefe507", "b2090d3f60304e1d9cce091d2ef282d8", "1dad821174ef4829ad6d060e32ed39f1", "81f693f2e13b4b19b2aafc8fe2d8005f", "f9db19f80ccf4af4aa914ef7af86b252", "c5ce0db140b144cfa4c7f1f8cc4c662b", "8fe2e86f93aa4d40b71dd8d0da8496a9"]} id="34bqZhsbClcS" outputId="15fe5321-847d-481e-e20c-7afbeba8ad8e"
from trl import SFTConfig,SFTTrainer
sft_trainer=SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=args
)

# %% colab={"base_uri": "https://localhost:8080/"} id="b0J2JfX7HBMO" outputId="41c4f66f-b911-481e-c10e-1104dc4ca2cb"
sft_trainer.get_num_trainable_parameters()

# %% [markdown] id="jI-hnCT9sBqc"
# ## Start Training

# %% colab={"base_uri": "https://localhost:8080/", "height": 704} id="xMyODhQhDo4s" outputId="73cea4b0-da6d-4c0f-f43f-320f4801342f"
sft_trainer.train()

# %% [markdown] id="PhgIPNIhsIkA"
# ## Test Model After SFT

# %% colab={"base_uri": "https://localhost:8080/"} id="y-qJWcJUkBzK" outputId="f6ec4ba3-ade9-4400-857a-b3d2291d032f"
# Generate with after ft
prompt="السلام لباس"
messages=[{"role":"user","content":prompt}]
formatted_prompt=tokenizer.apply_chat_template(messages,tokenize=False)
ids=tokenizer(formatted_prompt,return_tensors="pt").to(device)
output_ids=model.generate(**ids,max_new_tokens=100,
                          repetition_penalty=1.2)
output=tokenizer.decode(output_ids[0],skip_special_tokens=True)
print(output)

# %% [markdown] id="S3zwJTi0sLhM"
# ## Push To THe HUB

# %% colab={"base_uri": "https://localhost:8080/", "height": 232, "referenced_widgets": ["b0f4c3eaa51c4e88921323fc3964736b", "f8a206936264409fbf79ae3ceee2c1eb", "ca18412b0f1946fb81e6de6b2515b875", "a0dd6a92d18e406f9504507c95563f3b", "f1cd5b6af62547d9a60b4cb3a125fdae", "533c894f09f740f5a04aca489b4216fc", "c278405cec3842f1844e7ad858570e46", "0f761c2e487d4e15b2ef5de867dc293a", "55d61306731e4edbb04640050dcfc4be", "30613d8bf4ab4ab7b2fff6e76cb78017", "f797ace61cfe406e978fab512fb13758", "a95e972a6e164631b825ada1e768aa7b", "de529642518e4bacae515885d5093a80", "38375d2db12b41b68e1269cf6da52c2c", "3670a7de5e2e4d4aa025cc2053fc2a08", "00e6acc2d02d4cb8af04248a4fa4024d", "be78985f5a1945a5b64429219857f777", "6d4d1d06dc6540a0b6a3c4aec1a46c6c", "9fc5b59371d6487897ac298895dd5fe5", "719a4d5dfc7b4a778bbaf5cb177e322c", "828f340842074ad0a661f073320eb1bd", "4d4068417ec24b57a199475f788b11e5", "5eac2d8f1c7e42b39a3ca3be0dda01f3", "390e8ba6c26348b1927b103c038d6c78", "a5ba09ad4af84a8e8015339976678638", "7197c06b4b8c47e7a6a7f73ea27f218d", "57f5527ffd444d7ca47a5780faf6a8c9", "9c5b8dd8bde944eeab70f6de29be1762", "6d098661f35842e9bd71a497a4a9107b", "877a7b8da4984f308533ddd4c9e0504b", "3a5f6992cf2848e5a074916dfd489298", "41464ab171734082899180062772ad9f", "6b36c7ed6e61431b9c3031bd73935c28", "702325f1788f43f8ab2be3f0e610c1b4", "1e8063d1c4944515a9336bf33a8ae643", "ad21c24911f54931bb230d59a88b2750", "9d78666e01d84921949fddd82a964861", "28466df8eb48401f873aec2d17bcd61b", "c142b771d90842d6ad0017e1d6c6358c", "a271f17c2bf84dd590c09183ac87cece", "69f2d94536124017885a79d1444b3d91", "bc3d5790620a44f2ab42ddca57b3574f", "e8f1731efa8c4dab91a2651dee89f9fb", "0e8ee52ec4944d09a10e83b4c912f311"]} id="SrlfepdjF1ib" outputId="5fa153ed-cf86-4a4f-b47f-9f24cf10ff33"
sft_trainer.push_to_hub("abdeljalilELmajjodi/alatlas-sft-lora-gra")
