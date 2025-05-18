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
# * **What is Embedding Model ?**
# > - Processing text for NLP tasks requires a numeric representation of each word.
# >* Embedding is the process of converting
# data (text, image,...)
# into vectors representation
# \begin{equation}
# data \rightarrow vector
# \end{equation}
# > * Embedding capture the semandtic reationship between words.
# > * words with closer meanings or relationships are closer in the vector space than words that are less related.
# > * the dim size of an embedding vector is different from model to model, smaller vectors (lower dimensions) are more efficient to keep in memory or to process, while bigger vectors (higher dimensions) can capture intricate relationships, but are prone to overfitting.
#
#
# * **Static Vs Dynamic Embedding (Attention Models)?**
#
# <center>
#
# </center>
#
# > - **Static embedding:** Each word or token is assigned a fixed vector representation, regardless of its context within a sentence or document. These vectors are typically pre-trained on a large corpus of text and remain unchanged during model training or inference.
#
# <center>
#
# </center>
#
# > - **Dynamic embedding:** Also known as contextual embeddings, these representations are generated on-the-fly, taking into account the context of the word within the input sequence. This allows the model to capture the nuanced meanings of words based on their surrounding words.

# %% [markdown] id="iYKHGfERxtfu"
# <center>
#
# </center>

# %% [markdown] id="efnr5RusywjH"
# * ***BERT (Bidirectional encoder representations from transformers)***
#
# > BERT inspired from the Transformer architecture introduced in "Attention is all you need", to become an encoder-only transformer that can produce meaningful representations and understand language.
#
# > In the pretraining phase, BERT is trained to learn two tasks simultaneously:
# 1. **Masked Language Modeling**: is to predict masked words in a sentence (I [MASKED] this book before -> read)
# 2. **Next Sentence Prediction:** given two sentences, predict if A came before B or not. The special [SEP] token separates the two sentences and the task is similar to binary classification.
#
# * ***why is BERT important?***
#
# > BERT is among the first instances of Transformer-based contextualized, dynamic embeddings. When given a sentence as input, the layers of the BERT model use self-attention and feed-forward mechanisms to update and incorporate context from all other tokens in the sentence. The final output of each Transformer layer is a **contextualized representation of the word**.
#
# * **How BERT Embedding Model Works with SentenceTransformers?***
#
# >SentenceTransformers is a framework provides an easy method to compute embeddings for accessing, using, and training state-of-the-art embedding and reranker models.
#
#   
#
# 1.   Compute Embedding
#
# 2.   Calculate Similarity
#
#
#
# * ***Steps To Get Your Embedding Model***

# %% [markdown] id="elJ3Ujuv6W3i"
# # Fine-tuning Embedding Model (SFT)

# %% [markdown] id="IxhY1KPp6u1A"
# **In this stage, we can fine-tune our pretrained BERT model, or another selected embedding model.**
# * We will fine-tune our selected model using the **[SBERT](https://sbert.net/)** library.
# * To fine-tune an existing embedding model, you can use **[mteb leaderboard](https://huggingface.co/spaces/mteb/leaderboard)**.

# %% [markdown] id="pn_E7tzSaFk5"

# %% [markdown] id="HZC5tJWEZ6Nc"
# #### install requirements

# %% id="tP51QpbF6uKD"
# ! uv pip install -U sentence-transformers datasets

# %% [markdown] id="Kmibqr49x6tT"
# #### HF Login

# %% colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["812c32c719954b0abcc97757873a111d", "7bba1f66c25c492694f28b685db2accf", "f9c6ab5c9918486c9e134d47e6494e5f", "5143cba4c69d4d6a9fc3de3708f51bf5", "98baffbba7b9407fb44b1b5063deab67", "c02a67efdb754e2089168561a7973a8a", "79685685042242129405ba42f170296a", "e8a872b86e394428aaf7a5dbfac7864d", "ddb37e9bedf04fd191bb555b722a8764", "3640cd8ec38542b39cf233447a451569", "2a90e36c7a5e4394a6465f931db519b0", "22908e70cd7342fbb0651e3f8ec49400", "e6c81854a1134d5c98697faadbfb764d", "a2fa551acf5e4c01a206716da3e8376f", "1a36205cc6704720910b8939d17cb09d", "804d721151974c279ee396b30861fbbf", "4baa95ed918a4edb8e1e5f50504f02df", "67f9997e13b543bfbe63bc6de270ef87", "ce26aec134304264951b5fa56ba19c1a", "a24ac0281be04a30a3f540854e03aafc"]} id="ALMt-CcEx-dY" outputId="b48e5626-32f4-456e-ba5f-b1665b77a74b"
from huggingface_hub import login
login()

# %% [markdown] id="AuNoBqYMXFwV"
# #### Load Dataset

# %% id="CuGmsLJAXIni"
from datasets import load_dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 538, "referenced_widgets": ["b926c6e42de243dabd2b10c8a80f6444", "f982415528bb444d8d45abe1a03e74fe", "3bdeb510294e48b4905938699f6f5030", "a73fad23271740c39111b7199942cf3a", "b9f5c5155eaf4b428e2508150aee2b0e", "2b9fd127b62b4f829a2d57df9f76f962", "65d090fd4a9e4ee6bc3e939d6db9dd35", "dcebf459e9504627b335cc31ea726083", "e438e4c0b5b84c168fcff528a5f916f6", "8b56bc8382804c9794a96e6e394024f1", "4438377728744f25af934db26d8bd842", "03ed8b123a074e77bb5b4a2acdc7b388", "5e6d04d3d706467780955a10fef141d0", "b8e3bc9c2ca64c97b04a8fb6e6870524", "bfac66d9b3ab4403b1bcacf9c9746e4d", "cccea88fadbc4d71856b1ce7b258ccf2", "917590676772438b87283a85c9217aee", "e4b39fac7e14424ab0531f507e511e93", "1880845ff521440e8af6a7d9bd6ffa2c", "0e053ff8efed4718b1c6a66829e94536", "b2e5053e7df04a73b1e22181e0e8a48a", "c164ac66a63a4cbb8d0ace5e6f4f9c24", "dd8e1e6509d14a6c8d6ccf94b87b99c4", "91d523a38c2a4712ac084318fc39f8a1", "f168fc90877746a4a0e6466c5eddd276", "2a2c1532482f4c7fafcfe7f0ae107352", "020b45415eb14a529ae3dade2e6ec30c", "6534db6a86164302b6fd7d48dc634151", "038d3003881b4e1d83741650699e4774", "53aa7cb7d27f40838e501f35572d74cc", "5c85bc799602448bb75829ca7d74cd90", "9ca3b9e640914d6f96348657008c0970", "cb005077dd814997a96301862f4fbc14", "75d33bd609234fa7868f2abc4861dff2", "3fc8f5f76dd8443fac13c501a351f7cf", "d99fc5bea3f34236bc78a6fce58eb6ce", "e8de7c370b9e4b289c174767e7fa305f", "e74d158046264b82bd0a7a0b32dbf47c", "934a63b781e641ccae7f7e44ce184e06", "327ba73bbfab457780291b50c2f60c79", "1c98e679ffb14d0788c66446d4374ad5", "f8329bb1d5b046ec9c8b7f135eca81d7", "59f773b3b4d143bf8b8ad527a2ce4f03", "811b821c96494d258270170e5bacf2d5", "9b477ec1117a4d279cf8bceca85a3f69", "6a6a2a95db984e21ba52a26aef936ddb", "753728ee86504bc683f56a11d343c0dd", "65a8b407ca2d4d8488dfc54f53b6d4c5", "94dcd9fa73394f028c6a144da5b11dad", "953cd59db7984a1687892c660278c0c1", "0fa052ae994b41fd80f0a7d899cd0608", "e0eb0f17d7c54ea7b08be69901f67f74", "82907efd149e4e19bac74a4b5de2b00d", "e21bb99e92c64954bcd899623d80eb26", "f309182da8354c988601ce7885bbe697", "901af12513d949239af5ef3f40f816c2", "52b6443f6b044455bf675d7df3a8dfbc", "d1fb0dec37e84f1bbdcf2f6fa4a1aa4a", "49e1b7eb378a432fae197b4dfd4cab61", "cc811e3436334f39857ef5cb731dfa7d", "1a876e7efa904bc883a725f3bf689080", "f7d49c1ebbc445019cdaea18c11a070f", "994a740dbf4944b6a6cc82dbb7c09a90", "eebd24053adf4224983d7760e8962330", "66b13329f9aa4f23922478510eba3f5a", "36f64606829c426f93929f9997ad8876", "b2bdf4761eab45c488f1aa4920e0d58a", "c72289c28ab94f158b1ef311218891cc", "3406c86ce31b45ef804b6df6c8022d07", "5961979313f8444ba8c2da28cdaf5435", "e328d85beaa5447d9c540911f45e2f0f", "cc6abf66b97341719fb81470b3e1384b", "13d2d973282644bb972f37955e23213f", "7b80c1cddb2e47d5b06aee9d9a97acd0", "cb9798a4bcd544aaa1cb40c98dcb8169", "f20bfb5320004653bea611dcd89bc723", "687d9f1d1cfc43968f62a95693c81c0a"]} id="83ieQsI1XWHw" outputId="7ef9631c-0098-41f2-871b-d9bedd50459d"
ds_id="sentence-transformers/all-nli"
ds=load_dataset(ds_id,name="pair-score",split="train[:100]")
ds

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="IJaY_inoYCUG" outputId="e3304b2b-46b1-4af5-c347-c775f5f5195d"
ds.to_pandas().head(10)

# %% [markdown] id="KIPcI1Lylqcf"
# #### Split Dataset

# %% id="TgWhl2trlsym" colab={"base_uri": "https://localhost:8080/"} outputId="aa5b2384-b2c6-4e10-c1e7-a4628ee63697"
ds=ds.train_test_split(test_size=0.2,shuffle=True)
train_dataset,eval_dataset=ds["train"],ds["test"]
train_dataset,eval_dataset

# %% [markdown] id="sUjEE70OaQzC"
# #### Load Model

# %% id="2p06NXKaaSZx"
from sentence_transformers import SentenceTransformer

# %% colab={"base_uri": "https://localhost:8080/", "height": 281, "referenced_widgets": ["12ed114377ed4ed6bec4ce782d071026", "adee7291fe294591a7adf9508b5b0946", "2565717fc8534201bca38d32c757ded9", "f618f7650f5c45f7855abc093e8234d7", "f50b34a1271d40348d233e385f0666a5", "0e054260cdbf45eeba1fa3801a4bf045", "14c794f972134dafa29b9effd31fa132", "4d4bbda036e0451e8c62cbff7cd63d3f", "96e7344f70d74269895a580e4a4cf896", "b1cf5c94669a4a2aae571571138f985b", "6b5c95da2c0546dc827e7d1b08894019", "6a45b4614ac0427aa917d401d0745846", "e06a1e82331449db933bcb3337a8daa2", "5d9bf976cf304e76b6c8318d52325bc9", "3d1ff294288c4e3c98104a58b7f76c3c", "76fa330970304918b33f8b3704476aad", "c36ab835856340d0b577ff0e2bcf04fd", "a00133728b2e433f810929f4b8ab58e9", "4fde42a3170b440d9974f8f538f12152", "1bb844019b954c29aa16ecb17ade8167", "28d6b55f34b541da97dd13e2874f593b", "19416469b8004c65bde6db43556d7166", "d38899ab680a48dca922ab5a0311087c", "21c9910b9ff14cf8a680d6e1b4ded2a3", "7d28f97e7d1341a7bc889351c37ab255", "3a3ffd81aa9e49df84200eb91b42417e", "baa17ce7275e47a8889864dda4e94c74", "578d345e5ea94b32b23ecc857a242208", "c5008ccc1e8449ab883c64d0ca7459e3", "b7d0c2da3b0b44f7a53252f44dd1309d", "162c0bef8c8c4c7a960befa7af047136", "69ef9e95478143559abf37a39b8e9c3a", "8f2779d609b149dfac47ca6d68cc4d05", "5a6e1141e02f43fa972ced2a4273652f", "5ad600f1644d43e18746bc841914a096", "f9f9710423c1457c8d1df3f52fd32298", "9edc443fb9264eb09bcce32bd55c9c8e", "15cceec3e0534c5cbe4d0bcce8d40262", "c7ac736e1f2a4e09823e77a8a8c17ef0", "8af9bf199dda4677b719addbc1e1a7b4", "b8adfc28a3a84888a67f08d3285d1308", "d601945ea2b649939a181b787ab28bec", "f7ab24a7782b4d85a00077bf36ff00e1", "dae5308d00d74d8889c66dd137119275", "081f6a8c1a25440b9eda33e6a535f003", "e6f2c8c193074bf094d04e93d4c65523", "329e10ed59b44746af50eb0d54ac5281", "27a70b599c1b4a00bc63c18cbefe2464", "e52ea37ea3f14f87b7f1d5ca9800b51c", "a088c5f4923d4d7d8bb0078033d6029e", "50064ee37dce450bac31466629aa1285", "90c13bdc80f147fea3077db2f30fb230", "bf0a845eaea74fad9b6fc2a0762a5f1a", "40c1b956cc634a9888d5ae5694846ac6", "9148d946953849408dc40940d1b1949b", "7fa6320ce8894b419a5200dea96391eb", "a8b3bbf0e9094f3da48235286e6a9848", "e3be62f00f2c49438651f90d2bd38a86", "9f18e789d69b4d0a855f5611797b1753", "a26b6d5c65cf42398a29f5f81e56f889", "e63371bcd85843e592cbd8ff34927495", "a45c4a1ee70a40a5b26a6a5f846cdc60", "a701297cce8e418c81b0c20bc13d8edf", "0f79156f6cfb4bbab63f313bc35e580f", "c9065dc01be842c19f662a395e52c77b", "5c7f71b03ab948d3992b4b38e870f936"]} id="vksXpY9pasWm" outputId="749ed821-e8b1-47d9-d3a1-dbc6deeff53b"
model_id="abdeljalilELmajjodi/model"
model=SentenceTransformer(model_id)

# %% colab={"base_uri": "https://localhost:8080/"} id="V5NKxzQWbPIN" outputId="d2ade6f1-aa1e-46af-e37b-b57af40d7b0b"
# test model
sentence="hello world"
s_embedding=model.encode(sentence) # calculate embedding of sentence
model.similarity(s_embedding,s_embedding) # calculate the similarity

# %% [markdown] id="Yqr6POfKdDUM"
# #### Define Loss function(s)

# %% [markdown] id="T0IjVNLveCIs"
# [**How to select loss function?**](https://sbert.net/docs/cross_encoder/loss_overview.html)
#
# <center>
#
# </center>

# %% id="E7EaEKO4WmCe"
from sentence_transformers.losses import CoSENTLoss
loss=CoSENTLoss(model)

# %% [markdown] id="FRWp0TRsfzw9"
# #### Training Arguments

# %% id="T8noEfUfffER"
# Specify training arguments
from sentence_transformers.trainer import SentenceTransformerTrainingArguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="./result_model",
    eval_strategy="steps",
    learning_rate=5e-5,
    warmup_ratio=0.05,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    save_total_limit=1,
    bf16=True,
    fp16_full_eval=True,
    logging_steps=5,
    save_steps=10,
    eval_steps=10,
    report_to="wandb",
    push_to_hub=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    logging_first_step=True,
)

# %% [markdown] id="p9mmoMJBhu4D"
# #### Evaluator

# %% [markdown] id="dx33NNKkjrew"
# [**How to select the right evaluator?**](https://sbert.net/docs/sentence_transformer/training_overview.html#evaluator)
#
# <center>
#
# </center>

# %% id="G8gieTPRhwYt"
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
pair_score_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="pair-score-evaluator-dev",
)

# %% [markdown] id="1CId7NTrxkzQ"
# #### Create Trainer

# %% colab={"base_uri": "https://localhost:8080/", "height": 17, "referenced_widgets": ["bf5ee23be2044866ab30a9b8a4898690", "e6841934a46345ebb8fc6288242dd2a8", "a1c19c9d18fe4cf29d094b1d615e4813", "632a9a606d9e4167a13ce22f1a1828ba", "cf60e9839b814e74ad9fc6e84b582a8c", "b21a55a0aa054ada8a4b72cd26aecf20", "6b7524f54b12497db231bb49f7e6ac10", "80c16b2bcffd472fbe877e628625dd8f", "136477affdf242ebb86af79cc8eaeae7", "e2997ddc7a4f4275bd98386e6801be63", "bef5d7711b7845fc8b79c32ec20cb069"]} id="msECnMGExoiv" outputId="3140be71-c0f2-4266-bac5-a834fc81e4c1"
from sentence_transformers.trainer import SentenceTransformerTrainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=pair_score_evaluator,
)

# %% id="NuTKN3GvybpB" colab={"base_uri": "https://localhost:8080/", "height": 422} outputId="4e1cd6de-157f-40d7-e1eb-f8d5ce3d605b"
trainer.train()

# %% [markdown] id="LgC5yk4qyrXc"
# #### Push Model To HF 🤗

# %% id="KS_WXQdoxdXp" colab={"base_uri": "https://localhost:8080/", "height": 86} outputId="9c2261b7-ea25-41eb-d987-ad2cb9e2c9ea"
trainer.push_to_hub("abdeljalilELmajjodi/hack_ai_embbedding_model")

# %% [markdown] id="zL9UiCCF4vzD"
# <center>
#
#
# </center>

# %% [markdown] id="_WRaZySh4M4u"
# ## Show Your Work (Model Space) at Hugging Face

# %% [markdown] id="c9avqG_z5Cd7"
# **In this final step, you will create a Gradio Space on Hugging Face to showcase your work. Follow these steps:**
#
# 1. **Clone the repository** from this [space](https://huggingface.co/spaces/atlasia/Masked-LM-Moroccan-Darija).
#
# <center>
#
# </center>
#
# 2. **Open the code files** and **replace the model ID** with the ID of your own trained Embedding model.
#
# </center>
#
# </center>
#
# 3. **Update the example inputs** to be compatible with your model’s expected format (e.g., appropriate masked sentences).
# 4. **Save and commit your changes**, then save changes.
#

# %% id="qNmUedSD6dJ-"
