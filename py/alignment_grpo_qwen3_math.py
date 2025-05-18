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

# %% [markdown] id="d4kNy7yd4Too"
# # 🧠 Post-Training an LLM for Reasoning with GRPO in TRL
#
# Estimated time needed: **90** minutes on a free T4 (Google Colab)
#
# In this notebook, we guide you through the process of **post-training a Large Language Model (LLM)** using **Group Relative Policy Optimization (GRPO)**, a method introduced in the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300).
#
# GRPO is particularly effective for **scaling test-time compute for extended reasoning**, making it an ideal approach for tackling complex tasks such as mathematical problem-solving.
#
# ---
#
# #### 🧐 What is GRPO?
#
# **Group Relative Policy Optimization (GRPO)** is a reinforcement learning (RL) post-training technique developed and used in the training of [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1). It builds on concepts from PPO and DPO, but introduces a **group-wise reward normalization**, enabling better reasoning and more stable learning.
#
# Unlike earlier techniques that relied heavily on search heuristics, GRPO **relies exclusively on RL** to fine-tune the LLM post-SFT (Supervised Fine-Tuning), enhancing its capacity to solve nuanced, multi-step tasks.
#
# > 🔎 **Note**: Unlike DPO, GRPO does **not** use pairwise preference data. Instead, it relies on a **grouping of prompts by category or difficulty**, and optimizes based on **reward normalization within those groups**.
#
# The GRPO method is available in the [TRL library](https://huggingface.co/docs/trl/main/en/grpo_trainer#quick-start), and the Hugging Face Science team is actively working to reproduce the full DeepSeek-R1 training process via the [Open-R1 project](https://github.com/huggingface/open-r1).
#
# ---
#
# #### 🔄 Comparing GRPO, DPO, and PPO
#
# | Aspect                     | PPO (Proximal Policy Optimization)        | DPO (Direct Preference Optimization)         | GRPO (Group Relative Policy Optimization)           |
# |---------------------------|-------------------------------------------|----------------------------------------------|------------------------------------------------------|
# | **Type**                  | RL algorithm                              | Supervised objective (no reward model)       | RL-based with group-wise normalization               |
# | **Training Signal**       | Uses a learned reward model               | Uses pairwise preference labels              | Uses task-specific rewards, normalized across groups |
# | **Stability**             | Prone to instability in large-scale LLMs  | More stable due to no sampling/rollouts      | More stable than PPO via group normalization         |
# | **Compute Requirements**  | High (sampling, rollout + reward model)   | Low (no sampling or reward model inference)  | Medium-High (RL training, no reward model)           |
# | **Alignment Type**        | Reward-based RL                           | Implicit via preference-based supervision    | Reward-based RL on grouped task data                 |
# | **Strengths**             | Proven RL method, widely used             | Simplicity, fast training, stability         | Better reasoning ability, handles outliers           |
# | **Weaknesses**            | Reward model is hard to train well        | Might underperform in reasoning-heavy tasks  | Needs clear grouping logic and high-quality tasks    |
#
# ---
#
# #### 💡 Why GRPO?
#
# GRPO was specifically designed to **enhance reasoning ability** by promoting group-aware learning. It:
#
# - Encourages **relative improvement within groups of samples** (e.g., math questions of similar difficulty)
# - Promotes generalization across problem types
# - Improves robustness to reward outliers by normalizing over similar examples
#
# These advantages make GRPO especially promising for tasks that require **multi-step reasoning and consistency**, such as math, code generation, or logic-based problem-solving.
#
# ---
#
# #### 📘 About This Notebook
#
# We focus specifically on **post-training with GRPO** using Hugging Face's TRL library. This notebook provides:
#
# - A hands-on demonstration of using `GRPOTrainer`
# - An overview of how group-based preferences are formatted
# - A look at how this training compares to other RLHF techniques
#
# > For a deeper dive into the full DeepSeek-R1 training procedure, check out the [Open-R1 repository](https://github.com/huggingface/open-r1) (**HINT** you might need to check this afterwards for the exercise)
#
# ---
#
# #### 🧩 GRPO Training Pipeline (Illustrated)
#
# The diagram below highlights the main differences between **PPO** (Proximal Policy Optimization) and **GRPO** (Group Relative Policy Optimization), specifically the removal of the value model in GRPO. For more detailed information on the key differences, you can refer to the [full explanation here](https://www.philschmid.de/deepseek-r1).
#
# ![image](https://miro.medium.com/v2/resize:fit:1400/1*84PSf3d1-OGN10y_2H-XdQ.png)

# %% [markdown] id="fLy2KT9m4Toq"
# #### 1. Install Dependencies
#
# Let’s start by installing the essential libraries we’ll need for fine-tuning! 🚀
#

# %% colab={"base_uri": "https://localhost:8080/"} id="wdcWz4bhu7fK" outputId="906d9a7f-2055-47ef-e421-fab4392caa68"
# !pip install --q unsloth vllm math_verify pypdf wandb

# %% [markdown] id="mAoBTDVD4Tor"
# Import needed packages

# %% id="x7ib9Yvs4Tor"
# Python standard library
import os
import re
from typing import List

# Third-party libraries
import torch
import warnings
from datasets import load_dataset, Dataset
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from trl import GRPOConfig, GRPOTrainer

# Custom/project-specific libraries
from math_verify import LatexExtractionConfig, parse, verify
from unsloth import FastLanguageModel, is_bfloat16_supported

# %% [markdown] id="LpE9U7UM4Tos"
# Authenticate with your Hugging Face account to save and share your model directly from this notebook 🗝️.

# %% id="kHj52G5_4Tos"
# set device
device = "cuda" if torch.cuda.is_available() else "cpu" # set device to cuda if available

# Set the huggingface token (get your token from https://huggingface.co/settings/tokens)
os.environ["HF_TOKEN"] = "YOU API TOKEN"

# Set the wandb token (get your token from https://wandb.ai/authorize)
os.environ["WANDB_API_KEY"] = "YOU API TOKEN"

# Ignore warnings
warnings.filterwarnings("ignore")

# %% [markdown] id="oPux4F39Pms1"
# General Config

# %% id="_kxspiwuw0vW"
# ----------------------------------------
# ✅ Candidate 4-bit Models (memory-efficient, quantized)
# ----------------------------------------
# Note: These models are optimized with Unsloth's 4-bit quantization,
# providing faster inference and lower memory usage on limited hardware.
fourbit_models = [
    # Qwen series (good performance, various sizes)
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",     # ✅ Smallest Qwen model, fast & memory-efficient
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",       # ⚖️ Middle ground
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",       # 💪 Better reasoning, more memory needed
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",      # 🚀 High performance, needs more compute
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",      # 🧠 Best accuracy but resource-intensive
]
# 🔗 More models: https://huggingface.co/unsloth

# ----------------------------------------
# ✅ Model Configuration
# ----------------------------------------
MODEL = "unsloth/Qwen3-1.7B"  # Default model for training/inference (change as needed)
max_seq_length = 2048         # Allows longer reasoning chains (e.g., for math or code tasks)
lora_rank = 32                # LoRA rank: higher values = better adaptation, more VRAM usage
NEW_MODEL = "Qwen3_1.7B-GRPO-math-reasoning"

# ----------------------------------------
# ✅ Prompting Strategy
# ----------------------------------------
SYSTEM_PROMPT = """
Respond in the following format:
/nothink
<reasoning>
Briefly explain your reasoning. Be concise and avoid unnecessary detail.
</reasoning>
<answer>
answer here
</answer>
"""
# ----------------------------------------
# ✅ Dataset
# ----------------------------------------
DATASET = "lighteval/MATH-Hard"  # Benchmark dataset for evaluating math reasoning difficulty


# %% [markdown] id="S9ku5d2S4Tos"
# #### 2. Load Dataset 📁
#
# These models excel at tasks that require **complex reasoning**. A prime example is **mathematical problem-solving**, which often demands multi-step reasoning to arrive at a correct solution.
#
# For this project, we'll use the **lighteval/MATH-Hard** dataset on Hugging Face is a curated benchmark designed to evaluate large language models (LLMs) on challenging high school-level mathematics problems. It focuses exclusively on Level 5 questions from the original MATH dataset, which are the most difficult problems sourced from competitions like AMC 10/12 and AIME. These problems require multi-step reasoning and often involve algebra, geometry, number theory, and combinatorics.

# %% colab={"base_uri": "https://localhost:8080/", "height": 625, "referenced_widgets": ["1f0762b1933e476cbacc760646176f3b", "441e2d9195034996929fdeb015d71ae9", "46371ccc139a405ca9a361d3d3d34e53", "4af47a710aff4c439d51907f72f3b2f0", "09f122cc89b7461ab5cf22411ec2ca5a", "ea093fb38743482ea35f72e4940cdea8", "4982605b435743269b483d714a388281", "1f88f1bc8a414d00864ae1bb21292316", "3cd495f577384199bd58641bf804277c", "f1a731818c5c496a9ad39f481920596d", "6c20a243a486486da189c73f738d96b2", "fde05f6ee07b4ef7b6d9cb1bb6f5082b", "ae05ee7abc934533ab1b6fffd3cf22db", "aa6b3d012b3b4040bc55e5f743796127", "87b1170c2b8b4313940b72010f634f09", "2832218a2c6f4fe1b07cb1abfbb64b8e", "134b1d8e3d8644649a06f5be55e5a0b3", "776f1b20f9404b3886b573b7f0b8a2ae", "61828b95911b435fa467ed06d0bd5f21", "f9dd82b34ced4789a1488e2ca83f9d8a", "9867ed44382b4cea86e62b472719f4df", "4173f665e01d4f979a51fdf05a314c16", "69226a342a36414eb5a724bea1e783b6", "59693c85e6fc452eaa7b83332f47a164", "6950d14a64ea4ad29bc5a877fd94e8ed", "9a40fc2729dc424f900ea623b60dd3bf", "6731d23b43b04d94986ee0058f57ab01", "e678e1cbb11e42aa96e45d086f77dece", "a1716cb0c9534d7e9737341fc606e869", "616db87429c641d6b8d1d3f7dc1a33bb", "90f35ac764de481f820d7398d8a16af4", "ae8251af8531433f8fc9d490c876bb6f", "d237affea1844819a793974be0af0a67", "2c4519ee96f74a43b92f4f6dba9e55a4", "23566069aafb4e2785ce9c19b6071439", "3655eaebe29f46dcb480d9e9c0d336ce", "ee9b1e96d7504dafa1d9ff60d43a8981", "580a558aa3aa441ea4b4ba960d568008", "ff87a32aeae749769258be1a624a3147", "51951400d6594b80b7cec1dba473672a", "354a23548ad444e5ac8020e8343794dc", "5ed666d2787b462c90a2458544c1256a", "404a9b3a405a48aa9ce78ef5680fbbb9", "3afec8fe75214559bcf2ef1a93e4a76c", "9b3f04e6878a4bb080c9f9d70edcc10c", "9e997cd247b14f1486e9a4061e215ff1", "3115cde5dea7484d9172ef63db58f073", "3a92ecd3cab744099954cd379822eb38", "6a1598e1515d4d42866cbb57efae5a20", "feeaff8c9bfe4993b5aba17cf7a7065c", "4759dae32b4c47d087b0d2214ba9de6a", "60beebe3ac2e461bb0b259b9f4e18867", "c8682b761d5243a5bdea955e65cf3c8f", "a0ca2f59234d468c8082d372c315c1cf", "c3f383e3b23a45bca296c6c5f9c7bd40", "1c274d26f00f4ecda0caf49cda7c2818", "f7e62470d14b4168adcffaaca9b37b4a", "aa23a299b44844e6aa3d92618675ed43", "d17031d1e78e4adf8932c847a907e091", "61e301666d3040c29c8a01890eb7b3a0", "89323dd7c1924a659b39900e0d9c4159", "913e60c2ec5848b69f839e550a19a97f", "12ff6e25e0074915920c57abdad529c6", "18594250bcb143ec8c2e153faefdfaac", "f46e30f4dade411fb4592ac58187a29e", "dea2f8fc1f9e4129b7db63c867ce72ee", "927edc5b3ae047a7865b3b3e1a931d82", "30f101b6fc294e239ef12575cc2fd8da", "3e609b1e1e494c479d3d7a12ad211b7f", "6bf7221970ab42d1a21e8b42c026eee0", "ac865559abbd49498e8de29c5235d192", "fbeebda98b9840ca84203ab39367fdac", "33ca15d650304e6da693a5ab269c3747", "c36574355c604583bbad51854df2a67f", "48db375e57bf4d149428012a020e9a2d", "b6e117407e754b76ac7b69cd6b938754", "b6c6ec60d8154533b0ce7b951da3c6cb", "0edc7b5fc3f94536a7f68eca28b5d5ee", "8f6eb9953fcf45909ed4d3fa0e736d74", "4e8801144a7d41f3bd415e58c8000298", "ba5f55b0842549d3af536a743870d645", "d028c5101c3c4f35bfa10facbf90cac1", "6cd6362becdc476699c22e2f1bea6e46", "df728de9f9834fd78f240a607d618042", "917791e6c61a4088bb844f3456c238ea", "a1e2c21c85bf457fb653192661c595c3", "e6aae9a324df43eda988c435c38994ec", "ec3c01efb7f34c6dbbf399fc299c360c", "c1220708e0ce4235ad2a52de71b7d81b", "1020a93d4126436ca0a1a48b166f8690", "b8db1750cf5c43c0b01ac79d69d56f47", "d8cb59e09a3046ba84c336f6a6fb6325", "ea2e2fe1eca443f2b811a0599b66db03", "51d875cee6bd408d9afe835a0b5b8fae", "b6f074b1ad6d49248a8088f96676482f", "b81b5a8468cd484d8d7d792ecfd1bb32", "1b351627f88e4f1083d99bdb84a090bc", "897235e3a4334ebfbb1fcc50f42a5405", "d1080a88a1c04045a70f45aef20c0a52", "23eacbf19c4b4bdaa101215ec033cf64", "c3432c47a7b14c898ffa8f88b0cc0fdd", "1050b3dd8831452a88a8e3e8f52a7fee", "71e875dd0865445880eda3efbd995aa3", "2e5f7d5b3d6c445fbe78268c418181a7", "75da5620e2364beabccaf109e87aca9c", "f5479ae0fcbb41aa8bcf9b17446b671d", "8da03a911eae4122b420ce651625648b", "1d1687ffac4b437d9e27dbef99b2f267", "d2089aa4b28542e0820cebc8e2a2a46e", "d8f6f78e7a5b430ba521bd8adc2b3769", "b4e3d5d2ecd14e878453215bf70f4d68", "285fbe557f534584a3510f09c25f0ee7", "cfc55ec28eaf479ebb2c2870870c1c8b", "4409cbb0728840e2975e489d03dc2cc2", "aed8566bb34c4f3c87f1385e04e9af1c", "f167956ef40d4c758581274cfdd3dd98", "e25ef4a7bd244f20a7cf9ca9f9811222", "446a200635454f42aac96cdbc4db5dc1", "1060b16a245d443e92abf4845ff0251f", "c4bccdd529f24c2d81645be19e45f168", "8b97acd5ebaa476dad5f480a16ede0e1", "114aa446b71144cf983ede0d588ceb2d", "c3a7e064bc9d439686f34aa6ba37d893", "38e2cf67a1ca45769a849de6476b0afc", "065ff3c4e16642a799b2b92cd8deb420", "072a840610f34b768d750aab5a41295a", "19846adb2e084fe294c24ef2a3442103", "ac08e5a27d2a4660ba897799b9eb4a0c", "1a53bdcca528450ba860088a4c3574a5", "3fd8ebaf055842d98a36ae9fb73afb94", "6aafa5e384434417858ea04d16b93f15", "b47c3a5402204dc6a23a089c04863504", "8f5c4707aaf548ff87bd69f9d04ed5dc", "32efb61b18dd4ac397b197f14bc6b1c2", "ab0508508b68474b865cd12ef893dd1b", "04960bb2b1224265bf29b8dbfeaafed7", "f1d9d44834a546caa0221c2ca35158f6", "7ca96cb8240b4cb0951bdb887d349df4", "b0a06e0402b54a658411b5a3341fbff7", "e2952a1743dc47f2983092abb949f720", "b59b6b1b80c74823b2333c64cadf8dc0", "21a4c2ab90674ca69d1afddfb9f99a03", "2836a05ed9094c429d14a7522469ce00", "4ff55f29efdf4abd92bdb4e6e9549200", "276123b386ef4359a00a92d25554c52c", "9624ef076cfb4089a6e78b04d0a09cce", "7895735f840e4bdab7855b16d1cbd91a", "d016cda4c2014498a8f497c91341c5a6", "7c3e85341de144c48fff48881268145c", "2aa7f9464d1144dca2f5b0ef1d8bf5a3", "12676a4e4fae40119d35f0dbc36b1ea9", "de68e9065de24c898c14b244a09f7e07", "4fe5744ff90f4fc2a65ca73ef49d6b9a", "7cb6b16f6a0a462a9431b5a76e8222c7", "d76367f3cc8141508e9261829ec47b59", "61a74924dd794feab4a3413eebe16ef9", "6a588e7202504150a64dc056ed61b812", "2e82063b3eb0433582376fa21aff610c", "a42b0f69d8c949c69171bd3942be64e6", "b430335c88924e0b93cda3fcb7705e50", "de4bc416c1094972bf63f9833c8a0006", "c3e4773579aa4086aaf6fce923fd7c08", "eea588d9a1af41a6b4af1568ec9355a4", "bf730cd1f28b4a7bb20bd7d9b5e1047b", "f94e51303ffd4d12a101bc1b5aff336b", "b011faa8df7a4d96a126817730b58b94", "384fa4af02ae4cf8833be25d23c956b2", "825ccd9142b74f51b45253d35479582a", "8cf46882aeeb42b389dc7643d995eba2", "6594bff0cc0c41e49c329de7e6ce98df", "13915224b6964b18a2145cf75abaa193", "01f1c43a7cc04a7cba8da04e9bbd0be5", "c3291476a12b40a6a3cacbcb1f1d30b1", "8628074248bd4ac584e2f29b24eca3c1", "062cf41a98f34ddfba9d66184d09df50", "87aad3b44f8545ab936aa8fb1ddaefa8", "398c29ffaa4a460fb79f46fd9f1cdf7d", "1afe8a1fbb61435997532e138fd12e72", "75478a39062d49048e4616bcdaf21ccc", "c705526210fa4a7f80f75b6fe27f29e1", "acd58d24087f4a0b8359be821598c05d", "26db3663d0db4ba7b5d273bca1b543af", "40e526db3a1544e4ae1d23a4083cbecd", "5ed047df02a744578b26d174c9e20514", "108da5ec91384ba69f1997f7937f43bb", "d2bca3a7a2e24f97bdae160b6306f188", "fcd5a0400e544ebbb8c5b528fceae58c", "0c5a988d2a5947c4b07d4c1947d9d791", "3a67ba25098a47ad8ab3ee2bca419d5e", "59a3529861264cdaaa3768a6c7871124", "b4611acb230145b286d5ae1a68107eb2", "1f4e2aa85e874d9e8777b69934085b8d", "e3327c329b8446d1a1e6ca68af613e07", "305e560fb90a45b787d27db0f4da60c5", "25840d89ed454f51b7aba6ae79becde6", "661fcfd9f9994ac4ade625efae4732dd", "ec31166426704c2eaa84550679733c98", "3081de2abe564e55a1fe1b53571a5b81", "68edb3b0c7db473f906f66c212968d1d", "d1643e1f60dd41629e8ce5e4ccf20401", "9dd6d9ee27aa4bb7916ee94f1f1097d7", "3e349c4b62584015a15fc7d02cc350ed", "48f1b72037664f9489dbca98f5540f2d", "e3ead7d379ec4a5b84ad776fa5bf0421", "eeecb83d70344fe4aec802310d77907d", "d377ab77cec6422382f3a755ad9f08cc", "efea5523e4fb4dcdbde983bbdfa4a704", "cf79ea426be843468c351d7cc3a798b9", "66a81730d8424fa380c1430733ffde2f"]} id="rLqs8X-C4Tos" outputId="0e0da8de-fb2a-4626-bfa0-091b280cf2d6"
def get_math_questions(split="train") -> Dataset:
    # Load the raw dataset from the hub
    data = load_dataset(DATASET, 'default')[split]
    data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['problem']}
            ],
            'answer': x['solution'],
            'question': x['problem']
        }).remove_columns(['problem', 'solution','level','type'])
    return data

# To get the three splits, call:
train_dataset=  get_math_questions(split="train")
test_dataset =  get_math_questions(split="test")


# %% colab={"base_uri": "https://localhost:8080/"} id="KfDUrCTEnBKh" outputId="8db126a6-d229-4239-8364-5c82a3c1d70a"
print("train dataset",train_dataset)
print("test dataset",test_dataset)
print("train sample",train_dataset[0])

# %% [markdown] id="TVY7JgKS4Tot"
# ## 3. Post-Training the Base Model Using GRPO
#
#

# %% [markdown] id="W6K6GOIx4Tov"
# ### 3.1 Loading the Baseline Model
#
# To begin, we'll load [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). With only 1.7 billion parameters, it is lightweight and fits within the available resources. However, for better results, a larger [alternative](https://qwenlm.github.io/blog/qwen3/) should be considered.
#
# [Benchmark](https://dev.to/best_codes/qwen-3-benchmarks-comparisons-model-specifications-and-more-4hoa)
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 409, "referenced_widgets": ["56b3bda4c8094dce9977b877b62d501f", "344da8eb3158439796a3f11ca7c13b67", "5c817e1df2de44fda8af15f0763ae581", "77b208fbe4fa4f0bad6db78b68b54297", "6751790ff00b4885a55a8f0341c7479d", "fade6ce863ff414a9ab884c038e107b3", "0a12207afdc8496081a7a28f4d52ebcf", "5fbccab8205c4be5a7382cc281d3de00", "a26589b4d2f04c899101c8c38725b840", "05407e699e1c45d2a0c88461a935ba6a", "0f300eb9d6b9430c9316173098c3c159", "60073a70fb454cf8bcacdfc4a4970b57", "524f63d326aa45f8adcb6d031853664d", "10b7eae0dc594712a1165285f784acb5", "5d5b87c4547d46aa91501a6789bfb8a8", "92bb32ec511c487787978f0085aa8a09", "ec1a975a413c407e8942b50bda5a5fca", "aaff55c3781c4cbba5db8b9a814f594c", "cb11c8f917c64b6c86582b078b2ad4b4", "c54b5c21d999415fa2c5e45d1da11cf1", "083c319d82a2471fa2611e7069d2baae", "6856fb0a1d1f4d92acf44eac6e047790", "c129355d67324ee8a755146e14a5a81a", "b478fa9f770d47e4a2bf7b965fb7e996", "edc035a963e442c88a9a9903a690178b", "09e70b61079a4675b28c7ba2c9e33c61", "abdd7ab82f7d475da441f09549fecb49", "f74ea133e38a49b390e8bc260939ce5c", "e55e097fcb4f4b28bf55eafd987cdea2", "49732293a322484881aa008bbacdb36b", "804274e29ce6409396288d6d2d1396bc", "6a2b0f43358f4f23b3af13c38602546b", "4de5984f238b4cc1aac9449e15a99920", "9404d01d7b8a482c977a86a9db1ab902", "6db89670c0e14ce98b2156a521af4a15", "155c98aee2ec407b9ef538d1a6161df5", "26134272bd884e28a9dfd243490f8040", "e16cdd404e064aa386085a9434329cbb", "7e3efc8a1c5346a4b59e50fb21bd20d9", "1886688202884c159a43c59e78800f26", "ee7d6a92cbed4e3483c96980caf8f0d5", "d2942510f092413da2d532db6e19e948", "ee0986996a92462a80e6806c4130f58f", "71c6b377a1f7491a9a2fe20b9acc1430", "17bbe07d35e24142921584a731df0242", "4694221fc1dc4809a4d3f40ea93e0960", "3183c130f33f4838953ff4813ade3a19", "49b0986be8d1406793955c80a9f75f31", "4c6e89a3985a4236a333a450f96353ea", "2e967c0ca87f4671ba3f93b67ad6a156", "7bef1ba5217a4b6f8f4369bcd18f5fdf", "da50a9669351459c90292388d93ce41d", "0b5e1963ccb546f996dad389efcd2977", "ee28e957ef5d46a985eb83a7f1b433a6", "529a8ec79bfe4d819dd38f593e70aceb", "b60fa5b70fa04d418d08dba51c97d0b5", "02f0f17bb8fa465aad47fbf6cac77e5d", "f367bf026c7d401ca1081f45f39c30c6", "d383dac645374a6fa969309411ba65c4", "2af0412966584017abb527608bba097e", "b062a02970ff405ab3c44e80510291b1", "60710c64cc5e4379a88e1e24daacdcb0", "2d049266bcac498faec08fe26cbaebed", "1f6bf5464aad420fbdbadc7d5bc20faf", "bfb09454c28543eeb7b8cb4afaafc0ef", "04157e0ef90748e4b364f88f3c2d25b0", "c02d38ee8a5a473fa57c55aac3015dc5", "093c8825d4a64d73ba10a8753295da79", "72ffb467cdcd4ce0aaf7a229be321070", "8a331aa66f3a4647bbb219b1c4f18380", "dc3aa8bbfde148c485de44c0ef2aab0a", "dca9a9c3505f4d488d18428f6f664fa0", "749725ce5e134a64b12f0a8204177e06", "9b546639664a42bbbc429f652c23ad85", "9d4d5309c9994353a930c192373d097f", "a29ae1609c3446b893b7636ae10bda29", "bdce183645b74c66b96b7133726a93a4", "463e73ad15e94581a2b6897297822ab2", "1c47793285a4452ba6273f34b7109398", "1241edafab7e4a3fb06780c791d3c3e2", "ed17ba0eb30c442cbed13201c5f31ed8", "16d6167babb9437fbcc1c38db4b8602e", "ae6d1a9fa9f944e39d86d3fb52a114dc", "caf59c65e27f493797eb6a0f75dae432", "b610ef1ce6ad42d88c2479889c261c72", "0681c95841634f7db9948160ce546065", "4c95f385362d4d3598f2b8826a468e9d", "e1f7c91897e14d749763e52124bfe899", "9aad7e2f7cd24bae9183c40d0973636e", "e99f557c4fe448a09fe31fed01835761", "d56a5fd9cd4d4f7ea8779addd45d54a3", "a4c2567661f442cc9aa74ded595e7745", "96ae4276c1934e8a96765d388d25d199", "a489a9796fb74070a3905de04a90aed4", "ef09f91c9763486eb8d21ffd72ff90b2", "d082ed6c7dc2437bb8a4378c7a6d448e", "248e5a0cb0994f2d9802abf49df24049", "90b0a53136f04fd49919178422a3a6d0", "6b3b4e8c70c44a78b61665effa128e83"]} id="gAji1v2IvBhj" outputId="634d0c99-c6d1-4396-cd41-3a750b703ec1"
# Load the language model with optional 4-bit quantization and LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,              # Name or path of the pretrained model
    max_seq_length=max_seq_length,  # Sets maximum input size handled by the model
    load_in_4bit=True,              # Use 4-bit quantization to save GPU memory and speed up inference
    max_lora_rank=lora_rank,        # Sets the rank for the LoRA adaptation
    full_finetuning=False,          # If True, fine-tunes all weights; if False, only fine-tunes LoRA layers
    # fast_inference=True           # Optional: Enable vLLM-style fast inference (commented out here)
)


# %% [markdown] id="8XXFnfOhlkWA"
# This model would normally require ~6.8 GB of memory (assuming 32-bit floating point: 1.7B × 4 bytes), However since we decided to apply Quantization(load_in_4bit=True)
# This reduces memory footprint by 8× compared to 32-bit:
# From 4 bytes per parameter → 0.5 bytes per parameter
# 1.7B parameters × 0.5 bytes = ~0.85 GB

# %% colab={"base_uri": "https://localhost:8080/"} id="W3vslLVWqAQ6" outputId="7a854ad0-ad02-43bb-bd40-6f9ea3bb68b5"
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    num_params = param.numel()
    # if using DS Zero 3 and the weights are initialized empty
    if num_params == 0 and hasattr(param, "ds_numel"):
        num_params = param.ds_numel

    # Due to the design of 4bit linear layers from bitsandbytes
    # one needs to multiply the number of parameters by 2 to get
    # the correct number of parameters
    if param.__class__.__name__ == "Params4bit":
        if hasattr(param, "element_size"):
            num_bytes = param.element_size()
        elif not hasattr(param, "quant_storage"):
            num_bytes = 1
        else:
            num_bytes = param.quant_storage.itemsize
        num_params = num_params * 2 * num_bytes

    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params

print(f"trainable params: {trainable_params:,} || {all_param:,} || trainable%: {trainable_params/all_param:.2f}")


# %% [markdown] id="GT6lIIGZ4Tov"
# ### 3.2 Configuring LoRA
#
# Next, we will configure LoRA for model training. This technique will allow us to efficiently fine-tune the model with a reduced number of parameters, enabling faster and more resource-efficient training.

# %% colab={"base_uri": "https://localhost:8080/"} id="y8MD9w0dxOsw" outputId="81e8825d-e104-4e28-b6f0-d11b769b7555"
# Apply LoRA (Low-Rank Adaptation) using PEFT to the base model
model = FastLanguageModel.get_peft_model(
    model,

    r=lora_rank,  # LoRA rank: Controls number of trainable parameters. Common values are 8, 16, 32, 64.
                  # Higher rank → more capacity to adapt → better accuracy but slower and more memory usage.

    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",        # Attention projections
        "gate_proj", "up_proj", "down_proj"           # Feed-forward projections
    ],
    # These modules are typically the most sensitive and impactful in transformer adaptation.

    lora_alpha=lora_rank,  # Scaling factor for LoRA updates. Alpha = rank or 2×rank is a common rule of thumb.

    lora_dropout=0,        # Dropout applied to LoRA layers.
                           # 0 is best for most tasks and is optimized in frameworks like Unsloth.

    bias="none",           # Bias configuration: "none", "all", or "lora_only".
                           # "none" is memory-efficient and recommended when biases don’t significantly affect results.

    use_gradient_checkpointing="unsloth",  # Use gradient checkpointing to save memory during backpropagation.
                                           # "unsloth" mode is optimized for long-context tasks (e.g., long reasoning chains).

    random_state=3407,     # Ensures reproducibility of LoRA weight initialization.

    use_rslora=False,      # If True, enables Rank-Stabilized LoRA (adds rank flexibility).
                           # Off here to keep configuration standard and stable.

    loftq_config=None      # Optional: If using LoftQ quantization-aware LoRA training.
                           # Not used here — defaulting to standard LoRA without LoftQ.
)


# %% colab={"base_uri": "https://localhost:8080/"} id="d4fLUB7EoeXL" outputId="a5a35dc4-d204-4f8f-d5df-217d2451c18e"
model.print_trainable_parameters()


# %% [markdown] id="Vl0GaSrNAVc6"
# There was a significant reduction in traing size from  37,872,640 to  34,865,152, due to LORA (Low-Rank Adaptation), it is a parameter-efficient fine-tuning method that dramatically reduces the number of trainable parameters while maintaining performance. Here's how to understand and calculate its parameter efficiency.
#
# How LoRA Works
#
# Instead of updating the entire weight matrix during fine-tuning, LoRA approximates weight updates using low-rank decomposition:
# W' = W + ΔW = W + BA
# Where:
#
# - W is the original weight matrix (dimensions m×n)
# - B is a matrix of dimension m×r
# - A is a matrix of dimension r×n
# - r is the rank (much smaller than m and n) (32 in this case)
#
# Calculating Parameter Reduction
#
# For each weight matrix W with dimensions m×n:
#
# Full fine-tuning parameters: m×n
# LoRA parameters: r(m+n) = (m×r) + (r×n)
#
# For example, if you have a weight matrix of size 4096×4096 and r=16:
#
# Full parameters: 4096×4096 = 16,777,216
# LoRA parameters: 16×(4096+4096) = 131,072
# Reduction: ~99.2%
# ![image](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5dfbd169-eb7e-41e1-a050-556ccd6fb679_1600x672.png)

# %% [markdown] id="2b8GdNKD4Tov"
# ### 3.3 Loading Reward Functions
# In Group Relative Policy Optimization (GRPO), reward functions are essential because they guide the preference-based optimization of the policy by comparing the quality of generated outputs within a group. Unlike standard Reinforcement Learning (RL), where absolute rewards are assigned to single actions or trajectories, GRPO relies on relative comparisons—often derived from human preferences or heuristics—to update the policy.
# In this case, we will utilize these reward functions:
#
# 1. **Format Enforcement:** Ensures that the generation follows a specific format using `<think> </think> <answer> </answer>` tags for reasoning.  

# %% id="9sT1TQ1R4Tov"
def tag_presence_reward(completions: List[dict], **kwargs) -> List[float]:
    """Reward for presence of <reasoning> and <answer> tags"""

    print(completions[0])
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', content, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', content, re.DOTALL))
        reward = 0.5 * has_reasoning + 0.5 * has_answer
        rewards.append(reward)
    return rewards


# %% [markdown] id="rq5_g9jL4Tov"
# 2. **Solution Accuracy:** Verifies that checks whether each generated model completion matches the expected ground truth solution. If the model's output matches the gold solution (e.g., mathematical expression), it rewards the completion; otherwise, it assigns a penalty (0.0 or near 0). This helps reinforce accurate model behavior in tasks where precision is crucial, such as solving math problems or parsing structured data.correct.

# %% id="jeOHiSiZ4Tov"
def accuracy_reward(completions:List[dict], **kwargs)-> List[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['answer']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


# %% [markdown] id="aSloa6L2pK08"
# 3. **Semantic correctness:** reward uses a cross-encoder model (cross-encoder/stsb-roberta-base) to evaluate the similarity between generated responses and reference answers. When no valid answer is extracted (i.e., an empty response), the function assigns a reward of -1.0 to indicate failure in producing an answer.

# %% id="D_HuKspwL4k-"
def semantic_correctness(completions: List[str], **kwargs) -> List[float]:
    """answers semantic similarity using cross-encoder"""
    answers = kwargs['answer']
    model_ss = CrossEncoder('cross-encoder/stsb-roberta-base', device=device)
    responses = [completion[0]["content"] for completion in completions]
    inputs = list(zip(responses, answers))
    with torch.no_grad():
        similarities = model_ss.predict(inputs, show_progress_bar=False)
        similarities = torch.tensor(similarities).clone().tolist()
        # Set similarity to -1 if the response is an empty string
        similarities = [-1.0 if response == "" else similarity for response, similarity in zip(responses, similarities)]
        return similarities


# %% [markdown] id="g3loewUNNYYu"
# ==> Infer the original model on a a given question (to test the generation config)

# %% colab={"base_uri": "https://localhost:8080/"} id="L3_iR5xkvHl1" outputId="e74c7649-8ea5-40aa-b5f4-a07759a849bc"
# text = "Find all values of $x$ that satisfy the equation $|x-3|=2x+4$. Express your answers in simplest fractional form."
messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': train_dataset[5]['question']}
        ]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 2048, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# %% [markdown] id="gOE9BHeD4Tow"
# ### 3.4 Configuring GRPO Training Parameters
#
# To keep things simple, we’ll start by training for just one epoch and reducing the `max_completion_length`, `num_generations`, and `max_prompt_length` from their default values.

# %% colab={"base_uri": "https://localhost:8080/"} id="Q38jT1N81cPY" outputId="9cf7f22f-5bde-4435-bc92-4d01a78b2c1c"

# Configuration for the GRPO training setup
training_args = GRPOConfig(
    # use_vllm=True,  # Optional: Use vLLM for fast inference, but not compatible with Qwen 3 models (commented out)

    lr_scheduler_type="cosine",  # Cosine learning rate scheduler for smooth and natural decaying
                                # Good for tasks where we want gradual adjustments of the learning rate.

    per_device_train_batch_size=1,  # Batch size per GPU; 1 for memory efficiency on small GPUs like T4
                                    # Larger batch size will use more memory but may speed up training.

    gradient_accumulation_steps=1,  # Number of steps to accumulate gradients before performing an update.
                                    # Increase this if you want larger effective batch sizes without running out of memory.

    warmup_steps=5,  # Number of steps to warm-up the learning rate. A small value for a quick adjustment.
                    # You may want to adjust this if the model's training is unstable at the start.

    max_steps=50,  # Number of training steps. Typically set based on your available compute and the model’s convergence speed.
                   # For long training runs, this may need to be adjusted for fine-grained control.

    learning_rate=2e-4,  # Base learning rate. A moderate value to start with; can be reduced (e.g., to 2e-5) for longer runs.
                         # Lower values tend to work better for large models or fine-tuning tasks.

    optim="adamw_8bit",  # Use 8-bit AdamW optimizer to save memory and speed up training.
                         # Good choice for larger models where memory and speed are concerns.

    max_grad_norm=0.1,  # Gradient clipping to prevent exploding gradients; 0.1 is a standard value.
                       # You can adjust if training becomes unstable or to improve convergence.

    max_prompt_length=500,  # Maximum input length (tokens) for the model’s prompt.
                           # Useful to control memory usage and ensure you don’t exceed model’s input limit.

    max_completion_length=1024,  # Maximum output length (tokens) for generated completions.
                               # Adjust according to the expected complexity or verbosity of the model’s response.

    seed=3407,  # Random seed for reproducibility of experiments.

    report_to="wandb",  # Reporting to Weights & Biases

    output_dir="qwen3_1_7B_grpo_math",  # Directory where model checkpoints and logs are saved.
                             # You can adjust this to store results in a more appropriate location.
)

# %% [markdown] id="wwHPj5Mprg7_"
#  Another interesting parameter is reward_weights in order to Weights for each reward function. Must match the number of reward functions. If `None`, all "
#             "rewards are weighted equally with weight `1.0`."
#     

# %% [markdown] id="Ws9rBuEi4Tow"
# ### 3.5 Training the Model 🏃
#
# Now, let's configure the trainer and start training the model!
#

# %% id="3xdAxp-v1ikC"
# Initialize the GRPOTrainer with the necessary configurations
trainer = GRPOTrainer(
    model=model,  # Pretrained language model that will be fine-tuned during training.

    processing_class=tokenizer,  # Tokenizer used to process input text for the model.
                                 # Ensures proper encoding and decoding of text into model-friendly formats.

    reward_funcs=[               # List of reward functions to evaluate and optimize the model’s outputs.
        tag_presence_reward,     # Reward function focusing on the presence of specific tags or keywords.
        semantic_correctness,    # Reward function evaluating how semantically accurate the model’s response is.
        accuracy_reward          # Reward function for the accuracy of model predictions, based on ground truth.
    ],

    args=training_args,          # Training configurations such as batch size, learning rate, etc. (from previous GRPOConfig).

    train_dataset=train_dataset, # Training dataset used to fine-tune the model.
                                # Should contain relevant examples for the task the model is being adapted for.
)

# %% [markdown] id="mlHhdDaRe9Kr"
# These settings are tuned to reduce hallucinations while preserving fluency and diversity, which is critical for math-heavy or logic-intensive datasets like lighteval/MATH-Hard.
#

# %% id="G1E29wSJdj5T"
# Modify the generation config
trainer.generation_config.temperature = 0.7
trainer.generation_config.top_p = 0.8
trainer.generation_config.top_k = 20

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["6e728fa7f0304fe793f1bf3c44c28a21", "33cae56bee1746d59266dd615d300fc2", "5d965d7c3bc64c7fb228a5bceb8ebe75", "2f4333f9205443e7a304a3794a705d96", "bdae4c4dc5e24079b7375ae2fa099399", "7f4e98857c26488987b423d0908f1afe", "785d488c1c1647a0a4242228f9899eae", "297dd15827294d04b8b4feee1b4754de", "f82ddad9124a448a9c7e92b1d0de38e3", "5bad783524154add88ddee003e561c88", "28c418e7cfb64c2aaad642ca297aabfa", "cdfd3f79c18b4f8587b636919e5c1653", "660b80d6d57640e5b003cc49908ab7e4", "2c3e05b902714960a5f34dac38a5b097", "d4bdb636a98b48edae65a3f6018fd382", "7996e34623714fd2ad6031618d6cce92", "299e7f01c8314733a0b0ebff5e20ecec", "009193589a2a451aab316af18a36bbb3", "b6c156c97cd54a988fe70089bdd16baa", "9c4c1f456f094ebbaee8e81e4f521b5b", "371a3d2b10d342c98c845055cd91bba0", "9de725b3ca5d448496450994104940a4", "00ee37e9597048ceb277b411ce119e4c", "9fcbed8d122946e8bc86dc40e551a54e", "20b679c5c474466bbad8939153f7a5cd", "ae355ada677849b2944187b3a833bca2", "485c09d7a36b42a695f0795ad4c80e25", "850ec209a83b415e91e8679463e630c1", "2659377bc9cc4385a3dd232521a6dbb6", "7066688babc54f3ea460c3fc64eccbf6", "ae2f18f25c6644ee853f3cd23aa6f1ed", "5512cbc8b0b74d4a96e4d69989c53579", "bfc15ac959e44725bb3e5ed7626612ea", "a548e304620646258d3fb0a30d155c7c", "0d2665918eb949b3b178e43fb6567dd2", "4c033b837edd4e459823f60de6f5ef71", "1403375b3d544d0bbfd326fa60691a16", "d91beb418ebf49909423da2a7e985d68", "f10661d60203475c81b5bed2b83a6567", "951dcc8ac456411fb8158f0c9bc431d5", "70089ef84d004d7993999aad7d9bf528", "4fdcd0cd49ff4b3c97471b8487bfbc56", "6d57cc877c694bcc92e2c706ab0c16f0", "54fad36cb907448dbe40a0b18f82ce03", "3289c3a068bf4903965c495163279a0f", "5697520ed7b04ede91b68ed37317db3b", "a701bed73f84497fb48ce28922dfeeec", "f98f3f911ce64966a786ae6efa62a6ba", "5358f594353f4278bef46b2244c7d03e", "49496a17110b4e60bbb7b551cd216b50", "3aecac0718ad45fa9f92feeb3efe8fd7", "9daf314bc0e94a0e93a79f24773a281e", "9dbbd27dca11443f93a0412864084c5f", "1c066028dc7c4f398354eb4200e31185", "ed1d366ff18e410895ce3bdcfd713655", "275ceed926d84faf9a6efd8b13bbbb68", "1305f6c2de7a4da08ac83f879bdfb0c8", "ca45fa85e3ca4990b79976f174347075", "bb712e1134124cf2a23d3ae97d964e53", "2081734f1e3e4dd0b78c61b92a6f7f30", "d48c8ff69d76450f9c102a729958c6c0", "5a30817d0ec54bd88ca143d2ff76f105", "1982937c3c8443f0bf8b3d807513b29e", "548bb3f69ef844d9a91d99d314791d2b", "52769fa4d7224ad4809b0336082fc95b", "d3a633ee369249b88317d06538ef3809", "52277f6ff7024022b4bc7402d3843992", "723a4a11e08743ffa32b22a857c3440a", "c9cbe1911b80467b8d16d52eb16ac698", "2676d33b8f6c4d7981a0a54b1f281531", "e40487e7bcd8468cb086e9ab962519c4", "2c241e49bbeb4e1ba4df6f8da27b7bc8", "9ae06f521f4346aabb8c883560ac43cb", "b575069a2cc64d40a45bb7349ebb07cc", "7dad4b886e4042b2b1ebc9e695b59057", "77180db8b194475bb5fca647903e6aa2", "da98fe8cbd114698b053cc5fe41afd95"]} id="-4tgaRQzc8Rf" outputId="f807ef22-1521-4daa-d403-4e61f6560c4b"
# Start the training process
trainer.train()

# %% [markdown] id="yubuCTq4wwvM"
# !!!!!!**Normal GRPO Training Behavior: Loss Starting at Zero Then Increasing**!!!!!
# This is completely normal. Initially, your model policy (πθ) equals your reference policy (πref), so their KL divergence is zero, meaning your loss starts at zero.
# During training, as πθ optimizes toward higher rewards, it naturally diverges from πref, causing the KL penalty term (βDKL[πθ∥πref]) to increase - which increases your loss.
# The simplified math shows this clearly:
#
# When πθold = πθ (single exploration step)
# After simplification and considering normalized advantages (∑A^i = 0)
# The loss becomes: JGRPO(θ) = -β·DKL[πθ∥πref]
#
# Your increasing loss actually indicates successful training - your policy is moving away from the reference to maximize rewards, constrained by the KL penalty.

# %% [markdown] id="36EF3_DUfWa-"
# ### 3.6 Saving the Model to float16 for VLLM
# We can save the model to float16 directly. Select merged_16bit for float16 or merged_4bit for int4. We also allow lora adapters as a fallback. Use push_to_hub_merged to upload to your Hugging Face account!

# %% colab={"base_uri": "https://localhost:8080/", "height": 593, "referenced_widgets": ["ba04f32697af490a8f7b80684942549a", "b0340466490b4712b95155503f0eb40c", "7945b91ac5f34fc98d5197a5fe6da0f9", "696d242b6b4349c7a348e01e0ff7f962", "d1b888924b914e3e8a8fd0810ba803c6", "770fd9d0b06a4bd4a848b6628e1be158", "42a8120de45247608f3da72352b47ccd", "f6e6976dbe8548849599bb8e5d78b10a", "a71c0f50412a48608083c8793b812683", "21960505ddab421690ea090748ed6763", "9b733d44250e475fb11fa8dc00303bcd", "15f94ba92e5a48929fe466661f36debe", "51b1bc503aa04c35a3be459f471db14e", "4bc9ffe306094a39b4e513113936204c", "69c926d896fc47b18cba9501a16febdd", "3edaad17a63f4aa39e37597b7546ec0e", "683baf1cb2904166b8ceb63aafc8f3ed", "bf5ab852c7514f85adb5bd0e3f661cb5", "7d752f272fd24ca9bdfc57ba93627136", "9513e3bffdb846e9b0459a805642c757", "8d1e659391604bd3a8eca5e36f47121d", "01011805413d4113a65491dfec4a30ba", "6a16b6fdcdfb47ea87aba16b4a7c3774", "67ee60255bf94a55a99bfa4467e0b5a2", "2f394c7a47b641eca98a5363fa91a598", "2a14ef1ce0f748509a32ed7b4eb54430", "92dcee4346924e1cbe22161963dcb128", "bd667e5d35b14f24bf666f4d8d00133d", "fa762fb3d105439bad7bcf25574d8f27", "b24f5554baa64e5d9dcdb0c516479294", "c9484170efe24d93aea05d9bcf4f82de", "b34d80e3d41f45e99d1630e7f6a5f04a", "eb9575de16384940baca1d235deda81c", "0286e625752e44d8ab69cad15aff0235", "f8762f0e9bd649c3b1b07a3105637ef7", "6d0af433f4bb499cb84df817f1a64bb6", "09e3a96813154a68a91840793b2fc06c", "d3f820aa8adf467dbdeb8653165e2005", "7de864d4d0cb421d8b68dc20b9ea0887", "606138615b1042508f530bc5febf1589", "e4cf01f09ccc404f8f5d98d08411b1a9", "2092a5e84c0242b7ae04478bdf209cf2", "e0dae208f68f4c729104966b098c54b2", "438827288c7644e28f15f078bfe33167", "46c51b33de314db083cdc5c1de0ac8f9", "92d2588eb966425b91b958eca7509729", "cc7da85193f64075a7a9e59728a8dbe9", "a9abf48a087646599c45b5053aafe149", "bf4574eeabe74d338c5b6db98569c21f", "b2ccb29e2a7f4e658dab8ffcb202dbdc", "a0850d4c88f9443997d42e283832d2af", "c248b1e5b8704546b964b80c3fdb1287", "95dab0fcc14f49349b8d7529da2a336f", "e8ca0893a5b64f63a8c186fcf414f6fe", "c467232888514c22ab0d2833c100110d"]} id="cmGJnaxd4Tow" outputId="7321e091-4fd0-42cb-bc9c-5e8963aa9b96"
user_name = "YOUR_USEER_NAME"
model.save_pretrained_merged(NEW_MODEL, tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged(f"{user_name}/{NEW_MODEL}", tokenizer, save_method="merged_16bit")

NEW_MODEL

# %% [markdown] id="_lwq18a34Tox"
# We observe that the model demonstrates some reasoning capabilities, although these are limited. This can be attributed to several factors: the use of a small model, a limited subset of the dataset, and a short training duration to keep the process simple and practical for a notebook environment.
#
# Despite these constraints, this technique shows great promise. The release of DeepSeek-R1 and the adoption of this training approach could lead to significant breakthroughs in the coming months!

# %% [markdown] id="IF_BPKPYfwmj"
# ## 4. Test the model
#
# In case you didn't manage to finish the training, feel free to call and load the model from the hub running it on test dataset
#
#

# %% id="uQ2Q6OnY1m68"
# Load the language model with optional 4-bit quantization and LoRA
model_inf, tokenizer_inf = FastLanguageModel.from_pretrained(
    model_name=f"HackAI-2025/{NEW_MODEL}",              # Name or path of the pretrained model
    max_seq_length=max_seq_length,  # Sets maximum input size handled by the model
    max_lora_rank=lora_rank,        # Sets the rank for the LoRA adaptation
    full_finetuning=False,          # If True, fine-tunes all weights; if False, only fine-tunes LoRA layers
    fast_inference=True)

# %% [markdown] id="KuGo-hPiupCx"
# Let us test it on Math&Maroc competition exercise:

# %% id="c0dmPHLZNBeB"
# Extract text
reader = PdfReader("Assets/MMC_2024_day1.pdf")
all_text = '\n'.join([page.extract_text() for page in reader.pages[:1]]) # only eng page

# Find all problems (Problem 1:, Problem 2:, etc.)
pattern = r"Problem\s*(\d+)\s*:(.*?)(?=Problem\s*\d+:|$)"

problems = re.findall(pattern, all_text, re.DOTALL)

# problems is a list of tuples: (problem_number, problem_text)
for num, statement in problems:
    print(f"Problem {num}:\n{statement.strip()}\n")

# %% colab={"base_uri": "https://localhost:8080/"} id="2-4rPdKXJx5u" outputId="28c02329-c362-41f2-9f4c-bcba20846dd6"
messages = [
            {'role': 'user', 'content': problems[0]}
        ]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 2048, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# %% [markdown] id="hZLkuSJ-r7mC"
# ## 🏁 5.Team Exercise: GRPO Understanding Challenge
#
# ### Overview
# This exercise tests your understanding of Group Relative Policy Optimization (GRPO) without requiring additional model training. Your team's performance will be evaluated on a leaderboard to determine the best understanding of the concepts.
#
# ### 🧩 Exercise: Designing a GRPO-Based Fine-Tuning Strategy
#
# #### Task Description
# Your team must design an improved GRPO fine-tuning strategy for a math reasoning model by making strategic decisions about various components of the pipeline.
#
# #### Requirements
#
# 1. **Reward Function Design (35 points)**
#    - Design a comprehensive set of reward functions for math reasoning
#    - Explain how each reward function addresses a specific aspect of high-quality math solutions
#    - Justify the relative weighting of different reward components
#
# 2. **Training Configuration Optimization (35 points)**
#    - Recommend specific adjustments to the training hyperparameters
#    - Justify each adjustment with clear reasoning
#    - Provide a complete `GRPOConfig` code snippet with your optimized values
#
# 3. **Evaluation Methodology (30 points)**
#    - Design a robust evaluation protocol for your GRPO-trained model
#    - Specify metrics to track during and after training
#    - Describe how you would determine if the GRPO approach is working better than simpler alternatives
#
# ### 📊 Evaluation Criteria
#
# Your submission will be evaluated on:
#
# 1. **Technical Correctness:** Proper understanding of GRPO concepts
# 2. **Innovation:** Novel but practical ideas for improving the training process
# 3. **Implementation Feasibility:** How feasible your approach is to implement
# 4. **Justification Quality:** The depth and clarity of your reasoning
#
# ### 📝 Submission Format
#
# Submit a Markdown or Python file containing:
#
# ```python
# # Team Name: [Your Team Name]
# # Team Members: [List of team members]
#
# """
# REWARD FUNCTION DESIGN
# ---------------------
# [Your detailed response here]
# """
#
# """
# TRAINING CONFIGURATION OPTIMIZATION
# ---------------------------------
# [Your detailed response here]
# """
#
# """
# EVALUATION METHODOLOGY
# --------------------
# [Your detailed response here]
# """
#
# # Bonus: Sample code snippet for any one component of your solution
# ```
#
# ### 🏆 Leaderboard Assessment
#
# Your team's submission will be evaluated using a scoring system that assigns points based on:
#
# 1. **Correctness Score (50%):** Assessment of technical accuracy and proper GRPO understanding
# 2. **Innovation Score (30%):** Originality and effectiveness of your proposed strategies
# 3. **Clarity Score (20%):** Clear articulation and organization of ideas
#
# The total score (100 points maximum) will determine your team's position on the leaderboard.
#
# ### 📋 Hints for Success
#
# - Focus on the unique aspects of GRPO compared to other methods like PPO and DPO
# - Consider the specific challenges of math reasoning when designing rewards
# - Think about scalability and computational efficiency
# - Review the implementation details from the notebook carefully
#
#
# ## ⚠️ Important Disclaimer
#
# **This notebook demonstrates a simplified implementation of GRPO fine-tuning.** For production-level applications, this approach should be significantly expanded and refined. Specifically:
#
# 1. The training duration (50 steps) is far too short for meaningful learning
# 2. The reward functions are basic implementations that would benefit from more sophisticated alternatives
# 3. Proper group construction requires careful analysis of your specific dataset
# 4. Larger models (7B+ parameters) typically yield better reasoning capabilities
# 5. More extensive evaluation on diverse test sets is essential for real-world deployment
#
# For research or production applications, we recommend:
# - Increasing training steps by at least 100x
# - Using more sophisticated reward modeling techniques
# - Implementing proper group balancing and sampling strategies
# - Considering multi-stage training approaches (SFT → GRPO)
# - Developing robust evaluation suites for mathematical reasoning
#
# The simplicity of this notebook is designed for educational purposes and to fit within Colab's resource constraints.
