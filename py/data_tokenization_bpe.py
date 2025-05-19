# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/data_tokenization_bpe.ipynb)

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

# %% [markdown] id="5b9c8726"
# # 🏆 Tokenization Challenge: Understanding How AI Reads Text
#
# Welcome to the Tokenization Challenge! In this notebook, you'll learn about how AI models process text by breaking it down into smaller pieces called tokens. We'll build a simple tokenizer from scratch to understand the process better.

# %% [markdown] id="1b195729"
# ## 1. 🔍 What is Tokenization?
#
# Tokenization is like breaking down a sentence into smaller pieces that an AI can understand. Think of it like cutting a sentence into words or even smaller parts.
#
# ### Why Do We Need Tokenization?
# - AI models work with numbers, not text. Tokenization helps convert text into numbers.
# - It helps the AI understand patterns in language better.
#
# ### Different Ways to Tokenize:
# - **Word-level**: Splits text into words. Example: `"hello world"` → `["hello", "world"]`
# - **Character-level**: Splits into individual letters. Example: `"hello"` → `["h", "e", "l", "l", "o"]`
# - **Subword-level**: Splits into parts of words. Example: `"playing"` → `["play", "##ing"]`
#
# In this notebook, we'll learn about Byte Pair Encoding (BPE), which is used by many popular AI models like GPT. BPE is smart because it can break down words into meaningful parts that it sees often in text.

# %% [markdown] id="acf191c1"
# ## 2. 🤖 Let's See How Different AI Models Tokenize Text
#
# We'll use some pre-built tools to see how different AI models break down text. This will help us understand what we're trying to build.

# %% colab={"base_uri": "https://localhost:8080/"} id="159e5deb" outputId="20b53f5c-ced1-4e06-9a31-c567f012bc9b"
# !pip install transformers tiktoken --quiet

# %% colab={"base_uri": "https://localhost:8080/", "height": 521, "referenced_widgets": ["6b096241436844168c1500bfbabff87d", "6a19e07723394dca9cb4487579f2730a", "18d472a1a0f3431c82fd1fab1351182a", "2bba1a512ddd4833afb5828a2cea6b2f", "0b5999c3d3c444fdba5bba52800bf0db", "475ed9d52528435b96b601029161a26c", "f8e0f039738549e1ba8ad59b0d04852c", "d2f324f306864a7ba0c2d785babefb90", "e43e3064494d4de5986602e692130dd5", "6cd54ce50544411cbd9931d2547edc73", "2f3c4cc05d76423796f478115e0719fe", "0f35b03090924cfdb36817f7f4178719", "c26ec4686f2b4f1a95b99cf18e12f49e", "c43b4b3656bb498bb2f2d123ec38a01e", "2ebc469f1a04481a93e2726beb77c301", "0deb9e8a032f493a9674411d810532e8", "878da3aacf084ddaacd0a9a2c7de9577", "fb701ed97bbd4ec8a24b181e30e6bf42", "3a4c0cd931e640f193775c2fc0065569", "6e6c5d401581468380f1d68c7449e16f", "9ec295c2d1904151ba8d22a97b81c28b", "40725fc7b70f43278fbdcd4e4f23ebd7", "d21bbd327f944e47acefe8007fcec490", "039bd3eff90442d68c308b45552ee692", "802205f3e50145e18e4a6f942242e789", "93ba365e594e4b53a58429e52940134e", "1bc7367535324533a26dd03264fb44f1", "16a2b10a6fae4805a560fbacfc2317fb", "c299486788504ae58fd1e8f00e6291ee", "6a156ade90e746f59df3e8ec93268442", "b42e53098581401ab7fa5d4d25f7a3a2", "f61af50ae1fb4db2a4e53de6560e11a7", "1392b7b274d24b8b944394f439db52fc", "77f89bd9f8444ae4961fac6e06a26514", "32b255bb83e64f828fb15d6ba157c69c", "ccfaedc2bb1641218493f6ed75f0985d", "a2927132ba9c4c32b5599a6bbed9d94a", "78ce0af368594dff961540dea2786297", "b430f6d493b64638ab1bafdeac86c0ab", "666b32962d8341a1968fe202152df3ac", "131d48e3b69a481ebbc99e509f82a0bf", "7ff0be49241e483fbbf49bc0e904acd8", "ed6b8619289c45e9b5aacaaafa3fd511", "f9eff1041fd8456cb784332939f13352", "f857416c46824818afd9c41dc24073bf", "d7eae7e8a7c04914bd715294bf12dbcc", "3e297d4e74f345169fee23b84bf27041", "3839a35f61504b8ca93d5a07f5bab495", "5babd95558234b109bfa7eb9bf9b1ed8", "86084fb47317433e9bf28e8c1bf56b7d", "6fd2d045e0cf45119d911f88d66d64f0", "f589daa03cbb4d9aba72adc7c611fee1", "691f0476ec114509a44cda71ef638ca9", "acaa4511205e4bf69da1fc16bbf6ecd2", "b286a9f0eee341108d1f5dd38573aab5", "15f68b7a00c44366a3c9021288162e0d", "d473cda9c49b4a56b82fcc35faa65a52", "aa394645109a4ad18441c510aa2dbf17", "c4012819b4ed402a8bcb0f12edc5618d", "743ecfe72e9c49f9a5876a02c93f331e", "29d8caca39074b1a8f85f3754d2e224c", "209d239500fa4c51bc95eb35ee64f343", "7e2a956049754871b5e2a531d17afaa9", "826a1aa741bc4acca4d40b73e9181bbc", "a737a42da61c4050818bbe63fb83b41d", "f01c4f605f5640659c5f9c05559438e5", "5a65390726af4c8b939d99349cc58447", "b358cee5952d4a91b3fc17a5ea2442fb", "4555b9bc83b94f3baf87aeafd682d9ef", "e278513d0afb45df9a8e92a5fa09f549", "d7f60eaa0c2f4c34a018076899f4da16", "309f9590c15644a7bfa131c20002aecb", "f5e2aa29420a4eb58615d3d05995189c", "ce19c824ed7349c8add08cfe048fd020", "91fec637b52b4823abf48c56daf4e871", "178f48bd1a2b411cb8f4543977d16b01", "67854ea28bf94fe18b63952d8de2ff21", "806b45fa970c40af91c82b4f80fdc6ad", "e09f5a3c35584655bfa4c56da93db5ec", "0e8f18b63926476ab2217c28157f32bc", "40557aa634e34e8f84406f44f9b87e50", "887b261d651d44a289f71115d919c704", "66cb9e9ac6434aa7be3483e476fd6c89", "3dd945e4db154e6e90a8f1e2e05c1564", "6fd575e8eeb243029763654a7c6ef918", "e4a107bfb15c4f7aa7c32d29ad48182b", "4ddcbab68f9046e9a8359cc82ecba1fe", "a4f7b6df67e9433aaaa09244b0818be1", "0573f36368cc42ef9c9b87f8e8999761", "0b67b1ba8ed14acaafbbeb925fcf4d0e", "5e7fa8576d834ac7938f3363dbd885c8", "0955d99ce4dc4870915885c4920e4f41", "68d54a6d0af14aa1944d323452f6907d", "89d5b1bcdb464e149ceb7583242e303b", "bcc8f9774dcb462190b94adf116188a9", "64ee243c349e45918cac5549f8159c64", "bf6d032586ec41218e5facbd09527abd", "2a4537912fc942db92dafd3a3be0d05f", "2822d0895ac24597869d73b102e640f1"]} id="22d61830" outputId="e020959f-c5c2-455a-d559-f4fb4a687785"
from transformers import GPT2Tokenizer, BertTokenizer
import tiktoken

sentence = "Tokenization helps models understand the world."

# GPT2 (via Hugging Face)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("GPT2 Tokens:", gpt2_tokenizer.tokenize(sentence))
print("GPT2 input_ids:", gpt2_tokenizer(sentence).input_ids)

# BERT Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("BERT Tokens:", bert_tokenizer.tokenize(sentence))
print("BERT input_ids:", bert_tokenizer(sentence).input_ids)

# TikToken (used by OpenAI GPT models)
enc = tiktoken.get_encoding("gpt2")
print("TikToken input_ids:", enc.encode(sentence))

# %% [markdown] id="cacc30b9"
# ## 3. 🧠 Understanding Byte Pair Encoding (BPE)
#
# BPE is like learning to read by first learning letters, then common letter pairs, then longer combinations. Here's how it works:
#
# 1. Start with all individual letters as tokens
# 2. Find the most common pair of letters that appear together
# 3. Combine that pair into a new token
# 4. Repeat until we have enough tokens to handle most words
#
# Let's build our own simple BPE tokenizer step by step!

# %% [markdown] id="fee328ee"
# ## 4. 📄 Our Training Data
#
# We'll use a small dataset of stories to train our tokenizer. This will help us learn common patterns in text.

# %% colab={"base_uri": "https://localhost:8080/"} id="tqatvGRRUNv1" outputId="0fd7cecd-8f7b-4d36-b019-50f3fa9362b6"
# !pip install datasets

# %% id="7d4dd459" colab={"base_uri": "https://localhost:8080/", "height": 273, "referenced_widgets": ["096d0d2875db4365bbf3b5396c53eec7", "058dafd3fbfd472d808083315050e320", "bf066ce618c5451a914470777d52993c", "05808b3ce7e44c39bb97ccaf18bfc791", "2e530c498e034924abc358c8d26af935", "9115845c89a0457d81017f5d59ee0358", "25bc9f0aae6b496ba36438f56cf21172", "7511b6c19ee2429e894e40ea067508bd", "6cd14ca1a3a746d699199fa2a35cf2cc", "1137a4b946ea45e5af982439f9c47203", "6d6bf92f4b054117bbab7090d135c138", "e9d2f1143f804d679d2aed3a7a32ceb2", "31c56d1e348449e88de4b9841c48cb7d", "2304eb86ef4a40438cea1dd5b6c79db5", "42cad1a174d74c469194349a3059588c", "d4fd7f4e2f5c44a2b2c5ab4958809219", "e34fc60d9e0743699c907c64e99ef10d", "a6f9985f28d54bec8e1de39727b4a5b6", "07d2f0ad879e4488929cdb33eacfd787", "a95efe2d0e7140a19b33c3b086f33697", "bd09f85a696148f19669bc303f3fe4ac", "9bdde8661ab24601af306253eae43c25", "1aaa5ad9ef114cf586b92c4304a194d9", "5d546e2dc77848b2ac66ab4af6911f50", "6caee68576f9404f9d172de0f149a54f", "c662a102f9f74abf92508b9fac68c537", "90a9b7fc1f6c44c5b7c7eee42a0a92e3", "026f9740bd964ad092f59c473ab190f3", "f3adcdb6f1494703b9dcd98059a8e531", "8f22c0cd12af469cbdfa52b418c79d28", "01d624a7fcbd4c9090cd0231a87b24ec", "05ee4a298e1542019a5402a0e98f68db", "238b49be1a8d432395a049003a1451d1", "3296a064864b44ad96b693da1154fc44", "523465dc1b7d4a0c9e11745702630281", "ff536462928e4c3d8086fe286439766d", "572847149c0540bf9f5e20b74f916812", "d46caf3ad9ed4811988c35f64f759ba8", "e3404d6e42064719953b10487c5388d7", "1387f5d95a4e4cba8752595e4cd70917", "37d479bc4e7b46c5abea2b265f203365", "079d888c166041909cdf36d4a1a792f6", "98a3452422334b0cac1acd0a474da792", "279a4ae7b69141619a3fce568fc90f7d", "4d5b8d35fa9942cdace4d130a6ac2247", "e36d122a424d4d26bdae8bc60bf6b6d7", "ff7c65d1f2144fbf808d6f2fa7a70545", "fdebddde19f549658f173331d080bb7e", "e794b4a7a6394371b78db9feae74e622", "b953dc7947e745b1b3262078492ab6e2", "c3c5428e9fa148478ad8f9d9c73f45d1", "9575563e42f348458526a79f29b71f0b", "50d181c386214493924b63612bb135ca", "17221a66eb964fb09ac24ed3b18e2ad1", "fd09615a0dd0485f9fe780d61e3f08d2", "cda6be36b9134c6dbed9021b5c948547", "db7e60886f1648bc81404ac8ed16ee04", "2665421458f6482088655f5644235722", "8d10adba5aef4a06b215ec90f5323e7e", "f98df58be8924d87a281505cf4a145a2", "b2eb25a1bf7a4abf8b48455391926d47", "3a4b04e8ba8f41eba930021c97369151", "4419a440016a486c99af3e38b9cac38f", "cad4b3a9e9b9492a95531b56cb5a35ac", "de5c8f1ced484e0a88fc040841a94e8e", "fa245fe1d98b40cf8e4f72a5f986175c", "8cedd379f85f4c5583ba4f48c1e09ad9", "9e1cbd7c2aca428ebd17aa7c6c4aec01", "0622a129da6e47e5bc2972b37b05ba41", "73804d1c3e7f40b481267b2edb24621c", "d55fc2693a644153b630e435105f1af0", "3c918a3d2b444520960eca3eb42116c8", "7395992ded0e4374bcc8a00d3b24708e", "d1b7b9f1bf0148f699ccbe4e91d99074", "4f612fbf16a94755af634ae206ace27d", "9743bf99a7804b799556a8a9d680b442", "4c533657f92a46fdad43d1fcb17ddd04", "8f4354b2b62b4dc9b40135571172fb56", "549900e63d2c4b9e826e4626f92b5a94", "8338f8052a334115bebc1ce32aba3818", "0ae103bc5c1f4fa1a89c7a0e9ce79886", "be2ac5d6f58d40218bce3606936df785", "2799fcf2eee248908564a4f983835f3e", "c7d15120c61c4ef292272d51d817d202", "6b9d19cf5d514d868738ab14a4f219bf", "fee40ff6614d41ee912fcd04b657137f", "b6b41525b50043809ed42f2e5a0168fc", "161f539471ab41968430e1af8832502e"]} outputId="4d8c0a0b-8df4-446d-a867-edce0cb9d34b"
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train")

# Convert to a list of strings (using the 'text' field)
# For faster training feel free to only keep the first 1000 examples : corpus = corpus[:1000]
corpus = [example["text"] for example in dataset]

# %% [markdown] id="4a1d3666"
# ## 🔧 Step 1: Pre-tokenization
#
# First, we'll count how often each word appears in our text and split each word into its letters. This helps us find common patterns.

# %% colab={"base_uri": "https://localhost:8080/"} id="85975841" outputId="e4cd0e2c-9884-4d03-aa26-00ab0638a172"
from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    for word in text.split():
        word_freqs[word] += 1

# We also need to split each word into individual characters for pair counting and merging
splits = {word: [c for c in word] for word in word_freqs.keys()}
print(word_freqs)
print(splits)

# %% [markdown] id="TTOwryd_eO2f"
# Now we need to create our starting vocabulary with all the letters we see in our text

# %% id="cDRjZjiNeVWA" colab={"base_uri": "https://localhost:8080/"} outputId="683f4a31-5056-49cb-82c7-a799a7890408"
alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)

alphabet.sort()

# we need to add some special tokens like for example the end of text
vocab = alphabet + ["<|endoftext|>"]
vocab

# %% [markdown] id="ccd34cc6"
# ## 🔧 Step 2: Count Token Pairs
#
# Now we'll count how often pairs of letters appear together. This helps us find common patterns to combine.

# %% colab={"base_uri": "https://localhost:8080/"} id="4782b38a" outputId="47d4da61-19bc-4923-ce7e-ad0fb787785e"
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# %% [markdown] id="f6435d66"
# ## 🔧 Step 3: Merge Most Frequent Pair

# %% [markdown] id="Ns9vM0UDqK61"
# Let's find the pair of letters that appears most often in our text

# %% colab={"base_uri": "https://localhost:8080/"} id="0WlM0KxZqP7w" outputId="d77b8eb6-d0ae-4cf9-886d-39a98f2585ef"
best_pair = ""
max_freq = None

for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq

print(best_pair, max_freq)

# %% [markdown] id="ypek5u42Ouqr"
# Now let's combine this most frequent pair into a new token and update our vocabulary

# %% id="69f7f7b5"
def merge_pair(a, b, splits, vocab):
    vocab.append(a+b)
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a+b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits, vocab

# %% [markdown] id="e1047334"
# ## 🔁 Step 4: Full BPE Training Loop
#
# Now we'll put everything together to train our tokenizer. We'll keep merging pairs until we have enough tokens to handle most words.

# %% id="9bd6d7a5"
def train_bpe(vocab, splits, vocab_size=1000):
    merges = {}
    vocab = vocab.copy()
    splits = splits.copy()
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
                merges[best_pair] = best_pair[0] + best_pair[1]
        splits, vocab = merge_pair(*best_pair, splits, vocab)
    return merges

# This might take a lot of time, you try to reduce the vocab size to get a fertility of at least 2 (for fertility def check below)
merges_1000 = train_bpe(vocab, splits, vocab_size=1000)
merges_10000 = train_bpe(vocab, splits, vocab_size=10000)

# %% [markdown] id="14f0545f"
# ## ✅ Step 5: Tokenize New Words
#
# Now we can use our trained tokenizer to break down new words into tokens!

# %% colab={"base_uri": "https://localhost:8080/"} id="c36b8b86" outputId="97370cd9-c5c2-455a-d559-f4fb4a687785"
def tokenize(text, merges=merges):
    splits = [[l for l in word] for word in text.split()]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return {"input_ids": sum(splits, [])}

print(tokenize("Hello lower lowest", merges=merges_1000))
print(tokenize("Hello lower lowest", merges=merges_10000))

# %% [markdown] id="f133ed5a"
# ## 📖 How Good is Our Tokenizer?
#
# Let's measure how well our tokenizer works by looking at a few metrics:
#
# - **Average Sequence Length**: How many tokens we get for a piece of text (lower is better)
# - **Compression Ratio**: How much we can compress text (higher is better)
# - **Fertility**: Average number of tokens per word (lower is better)

# %% colab={"base_uri": "https://localhost:8080/"} id="AWmLO3bpNdEF" outputId="796e5aa8-8360-42a5-e563-21cd045f004c"
def tokenization_efficiency(texts, tokenizer, merges=merges):
    total_chars = sum(len(text) for text in texts)
    if merges is None:
        total_tokens = sum(len(tokenizer(text)["input_ids"]) for text in texts)
    else:
        total_tokens = sum(len(tokenizer(text, merges=merges)["input_ids"]) for text in texts)

    avg_tokens_per_text = total_tokens / len(texts) if texts else 0
    compression_ratio = total_chars / total_tokens if total_tokens else 0
    fertility = total_tokens / sum(len(text.split()) for text in texts)

    return {
        "average_tokens_per_text": round(avg_tokens_per_text, 3),
        "compression_ratio": round(compression_ratio, 3),
        "fertility": round(fertility, 3)
    }

print("Our Tokenizer with vocab_size 1000: ", tokenization_efficiency(corpus[:100], tokenize, merges_1000))
print("Our Tokenizer with vocab_size 10000: ", tokenization_efficiency(corpus[:100], tokenize, merges_10000))
print("GPT2 Tokenizer : ", tokenization_efficiency(corpus[:100], GPT2Tokenizer.from_pretrained("gpt2"), merges=None))

# %% [markdown] id="HDBobNhANkod"
# ## 🎯 Summary
#
# Congratulations! You've built your own tokenizer from scratch! Here's what we learned:
#
# 1. How AI models break down text into tokens
# 2. How to build a simple BPE tokenizer
# 3. How to measure how well a tokenizer works
#
# 🎓 Try using this tokenizer on different types of text or in different languages to see how it performs!
