# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/speech_wav2vec_dodaaudio.ipynb)
# ---
# # 🎤 Welcome to Speech Recognition with AI! 🇲🇦
#
# In this notebook, you'll learn how computers turn Moroccan Darija speech into text using AI. No experience needed! We'll guide you step by step, explain every new word, and show you how to use powerful models like Wav2Vec2.
#
# **What will you do?**
# 1. Listen to and visualize real Moroccan speech data.
# 2. Learn what a speech recognition model is, and how it works.
# 3. Try out a pre-trained model (so you don't need a big computer!).
# 4. (Optional) See how to train your own model if you want to go further.
#
# **All code runs on free Google Colab!** If training is slow, we'll show you how to use a ready-made model.
#
# Let's get started! 🚀
#
# ---
#
# ## What is Speech Recognition?
#
# **Speech Recognition** (also called Automatic Speech Recognition, or ASR) is when a computer listens to your voice and writes down what you said. It's used in voice assistants, subtitles, and more.
#
# ## What is Wav2Vec2?
#
# **Wav2Vec2** is a special AI model that can turn raw audio (your voice) into text. It was trained on thousands of hours of speech, so it can understand many accents and ways of speaking.
#
# ---
#
# ## Moroccan Context
#
# In this notebook, we'll use a dataset of Moroccan Darija speech. This is perfect for local projects and helps AI understand our unique way of speaking!
#
# ---
#
# ## How to Use This Notebook
#
# - Run each cell one by one (Shift+Enter in Colab)
# - Read the short explanations before each code cell
# - If you see a word you don't know, check the explanation or ask your mentor
#
# ---
#
# ## TODO: Add a fun image about speech recognition here!
# ---

# %% [markdown] id="J89BBMBuPaYQ"
# # Section 1: Welcome to the World of Speech Recognition!
#
# Hello and welcome to this interactive introduction to Automatic Speech Recognition (ASR), also known as Speech-to-Text (STT)!
#
# ASR is a fascinating field of artificial intelligence that enables computers to understand and transcribe human speech into text. It's a technology that bridges the gap between human language and machine comprehension, unlocking a vast array of applications that make our lives easier, more productive, and more accessible.
#
# In this Colab notebook, our primary objective is to explore the world of speech recognition.
#   1. You'll learn how to handle speech data, understand its characteristics through visualizations
#   2. We will then dive into a powerful model called Wav2Vec2, understanding its architecture and the clever techniques like self-supervised and contrastive learning that make it so effective.
#   3. We will fine-tune this pre-trained model on a specific dataset (Darija Speech ^^)
#   4. Finally, you'll learn how to test your model's performance and even share it with the world by deploying it on the Hugging Face Hub.

# %% [markdown] id="ss2gCskC9DH8"
# # 👋 Need Help with Anything Speech-Related?
# If you're interested in incorporating text-to-speech or any speech component into your project, feel free to reach out to Yassine El Kheir at yassine.el_kheir@dfki.de.
# I'd be happy to give you a hand and share some helpful tips!

# %% [markdown] id="Pg_9Wv2aQvk4"
# # Section 2: Getting Our Hands Dirty - Exploring Speech Data
#
# ## 2.1 Setting Up Our Toolkit: Installing Libraries
#
# We'll use a few Python libraries to work with speech data. If you're running this on Colab, just run the cell below to install them. (If you see an error about a missing library, run this cell again!)
#
# - `datasets`: lets us easily download and use speech datasets
# - `torchaudio`: helps us load and process audio files
# - `librosa`: for audio analysis and making cool visualizations
# - `matplotlib`: for plotting graphs and images
# - `IPython`: lets us play audio directly in the notebook
#
# **Run this cell if you need to install the libraries:**
# ```python
# !pip install -U datasets torchaudio librosa matplotlib IPython
# ```

# %% colab={"base_uri": "https://localhost:8080/"} id="jWbBlQXzPIJe" outputId="ddef743f-576b-408b-c2a0-ba1b23701203"
# !pip install -U datasets
# !pip install torchaudio
# !pip install librosa
# !pip install matplotlib
# !pip install IPython

# %% [markdown] id="L7GEisZ4RQAO"
# ## 2.2 Loading a Speech Dataset from Hugging Face
#
# Hugging Face Hub is an amazing resource that hosts thousands of datasets and pre-trained models. For this tutorial, we'll use the `atlasia/DODa-audio-dataset`. This dataset contains moroccan darija audio samples that we can use to explore and later, to fine-tune our model. To load this dataset, we will use the `load_dataset` function from the `datasets` library.
#
# ---
# ## Step 2: Load Moroccan Speech Data
#
# A **dataset** is just a big collection of data. Here, we'll use a dataset of Moroccan Darija speech from Hugging Face. Each entry has:
# - The audio (what was said)
# - The text in Darija (Latin and Arabic letters)
# - The English translation
#
# We'll use the `atlasia/DODa-audio-dataset`, which has real Moroccan voices. This helps our AI learn how we really speak!
#
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="SytdujR-RNsC" outputId="f4eed0f1-6ee7-4652-fa65-dab7c7145a7d"
from datasets import load_dataset
from huggingface_hub import login

token = "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"

# Authenticate your session
login(token=token)

# Load the dataset
dataset_name = "atlasia/DODa-audio-dataset"
dataset = load_dataset(dataset_name, split="train[:50%]")  # 1% for testing
print(dataset)

# Show an example
if 'dataset' in locals() and dataset:
    print("\nExample entry:")
    print(dataset[0])


# %% [markdown] id="iPqV4SaSVqex"
# This snippet loads a small portion of the dataset and prints its structure, allowing you to inspect the available fields for each audio sample. Each entry includes:
#
# - **`audio`**: the raw waveform (`array`), file path (`path`), and sampling rate (`sampling_rate`)
# - **`darija_Latn`**: the transcription of the speech in Darija using Latin characters  
# - **`darija_Arab_new`**: the processed Arabic script version of the transcription  
# - **`darija_Arab_old`**: an earlier version of the Arabic transcription  
# - **`english`**: the English translation of the utterance
#
# These fields provide a rich representation of the data across scripts and languages.
#
# ## 2.3 Listening to Speech 🎧
#
# Before training any AI, it's good to listen to your data! This helps you understand:
# - How people really speak (accents, speed, background noise)
# - If the recordings are clear or noisy
#
# Let's listen to a few Moroccan Darija samples from our dataset. Try to notice different accents or any background sounds!

# %% colab={"base_uri": "https://localhost:8080/", "height": 557} id="4TkeKtKGoSIc" outputId="49cfa14f-ef68-4fe5-c2b9-a4b2141e5f2e"
import IPython.display as ipd
import numpy as np

if 'dataset' in locals() and dataset and len(dataset) >= 3:
    for i in range(3):
        sample = dataset[i]
        audio_data = np.array(sample["audio"]["array"], dtype=np.float32)
        sampling_rate = sample["audio"]["sampling_rate"]

        print(f"\n--- Sample {i + 1} ---")
        print(f"Darija (Latin): {sample['darija_Latn']}")
        print(f"Darija (Arabic - new): {sample['darija_Arab_new']}")
        print(f"Darija (Arabic - old): {sample['darija_Arab_old']}")
        print(f"English translation: {sample['english']}")
        print(f"Sampling Rate: {sampling_rate} Hz")
        display(ipd.Audio(audio_data, rate=sampling_rate))
else:
    print("Dataset not loaded or contains fewer than 3 samples.")


# %% [markdown] id="yoX7U_CAWjIl"
# ## 2.4 Visualizing Speech: Waveforms and Spectrograms
#
# Listening to audio is intuitive, but visualizing it can reveal patterns and characteristics that are not immediately obvious to the ear. Two common ways to visualize audio are waveforms and spectrograms.
#
# **Waveforms:** A waveform is a graphical representation of an audio signal that shows the changes in air pressure over time. It's the most direct visual representation of the raw audio data. From a waveform, you can get a sense of the loudness and a rough idea of the speech segments versus silence.
#
# **Spectrograms:** A spectrogram is a more informative visualization. It shows the spectrum of frequencies of a signal as it varies with time. In a typical spectrogram, the horizontal axis represents time, the vertical axis represents frequency, and the intensity or color of each point represents the Amplitude/Energy of that particular frequency at that particular time. Spectrograms are incredibly useful in speech recognition because they highlight the phonetic(characters sounds) content of speech, making different sounds visually distinct. They essentially show *what* frequencies are present *when* and *how strongly*.
#
# Let's generate and display these visualizations for one of our audio samples.
#
#

# %% [markdown] id="aOKavtIEn3qy"
# ## 🔍 Instructions
#
# 1. Play the audio using the embedded player.
# 2. Observe the **waveform** — it shows amplitude over time.
# 3. Look at the **Mel spectrogram** — it shows how energy is distributed over frequency and time.
# 4. Try to **link sounds** to parts of the spectrogram:
#    - Vowels (like /a/, /i/) → smooth bands in low-to-mid frequencies
#    - Fricatives (like /s/, /sh/) → noise in high frequencies
#    - Stops (like /b/, /t/) → sudden bursts or gaps
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 997} id="o0VntrdHWhP8" outputId="db52b8fb-0b9a-448a-ab79-4ac5f0bce3c3"
import librosa
import librosa.display
import matplotlib.pyplot as plt

if 'dataset' in locals() and dataset and len(dataset) > 0:
    sample = dataset[0] # Take the first sample again or a different one
    audio_data = np.array(sample["audio"]["array"], dtype=np.float32)
    sampling_rate = sample["audio"]["sampling_rate"]

    # 1. Visualize the Waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio_data, sr=sampling_rate)
    plt.title(f'Waveform (Sampling Rate: {sampling_rate} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    # 2. Visualize the Spectrogram (Mel Spectrogram is common)
    # A Mel spectrogram uses the Mel scale for frequencies, which is closer to human auditory perception.
    S = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_dB, sr=sampling_rate, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram (Sampling Rate: {sampling_rate} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Mel)')
    plt.tight_layout()
    plt.show()

else:
    print("Dataset not loaded or empty, cannot visualize audio.")


# %% [markdown] id="i1Qj-1E7xZ48"
# Try to **link what you hear with what you see** in the spectrogram. This is the core of understanding sounds visually!
#
# If you would like to understand Frequencies further, please contact yassine.el_Kheir@dfki.de
#

# %% [markdown] id="F71I2SFh10ey"
# # Section 3: The Journey from Sound to Text - Understanding the Pipeline and Its Hurdles
#
# Wav2Vec2 is a pretrained model for Automatic Speech Recognition (ASR) and was released in [September 2020](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) by Alexei Baevski, Michael Auli, and Alex Conneau.
#
# Using a novel contrastive pretraining objective, Wav2Vec2 learns powerful speech representations from more than 50.000 hours of unlabeled speech. Similar, to [BERT's masked language modeling](http://jalammar.github.io/illustrated-bert/), the model learns contextualized speech representations by randomly masking feature vectors before passing them to a transformer network.
#
# ![wav2vec2_structure](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/wav2vec2.png)
#
# For the first time, it has been shown that pretraining, followed by fine-tuning on very little labeled speech data achieves competitive results to state-of-the-art ASR systems. Using as little as 10 minutes of labeled data, Wav2Vec2 yields a word error rate (WER) of less than 5% on the clean test set of [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) - *cf.* with Table 9 of the [paper](https://arxiv.org/pdf/2006.11477.pdf).

# %% [markdown] id="HQITTqSe2MjJ"
#
#
# Wav2Vec2 is fine-tuned using Connectionist Temporal Classification (CTC), which is an algorithm that is used to train neural networks for sequence-to-sequence problems and mainly in Automatic Speech Recognition and handwriting recognition.
#
# I highly recommend reading the blog post [Sequence Modeling with CTC (2017)](https://distill.pub/2017/ctc/) very well-written blog post by Awni Hannun.

# %% [markdown] id="HCaDqtz92XnY"
# First, let's try make sure you have GPU set ...

# %% colab={"base_uri": "https://localhost:8080/"} id="OY-2Gd5povEG" outputId="9a04db1c-cddb-496f-ce6e-5478637a67c0"
# gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

# %% [markdown] id="dCKA8Ga65483"
# ## No GPU? No problem!  
# Just activate the GPU in Colab (ask a mentor if you need help), then start from here.

# %% [markdown] id="kWI7Q_F73PFu"
# ## 3.1 Preparing for Fine-tuning: Data is Key
#
# Before we dive into the code, let's discuss data preparation for fine-tuning Wav2Vec2:
#
# 1.  **Labeled Dataset:** You need a dataset where each audio sample has an accurate corresponding text transcription. We will continue using a subset of the `atlasia/DODa-audio-dataset` for this demonstration.
# 2.  **Consistent Sampling Rate:** Wav2Vec2 models are pre-trained with audio at a specific sampling rate (commonly 16 kHz). All audio in your fine-tuning dataset *must* be resampled to match this rate. The `datasets` library can help with this.
# 3.  **Vocabulary Definition:** The model needs a vocabulary, which is the set of all possible characters (or subwords) it can predict. This vocabulary is created from the transcriptions in your training dataset.
# 4.  **Processor/Tokenizer:** A `Wav2Vec2Processor` (or `Wav2Vec2CTCTokenizer` for older versions) handles both audio preprocessing (like resampling and normalization) and text tokenization (converting transcriptions into sequences of IDs based on the vocabulary).
#
# ## 3.2 The Fine-tuning Process: An Overview
#
# The fine-tuning process generally involves these steps:
#
# 1.  **Set up the Environment:** Install necessary libraries like `transformers`, `datasets`, `evaluate`, and `accelerate` (for efficient training).
# 2.  **Load Dataset and Processor:** Load your speech dataset and a pre-trained Wav2Vec2 processor (which includes a tokenizer).
# 3.  **Preprocess Data:** Resample audio, tokenize transcriptions, and prepare the data in a format suitable for the model.
# 4.  **Load Pre-trained Model:** Load a pre-trained Wav2Vec2 model suitable for ASR (e.g., `Wav2Vec2ForCTC`).
# 5.  **Define Training Configuration:** Specify training arguments like learning rate, batch size, number of epochs, and evaluation strategy.
# 6.  **Define Evaluation Metrics:** Choose metrics to evaluate your model, typically Word Error Rate (WER) and Character Error Rate (CER).
# 7.  **Define Data Collator:** A data collator is responsible for batching your processed data samples and padding them so that all sequences in a batch have the same length.
# 8.  **Instantiate and Run Trainer:** Use the Hugging Face `Trainer` class, which simplifies the training loop, handles evaluation, and saves checkpoints.
#
# Let's get to the code!

# %% id="LoIBjgeS3Ogk"
# !pip install transformers[torch] evaluate accelerate torchaudio librosa
# !pip install -U datasets

# %% [markdown] id="JxvfD5jJ3lZH"
# ### 3.2. Load Dataset and Create Vocabulary/Processor
#
# We will use the `atlasia/DODa-audio-dataset` again. We need to extract all unique characters from our transcriptions to build a vocabulary.
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 310, "referenced_widgets": ["5b50a9d591be46ea80ea56445d191a1b", "55479db9fa884146bc412f01e5a5e0b4", "a2c1466c8caa4fb2b4c300ee19c5d5ff", "b9d85ebb57c84439b70d886766ddd550", "abb274cce9954bf9adf72f4af6f2ed55", "7f6bcac9d82c43fcb34512baebde607b", "b4ffba4e4ed9487b811d4d8e0868ccb6", "8ba91cd6fc544e408b924bd0d7f38b77", "adf6d44a94f4456bb07478fbac618566", "1c7a9bee5e6b4595a7afb11e3bf441b6", "5741b298b79441cba7e9891c8f61c499", "8fee0a3f07a043319cb22bfe7191cce9", "432ad956ce5d4014beb911a7f9f4c4a4", "add05af3ab844e97a7cb1047ae36b3dd", "0bb0a25607644e34954e681e752847d3", "448b6d29d34343c2bd2b9dfd4d6b9e57", "aac037c5fa8a483db8557c6a53fecc8e", "5226c138e05c4df387378f1ff93895b0", "04e41ef9bd6c4d27a666fc1d389163de", "84d377dfa8d04962a8aa220a50858cfe", "3da26e8f8be64fb1ab57b8c6947e8ba1", "d01fbd0f38fb4f9096f4b0adf6882a29", "137e1ce26e2d44728744e1fd07b8ac2c", "63c84fb93ba743daab54c76ec5c5f020", "d4b452a12a294f92b476e6364d405b9a", "794799eeab4d4bc7b8f98217625d30ea", "191e2698304e4f46b5780c07dc4c1c19", "9771afb474324224b611b0d5cc78d60a", "6435098bbf854011b488f271c330f66a", "9eec9f75c78a4476a106d84987680570", "47a741adab914bb08c35519381f18df2", "91d7c1b6e27f4cca9e4a3fe98ffc4e71", "b4f5e97f07eb4a279fb306ed7962ead5", "5203d1d126be424581217c3630fc19c4", "7ce44e1657114ca9a522adefc1df2a6d", "3c6a554fba09482290b450507c2f99e1", "70f684dadfd84bae940b141a6090711c", "32133ce0bd4d4963aabf95240bd10709", "5f79f4d9668b48eeb40cb714f7b26d57", "34e428951a104a95945885cf3a82f638", "d41fb3f66c034406aa21bdee01839ca5", "b642855f63f54e7397d3198a05138e56", "3aba6a0492604b18b73765d1b4848881", "07212f43430a4346a9a533e8dbff57fb", "4452862f35e84df78b40f70aa1cd4737", "f99945c834f9451ca7011b20309481e4", "2f80b059f6e84753b76d99405b4c8a34", "812e24b959c34eeebdd14b831fd1434c", "e40cad78436a4fff8f0b51cc140fb9af", "17051eabb679413bac267b17b6dc9c2d", "f82bf0383cc742b0a9daa3e56163b809", "db51108f7b9445689a399985d5e7b5c4", "cb2fe3717fca409fb21493f169bef6ae", "388e447414e4443aa97ec9c825788882", "8a175fcef9c74f189f692dcf9050edaf", "550b5622e54b48ee976e38da5951c5ac", "792d557ed9c9432a8fa73a85b101d7cb", "c6f33b033de14ffea566944c627dd523", "dc4f1ee4d542452ebd651c45b4df3257", "122829d4e0da4549aa8194cb607ed1a5", "04577d29a77f430fb432096c8506efa0", "9587f95d97554b74ae46de4510ea3ee9", "8eae73a745f049798a9af08ce46dba13", "8829990eea9649f08c58bed58552f7b3", "da603794eac0429db96022eb74440463", "0eec4df1285b4a5ebd9da3b9bd1b8682", "64d81607f27c4f14b8de626d7ea2a67d", "9a117d5e09584ccf948cf84b1ebd579e", "60807204e1ab4637b3836184186ef1e7", "24783ed108864b35882129cfaef633d1", "aa1728ee0ee546fb881d23ef89996bb4", "97218a29190c4870b3de6be1455fde62", "50fd0ef78a8746cca02de0f31df01759", "3cd4371727974675a5ee24336958a771", "b435edc2e88a483b8267875fc05cd5e4", "862900e1e12b4591aec300283bc67ccb", "97484432121740f8bdee4ea3ac330b64"]} id="XITXfAzN24cu" outputId="d7477634-5bc2-4a85-b722-ffd991ec8155"
from datasets import load_dataset, Audio
import re
from huggingface_hub import login


token = "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"

# Authenticate your session
login(token=token)

# --- 1. Load a small subset of the dataset for this demo ---
dataset_name = "atlasia/DODa-audio-dataset"
try:
    raw_dataset = load_dataset(dataset_name, split="train[:50%]", token=token) # Using only 20 samples for speed
    print("Raw dataset loaded:", raw_dataset)
except Exception as e:
    print(f"Error loading dataset: {e}. Please ensure authentication if needed.")
    raw_dataset = None


# %% [markdown] id="Q1kGEIT46QTy"
# We'll use the ***`darija_Arab_new`*** column as our target — it's already preprocessed (Yaaay! 🎉... faster coding ahead 🚀).
#

# %% id="bUviXALg3wB4"
# --- 2. Define a function to extract all characters from the transcriptions ---

def extract_all_chars(batch):
    transcription_key = "darija_Arab_new"
    all_text = " ".join(t for t in batch[transcription_key] if t is not None)
    vocab = list(set(all_text.lower())) # Convert to lowercase and get unique chars
    return {"vocab": [vocab], "all_text": [all_text]}


# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["07f4531cfd224055bd5c97fb469807e9", "c423936059d24e519efab0a7f4e34ba5", "a682291dd2644dd0854eff2273989f50", "fb63eb6f47244539aa1ceb607b7c4e59", "61a118d64f3e462597de1f22f9fddcb7", "648e13f2e7e9467494e1dcf99f160bd6", "1430c50e55704a6db086f76f033b8c32", "d234a887552a48c79f4739419d59111b", "a1d264ceeff74dc78a5b72163e54dee0", "d665913f5409459e8b52fe1c57d8c55a", "f709119137e84c5e8b6ed553fbc73e20"]} id="u5AEk8N23yX4" outputId="9c5359cd-155c-4a24-beca-f4240bf53f26"
# --- 3. Create the vocabulary ---

vocabs = raw_dataset.map(extract_all_chars, batched=True, batch_size=8, keep_in_memory=True, remove_columns=raw_dataset.column_names)
vocab_list = list(set(c for vocab_item in vocabs["vocab"] for c in vocab_item))

# %% id="QcWuA4rY7U4s"
vocab_list

# %% [markdown] id="MGyvnW55_Bze"
# ### 🧠 Let's Talk About Vocabulary!
#
# When you look at the list below, you'll see lots of strange or unnecessary characters — things like punctuation marks, special diacritics, and even numbers.
#
# ```python
# # This is an example of a "messy" vocabulary extracted directly from raw text data
# messy_vocabulary_example = ['','ش','ر','َ','؟','ب','0','ص','ؤ','ث','پ','9',',','3','ء','?',
#  'ة','خ','غ','ي','ف','-','ى','2','أ','ك','ظ','و','1','ّ','ُ','ت',
#  'ض','ح','س','ئ','د','5','ه','ل','ڤ','ط','ڭ','إ','!','"',' ','ا',
#  ':','6','ق','ذ','ز','ن','ج','.','،','ع','ً','م','آ']
# ```
#
# ### ❓ But wait... why does our vocabulary look like this?
# Because it was built automatically from data not fully cleaned, not filtered. So, yes:
#
# ✅ it includes Arabic letters (like ش, ر, ب)
# ✅ but also punctuation (like ؟, ,, .), numbers (0-9), and even other symbols (like ','،, َ, ً, ُ, ّ)
#
# ### 💡 Hint for You & Your Thoughts
# Try to think about:
#
# Which of these symbols are actually important for recognizing spoken words?
# For example: do we need to distinguish between "hello." and "hello!" or "hello,"? Does these sounds different? Do we need punctutations??
#
# Which symbols might be useless or even harmful during training of a speech recognition model?
# Could they confuse the model? => Can you select some?
#
# If you were to clean this vocabulary:
#
# What would you keep, and what would you remove or replace?
#
# Let's do some further cleaning ...

# %% id="j9CTzMGOE7y_"
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    if batch["darija_Arab_new"] == None:
      batch["text"] = "خاوي"
    else:
      batch["text"] = re.sub(chars_to_ignore_regex, '', batch["darija_Arab_new"]).lower() + " "
    return batch

def remove_digits(batch):
    ## to be implemented ...
    return batch

def normalize_hamza(batch):
    ## to be implemented ...
    return batch


# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["d7d679602f5f4e8cba69fbc01d9fdc43", "bef334e73b2544ee8143594c9007a44e", "78c3c4b519dd4142968b6dc34cb7a213", "e11937f0e06a4b47b80bc5abe933fb3b", "dbf695974f9a4097bb6486418d346c50", "a3f33cb391964a21ba2155e9a7e34fc7", "64318fb343bd47c5b7426f23356b73fe", "fb2753188b56428ba301a69cfddd3db2", "dfd22019ab074fed8ec9725669e7fb04", "be66b6ff880f43a2b19e4ae8f2d3cb91", "831ef58813184ab18157a495184fd757"]} id="drNgCBhKF4iZ" outputId="a5b4f835-9cb1-4f1f-da0b-e5821c518edf"
raw_dataset = raw_dataset.map(remove_special_characters)
# raw_dataset = raw_dataset.map(remove_digits)

# %% colab={"base_uri": "https://localhost:8080/"} id="SV44UzmhGWFo" outputId="b451bde7-41e8-4927-cba8-374f9f171c72"
raw_dataset


# %% [markdown] id="K0BkstwWGXlT"
# Do you see a new column names `"text"` ?? Make sure you have:
# Then now, let's make a cleaner Vocab ...

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["e35a2fca802b4852b0c3216c76fd0002", "3d9ac0daa7bb4f53ad83125a0c76bbd6", "aafb5be4f3264879875f616d1015d5de", "61aead591e334dab8329324094a1d620", "5a33489b5bdb436d8de0f49ebd4f23c5", "db56a9dbf95d4d7fa53e4fddbf132d5c", "97e6e450d91448a8ac8b9c1decf11268", "c12ba7447c214911945fbc61930c1928", "6a721f10295d41b4b35bfa32c6f45ae9", "6f792cfae3c548458b50bb0232f2b881", "5d4e6370e18a4bb68d63b7f528d3a5a5"]} id="KXTsqst5Gh67" outputId="1164efc8-7003-407b-f76e-c496797c98e8"
def extract_all_chars(batch):
    transcription_key = "text"
    all_text = " ".join(t for t in batch[transcription_key] if t is not None)
    vocab = list(set(all_text.lower())) # Convert to lowercase and get unique chars
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = raw_dataset.map(extract_all_chars, batched=True, batch_size=8, keep_in_memory=True, remove_columns=raw_dataset.column_names)
vocab_list = list(set(c for vocab_item in vocabs["vocab"] for c in vocab_item))

# %% id="A5Fqcs0X7iDW"
## make the dictionary as json: mapping characters == Numbers
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

# %% [markdown] id="ciX28_eSBErb"
# To make it clearer that `" "` has its own token class, we give it a more visible character `|`. In addition, we also add an "unknown" token so that the model can later deal with characters not encountered in Timit's training set.
#
# Finally, we also add a padding token that corresponds to CTC's "*blank token*". The "blank token" is a core component of the CTC algorithm. For more information, please take a look at the "Alignment" section [here](https://distill.pub/2017/ctc/).

# %% id="npbIbBoLgaFX"
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# %% colab={"base_uri": "https://localhost:8080/"} id="znF0bNunsjbl" outputId="c11262b1-96f4-41a1-ba5b-3fe60443d9fb"
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

# %% [markdown] id="qosTfNd6CCxG"
# ### 3.3. Create Tokenizer -- Wav2Vec2CTCTokenizer

# %% [markdown] id="SFPGfet8U5sL"
# **Cool**, now our vocabulary is complete and consists of less than 63 tokens, which means that the linear layer that we will add on top of the pretrained Wav2Vec2 checkpoint will have an output dimension of (total number of vocab).

# %% [markdown] id="1CujRgBNVRaD"
# Let's now save the vocabulary as a json file.

# %% id="ehyUoh9vk191"
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# %% [markdown] id="SHJDaKlIVVim"
# In a final step, we use the json file to instantiate an object of the `Wav2Vec2CTCTokenizer` class.

# %% id="xriFGEWQkO4M"
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# %% [markdown] id="TJOZ-uxUCNo3"
# ### 3.3. Create Feature Extractor -- Wav2Vec2FeatureExtractor

# %% [markdown] id="KuUbPW7oV-B5"
# *A* Wav2Vec2 feature extractor object requires the following parameters to be instantiated:
#
# - `feature_size`: Speech models take a sequence of feature vectors as an input. While the length of this sequence obviously varies, the feature size should not. In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal ${}^2$.
# - `sampling_rate`: The sampling rate at which the model is trained on.
# - `padding_value`: For batched inference, shorter inputs need to be padded with a specific value
# - `do_normalize`: Whether the input should be *zero-mean-unit-variance* normalized or not. Usually, speech models perform better when normalizing the input
# - `return_attention_mask`: Whether the model should make use of an `attention_mask` for batched inference. In general, models should **always** make use of the `attention_mask` to mask padded tokens. However, due to a very specific design choice of `Wav2Vec2`'s "base" checkpoint, better results are achieved when using no `attention_mask`. This is **not** recommended for other speech models. For more information, one can take a look at [this](https://github.com/pytorch/fairseq/issues/3227) issue. **Important** If you want to use this notebook to fine-tune [large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60), this parameter should be set to `True`.

# %% id="kAR0-2KLkopp"
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

# %% [markdown] id="YSTrKB6TCfCK"
# ### 3.4. Create Processor -- Wav2Vec2Processor
#

# %% [markdown] id="qUETetgqYC3W"
# Great, Wav2Vec2's feature extraction pipeline is thereby fully defined!
#
# To make the usage of Wav2Vec2 as user-friendly as possible, the feature extractor and tokenizer are *wrapped* into a single `Wav2Vec2Processor` class so that one only needs a `model` and `processor` object.

# %% id="KYZtoW-tlZgl"
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# %% [markdown] id="sdc1QZLJC1Kb"
# ### 4. Excute everything on our Data ...

# %% [markdown] id="k3Pbn5WvOYZF"
# Finally, we can process the dataset to the format expected by the model for training. We will make use of the `map(...)` function.
#
# First, we load and resample the audio data, simply by calling `batch["audio"]`.
# Second, we extract the `input_values` from the loaded audio file. In our case, the `Wav2Vec2Processor` only normalizes the data.
#
# Third, we encode the transcriptions to label ids (using tokenizer =? what computers can understands).
#
# **Note**: This mapping function is a good example of how the `Wav2Vec2Processor` class should be used. In "normal" context, calling `processor(...)` is redirected to `Wav2Vec2FeatureExtractor`'s call method. When wrapping the processor into the `as_target_processor` context, however, the same method is redirected to `Wav2Vec2CTCTokenizer`'s call method.
# For more information please check the [docs](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#transformers.Wav2Vec2Processor.__call__).

# %% id="eJY7I0XAwe9p"
def prepare_dataset(batch):
    ## get speech arrays
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


# %% [markdown] id="hVMZhH4-nP8-"
# Let's apply the data preparation function to all examples.

# %% colab={"base_uri": "https://localhost:8080/", "height": 84, "referenced_widgets": ["659bedce101747c493de10c771fb6a8e", "e17608affe7c4b6794b0fb22003bd9e6", "9939956edf924e68b21456ac0cdd498e", "d4f890cdb936461198b2b3d1754e4ed0", "91f8f00bbb4345bab3979b748ad0fa57", "734a836dd8ed4ffa86d3a347fa2a8a9f", "6e4268ef91214ff789a3994564dd777f", "c7668edd004140c1a22fd3fc1c20667c", "1731da86024f4b098e6989c11f9375f2", "778cdca5b0c9437ba1d29d2a84650bfe", "5c4ff0f257bb41ecac022412c3b0f5ef"]} id="-np9xYK-wl8q" outputId="6c0cbafd-0696-4016-d0e2-33ecaf2a76d3"
## it should take 6 minutes Max
raw_dataset = raw_dataset.map(prepare_dataset)

# %% [markdown] id="WMXfFBv-HOuh"
# Awesome, now we are ready to start training!

# %% [markdown] id="m4u9pAQUKRWd"
# # Section 5: Final Exercise – Show Us Your Acoustic Talents
#
# Your task is to create three speech samples and test them using a pretrained Arabic model from Hugging Face. The challenge? Push the system to its limits and try to generate samples that result in a Word Error Rate (WER) above 50%.
#
# How you challenge the model is entirely up to you — be creative! You might try background noise, strong accents, unusual phrasing, or any technique that makes recognition more difficult. We're here to see how well you can break the system and showcase your audio manipulation skills.

# %% id="AzRk2nr98jZb"
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, Audio
from jiwer import wer
import os

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("boumehdi/wav2vec2-large-xlsr-moroccan-darija")
model = Wav2Vec2ForCTC.from_pretrained("boumehdi/wav2vec2-large-xlsr-moroccan-darija").cuda()

# Function to map each audio sample to predicted text
def map_to_result(batch):
    with torch.no_grad():
        # Preprocess input audio
        input_values = processor(batch["audio"]["array"], sampling_rate=16000).input_values[0]
        input_values = torch.tensor(input_values).to("cuda")

        # Forward pass through the model
        logits = model(input_values.unsqueeze(0)).logits

        # Decode predictions
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]

    return batch

# Load audio files into a Dataset object
def load_audio_files(audio_paths):
    data = {"audio": audio_paths}
    dataset = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=16000))
    return dataset

# Main function to process files and compute WER
def evaluate_samples(audio_folder, references):
    # List audio file paths
    audio_files = [os.path.join(audio_folder, f"{i}.wav") for i in range(3)]

    # Load dataset
    dataset = load_audio_files(audio_files)

    # Transcribe
    results = dataset.map(map_to_result)

    # Compute WERs
    wers = []
    for i, ref in enumerate(references):
        hyp = results[i]["pred_str"]
        error = wer(ref, hyp)
        wers.append((i, error, ref, hyp))

    return wers


# %% id="HcNmEd5X8_K9"
references = [
    "سجل النص الصحيح الأول هنا",
    "سجل النص الصحيح الثاني هنا",
    "سجل النص الصحيح الثالث هنا"
]

results = evaluate_samples("./audio_folder", references)

# Print out WER results
for idx, error, ref, hyp in results:
    print(f"Sample {idx} - WER: {error:.2f}")
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
    print("-----------")


# %% [markdown] id="T8EKbk5Z8ofB"
# # 👋 Need Help with Anything Speech-Related?
# If you're interested in incorporating text-to-speech or any speech component into your project, feel free to reach out to Yassine El Kheir at yassine.el_kheir@dfki.de.
# I'd be happy to give you a hand and share some helpful tips!

# ---
# 🎉 **Congrats! You've completed the speech-to-text challenge.**
#
# - Try more Moroccan audio, or even your friends' voices!
# - Share your results with your classmates or mentors.
# - If you're curious, explore other Moroccan datasets or models on [Hugging Face](https://huggingface.co/models?language=ar&search=moroccan).
#
# If you have questions, don't hesitate to ask. Keep experimenting and have fun with AI!
# ---
