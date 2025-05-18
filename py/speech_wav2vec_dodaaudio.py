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
# Before we can start playing with audio, we need to install a few essential Python libraries. These libraries provide the tools we need to load datasets, manipulate audio signals, and create visualizations. We'll be using:
#
# *   `datasets`: To easily download and access datasets from the Hugging Face Hub.
# *   `torchaudio`: A PyTorch library for audio I/O and basic signal processing.
# *   `librosa`: A powerful library for audio analysis, especially useful for feature extraction like spectrograms (No worries, we will later what's that...).
# *   `matplotlib`: The go-to library for plotting and creating static, animated, and interactive visualizations in Python.
# *   `IPython`: Used here specifically for its ability to play audio directly within the Colab notebook.
#
# Let's install them using pip. Run the following code cell in your Colab environment:
#

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
# ## 2.3 Listening to Speech
#
# One of the first things you'll want to do with a speech dataset is to actually listen to some of the audio samples. This gives you a direct sense of the data quality, noise levels, and content. We can use `IPython.display.Audio` to play audio directly in our notebook.

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
# When you look at the list below, you’ll see lots of strange or unnecessary characters — things like punctuation marks, special diacritics, and even numbers.
#
# ```python
# # This is an example of a "messy" vocabulary extracted directly from raw text data
# messy_vocabulary_example = ['’','ش','ر','َ','؟','ب','0','ص','ؤ','ث','پ','9',',','3','ء','?',
#  'ة','خ','غ','ي','ف','-','ى','2','أ','ك','ظ','و','1','ّ','ُ','ت',
#  'ض','ح','س','ئ','د','5','ه','ل','ڤ','ط','ڭ','إ','!','"',' ','ا',
#  ':','6','ق','ذ','ز','ن','ج','.','،','ع','ً','م','آ']
# ```
#
# ### ❓ But wait... why does our vocabulary look like this?
# Because it was built automatically from data not fully cleaned, not filtered. So, yes:
#
# ✅ it includes Arabic letters (like ش, ر, ب)
# ✅ but also punctuation (like ؟, ,, .), numbers (0-9), and even other symbols (like ’, ،, َ, ً, ُ, ّ)
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
# # Section 4: Training & Evaluation
#
# The data is processed so that we are ready to start setting up the training pipeline. We will make use of 🤗's [Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer) for which we essentially need to do the following:
#
# - Define a data collator. In contrast to most NLP models, Wav2Vec2 has a much larger input length than output length. *E.g.*, a sample of input length 50000 has an output length of no more than 100. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning Wav2Vec2 requires a special padding data collator, which we will define below
#
# - Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a `compute_metrics` function accordingly
#
# - Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.
#
# - Define the training configuration.
#
# After having fine-tuned the model, we will correctly evaluate it on the test data and verify that it has indeed learned to correctly transcribe speech.

# %% [markdown] id="CzWzW6CiKncL"
# ### 4.1 Define Data Collator
#
# This class will take care of padding our inputs and labels dynamically per batch.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="dxOSOQC_KQ6x" outputId="ea29bf20-3acf-49e8-909a-fe1f1abfe7e9"
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

if processor:
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    print("Data collator defined.")
else:
    data_collator = None
    print("Some Problem is there, call Mentor.")


# %% [markdown] id="bbHJHtoFLFcp"
# ### 4.2 Define Evaluation Metrics (WER & CER)
#
# Word Error Rate (WER) and Character Error Rate (CER) are standard metrics for ASR.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="gAQX3eXkLeB7" outputId="2ac683ce-ed39-4a5f-c8da-b656ac6a9929"
# !pip install jiwer

# %% [markdown] id="YIS9UN1fMKD8"
# We introduced WER and CER earlier, but let's delve deeper. These are the standard metrics to evaluate the performance of an ASR system.
#
# *   **Word Error Rate (WER):** This metric measures errors at the word level. It's calculated by comparing the predicted sequence of words with the reference (ground truth) transcription. The formula is:
#
#     `WER = (S + D + I) / N`
#
#     Where:
#     *   `S` is the number of substitutions (words in the prediction that are different from the reference at the same position, e.g., reference "hello world", prediction "hallo world" -> 1 substitution).
#     *   `D` is the number of deletions (words in the reference that are missing in the prediction, e.g., reference "hello brave new world", prediction "hello new world" -> 1 deletion, "brave").
#     *   `I` is the number of insertions (words in the prediction that are not in the reference, e.g., reference "hello world", prediction "hello there world" -> 1 insertion, "there").
#     *   `N` is the total number of words in the reference transcription.
#
#     A lower WER is better, with 0% being a perfect transcription. WER can sometimes be greater than 100% if the prediction is much longer than the reference and has many errors.
#
# *   **Character Error Rate (CER):** This metric is similar to WER but operates at the character level. It's useful for languages without clear word boundaries (like Chinese or Japanese) or to get a more granular view of errors, especially for out-of-vocabulary words that might be partially correct at the character level.
#
#     `CER = (S_char + D_char + I_char) / N_char`
#
#     Where `S_char`, `D_char`, `I_char` are substitutions, deletions, and insertions at the character level, and `N_char` is the total number of characters in the reference transcription.
#
#     Again, lower CER is better. CER is often more sensitive to minor misspellings or phonetic errors that WER might not capture if the word is still mostly correct.
#
# Both metrics provide valuable insights. WER gives a more practical sense of how understandable the transcription is, while CER can help diagnose issues at a finer level.
#
#
# Why do we use metrics like WER and CER? Why not simply use accuracy — for example, assigning 0 if the predicted sentence exactly matches the reference, and 1 otherwise?
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["50853fa6c3ac4b998bd781709e076fe8", "152d36e115d64d2ba6c565bf40571d62", "4615760c87e84c2da67236f00292a822", "6103723df26d4a7289fc361aecf035c3", "8c2d42697657427e90c1234ae22901f8", "fb21e72ae37044f1b20061bb78805e2b", "39c385ab46ed4155b7be29d4c806cfba", "3722013457eb4c8082393c964c96002d", "ae9f309d8b3f41a38b09746dac076c38", "bfb25a3f27e840018ce63cb960b144cb", "1d19e55f5b7b463c90d587b0517844ab"]} id="ZjxmLj9bLFwG" outputId="d129e842-ae75-4884-df64-41f26a14f4f7"
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids)
    label_ids_cleaned = []
    for label_seq in pred.label_ids:
        label_ids_cleaned.append([token_id for token_id in label_seq if token_id != -100 and token_id != processor.tokenizer.pad_token_id])
    label_str = processor.batch_decode(label_ids_cleaned, group_tokens=False) # group_tokens=False for char-level

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

print("Metrics functions defined.")

# %% [markdown] id="SA8q1DmPM1UB"
# ### 4.3 Load Pre-trained Model
#
# We'll load a pre-trained Wav2Vec2 model designed for CTC (Connectionist Temporal Classification), which is the common loss function for this type of ASR.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 188, "referenced_widgets": ["27b5fc973db4467a9a8f6cc37661d487", "f9404cef0c5d4f41863c1037341c8dac", "cd393865d1f947b0bbd6f2cabfd9191d", "6de52cc159854f3fb30731003ee98af4", "db86b196f51e4a868a197ddf11da9b4a", "22b9ae1251d6453ca8cfd624a1f9c705", "b4dc12ee5c53412dbc4a6793052c59f3", "12e33c4c63c2421cafaf16dff191a15c", "14289f892cc74891b484a508a7cf03cd", "e3220316a00b428c8d3aa67d58865c4b", "befdb38407514f70843a4901153467dd", "d9384445168743db9461b966c2dbaf4a", "9c5848639a1a4a61a6f55dfaab3e332f", "33c56ecbff6d4b3183f471f35765893b", "0eb6888c5f1c4963bafba011626778cf", "072def5528954464b0fba9076649098c", "f8dbf164c30f4533ae220eb1ff0f389e", "9c55e8aa823646f4b155947ac88e890e", "d52fa347bad549a0acfd7003284d6cb0", "bcb48b3ad7df45e3aea29f2e3cf88d1b", "ea9c6a1142e04ca6a0cac678a7cc0fb5", "3c42b93ddc1f444299988fbbf58b33a2"]} id="poN-Z1OvM0jZ" outputId="e014d106-7baf-4729-9fca-f17fde1149e0"
from transformers import Wav2Vec2ForCTC

if processor:
    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base",  # A common base model
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer) # Ensure vocab size matches our tokenizer
        )
        # Freeze feature encoder layers if you want to train only the top layers (common practice)
        # model.freeze_feature_encoder()
        print("Pre-trained model loaded.")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        model = None
else:
    model = None
    print("Processor not available, skipping model loading.")


# %% [markdown] id="G0xRpwR8M8-i"
# ### 4.4 Define Training Arguments
#
# These arguments control the training process.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="dNifwRNxMp_6" outputId="00cef00e-87e6-4e4e-9aa2-efdd5857ff43"
from transformers import TrainingArguments

# Define a directory for saving model outputs (checkpoints, logs)
output_dir = "./wav2vec2-finetuned-hackai-demo"

# These are example arguments. Adjust them based on your resources and dataset size.
# For a quick demo, we use very few steps.
training_args = TrainingArguments(
    output_dir=output_dir,
    group_by_length=True, # Speeds up training by batching similar length inputs
    per_device_train_batch_size=2, # Reduce if OOM, increase if GPU memory allows
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    num_train_epochs=3, # For demo.
    save_steps=100, # Save checkpoint every N steps (adjust based on training length)
    eval_steps=100, # Evaluate every N steps
    logging_steps=10, # Log metrics every N steps
    learning_rate=3e-4, # Common starting point for Wav2Vec2 fine-tuning
)
print("Training arguments defined.")

# %% [markdown] id="6Rkb-JzINkWv"
# ### 4.5 Instantiate the Trainer
#
# Now, we bring everything together using the `Trainer` class.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="gxQ_dEX_NjVi" outputId="1437bbba-37cf-4bc8-f256-c71f4b9e0f36"
from transformers import Trainer
import numpy as np # ensure numpy is imported
import dataclasses # ensure dataclasses is imported
from typing import Dict, List, Optional, Union # ensure typing is imported

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=raw_dataset, # Use the small preprocessed dataset
    eval_dataset=raw_dataset,  # For demo, using same small set for eval. Ideally, use a separate validation set.
    tokenizer=processor.feature_extractor, # Important for the Trainer to handle feature extraction correctly
    report
)

# %% [markdown] id="rKJuR0p5NxP1"
# ### 4.6 Start Fine-tuning!
#
# This is where the actual training happens.
#

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 175} id="TYHTZp6nNfzG" outputId="35c36efd-f50a-45b8-944b-26407e9a8e8c"
if trainer:
    print("Starting fine-tuning...")
    try:
        trainer.train()
        print("Fine-tuning completed.")
        # Save the final model and processor
        model_save_path = f"{output_dir}/final_model"
        processor.save_pretrained(model_save_path)
        trainer.save_model(model_save_path)
        print(f"Final model and processor saved to {model_save_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        print("If you are getting CUDA out of memory, try reducing batch_size or gradient_accumulation_steps, or use a smaller model (Call a Mentor to assess).")
else:
    print("Trainer not instantiated, Go back Hahaha, no skipping.")


# %% [markdown] id="NyU_Etn973Rx"
# # Section 5: Final Exercise – Show Us Your Acoustic Talents
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
