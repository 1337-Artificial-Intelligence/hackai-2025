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
#     language: python
#     name: python3
# ---

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/speech_forced_alignment.ipynb)

# %% [markdown]
# # Word-Level Force Alignment Tutorial
#
# In this tutorial, you'll learn how to automatically align words in a transcript with their exact timestamps in an audio recording. This is called "force alignment" and it's super useful for creating subtitles, language learning apps, and more!
#
# ## Learning Objectives
# - Understand what force alignment is and why it's useful
# - Learn how to use wav2vec to align Arabic speech with text
# - Create an interactive tool where you can click on words to hear them
#
# ## What is Force Alignment?
#
# Force alignment is like creating a precise timeline of when each word is spoken in an audio recording. It takes:
# - An audio recording
# - A text transcript of what was said
#
# And tells you exactly when each word starts and ends in the audio.
#
# For example, if someone says "مرحبا كيف حالك" (Hello, how are you), force alignment would tell you:
# - "مرحبا" occurs from 0.2 to 0.5 seconds
# - "كيف" occurs from 0.6 to 0.8 seconds
# - "حالك" occurs from 0.9 to 1.2 seconds
#
# ## Why is Force Alignment Useful?
#
# - **Subtitles**: Create perfectly timed subtitles for videos
# - **Language Learning**: Build apps where students can click words to hear pronunciation
# - **Video Editing**: Quickly find specific phrases in long recordings
# - **Dubbing**: Match new audio to original timing for natural-sounding dubs
#
# ## Setup
#
# First, let's install the necessary libraries:

# %% 
# !pip install torch torchaudio transformers matplotlib librosa ipywidgets

# %% [markdown]
# ## Loading and Preparing Our Data
#
# We'll use a short Arabic audio sample with its transcript. The code below:
# 1. Loads the audio file
# 2. Reads the transcript
# 3. Splits the text into words
# 4. Shows you the audio waveform

# %%
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import re
import librosa
import librosa.display
import IPython.display as ipd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set up matplotlib for Arabic text
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

def split_arabic_text(text):
    words = re.split(r'\s+', text.strip())
    return [word for word in words if word]

# Load the audio and transcript
audio_path = "sample_data/arabic_sample.wav"
transcript_path = "sample_data/arabic_transcript.txt"

# Read the transcript
with open(transcript_path, 'r', encoding='utf-8') as f:
    transcript = f.read().strip()

words = split_arabic_text(transcript)
print(f"Transcript: {transcript}")
print(f"Words ({len(words)}): {words}")

# Load and display the audio
y, sr = librosa.load(audio_path)
ipd.Audio(y, rate=sr)

# %% [markdown]
# ## Visualizing the Audio
#
# Let's look at the audio waveform to understand our data better:

# %%
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Using Wav2Vec for Force Alignment
#
# We'll use wav2vec, a powerful AI model that can understand speech. Here's what we do:
# 1. Load the pre-trained Arabic model
# 2. Process our audio to the right format
# 3. Get the model's predictions

# %%
# Load the Arabic model
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load and process audio
waveform, sample_rate = torchaudio.load(audio_path)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample to 16kHz (required by wav2vec)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Get model predictions
input_values = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values
with torch.no_grad():
    logits = model(input_values).logits

# Convert to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)
probs = probs.squeeze().detach().cpu()
print(f"Shape of predictions: {probs.shape}")

# %% [markdown]
# ## Understanding the Model's Output
#
# The model gives us a matrix of probabilities. Each row represents a tiny slice of time (20ms), and each column represents a possible Arabic character.
#
# Let's see what characters the model knows:

# %%
vocab = processor.tokenizer.get_vocab()
id_to_char = {v: k for k, v in vocab.items()}
print("First 10 characters in vocabulary:", dict(list(id_to_char.items())[:10]))

# %% [markdown]
# ## Finding Word Timestamps
#
# Now we'll use the model's predictions to find exactly when each word is spoken. We do this by:
# 1. Creating a "trellis" (a fancy word for a probability map)
# 2. Finding the most likely path through this map
# 3. Converting this path into word timestamps

# %%
from dataclasses import dataclass

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")
    
    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens[1:]])
    return trellis

# Get the alignment
tokens = processor.tokenizer(transcript, add_special_tokens=False).input_ids
trellis = get_trellis(probs, tokens)

# %% [markdown]
# ## Creating Word Segments
#
# Now we'll convert our alignment into actual word timestamps:

# %%
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator=" "):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

# Get word segments
path = backtrack(trellis, probs, tokens)
segments = merge_repeats(path)
word_segments = merge_words(segments)

# Print the results
for word in word_segments:
    print(word)

# %% [markdown]
# ## Interactive Word Playback
#
# Now you can click on any word to hear it! Try it out:

# %%
def display_segment(i):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / 16000:.3f} - {x1 / 16000:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=16000)

# Try clicking on different words to hear them!
display_segment(0)  # First word
display_segment(1)  # Second word
display_segment(2)  # Third word

# %% [markdown]
# ## Conclusion
#
# Congratulations! You've learned how to:
# 1. Use wav2vec to align Arabic speech with text
# 2. Create word-level timestamps
# 3. Build an interactive tool for word playback
#
# This technology is used in many real-world applications like:
# - Automatic subtitle generation
# - Language learning apps
# - Voice-controlled systems
#
# Try it with your own audio files and transcripts!

# %% [markdown]
# ## TODO: Add Images
# - Add a diagram showing how force alignment works
# - Add a visualization of the trellis matrix
# - Add a screenshot of the interactive word playback interface
