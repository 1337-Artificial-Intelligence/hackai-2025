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

# %% [markdown] id="2d1kLxXThW_1"
# # Word-Level Force Alignment Tutorial
#
# This beginner-friendly tutorial will show you how to align words in a transcript with their exact timestamps in an audio recording using wav2vec.

# %% [markdown] id="W1QMG7zChW_2"
# ## 1. What is Force Alignment?
#
# Force alignment is a technique that automatically maps words in a transcript to their exact timestamps in an audio recording. Think of it as creating a precise timeline of when each word was spoken.
#
# In simple terms, force alignment takes:
# - An audio recording
# - A text transcript of what was said
#
# And it tells you exactly when each word starts and ends in the audio.
#
# For example, if you have an audio recording saying "Hello, how are you today?" and the corresponding text, force alignment would tell you that:
# - "Hello" occurs from 0.2 to 0.5 seconds
# - "how" occurs from 0.6 to 0.8 seconds
# - "are" occurs from 0.9 to 1.0 seconds
# - "you" occurs from 1.1 to 1.3 seconds
# - "today" occurs from 1.4 to 1.8 seconds
#
# This timestamp information is incredibly valuable for many applications, especially when working with media content.
#

# %% [markdown] id="2v06c898hW_2"
# ## 1. Why is Force Alignment Useful?
#
# Force alignment has several practical applications in media production:
#
# - **Subtitles and Captions**: Creates perfectly timed subtitles that appear exactly when words are spoken
# - **Video Editing**: Quickly locate specific spoken phrases in long recordings
# - **Dubbing and Voice-overs**: Match new audio to original timing for natural-sounding dubs
# - **Language Learning**: Create interactive materials where students can click on words to hear pronunciation
#
# In this tutorial, we'll focus on creating an interactive visualization where you can click on words to hear them played back.

# %% [markdown] id="W-Ozesn0hW_2"
# ## 3. Setup
#
# First, let's install the necessary libraries. Run this cell if you haven't installed these packages yet.

# %% colab={"base_uri": "https://localhost:8080/"} id="MF02QPjnhW_2" outputId="ba800295-4d74-4a83-f473-8ddeb735b5bd"
# run if needed
# !pip install torch torchaudio transformers matplotlib librosa ipywidgets

# %% [markdown] id="9ZbiNLTyhW_2"
# ## 4. Preparing the Audio and Transcript
#
# For this tutorial, we'll use a short Arabic audio sample with its corresponding transcript.

# %% id="qqCI7HaShW_2"
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import re
import librosa
import librosa.display
import IPython.display as ipd
from ipywidgets import Button, HBox, VBox, Layout, Output
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import json

# Set up matplotlib for Arabic text
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# Function to split Arabic text into words
def split_arabic_text(text):
    # Split by whitespace while preserving Arabic characters
    words = re.split(r'\s+', text.strip())
    return [word for word in words if word]  # Remove any empty strings


# %% colab={"base_uri": "https://localhost:8080/", "height": 111} id="LNHHdq56hW_2" outputId="b5c7f8a7-995b-45d4-fa21-72066cf39cb3"
# Load the audio file and transcript
# Note: Replace with your own audio and transcript files
audio_path = "sample_data/arabic_sample.wav"
transcript_path = "sample_data/arabic_transcript.txt"

# Read the transcript and split into words
with open(transcript_path, 'r', encoding='utf-8') as f:
    transcript = f.read().strip()

words = split_arabic_text(transcript)
print(f"Transcript: {transcript}")
print(f"Words ({len(words)}): {words}")

# Load and display the audio
y, sr = librosa.load(audio_path)
ipd.Audio(y, rate=sr)

# %% [markdown] id="AilvDgIwhW_2"
# ## 5. Visualize the Audio Waveform
#
# Let's look at the audio waveform to get a better understanding of our data.

# %% colab={"base_uri": "https://localhost:8080/", "height": 307} id="36J9ZbX1hW_2" outputId="d1ce8a0f-5c71-4bde-8125-f5a146ea9481"
plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# %% [markdown] id="xWkIi1PkhW_2"
# ## 6. Word-Level Force Alignment with Wav2Vec
#
# Now, let's perform word-level force alignment using the wav2vec model.

# %% colab={"base_uri": "https://localhost:8080/"} id="2VKn_ykIhW_3" outputId="e6ef1463-cbb5-4b4f-8967-71dfa1b97d36"
# Load pre-trained model and processor for Arabic
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load the audio with torchaudio for processing
waveform, sample_rate = torchaudio.load(audio_path)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample (wav2vec2 expects 16kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Convert audio to features
input_values = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values

# Get the CTC probabilities (it gives series of characters predictions)
with torch.no_grad():
    logits = model(input_values).logits

# Get the probability matrix
probs = torch.nn.functional.softmax(logits, dim=-1)
probs = probs.squeeze().detach().cpu()
print(probs.shape)

# %% [markdown] id="XRvVSYVE5S4s"
# **`probs.shape`** returns a shape of **`(419, 51)`**.
# This means there are 419 frames, where each frame represents 20 milliseconds of audio.
#
# For every 20 ms segment, the model outputs a probability vector of size 51, corresponding to the likelihoods of 51 possible character predictions — such as ألف، باء، تاء and so on.

# %% [markdown] id="AMq1kpEL6YDR"
# To understand which index corresponds to which character, we retrieve the vocabulary using `processor.tokenizer.get_vocab()`.
#
# This allows us to interpret the output probabilities by mapping each index in the vector to its corresponding Arabic character or special symbol.

# %% colab={"base_uri": "https://localhost:8080/"} id="xccuTSxX6AbX" outputId="51dab1aa-a125-464f-a6d6-8c7f6312f713"
# Get the character mapping
vocab = processor.tokenizer.get_vocab()
id_to_char = {v: k for k, v in vocab.items()}
print(id_to_char)


# %% [markdown] id="apoYRQ1S6xcO"
# Let's now use the probability matrix to get the alignmnet for each characters.

# %% [markdown] id="LH1FxjOM8bQe"
# ## 7. Alignment Probability -- Trellis
#
# From the emission matrix, we generate the **trellis**, which represents the probability accumulations of transcript labels at each time frame.
#
# The trellis is a 2D matrix with  
# - **time axis** indexed by `t`  
# - **label axis** indexed by `j`, representing the transcript labels  
#

# %% id="IbbmV4YO8eiC"
def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],)
    return trellis

# let's tokenize our sentence:
sentence="هذا الشارع سوي لممارسه رياضه الركض يوميا و لم التقي باي شخص علي الاطلاق"
tokens = processor.tokenizer(sentence, add_special_tokens=False).input_ids

# let's get trellis
trellis = get_trellis(probs, tokens)

# %% [markdown] id="7esfsTnn-3za"
# ## 8. Find the most likely path (backtracking)
#
# Once the trellis is generated, we will traverse it following the elements with high probability. => this is called backtracking
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="JARKJcOw-v0F" outputId="7838056b-7ba7-4f9f-84cd-cb6589d0c8ce"
from dataclasses import dataclass

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

## Based ion Viterbi optimization
def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


path = backtrack(trellis, probs, tokens)
for p in path:
    print(p)


# %% [markdown] id="ep_kjBbV_Zsp"
# ## 9. Segment the path
#
# Now this path contains repetations for the same labels you can notice (token_index) as, so let’s merge them to make it close to the original transcript.
#
# When merging the multiple path points, we simply take the average probability for the merged segments.

# %% colab={"base_uri": "https://localhost:8080/"} id="zyW9Wx3V_Ulw" outputId="1642c6f3-9fb9-41bf-9678-007dbc7c5045"
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


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


segments = merge_repeats(path)
for seg in segments:
    print(seg)


# %% [markdown] id="wPYXGOaF_qCB"
# ## 10. Merge the segments into words
#
# Now let’s merge the words.
# Then, finally, we segment the original audio into segmented audio and listen to them to see if the segmentation is correct.

# %% colab={"base_uri": "https://localhost:8080/"} id="MqbtJCH5_jgG" outputId="85d4fbca-5f5a-4345-a07d-461e48526f13"
# Merge words
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


word_segments = merge_words(segments)
for word in word_segments:
    print(word)

# %% id="_vuBTuyt_0vm"
import IPython

def display_segment(i):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / 16000:.3f} - {x1 / 16000:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=16000)


# %% colab={"base_uri": "https://localhost:8080/", "height": 93} id="WAD2AlBt_6ki" outputId="699a65d6-dabb-4805-fbc9-67eb2a7dc641"
display_segment(0)

# %% colab={"base_uri": "https://localhost:8080/", "height": 93} id="a7XWGhmcAQYL" outputId="8402baf7-9fbd-4dbf-d673-c3364b159cd4"
display_segment(1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 93} id="27fIUyYIASFR" outputId="633cbdf7-a473-4fbf-970c-20434a4c69ad"
display_segment(2)

# %% colab={"base_uri": "https://localhost:8080/", "height": 93} id="t8IB8pi5AUYa" outputId="be658130-8b57-4255-d8dd-d287db2083d4"
display_segment(10)

# %% [markdown] id="RcAP1sZthW_3"
# ## 11. Conclusion
#
# In this tutorial, you've learned how to:
#
# 1. Understand what force alignment is and why it's useful
# 2. Use wav2vec to perform word-level alignment on Arabic audio
# 3. Create an interactive visualization where you can click on words to hear them
#
# This technique can be applied to any language supported by wav2vec models and is useful for many applications including subtitling, language learning, and audio editing.
