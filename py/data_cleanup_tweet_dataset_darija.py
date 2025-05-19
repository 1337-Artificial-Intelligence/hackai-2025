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
# %% [markdown]

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1337-Artificial-Intelligence/hackai-2025/blob/main/new_notebooks/data_cleanup_tweet_dataset_darija.ipynb)

# %% [markdown]
# # Data Cleanup - Tweet Dataset (Darija)
#
# In this notebook, you'll learn how to clean and analyze text data from social media. We'll work with a dataset of Moroccan Darija tweets and learn essential text preprocessing techniques that are commonly used in Natural Language Processing (NLP).

# %% [markdown]
# ## What you'll learn:
# - How to load and explore text data
# - Basic text cleaning techniques
# - Tokenization (splitting text into words)
# - Removing stop words (common words that don't add much meaning)
# - Basic text analysis and visualization

# %% [markdown]
# ## Why is this important?
# Before we can use text data for AI tasks like sentiment analysis or text generation, we need to clean and prepare it. This is called "text preprocessing" and it's a crucial first step in any NLP project.

# %% [markdown]
# ## Let's get started!

# %% [markdown]
# ### 1. Install and import necessary packages
# First, we need to install and import the tools we'll use:

# %%
# Install required packages
!pip install datasets pandas regex nltk matplotlib seaborn wordcloud arabic-reshaper python-bidi

# Import the packages
from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

# Download NLTK data
nltk.download('punkt')

# Set pandas to show full text
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# ### 2. Load the Dataset
# We'll use a dataset of Moroccan Darija tweets from Hugging Face. This dataset contains tweets with their sentiment labels.

# %%
# Load the dataset
darija_dataset = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", "arq")

# Convert to pandas DataFrame for easier handling
darija_dataset_df = pd.DataFrame(darija_dataset['train'])

# Let's look at the first few tweets
print("First 5 tweets in the dataset:")
darija_dataset_df.head()

# %% [markdown]
# ### 3. Text Cleaning
# Social media text often contains emojis, usernames, and other elements we want to remove. Let's clean our tweets step by step.

# %% [markdown]
# #### 3.1 Remove Emojis
# Emojis are fun but they can make text analysis more complicated. Let's remove them:

# %%
def remove_emojis(tweet):
    emoj = re.compile("["
       u"\U0001F600-\U0001F64F"  # emoticons
       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
       u"\U0001F680-\U0001F6FF"  # transport & map symbols
       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
       u"\U0001F923"
       u"\U0001F97A"
       u"\U0001F914""]+", re.UNICODE)
    return re.sub(emoj, '', tweet)

# Apply the function to our tweets
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_emojis)
print("Tweets after removing emojis:")
darija_dataset_df.head()

# %% [markdown]
# #### 3.2 Remove Usernames
# Twitter usernames start with @. Let's remove them:

# %%
def remove_user(tweet):
    user_re = "@[A-Za-z0-9]+"
    return re.sub(user_re, ' ', tweet)

# Apply the function
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_user)
print("Tweets after removing usernames:")
darija_dataset_df.head()

# %% [markdown]
# #### 3.3 Remove Latin Letters
# Since we're working with Darija, let's remove Latin letters:

# %%
def remove_latin(tweet):
    latin_re = "[A-Za-z]+"
    return re.sub(latin_re, ' ', tweet)

# Apply the function
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_latin)
print("Tweets after removing Latin letters:")
darija_dataset_df.head()

# %% [markdown]
# #### 3.4 Remove Punctuation
# Punctuation marks can be distracting for analysis:

# %%
def remove_punctuation(tweet):
    punct_re = "[^\w\s]+"
    return re.sub(punct_re, ' ', tweet)

# Apply the function
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_punctuation)
print("Tweets after removing punctuation:")
darija_dataset_df.head()

# %% [markdown]
# ### 4. Tokenization
# Tokenization means splitting text into individual words (tokens):

# %%
def tokenize(tweet):
    return word_tokenize(tweet)

# Apply tokenization
darija_dataset_df["tweet_token"] = darija_dataset_df['tweet'].apply(tokenize)
print("Tweets after tokenization:")
darija_dataset_df.head()

# %% [markdown]
# ### 5. Remove Stop Words
# Stop words are common words that don't add much meaning (like "the", "and", etc.). Let's remove them:

# %%
# TODO: Add darija_stop_words.csv file
# For now, we'll use a small example list
darija_stop_words = ["و", "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "هؤلاء"]

def remove_stop(all_tokens, stop_lst):
    stop_lst = {stp_wrd.strip() for stp_wrd in stop_lst}
    return [token.strip() for token in all_tokens if token.strip() not in stop_lst]

# Apply stop word removal
darija_dataset_df['tweet_token'] = darija_dataset_df['tweet_token'].apply(remove_stop, args=(darija_stop_words,))
print("Tweets after removing stop words:")
darija_dataset_df.head()

# %% [markdown]
# ### 6. Text Analysis
# Now that our data is clean, let's analyze it!

# %% [markdown]
# #### 6.1 Count Word Frequencies
# Let's see which words appear most often:

# %%
from collections import Counter

# Get all tokens
all_tokens = [token for list_token in darija_dataset_df['tweet_token'] for token in list_token]

# Count word frequencies
word_counts = Counter(all_tokens)

# Show top 20 most common words
print("Top 20 most common words:")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")

# %% [markdown]
# #### 6.2 Visualize Word Frequencies
# Let's create a bar plot of the most common words:

# %%
def plot_top_words(word_counts, n=20):
    top_n = word_counts.most_common(n)
    words, counts = zip(*top_n)
    
    plt.figure(figsize=(12, 6))
    reshaped_words = [arabic_reshaper.reshape(word) for word in words]
    bidi_words = [get_display(word) for word in reshaped_words]
    
    sns.barplot(x=list(counts), y=bidi_words, palette="viridis")
    plt.title(f"Top {n} Most Common Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()

# Create the plot
plot_top_words(word_counts)

# %% [markdown]
# #### 6.3 Create a Word Cloud
# A word cloud is a visual representation of text data where the size of each word indicates its frequency:

# %%
# TODO: Add NotoNaskhArabic font file
# For now, we'll use a default font
cloud = WordCloud(background_color="white").generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 6))
plt.imshow(cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Summary
# In this notebook, you learned:
# 1. How to load and explore text data
# 2. Basic text cleaning techniques (removing emojis, usernames, etc.)
# 3. How to tokenize text
# 4. How to remove stop words
# 5. Basic text analysis and visualization

# %% [markdown]
# ## Next Steps
# - Try different cleaning techniques
# - Experiment with different visualizations
# - Use the cleaned data for sentiment analysis or other NLP tasks

# %% [markdown]
# ## Resources
# - [Hugging Face Datasets](https://huggingface.co/datasets)
# - [NLTK Documentation](https://www.nltk.org/)
# - [Pandas Documentation](https://pandas.pydata.org/)
# - [Matplotlib Documentation](https://matplotlib.org/)
