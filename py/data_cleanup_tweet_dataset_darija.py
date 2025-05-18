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

# %% [markdown] id="FvG3HBJ_HVj_"
# # Data Cleanup - Tweet Dataset (Darija)
#
#

# %% [markdown] id="g8FqPxdqHYib"
# 📌 Challenge Description:
#
# In this challenge, you will clean a tweet dataset from Hugging Face (`shmuhammad/AfriSenti-twitter-sentiment`), focusing on the Moroccan Darija subset. The goal is to preprocess the text data by removing emojis, usernames, and applying custom list of Darija stop words. The cleaned data will then be ready for n-gram analysis.

# %% [markdown] id="mNThxcvS2O4c"
# 📊 Dataset Summary:
# AfriSenti is the largest sentiment analysis dataset for under-represented African languages, covering 110,000+ annotated tweets in 14 African languages (Amharic, Algerian Arabic, Hausa, Igbo, Kinyarwanda, Moroccan Arabic, Mozambican Portuguese, Nigerian Pidgin, Oromo, Swahili, Tigrinya, Twi, Xitsonga, and Yoruba).

# %% [markdown] id="vVGZTiRuZWjF"
# # Part 1: Data Cleaning

# %% colab={"base_uri": "https://localhost:8080/"} id="VZKBt4Wb2NhP" outputId="78af0152-63bf-43ec-99e8-a6cd671b9825"
#Installing Necessary Packages
# !pip install datasets pandas regex


# %% id="6-Nt7JiyZWjH"
#Import Necessary Packages
from  datasets  import load_dataset
import pandas as pd
import re
pd.set_option('display.max_colwidth', None)

# %% [markdown] id="0Fg2oYRo2rlx"
# ## Load the Dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="YbIaoq5E2xC2" outputId="08070100-8278-4a40-9541-108d1136ae7f"
"""
Load the `shmuhammad/AfriSenti-twitter-sentiment` dataset from the Hugging Face `datasets` library, and specify the "arq" subset for Moroccan Darija.
"""
darija_dataset = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", "arq")

# %% id="NiRI7cV5ZWjI"
""" Identify and select the subset of tweets of the train datasets"""
darija_dataset_df = pd.DataFrame(darija_dataset['train'])
darija_dataset_df.head()


# %% [markdown] id="3VEqLcwbZWjJ"
# ## Remove Emojis From Tweets

# %% id="i-kfuQSjZWjK"
def remove_emojis(tweet):
    emoj = re.compile("["
       u"\U0001F600-\U0001F64F"  # emoticons
       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
       u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U0001F923"
                      u"\U0001F97A"
                      u"\U0001F914""]+", re.UNICODE)
    return re.sub(emoj ,'', tweet)


# %% [markdown] id="k7zlZCxVZWjL"
# You can use this ressource emojis unicode: https://apps.timwhitlock.info/emoji/tables/unicode

# %% id="lg5hZSRAZWjL"
# remove the emojis from the dataset
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_emojis)

# %% colab={"base_uri": "https://localhost:8080/"} id="syymGzvvZWjM" outputId="f0cee374-b1d2-4324-a20e-08beecf69f4d"
darija_dataset_df.head()


# %% [markdown] id="NsLwBXGKZWjN"
# ## Remove user name from tweets

# %% id="K80cxzP3ZWjN"
def remove_user(tweet):
    user_re =  "@[A-Za-z0-9]+"
    return re.sub(user_re, ' ', tweet)


# %% id="D17kzaSFZWjN"
#remove the user names from the dataset
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_user)

# %% colab={"base_uri": "https://localhost:8080/"} id="Kl_VeIHnZWjO" outputId="cd2f99f5-c2b9-4f99-95ab-68c4354a3dd9"
darija_dataset_df.head()


# %% [markdown] id="FlH4Kx3OZWjP"
# ## Remove latin letter

# %% id="5_jyrkVmZWjP"
def remove_latin(tweet):
    latin_re =  "[A-Za-z]+"
    return re.sub(latin_re, ' ', tweet)


# %% id="7br7LFUAZWjP"
#remove latin words
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_latin)
darija_dataset_df.head()

# %% [markdown] id="WYHglBD-ZWjS"
# ## Removing Dublicate Rows

# %%
darija_dataset_df

# %% colab={"base_uri": "https://localhost:8080/"} id="txDycFxH6RMY" outputId="b6436162-eab7-48bf-d3a1-cd839077e4d7"
darija_dataset_df.drop_duplicates(inplace = True)
darija_dataset_df

# %% [markdown] id="gUGlJd6pZWjP"
# ## Remove Punctuation

# %% id="OUao2kxzZWjQ"
import re
def remove_ponct(tweet):
    ponct_re =  "[^\w\s]+"
    return re.sub(ponct_re, ' ', tweet)


# %% id="hvUmWQ6sZWjQ"
#remove ponctuation
darija_dataset_df['tweet'] = darija_dataset_df['tweet'].apply(remove_ponct)
darija_dataset_df.head()

# %% [markdown] id="PJMkjpdlZWjT"
# ## Tokenization

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="T6ooyvj0ZWjT" outputId="e51c9d0e-9539-4a0e-c012-e569d70b45bc"
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
def Tokenize(tweet):
    return word_tokenize(tweet)

darija_dataset_df["tweet_token"] = darija_dataset_df['tweet'].apply(Tokenize)
darija_dataset_df.head()

# %% [markdown] id="wJwUQg-6ZWjQ"
# ## Remove Stops Words
#

# %%
#Import darija stopwords dataset

darija_stop_words_df = pd.read_csv("/content/darija_stop_words.csv")
darija_stop_words = darija_stop_words_df['word'].tolist()


# %% id="vks_JspMZWjR"
def remove_stop(all_tokens, stop_lst):
    stop_lst = {stp_wrd.strip() for stp_wrd in stop_lst}  # Convert stop_lst to a set for faster lookups
    return [token.strip() for token in all_tokens if token.strip() not in stop_lst]


# %% id="9uW3telVZWjS"
#remove darija stopwords
darija_dataset_df['tweet_token'] = darija_dataset_df['tweet_token'].apply(remove_stop, args=(darija_stop_words,),)

# %% id="1LfzAGbF9TV8"
all_token = [token for list_token in darija_dataset_df['tweet_token'] for token in list_token]

# %% [markdown] id="ImfH7i4xZWjS"
# # Part 2: N-Grams Analysis

# %% colab={"base_uri": "https://localhost:8080/"} id="qkxaDDFHZWjT" outputId="bf7fcfc8-14b4-40ed-ec17-8af57c0f514d"
#Install Necessary Librairies
# !pip install nltk pandas matplotlib seaborn wordcloud networkx arabic-reshaper python-bidi

# %% [markdown] id="ODPBUDIbZWjU"
# ## Unigrams

# %% id="T7YGUfKUZWjU"
from collections import Counter
unigram_counts = Counter(all_token)

# %% [markdown] id="9ISxDkVaZWjU"
# ## Barplot of Unigrams

# %% id="rUT8XP_WG1V3"
#Import Necessary Packages
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display


# %% colab={"base_uri": "https://localhost:8080/", "height": 680} id="EDX8Fhv9ZWjV" outputId="4bc80e00-0fb9-4e4e-eaf5-48832da96f04"
def plot_top_n_grams(ngram_counts, n, title):
    top_n = ngram_counts.most_common(n)
    words, counts = zip(*top_n)
    plt.figure(figsize=(12, 6))
    reshaped_words = [arabic_reshaper.reshape(word) for word in words]
    bidi_words = [get_display(word) for word in reshaped_words]
    sns.barplot(x=list(counts), y=bidi_words, palette="viridis")
    plt.title(f"Top {n} {title}")
    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    plt.show()

plot_top_n_grams(unigram_counts, 20, "Unigrams")

# %% [markdown] id="EBZk5OpTG-uF"
# ## WordCloud

# %% id="TH37fLu4HBjp"
#Import Necessary Librairies
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# %% colab={"base_uri": "https://localhost:8080/", "height": 285} id="U2ZS5LvNZWjV" outputId="0bb1e966-28f5-4da2-a442-588297ce42c6"
cloud_ar = WordCloud(font_path='/content/NotoNaskhArabic-VariableFont_wght.ttf', background_color="white").generate_from_frequencies(unigram_counts)
plt.imshow(cloud_ar, interpolation='bilinear')
plt.axis('off')
plt.savefig("ary")
plt.show()

# %% [markdown] id="EbkjkAD1u_I6"
#
