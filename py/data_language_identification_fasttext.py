# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/data_language_identification_fasttext.ipynb)

# %% [markdown]
# # Language Identification using FastText
# 
# **Time Estimate**: 1 hour
# 
# ## Learning Objectives
# - Understand what language identification is and why it's important
# - Learn how to use FastText for text classification
# - Build a simple model to identify Moroccan Darija vs other Arabic dialects
# 
# ## Prerequisites
# - Basic Python knowledge
# - Understanding of text classification (briefly explained below)
# 
# ## What is Language Identification?
# Language identification is the task of determining which language a given text is written in. In this notebook, we'll focus on identifying Moroccan Darija (Moroccan Arabic) from other Arabic dialects.
# 
# ## What is FastText?
# FastText is a library for efficient text classification and word representation learning. It's particularly good at handling text in different languages and can work well even with limited training data.
# 
# ## What is Text Classification?
# Text classification is a machine learning task where we teach a computer to categorize text into predefined groups. In our case, we're teaching it to categorize text as either Moroccan Darija or another Arabic dialect.

# %% [markdown]
# # Setup (5 minutes)

# %%
# Install required packages
! pip install fasttext datasets pandas scikit-learn seaborn arabic-reshaper wordcloud python-bidi

# %%
import fasttext
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset

# %% [markdown]
# # Load and Explore Data (10 minutes)

# %%
# Load the Darija-LID dataset
data = load_dataset('atlasia/Darija-LID')
train_data = data['train'].to_pandas()
test_data = data['test'].to_pandas()

# %% [markdown]
# Let's look at some examples from our dataset:

# %%
print("Example texts from our dataset:")
print("\nMoroccan Darija example:")
print(train_data[train_data['label'] == 'ary']['text'].iloc[0])
print("\nOther dialect example:")
print(train_data[train_data['label'] == 'other']['text'].iloc[0])

# %% [markdown]
# # Data Preprocessing (10 minutes)
# 
# Before training our model, we need to clean our text data. This involves:
# - Removing URLs and numbers
# - Removing special characters
# - Converting text to lowercase
# - Removing extra spaces

# %%
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # remove urls
    text = re.sub(r'http\S+', '', text)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove Latin characters but keep Arabic text
    text = re.sub(r'[a-zA-Z]', '', text)
    # remove emojis but keep Arabic text
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    text = text.lower()
    return text

train_data['processed_text'] = train_data['text'].apply(preprocess_text)
test_data['processed_text'] = test_data['text'].apply(preprocess_text)

# %% [markdown]
# # Data Visualization (10 minutes)
# 
# Let's visualize our data to understand it better:

# %%
# Distribution of text lengths
train_data['text_length'] = train_data['processed_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 8))
sns.histplot(
    data=train_data.assign(text_length_clipped=train_data['text_length'].clip(upper=30)),
    x='text_length_clipped',
    hue='label',
    bins=30,
    palette={'ary': '#2ecc71', 'other': '#e74c3c'},
    multiple="layer",
    stat='percent'
)
plt.title('Distribution of Text Lengths by Dialect')
plt.xlabel('Number of Words')
plt.ylabel('Percentage')
plt.show()

# %% [markdown]
# # Train FastText Model (15 minutes)
# 
# Now we'll train our FastText model. FastText requires data in a specific format:
# - Each line should start with `__label__` followed by the label
# - Then a space and the text

# %%
# Prepare data in FastText format
data_train = train_data[['label', 'processed_text']].copy()
data_train['label'] = '__label__' + data_train['label']

data_test = test_data[['label', 'processed_text']].copy()
data_test['label'] = '__label__' + data_test['label']

# Save data
data_train.to_csv('data_train.txt', header=None, index=None, sep=' ', mode='w')
data_test.to_csv('data_test.txt', header=None, index=None, sep=' ', mode='w')

# %%
# Train the model
model = fasttext.train_supervised(
    'data_train.txt',
    lr=0.1,
    epoch=5,
    dim=100,
    minCount=5,
    wordNgrams=2,
    bucket=200,
    loss='softmax'
)

# %% [markdown]
# # Evaluate Model (10 minutes)
# 
# Let's see how well our model performs:

# %%
# Test the model
with open('data_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

texts = [' '.join(x.split()[1:]) for x in lines]
preds = model.predict(texts)

# Get predictions and true labels
y_hat = [x[0].split('__label__')[1] for x in preds[0]]
y_true = [x.split('__label__')[1].split()[0] for x in lines]

# Print results
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_true, y_hat))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_hat)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# # Try It Yourself!
# 
# Now you can try the model with your own text:

# %%
# Example usage
test_text = "كيفاش حالك"  # Replace with your own text
processed_text = preprocess_text(test_text)
prediction = model.predict(processed_text)
print(f"Predicted dialect: {prediction[0][0].split('__label__')[1]}")

# %% [markdown]
# # Next Steps
# - Try different preprocessing steps
# - Experiment with different FastText parameters
# - Try the model on different Arabic dialects
# - Learn about other text classification methods
