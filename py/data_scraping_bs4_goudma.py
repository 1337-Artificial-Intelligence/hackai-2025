# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Web Scraping with BeautifulSoup4
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/data_scraping_bs4_goudma.ipynb)

# %% [markdown]
# ## Learning Objectives
# By the end of this notebook, you will be able to:
# - Understand what web scraping is and why it's useful
# - Use BeautifulSoup4 to parse and extract data from websites
# - Create a simple dataset from scraped web content
# - Save your scraped data to a structured format

# %% [markdown]
# ## What is Web Scraping?
# Web scraping is the process of automatically extracting data from websites. It's like having a robot that can read web pages and collect information for you. This is useful for:
# - Collecting data for analysis
# - Monitoring prices or news
# - Creating datasets for machine learning
# - Automating data collection tasks

# %% [markdown]
# ## What is BeautifulSoup4?
# BeautifulSoup4 is a Python library that helps us parse (read and understand) HTML and XML documents. Think of it as a tool that can:
# - Take messy HTML code and make it organized
# - Help us find specific elements on a webpage
# - Extract text, links, and other data easily

# %% [markdown]
# ## Let's Get Started!
# First, we need to install the required libraries:

# %% id="NwvNYma2mYiK"
# Install required libraries
!pip install beautifulsoup4 requests pandas tqdm -q

# %% [markdown]
# ## Importing Libraries
# We'll use:
# - `requests`: to download web pages
# - `BeautifulSoup`: to parse HTML
# - `pandas`: to organize our data
# - `tqdm`: to show progress bars

# %% id="YUacjMzzn4o6"
from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm

# %% [markdown]
# ## Our Goal
# We'll scrape news articles from goud.ma, a Moroccan news website. We'll collect:
# - Article titles
# - Article content
# - Article images
# - Article links

# %% [markdown]
# TODO: Add screenshot of goud.ma homepage showing the articles we want to scrape

# %% [markdown]
# ## Step 1: Download the Web Page
# First, we need to get the HTML content of the webpage. We'll use `requests` to do this.

# %% id="HoPOQznttz7g"
# Send a request to the website
target = "https://www.goud.ma/topics/%d8%a7%d9%84%d8%b1%d8%a6%d9%8a%d8%b3%d9%8a%d8%a9/"
page = requests.get(target, headers={"User-Agent": "XY"})

# Check if the request was successful
if page.status_code == 200:
    print("✅ Successfully downloaded the webpage!")
else:
    print("❌ Failed to download the webpage")

# %% [markdown]
# ## Step 2: Parse the HTML
# Now we'll use BeautifulSoup to parse the HTML and make it easier to work with.

# %% id="L3_MQbeqzJPC"
# Parse the HTML content
page_soup = BeautifulSoup(page.text, "html.parser")

# %% [markdown]
# ## Step 3: Find Articles
# We'll look for article elements with the class "card". These contain our news articles.

# %% id="GHP8O2hOzo4F"
# Find all article elements
articles = page_soup.find_all(name="article", class_="card")[:6]  # Get first 6 articles
print(f"Found {len(articles)} articles!")

# %% [markdown]
# ## Step 4: Extract Article Links
# For each article, we'll get its link to access the full content.

# %% id="1rxW-0dHoPZ1"
# Extract links from articles
articles_links = [
    article.find("a", class_="stretched-link").get("href")
    for article in articles
]
print("Article links:", articles_links)

# %% [markdown]
# ## Step 5: Extract Article Content
# Now we'll visit each article page and extract:
# - Title
# - Content
# - Image URL

# %% id="YQIRlaVG3Slr"
# Let's look at one article first
link = articles_links[0]
article_page = requests.get(link, headers={"User-Agent": "XY"}).text
article_soup = BeautifulSoup(article_page, "html.parser")

# Extract data from the article
article_img = article_soup.find("img", class_="img-fluid wp-post-image").get("src")
article_title = article_soup.find("h1", class_="entry-title").text
article_content = article_soup.find("div", class_="post-content").text.strip()

print(f"Title: {article_title}")
print(f"Image URL: {article_img}")
print(f"Content preview: {article_content[:200]}...")

# %% [markdown]
# ## Step 6: Scrape All Articles
# Now let's do this for all articles and save the data in a structured format.

# %% id="895z_7DI_ajx"
# Create a dictionary to store our data
data = {"titles": [], "content": [], "images": []}

# Scrape each article
for link in tqdm(articles_links, desc="Scraping articles"):
    # Get article page
    page_html = requests.get(link, headers={"User-Agent": "XY"}).text
    page_soup = BeautifulSoup(page_html, "html.parser")
    
    # Extract data
    img = page_soup.find("img", class_="img-fluid wp-post-image").get("src")
    title = page_soup.find("h1", class_="entry-title").text
    content = page_soup.find("div", class_="post-content").text.strip()
    
    # Save data
    data["titles"].append(title)
    data["content"].append(content)
    data["images"].append(img)

# %% [markdown]
# ## Step 7: Save the Data
# Let's save our scraped data in a pandas DataFrame for easy viewing and analysis.

# %% id="ragUnpUMCNYl"
# Create a DataFrame
df = pd.DataFrame(data)
df

# %% [markdown]
# ## Congratulations! 🎉
# You've successfully:
# 1. Scraped a website using BeautifulSoup4
# 2. Extracted structured data from web pages
# 3. Created a dataset from web content

# %% [markdown]
# ## Next Steps
# - Try scraping a different website
# - Add more data fields (like dates, authors)
# - Save the data to a CSV file
# - Use the data for analysis or machine learning

# %% [markdown]
# ## Optional: Save to HuggingFace
# If you want to share your dataset, you can upload it to HuggingFace:

# %% id="IEfW3it9BdzA"
# !pip install datasets -q

# %% id="aAKgtv2LCf-j"
from datasets import Dataset

# Convert to HuggingFace dataset
ds = Dataset.from_pandas(df)

# To upload to HuggingFace, you'll need:
# 1. A HuggingFace account
# 2. A write token
# 3. A dataset name
# Uncomment and fill these to upload:
# HF_WRITE_TOKEN = ""  # Your HuggingFace write token
# HF_DATASET_REPO = "username/datasetname"  # Your dataset name
# ds.push_to_hub(HF_DATASET_REPO, token=HF_WRITE_TOKEN)
