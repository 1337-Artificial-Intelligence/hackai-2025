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
# # Scrape your first website using Beautifulsoup

# %% [markdown] id="_AO2lQ9-m1pS"
# with Beautiful Soup we can:
#
# * Parsing: It takes messy HTML or XML code and transforms it into a structured, easily searchable format.
# * Navigation: You can easily move through the parsed document, finding specific elements like tags, attributes, and text.
# * Searching: Beautiful Soup provides tools to find elements based on their tags, attributes, or content.
# * Extraction: Once you've located the data you want, Beautiful Soup helps you extract it cleanly.

# %% [markdown] id="c4ytLvGToTUT"
# * install beautifulsoup with pip

# %% id="NwvNYma2mYiK"
# install beautiful soup
# !pip install beautifulsoup4 -q

# %% [markdown] id="vXMg4bcCoajJ"
# * to scrap any web page we will need also **requests** to send any type of http (get,post,put,delete) request to our target.

# %% id="YUacjMzzn4o6"
from bs4 import BeautifulSoup
import requests #with requests we can to any http request (get,post,put,delete)

# %% [markdown] id="8-OPugpeqKGK"
# * in our tutorial we will try to **scrap 6 post news** from https://www.goud.ma, as in the images below ...

# %% [markdown] id="IivkXI3pELpm"
# <center>
#
#
# </center>

# %% [markdown] id="mUSSjdiOvNDj"
# * Before beginning our scraping first of all we need to analyze the HTML of our targeted website to extract which HTML object we will target.
#
# * as in the image below we will target the article object with the `´card´` class, then inside of it we will extract the href link to the content of the article, and we will repeat the same things with other articles.

# %% [markdown] id="WQTvqFbixv7d"

# %% [markdown] id="YZxQ3nsZt0h8"
# * to do what explained above, we need to follow the next steps:
# 1. send a GET request to our target website.
# 2. parse the response with bs4.
# 3. extract the target HTML object we want using its HTML tag (name) in our case `article` and its class `card` or id.

# %% colab={"base_uri": "https://localhost:8080/"} id="HoPOQznttz7g" outputId="e150e03f-5fc2-44bb-f876-1d0b582719fd"
# 1.1 send get request
target="https://www.goud.ma/topics/%d8%a7%d9%84%d8%b1%d8%a6%d9%8a%d8%b3%d9%8a%d8%a9/"
page=requests.get(target,headers={"User-Agent": "XY"})
if page.reason=="OK":
  print("[INFO] Request is Valide")
else:
  print("[INFO] Request is not Valide")

# %% colab={"base_uri": "https://localhost:8080/"} id="sj_3967Is0BX" outputId="294076fe-dffe-49ae-fecd-c5b488f096ed"
# 1.2 extract html from the page
page_html=page.text
print(page_html[:50])

# %% id="L3_MQbeqzJPC"
# 2. parse html with bs4
page_soup=BeautifulSoup(page_html,"html")

# %% colab={"base_uri": "https://localhost:8080/"} id="GHP8O2hOzo4F" outputId="4a430c2b-2861-48e6-b5b3-9015d3f755b6"
# 3. find article html object
articles=page_soup.find_all(name="article",class_="card")[:6] # 6 articles
articles

# %% [markdown] id="LObqA8y0FCIV"

# %% [markdown] id="AW2xCiaJ1pl7"
# * after finishing the first step, now we will move to extract the content of each article as in the image below.
# * to do this we need to extract the html object `a` with class `stretched-link` class from the articles above.
# * then extract `href` link

# %% colab={"base_uri": "https://localhost:8080/"} id="1rxW-0dHoPZ1" outputId="631f4205-d08f-4e93-82f0-fce6e57efa9b"
articles_links=[
    article.find("a",class_="stretched-link"). # 1
    get("href") # 2
    for article in articles]
articles_links

# %% [markdown] id="QEy6Nhj639yt"
# * After extracting each article link, now we will try to **extract helpful data from the article**, like `title`, `image`, and `content`.
# * we will follow the same steps as before.
#
# => let's do it with one article link

# %% id="YQIRlaVG3Slr"
# step 1
link=articles_links[0]
page0_html=requests.get(link,headers={"User-Agent": "XY"}).text
page0_soup=BeautifulSoup(page0_html,"html")

# %% id="UxcvZtYN6Ywb"
# step2
page0_img=page0_soup.find("img",class_="img-fluid wp-post-image").get("src")
page0_title=page0_soup.find("h1",class_="entry-title").text
page0_content=page0_soup.find("div",class_="post-content").text.strip()

# %% colab={"base_uri": "https://localhost:8080/"} id="Dlbg4hf06vju" outputId="bc6ca5cf-94c0-485f-b57f-f7b8f2afb59f"
print(f"img src:\n{page0_img}")
print(f"title:\n{page0_title}")
print(f"content:\n{page0_content}")

# %% id="895z_7DI_ajx"
# if you want to save images in your local
imgcontent=requests.get(page0_img).content
with open("image.jpg","wb") as i:
  i.write(imgcontent)

# %% colab={"base_uri": "https://localhost:8080/"} id="pbc44wRe-bB1" outputId="8c8b9932-a69f-43aa-8549-7f5a565d24ab"
# repeat the same with other 6 articles
from tqdm import tqdm
data={"titles":[],"content":[],"images":[]}
for link in tqdm(articles_links):
  # step 1
  pagei_html=requests.get(link,headers={"User-Agent": "XY"}).text
  pagei_soup=BeautifulSoup(pagei_html,"html")
  # step2
  pagei_img=pagei_soup.find("img",class_="img-fluid wp-post-image").get("src")
  pagei_title=pagei_soup.find("h1",class_="entry-title").text
  pagei_content=pagei_soup.find("div",class_="post-content").text.strip()
  # save
  data["titles"].append(pagei_title)
  data["content"].append(pagei_content)
  data["images"].append(pagei_img)

# %% colab={"base_uri": "https://localhost:8080/", "height": 237} id="ragUnpUMCNYl" outputId="2f63216b-d02c-4a73-ca45-eabcd87cfadf"
import pandas as pd
df=pd.DataFrame(data)
df

# %% [markdown] id="KseCZTjBBrqM"
# * The step of scraping is finished ... ✅
#
# ## Push scraped dataset to HuggingFace
# * Now we will push our scrapped dataset to huggingface 🤗 as the last step

# %% colab={"base_uri": "https://localhost:8080/"} id="IEfW3it9BdzA" outputId="abdb93ac-6a9b-46e8-8e64-f63aaaf99925"
# ! pip install datasets -q # install datasets by hf

# %% colab={"base_uri": "https://localhost:8080/", "height": 133, "referenced_widgets": ["a51a7a1f8f124c67831bac917aa9f521", "bd8aa96e01934d8baef8a468995f065b", "53bda4c7de514fdaafb41b9375ea784d", "d7ae3bdfa3d54ea3befee67774f49e41", "2172e01bc5a744bfa395c0675ec4ee20", "76a0d523ec5e4678bc8b5722434145d3", "1b248e1599074b05ab16ba7c579f3008", "866080c1452849ff9d249d3301a362c5", "e6700f60a3fc45a7836cef27030ec294", "6b7d8089bda64f57b9d63d77ed7e1838", "9b81d21461d54381aa58c604bc7690a4", "e39a9053406f4cd391787b480a7a1e53", "49e146d67f8c4592a8e9307224ca31b6", "6b430a7ff3f2416b821a2a2f131ec4a0", "6112c86863004b13ad37e0fd8e683ae8", "fcc0f933ce3343a3ac7f19416163891e", "f8a63efd99174cc994a9db8dc636b78c", "834795d74baf427d90956e9540bcc46c", "c92b80cfe31d464186cf75d7219a7382", "622bf4ed92b249bbb4dc244cba13aedd", "6801191e611f4d9586dc0c6511639b6f", "bb30d1c3c1b74f95b98672151d74fdf0"]} id="aAKgtv2LCf-j" outputId="052e45af-ac48-439d-960f-9bbcedd6ba20"
from datasets import Dataset
HF_WRITE_TOKEN="" # your hf write token
HF_DATASET_REPO="atlasia/good25" # username/datasetname
ds=Dataset.from_pandas(df)
ds.push_to_hub(HF_DATASET_REPO,token=HF_WRITE_TOKEN)

# %% id="jIRm__kiBncH"
