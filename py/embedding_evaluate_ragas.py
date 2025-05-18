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

# %% [markdown] id="wa8ykQk92aLX"
# # Evaluation with RAGAS and Advanced Retrieval Methods Using LangChain
#
# In the following notebook we'll discuss a major component of LLM Ops:
#
# - Evaluation
#
# We're going to be leveraging the [RAGAS]() framework for our evaluations today as it's becoming a standard method of evaluating (at least directionally) RAG systems.
#
# We're also going to discuss a few more powerful Retrieval Systems that can potentially improve the quality of our generations!
#
# Let's start as we always do: Grabbing our dependencies!

# %% id="5BN13TZlSCv4"
# !pip install -U -q langchain openai ragas arxiv pymupdf chromadb wandb tiktoken

# %% colab={"base_uri": "https://localhost:8080/"} id="8Lhqp5rUThG-" outputId="c33f7eee-b819-40bd-dc75-ce90721a6a94"
import os
import openai
from getpass import getpass

openai.api_key = getpass("Please provide your OpenAI Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key

# %% [markdown] id="DV_BOewX8CW0"
# ### Data Collection
#
# We're going to be using papers from Arxiv as our context today.
#
# We can collect these documents rather straightforwardly with the `ArxivLoader` document loader from LangChain.
#
# Let's grab and load 5 documents.
#
# - [`ArxivLoader`](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.arxiv.ArxivLoader.html)

# %% colab={"base_uri": "https://localhost:8080/"} id="DTDNFXaBSO2j" outputId="3b24521d-5c6f-466b-d818-46ce68d359ee"
from langchain.document_loaders import ArxivLoader

base_docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5).load()
len(base_docs)

# %% colab={"base_uri": "https://localhost:8080/"} id="nNPAWPgNSyGP" outputId="b2f80fc8-792c-489a-b8d4-9f98678c679a"
for doc in base_docs:
  print(doc.metadata)

# %% [markdown] id="Z7ht6bJX9PAY"
# ### Creating an Index
#
# Let's use a naive index creation strategy of just using `RecursiveCharacterTextSplitter` on our documents and embedding each into our `VectorStore` using `OpenAIEmbeddings()`.
#
# - [`RecursiveCharacterTextSplitter()`](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
# - [`Chroma`](https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html?highlight=chroma#langchain.vectorstores.chroma.Chroma)
# - [`OpenAIEmbeddings()`](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html?highlight=openaiembeddings#langchain-embeddings-openai-openaiembeddings)

# %% id="xne8P5dQTUiR"
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250)

docs = text_splitter.split_documents(base_docs)

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# %% colab={"base_uri": "https://localhost:8080/"} id="cnRzYx4c_2mZ" outputId="59d9bdd8-0414-4e8b-c285-bf3a2760e26a"
len(docs)

# %% colab={"base_uri": "https://localhost:8080/"} id="WyUh8EVI_6TZ" outputId="643fca9d-77c0-4296-d953-ec62d6de8954"
print(max([len(chunk.page_content) for chunk in docs]))

# %% [markdown] id="0f9kNIUUTxdT"
# Let's convert our `Chroma` vectorstore into a retriever with the `.as_retriever()` method.

# %% id="bwbdftltT29h"
base_retriever = vectorstore.as_retriever(search_kwargs={"k" : 2})

# %% [markdown] id="DBPZQUt4UBPl"
# Now to give it a test!

# %% id="r0Pie4xqUCkW"
relevant_docs = base_retriever.get_relevant_documents("What is Retrieval Augmented Generation?")

# %% colab={"base_uri": "https://localhost:8080/"} id="Z_CiHTD0UKj7" outputId="fab040cf-971f-440a-a6aa-93873c8e152f"
len(relevant_docs)

# %% [markdown] id="D8MKsT6JTgCU"
# ## Creating a Retrieval Augmented Generation Prompt
#
# Now we can set up a prompt template that will be used to provide the LLM with the necessary contexts, user query, and instructions!

# %% id="ijSNkTAjTsep"
from langchain.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# %% [markdown] id="BYHnPaXl-cvJ"
# ### Setting Up our Basic QA Chain
#
# Now we can instantiate our basic RAG chain!
#
# We'll follow *exactly* the chain we made on Tuesday to keep things simple for now - if you need a refresher on what it looked like - check out last week's notebook!

# %% id="-TsjUWjbUfbW"
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

# %% [markdown] id="zO69de-F-oMD"
# Let's test it out!

# %% colab={"base_uri": "https://localhost:8080/"} id="2FS5NxC6UyU2" outputId="2520bf73-9e62-435c-a213-c26d0655a913"
question = "What is RAG?"

result = retrieval_augmented_qa_chain.invoke({"question" : question})

print(result)

# %% [markdown] id="wyazkAIu85KL"
# ### Ground Truth Dataset Creation Using GPT-3.5-turbo and GPT-4
#
# The next section might take you a long time to run, so the evaluation dataset is provided.
#
# The basic idea is that we can use LangChain to create questions based on our contexts, and then answer those questions.
#
# Let's look at how that works in the code!

# %% id="V24T_gpPatAO"
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

question_schema = ResponseSchema(
    name="question",
    description="a question about the context."
)

question_response_schemas = [
    question_schema,
]

# %% id="ebbmazGrdPap"
question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
format_instructions = question_output_parser.get_format_instructions()

# %% id="qorL4TPGXJQ7"
question_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

# %% id="oPqC1_MXdRuh"
from langchain.prompts import ChatPromptTemplate

qa_template = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=docs[0],
    format_instructions=format_instructions
)

question_generation_chain = bare_template | question_generation_llm

response = question_generation_chain.invoke({"content" : messages})
output_dict = question_output_parser.parse(response.content)

# %% colab={"base_uri": "https://localhost:8080/"} id="aKFw9kyZd7eB" outputId="bbcf9e15-58be-4899-f102-6cae59c45eb0"
for k, v in output_dict.items():
  print(k)
  print(v)

# %% id="dtASDdhLfd89"
# !pip install -q -U tqdm

# %% id="Zolpr3CYeEYm" colab={"base_uri": "https://localhost:8080/"} outputId="a7962cf2-4cdf-4c7a-b776-a0c2478042e1"
from tqdm import tqdm

qac_triples = []

for text in tqdm(docs[:10]):
  messages = prompt_template.format_messages(
      context=text,
      format_instructions=format_instructions
  )
  response = question_generation_chain.invoke({"content" : messages})
  try:
    output_dict = question_output_parser.parse(response.content)
  except Exception as e:
    continue
  output_dict["context"] = text
  qac_triples.append(output_dict)

# %% colab={"base_uri": "https://localhost:8080/"} id="mKBdQHK7Y2Vw" outputId="73c7d139-be2d-483f-9f70-b6c4aae91506"
qac_triples[5]

# %% id="vNB9Z2DrX2TC"
answer_generation_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

answer_schema = ResponseSchema(
    name="answer",
    description="an answer to the question"
)

answer_response_schemas = [
    answer_schema,
]

answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
format_instructions = answer_output_parser.get_format_instructions()

qa_template = """\
You are a University Professor creating a test for advanced students. For each question and context, create an answer.

answer: a answer about the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=qac_triples[0]["context"],
    question=qac_triples[0]["question"],
    format_instructions=format_instructions
)

answer_generation_chain = bare_template | answer_generation_llm

response = answer_generation_chain.invoke({"content" : messages})
output_dict = answer_output_parser.parse(response.content)

# %% colab={"base_uri": "https://localhost:8080/"} id="Rk-_lRR6fn5U" outputId="fb014a65-a56f-49be-8ecf-9ca5527aa803"
for k, v in output_dict.items():
  print(k)
  print(v)

# %% id="yCdH0e9rrAKd" colab={"base_uri": "https://localhost:8080/"} outputId="629d8791-dedb-47c7-b5a0-adaa26f142cd"
for triple in tqdm(qac_triples):
  messages = prompt_template.format_messages(
      context=triple["context"],
      question=triple["question"],
      format_instructions=format_instructions
  )
  response = answer_generation_chain.invoke({"content" : messages})
  try:
    output_dict = answer_output_parser.parse(response.content)
  except Exception as e:
    continue
  triple["answer"] = output_dict["answer"]

# %% id="rrHXgH9Qs1ep"
# !pip install -q -U datasets

# %% id="uAvGGTyXsoHQ"
import pandas as pd
from datasets import Dataset

ground_truth_qac_set = pd.DataFrame(qac_triples)
ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})


eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

# %% colab={"base_uri": "https://localhost:8080/"} id="q_FHUnAPVseB" outputId="1a907389-e62b-4686-b3d7-e8707acfbd47"
eval_dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="W8CaCUeBVu4l" outputId="72cbf3c8-c698-4682-821a-8566f36f6adb"
eval_dataset[0]

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["c4bc64d47f2e4a239cf7156e7812887d", "b4a71cf676584200b46922fb976d1d50", "65f75f3d72cd415faffb32999893738d", "5367cbe7ca4d43b089ff0d5d7cd17d3c", "31390facfbeb455fa83067fc23d30718", "22fc25617c664e4aa02b7457832c1bef", "b556054df18d47aa8cb58ac482ca31bb", "0fff4a6672c848a38945e19101c46168", "de662cb67d054f08a15d191923d40369", "b53ebcf672a244978391cae559db5bf2", "f2de6a2c1f4b4a2fa5196028c0da0754"]} id="Nhp5X4M8zqrm" outputId="7e3c36df-12a3-4ea7-a772-dc1e0bc61568"
eval_dataset.to_csv("groundtruth_eval_dataset.csv")

# %% [markdown] id="7Al5cagr-rvL"
# ### Evaluating RAG Pipelines
#
# If you skipped ahead and need to load the `.csv` directly - uncomment the code below.
#
# If you're using Colab to do this notebook - please ensure you add it to your session files.

# %% id="QJhes58R66-P"
# from datasets import Dataset
# eval_dataset = Dataset.from_csv("groundtruth_eval_dataset.csv")

# %% colab={"base_uri": "https://localhost:8080/"} id="5fAD8c_kthWc" outputId="e722498d-3179-4cf1-e206-29a27163ace5"
eval_dataset

# %% [markdown] id="IqFYbjLK-6X7"
# ### Evaluation Using RAGAS
#
# Now we can evaluate using RAGAS!
#
# The set-up is fairly straightforward - we simply need to create a dataset with our generated answers and our contexts, and then evaluate using the framework.

# %% id="1eBoHaf5t4w8"
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate

def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["response"].content,
         "contexts" : [context.page_content for context in answer["context"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result


# %% [markdown] id="J4c4Jd8G_lXY"
# Lets create our dataset first:

# %% colab={"base_uri": "https://localhost:8080/"} id="R7oXgcjkuopO" outputId="6db1a904-90a2-4e47-85da-a442ebdc56b1"
from tqdm import tqdm
import pandas as pd

basic_qa_ragas_dataset = create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="nfzaFWEMZ5l_" outputId="60ca3e05-209a-4375-f24c-a23ad06e525e"
basic_qa_ragas_dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="8Vv1NsRGZ6m5" outputId="2ef4fbd9-011e-42b7-88d9-11c79974d87e"
basic_qa_ragas_dataset[0]

# %% [markdown] id="Obbgw3im_n01"
# Save it for later:

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["86761a36bdb04fed9f1dae6e74da54ce", "e079c0eb3d9c4ae3a40491f059a864f7", "d4bde97f3895450782030cad7808ea59", "1d17b8847ab34a1d8fc79430e8b64a12", "fb953d8d116e4cc389dbc23a0055873f", "bf27f78edb15477492fb2d5dd8dc5137", "ee96a569b8a54944bcac1b1bc164e53e", "e556e254882340168833ecf1a7153a90", "c077837b8b044c2fb14dcd8fbc44a37b", "e5c8b5933c754a1192456f43682276c5", "84abf77081a04bb686f794b49e04761d"]} id="6FG8x4i3yZ2B" outputId="52eb909f-69be-4c80-d9bc-77c2842bf14d"
basic_qa_ragas_dataset.to_csv("basic_qa_ragas_dataset.csv")

# %% [markdown] id="A5I_d_RT_pFr"
# And finally - evaluate how it did!

# %% colab={"base_uri": "https://localhost:8080/"} id="ywp3Rwavy9pc" outputId="7a3948ce-b743-4d95-a581-17e035215f3f"
basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="m4oYnKTn15gY" outputId="ba48e4d2-5559-4748-8bc6-0101074a5c0e"
basic_qa_result


# %% [markdown] id="SwhBxlxYAdno"
# ### Testing Other Retrievers
#
# Now we can test our how changing our Retriever impacts our RAGAS evaluation!
#
# We'll build this simple qa_chain factory to create standardized qa_chains where the only different component will be the retriever.

# %% id="qnfy4VNkzZi2"
def create_qa_chain(retriever):
  primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  created_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | primary_qa_llm,
         "context": itemgetter("context"),
      }
  )

  return created_qa_chain


# %% [markdown] id="vOPp4Xq7AvEx"
# #### Parent Document Retriever
#
# One of the easier ways we can imagine improving a retriever is to embed our documents into small chunks, and then retrieve a significant amount of additional context that "surrounds" the found context.
#
# You can read more about this method [here](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)!
#
# The basic outline of this retrieval method is as follows:
#
# 1. Obtain User Question
# 2. Retrieve child documents using Dense Vector Retrieval
# 3. Merge the child documents based on their parents. If they have the same parents - they become merged.
# 4. Replace the child documents with their respective parent documents from an in-memory-store.
# 5. Use the parent documents to augment generation.

# %% id="67I6QJAJ0Un7"
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings())

store = InMemoryStore()

# %% id="zfk5RYUt00Pw"
parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# %% id="68c1t4o104AK"
parent_document_retriever.add_documents(base_docs)

# %% [markdown] id="MTH0MDolBndm"
# Let's create, test, and then evaluate our new chain!

# %% id="KMjLfqOC09Iw"
parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="Rv8bAHPN1H4P" outputId="faa6bf43-1604-4468-9faf-bbefd8e48281"
parent_document_retriever_qa_chain.invoke({"question" : "What is RAG?"})["response"].content

# %% colab={"base_uri": "https://localhost:8080/"} id="OQJRIQmU1WTw" outputId="295a9011-684d-4c38-e409-867022603608"
pdr_qa_ragas_dataset = create_ragas_dataset(parent_document_retriever_qa_chain, eval_dataset)

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["bf7f045bdbe24360ad2aa2f4c8f02e79", "761e3c6035bf49429b3035145451d2df", "ffb7c97e7af648aaa13a43427154140e", "95d92c5c74e845779337eb727c2bbfc0", "6e7fb9a1d1454fcd9bcf5b5f748fb975", "207785da1f404d6ea0e8c63655a7aa51", "71237d176b2c4138a5e0346d10482257", "ed457547dc154f6bbce4ec970bb09c76", "c9f479f81119450bb451d5830361467c", "56098a347ea94ea3b10bd8d2ec0d4288", "7940d9e3f5fa4592b58dec3fdb55595a"]} id="d9vfKnCL1jtB" outputId="1d7421a8-b564-4da3-cd74-d1eb3f8311f7"
pdr_qa_ragas_dataset.to_csv("pdr_qa_ragas_dataset.csv")

# %% colab={"base_uri": "https://localhost:8080/"} id="qfB1H9S_1mW3" outputId="426d7b1b-2b0c-40d3-e7d7-39da43363f06"
pdr_qa_result = evaluate_ragas_dataset(pdr_qa_ragas_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="5nFyYCdL2Nco" outputId="bdde7173-c649-40bc-cbc3-e38a70c9f50a"
pdr_qa_result

# %% [markdown] id="JaNk6o7_BqX8"
# #### Ensemble Retrieval
#
# Next let's look at ensemble retrieval!
#
# You can read more about this [here](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble)!
#
# The basic idea is as follows:
#
# 1. Obtain User Question
# 2. Hit the Retriever Pair
#     - Retrieve Documents with BM25 Sparse Vector Retrieval
#     - Retrieve Documents with Dense Vector Retrieval Method
# 3. Collect and "fuse" the retrieved docs based on their weighting using the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) algorithm into a single ranked list.
# 4. Use those documents to augment our generation.
#
# Ensure your `weights` list - the relative weighting of each retriever - sums to 1!

# %% id="zz7dl1GD5-L-"
# !pip install -q -U rank_bm25

# %% id="Vs8wxT9b5pRA"
from langchain.retrievers import BM25Retriever, EnsembleRetriever

text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=75)
docs = text_splitter.split_documents(base_docs)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.75, 0.25])

# %% id="cv69YDpF6PrJ"
ensemble_retriever_qa_chain = create_qa_chain(ensemble_retriever)

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="6lSszzrf6UmP" outputId="8a5893d5-4095-42e5-aecf-66d849512321"
ensemble_retriever_qa_chain.invoke({"question" : "What is RAG?"})["response"].content

# %% colab={"base_uri": "https://localhost:8080/"} id="abUgTGDT6UrV" outputId="749ae6db-75b9-48a7-e743-a8aecdcbd802"
ensemble_qa_ragas_dataset = create_ragas_dataset(ensemble_retriever_qa_chain, eval_dataset)

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["632135599e39470aac1a3bb3d3de0ca4", "567176b50d074a1c9d384a8df8c3ff4c", "6bfa39f2c84e4f08b5ffb9a416398824", "911b40169865413b98643789150e5495", "2add9d30edd84152bb6d7bfa4ed2d910", "f34ffd11b98045b9bae326dd48a54896", "e3e19fc9963c4bf39293bda7c2030d5a", "c15c27b0523a429e9d77f439d47ada90", "256a0dbb2f104d928e181d2a882a9867", "68790cfe4ea14fc4a63275f3e99c468f", "e7df092ac205443e8baa3646ff1eae5b"]} id="bGHipYsf7phk" outputId="a70b0d3e-870f-49b8-a16a-5b7d4623c33f"
ensemble_qa_ragas_dataset.to_csv("ensemble_qa_ragas_dataset.csv")

# %% colab={"base_uri": "https://localhost:8080/"} id="Ozo0jkvx7r1d" outputId="f5770c52-4614-4172-8834-d48bd4005218"
ensemble_qa_result = evaluate_ragas_dataset(ensemble_qa_ragas_dataset)

# %% colab={"base_uri": "https://localhost:8080/"} id="hvabdcbh793a" outputId="56daa44b-841b-4924-9242-77c2bc93f86e"
ensemble_qa_result

# %% [markdown] id="O4vXVWqiCcSI"
# ### Conclusion
#
# Observe your results in a table!

# %% id="PmBoVQ5hV3Kc" outputId="12196187-5cbf-40b2-f35f-ab45616e71a4" colab={"base_uri": "https://localhost:8080/"}
basic_qa_result

# %% colab={"base_uri": "https://localhost:8080/"} id="Ax1JLXKxUsXF" outputId="b83ba792-7e66-44ff-b219-0f3890a5fe8b"
pdr_qa_result

# %% colab={"base_uri": "https://localhost:8080/"} id="drxLlO3zUpyQ" outputId="3b595607-2fa5-4590-d2e6-9707aa7bb283"
ensemble_qa_result

# %% [markdown] id="r6YPGf-2l0Kx"
# We can also zoom in on each result and find specific information about each of the questions and answers.

# %% id="SkxLtk43ikka"
ensemble_qa_result_df = ensemble_qa_result.to_pandas()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3JZMhUg3jSvE" outputId="95d1cf44-88ee-4056-b698-64237b119fe0"
ensemble_qa_result_df


# %% [markdown] id="0jXR7ckel-v8"
# We'll also look at combining the results and looking at them in a single table so we can make inferences about them!

# %% id="BE5KKE_JkcD3"
def create_df_dict(pipeline_name, pipeline_items):
  df_dict = {"name" : pipeline_name}
  for name, score in pipeline_items:
    df_dict[name] = score
  return df_dict


# %% id="L1mPqYdqk4iQ"
basic_rag_df_dict = create_df_dict("basic_rag", basic_qa_result.items())

# %% id="ntJPzwy9lI46"
pdr_rag_df_dict = create_df_dict("pdr_rag", pdr_qa_result.items())

# %% id="R0fkbIQElPza"
ensemble_rag_df_dict = create_df_dict("ensemble_rag", ensemble_qa_result.items())

# %% id="Bc4T1E83lVbE"
results_df = pd.DataFrame([basic_rag_df_dict, pdr_rag_df_dict, ensemble_rag_df_dict])

# %% colab={"base_uri": "https://localhost:8080/", "height": 163} id="cv_9wNYGlibg" outputId="6580c17c-543a-4d54-8577-104c0173f368"
results_df.sort_values("answer_correctness", ascending=False)

# %% [markdown] id="YPocfrNFiYWi"
# ### ❓QUESTION❓
#
# What conclusions can you draw about the above results?
#
# Describe in your own words what the metrics are expressing.

# %% id="fbhz-vD4JPtN"
retrieval_augmented_qa_chain = (
    RunnableParallel({
        'context': itemgetter('question') | base_retriever,
        'question': RunnablePassthrough()
    }) | {
        'response': prompt | primary_qa_llm | parser,
        'context': itemgetter('context')
    }
)
