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

# %% [markdown] id="QonfOON8PuED"
# ### 🏆 VertexAI Training
# 📌 Description
#
# This notebook outlines the steps to run a custom training job on Google Cloud Vertex AI Training.

# %% [markdown] id="cyyGi5lxPvG-"
# ### ✅ Step 1: Authenticate & Set Up Your Google Cloud Environment
#

# %% id="PsI9ROLPPeST" colab={"base_uri": "https://localhost:8080/"} outputId="08aa2506-ad0a-45ee-8692-3eb8684c5d3a"
# 1. Log in to your Google Cloud account
# !gcloud auth login

# %% id="RtAXN8SzP_kO" colab={"base_uri": "https://localhost:8080/"} outputId="3312d955-b7e1-43c8-bb9b-213bc44cb489"
# Create a project if you haven't already done so
PROJECT_ID = "hackai-1337-2025-test"
PROJECT_NAME = "My Test Project"

# !gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME"

# Set it as your active project
# !gcloud config set project "$PROJECT_ID"

# %% id="CGrVhiV8TP6m"
# Link Billing Account

# Output Example:
# ACCOUNT_ID            NAME                OPEN  MASTER_ACCOUNT_ID
# 01A2B3-XXXXXX-YYYYYY  My Billing Account  True
# Your BILLING_ACCOUNT_ID is the value in the ACCOUNT_ID column (01A2B3-XXXXXX-YYYYYY)

# !gcloud billing accounts list

# %% id="vUpFizuzTBol" colab={"base_uri": "https://localhost:8080/"} outputId="a5d9e78c-93d2-4515-e33e-c6c91d7ec1cf"
BILLING_ACCOUNT_ID = "BILLING_ACCOUNT_ID_HERE"

# !gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"

# %% [markdown] id="mhPB7RCvSnw3"
# ### ✅ Step 2: Enable required APIs
#

# %% id="WEDru2-5SWoc"
# !gcloud services enable compute.googleapis.com # For Compute Engine API
# !gcloud services enable artifactregistry.googleapis.com # For Artifact Registry API
# !gcloud services enable aiplatform.googleapis.com # For Vertex AI API
# !gcloud services enable cloudbuild.googleapis.com # For Cloud Build API

# %% id="YI7TpNBbo6Vp" colab={"base_uri": "https://localhost:8080/"} outputId="5141e246-f1f4-4ac1-d208-a09164d76d61"
USER_EMAIL = "YOUR_GCP_EMAIL"

# !gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member="user:$USER_EMAIL" \
#     --role="roles/cloudbuild.builds.editor"

# %% [markdown] id="uWHy5UrZY-8X"
# ### ✅ Step 3: Convert Your Training Notebook to Python Script
# This step assumes that you already have a notebook containing the code for your model's training step. We will generate a .py file from your notebook.
# Upload your notebook and run the following command.

# %% id="JlOphN25YrMa" colab={"base_uri": "https://localhost:8080/"} outputId="d3071fc9-7472-49f3-b5c8-6024832f3212"
# !jupyter nbconvert notebook_name.ipynb --to python

# %% id="nq-gEVcYS-eI"
# Create a trainer directory and move the .py code to it
# !mkdir trainer
# # !mv notebook_name.py trainer/task.py
# !mv notebook_name.py trainer/task.py

# %% id="ro3sHMfqbpmm"
# Create your requirements.txt with your needed libraries needed for fintenuning

packages = """
transformers
datasets
"""

# !echo "$packages" > requirements.txt

# %% [markdown] id="NkCpF06_cOXT"
# ### ✅ Step 4: Create a Dockerfile

# %% id="axDCUPzNcA-T"
dockerfile_content = """
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

WORKDIR /

COPY trainer /trainer
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "trainer.task"]
"""

with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)

# %% [markdown] id="6kD8DpV3dZFr"
# ### ✅ Step 5: Build and Push Container

# %% colab={"base_uri": "https://localhost:8080/", "height": 313} id="izKbGNqkii0U" outputId="b911a5d7-7bab-4ae1-f66d-25d84494285a"
REPO_NAME = "hackai-docker-repo"

# !gcloud artifacts repositories create "$REPO_NAME" --repository-format=docker \
# --location=us-central1 --description="Docker repository"

# !gcloud auth configure-docker us-central1-docker.pkg.dev

IMAGE_URI=f"us-central1-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/my_image:latest"
IMAGE_URI

# %% id="VDeGrLjLfHCX"
# !mkdir -p build_context/trainer
# !cp Dockerfile requirements.txt build_context/
# !cp -r trainer/task.py build_context/trainer/

# %% colab={"base_uri": "https://localhost:8080/"} id="daqBmGrXiBdp" outputId="8586f353-7f2b-442d-ccfd-5856f9070982"
# !cd build_context
# !gcloud builds submit --tag="$IMAGE_URI" --project="$PROJECT_ID"

# %% [markdown] id="-NMcVVQopXgY"
# ### ✅ Step 6: Run the Training Job

# %% id="ySXDf4KBpnNg" colab={"base_uri": "https://localhost:8080/"} outputId="93e5997b-bdf0-4cc0-c4c2-1d97cadd429f"
# Create a bcuket name (mandatory for the job to run)
BUCKET_NAME="gs://hackai-training-bucket"
REGION = "us-central1"
# !gcloud storage buckets create $BUCKET_NAME --location=$REGION

# %% id="4hRs18DQnWeh"
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name='my-training-job',
    container_uri=IMAGE_URI,
    staging_bucket=BUCKET_NAME
)

job.run(
    replica_count=1,
    machine_type='n1-standard-8', # To be customized
    accelerator_type='NVIDIA_TESLA_V100', # # To be customized
    accelerator_count=1
)


# %% [markdown] id="Yat288ewqLW6"
# #### Waiting for the job to finish
#

# %% id="CauJGU0-qkCF"
