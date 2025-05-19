# %% [markdown]
# # 🚀 Fine-tuning Language Models with GRPO for Math Reasoning
# 
# Welcome to this hands-on workshop on fine-tuning language models using Group Relative Policy Optimization (GRPO)! In this session, we'll learn how to improve a model's mathematical reasoning abilities through a technique called GRPO.
# 
# ## 🎯 What You'll Learn
# 
# 1. Understanding GRPO and why it's useful for math reasoning
# 2. Setting up a model for fine-tuning
# 3. Training the model with GRPO
# 4. Testing the improved model
# 
# ## ⏱️ Time Breakdown (1 Hour)
# - Introduction & Setup (15 minutes)
# - Model Training (30 minutes)
# - Testing & Discussion (15 minutes)
# 
# ## 🔗 Quick Links
# - [Open in Colab](https://colab.research.google.com/github/your-repo/alignment_grpo_qwen3_math.ipynb)
# - [Hugging Face Model](https://huggingface.co/Qwen/Qwen3-1.7B)
# 
# ---
# 
# ## 1. What is GRPO? 🤔
# 
# GRPO (Group Relative Policy Optimization) is a technique that helps language models get better at specific tasks by:
# - Learning from examples in groups
# - Getting feedback on their answers
# - Improving step by step
# 
# Think of it like learning math:
# - You solve problems
# - Your teacher checks your work
# - You learn from your mistakes
# - You get better over time
# 
# ![GRPO Learning Process](https://miro.medium.com/v2/resize:fit:1400/1*84PSf3d1-OGN10y_2H-XdQ.png)
# 
# ## 2. Setting Up Our Environment 🛠️
# 
# First, let's install the tools we need:

# %% [code]
!pip install -q transformers datasets trl torch sentence-transformers pypdf math_verify

# %% [markdown]
# ## 3. Loading Our Model 🎯
# 
# We'll use Qwen3-1.7B, a small but powerful model that's perfect for learning:
# - Small enough to run on free Colab
# - Good at understanding math
# - Fast to train

# %% [code]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% [markdown]
# ## 4. Preparing Our Math Problems 📚
# 
# We'll use a dataset of math problems to train our model. Each problem has:
# - A question
# - A step-by-step solution
# - The final answer

# %% [code]
from datasets import load_dataset

# Load math problems
dataset = load_dataset("lighteval/MATH-Hard", 'default')
print("Sample problem:", dataset['train'][0]['problem'])

# %% [markdown]
# ## 5. Training with GRPO 🚂
# 
# Now comes the exciting part! We'll train our model using GRPO to make it better at solving math problems.
# 
# The training process:
# 1. Model tries to solve a problem
# 2. We check if the answer is correct
# 3. Model learns from its mistakes
# 4. Repeat!

# %% [code]
from trl import GRPOConfig, GRPOTrainer

# Set up training
training_config = GRPOConfig(
    learning_rate=2e-4,
    max_steps=50,  # Quick training for demo
    per_device_train_batch_size=1
)

# Start training
trainer = GRPOTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset['train']
)

trainer.train()

# %% [markdown]
# ## 6. Testing Our Model 🧪
# 
# Let's see how well our model can solve math problems now!

# %% [code]
def solve_math_problem(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0])

# Try it out!
test_question = "If x + y = 10 and x - y = 4, what is x?"
print("Question:", test_question)
print("Model's answer:", solve_math_problem(test_question))

# %% [markdown]
# ## 🎉 Congratulations!
# 
# You've just:
# 1. Learned about GRPO
# 2. Set up a language model
# 3. Trained it to solve math problems
# 4. Tested its abilities
# 
# ## 📚 Further Learning
# - Try different math problems
# - Experiment with training settings
# - Learn more about GRPO in the [documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
# 
# ## ⚠️ Note
# This is a simplified version for learning. Real-world applications would need:
# - More training time
# - Better reward functions
# - More evaluation
# - Larger models