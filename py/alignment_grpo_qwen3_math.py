# %% [markdown]
# # 🚀 Fine-tuning Language Models with GRPO for Math Reasoning
# 
# This notebook demonstrates how to improve a language model's math reasoning abilities using Group Relative Policy Optimization (GRPO).
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/hackai-challenges/blob/main/py/alignment_grpo_qwen3_math.py)
# 
# ## 📚 What You'll Learn
# 
# 1. How to fine-tune a language model for better math reasoning
# 2. Understanding GRPO and its advantages over other methods
# 3. Implementing reward functions for math problem-solving
# 
# ## 🎯 Why GRPO?
# 
# GRPO (Group Relative Policy Optimization) is a powerful technique that helps language models learn better by:
# - Grouping similar problems together
# - Learning from relative performance within groups
# - Improving reasoning step by step
# 
# ## 🔧 Setup and Dependencies

# %% [markdown]
# ### 1. Install Required Libraries
# First, let's install the necessary packages:

# %%
!pip install -q transformers datasets trl torch sentence-transformers pypdf math_verify

# %% [markdown]
# ### 2. Import Libraries and Set Up Environment

# %%
import re
from typing import List

# Third-party libraries
import torch
import warnings
from datasets import load_dataset, Dataset
from pypdf import PdfReader
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from trl import GRPOConfig, GRPOTrainer

# Custom/project-specific libraries
from math_verify import LatexExtractionConfig, parse, verify
from unsloth import FastLanguageModel, is_bfloat16_supported

# %% [markdown]
# ### 3. Basic Configuration
# Let's set up our basic configuration for the model and training:

# %%
# Set device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Model configuration
MODEL = "unsloth/Qwen3-1.7B"  # Small, efficient model good for learning
max_seq_length = 2048         # Length for input/output
NEW_MODEL = "Qwen3_1.7B-GRPO-math-reasoning"

# Prompt template for consistent responses
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Show your step-by-step thinking process
</reasoning>
<answer>
Your final answer here
</answer>
"""

# Dataset for training
DATASET = "lighteval/MATH-Hard"  # Math problems dataset

# %% [markdown]
# ### 4. Load and Prepare the Dataset
# We'll use a dataset of math problems to train our model:

# %%
def get_math_questions(split="train") -> Dataset:
    """Load and prepare math problems dataset."""
    data = load_dataset(DATASET, 'default')[split]
    data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['problem']}
            ],
            'answer': x['solution'],
            'question': x['problem']
        }).remove_columns(['problem', 'solution','level','type'])
    return data

# Load training and test datasets
train_dataset = get_math_questions(split="train")
test_dataset = get_math_questions(split="test")

# %% [markdown]
# ### 5. Define Reward Functions
# These functions help the model learn what makes a good math solution:

# %%
def accuracy_reward(completions: List[dict], **kwargs) -> List[float]:
    """Reward function that checks if the answer matches the correct solution."""
    solutions = kwargs['answer']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        try:
            # Parse and verify the solution
            gold_parsed = parse(solution, extraction_mode="first_match", 
                              extraction_config=[LatexExtractionConfig()])
            answer_parsed = parse(content, extraction_mode="first_match", 
                                extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# %% [markdown]
# ### 6. Configure Training Parameters
# Set up the training configuration for GRPO:

# %%
# Training configuration
training_args = GRPOConfig(
    lr_scheduler_type="cosine",          # Smooth learning rate adjustment
    per_device_train_batch_size=1,       # Small batch size for memory efficiency
    gradient_accumulation_steps=1,       # Accumulate gradients for stability
    warmup_steps=5,                      # Quick warmup for faster learning
    max_steps=50,                        # Number of training steps
    learning_rate=2e-4,                  # Learning rate
    optim="adamw_8bit",                  # Memory-efficient optimizer
    max_grad_norm=0.1,                   # Prevent gradient explosion
    max_prompt_length=500,               # Maximum input length
    max_completion_length=1024,          # Maximum output length
    seed=3407,                           # For reproducibility
    output_dir="qwen3_1_7B_grpo_math"    # Save directory
)

# %% [markdown]
# ### 7. Train the Model
# Now we'll train our model using GRPO:

# %%
# Initialize the trainer
trainer = GRPOTrainer(
    model=model,                    # Our language model
    processing_class=tokenizer,     # Text processor
    reward_funcs=[accuracy_reward], # Reward function
    args=training_args,            # Training configuration
    train_dataset=train_dataset    # Training data
)

# Start training
trainer.train()

# %% [markdown]
# ## 🎉 Congratulations!
# You've successfully fine-tuned a language model using GRPO for better math reasoning!
# 
# ## 📝 Key Takeaways
# 1. GRPO helps models learn better by comparing performance within groups
# 2. Reward functions guide the model to produce better solutions
# 3. Step-by-step reasoning is crucial for math problem-solving
# 
# ## 🔍 Next Steps
# 1. Try different reward functions
# 2. Experiment with different model sizes
# 3. Test on more complex math problems
# 
# ## ⚠️ Note
# This is a simplified version for learning purposes. For production use, you would need:
# - More training steps
# - Better reward functions
# - Proper evaluation metrics
# - Larger models for better performance