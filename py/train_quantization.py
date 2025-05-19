# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/train_quantization.ipynb)

# %% [markdown]
# # Quantization Challenge: Making AI Models Smaller and Faster
#
# In this notebook, you'll learn how to make AI models smaller and faster using quantization. This is super important for running AI on phones and other small devices!
#
# ## What you'll learn:
# 1. What quantization is and why it's useful
# 2. How to make models smaller using different number formats
# 3. How to use quantization in real AI models
#
# Let's get started!

# %% [markdown]
# ## 1. What is Quantization?
#
# Quantization is like compressing a photo to make it smaller. Instead of using big numbers (32 bits), we use smaller numbers (8 bits or 4 bits) to store our AI model.
#
# ### Why is this useful?
# - **Smaller models**: Takes less space on your phone
# - **Faster**: Runs quicker on your device
# - **Less power**: Uses less battery
# - **Works on more devices**: Can run on phones and small computers
#
# ### Key Terms (Don't worry, we'll explain these as we go!):
# - **Bits**: The smallest unit of computer data (like a light switch: on/off)
# - **Float32**: The normal way computers store numbers (32 bits)
# - **Int8**: A smaller way to store numbers (8 bits)
# - **Scale**: A number that helps us convert between big and small numbers

# %% [markdown]
# ## 2. How Computers Store Numbers
#
# Let's look at how computers store different types of numbers:

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Setup for displaying figures
plt.style.use('ggplot')

# %% [markdown]
# ### 2.1 Integer Numbers
#
# Integers are whole numbers (like 1, 2, 3). They can be:
# - **Unsigned**: Only positive numbers (0 to 255 for 8 bits)
# - **Signed**: Both positive and negative numbers (-128 to 127 for 8 bits)

# %%
# Show different integer ranges
int_types = {
    'int8': (np.iinfo(np.int8).min, np.iinfo(np.int8).max, 8),
    'uint8': (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max, 8),
    'int16': (np.iinfo(np.int16).min, np.iinfo(np.int16).max, 16),
    'uint16': (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max, 16)
}

for dtype, (min_val, max_val, bits) in int_types.items():
    print(f"{dtype}: Range [{min_val}, {max_val}], Bits: {bits}, Values: {2**bits}")

# %% [markdown]
# ### 2.2 Floating-Point Numbers
#
# Floating-point numbers can store decimals (like 3.14). They use:
# - **Sign bit**: Is the number positive or negative?
# - **Exponent bits**: How big is the number?
# - **Mantissa bits**: What are the decimal places?

# %%
# Show floating-point ranges
float_types = {
    'float16': (np.finfo(np.float16).min, np.finfo(np.float16).max, 16),
    'float32': (np.finfo(np.float32).min, np.finfo(np.float32).max, 32)
}

for dtype, (min_val, max_val, bits) in float_types.items():
    print(f"{dtype}: Range [{min_val}, {max_val}], Bits: {bits}")

# %% [markdown]
# ### 2.3 Visual Comparison
#
# Let's see how different number formats look:

# %%
def visualize_number_line(dtype, num_points=1000):
    if dtype == 'int8':
        values = np.linspace(-128, 127, num_points)
        representable = np.arange(-128, 128)
    elif dtype == 'uint8':
        values = np.linspace(0, 255, num_points)
        representable = np.arange(0, 256)
    elif dtype == 'float16':
        values = np.linspace(-10, 10, num_points)
        dense_near_zero = np.array([np.float16(x) for x in np.linspace(-0.1, 0.1, 40)])
        medium_range = np.array([np.float16(x) for x in np.linspace(-1, 1, 30)])
        sparse_range = np.array([np.float16(x) for x in np.linspace(-10, 10, 30)])
        representable = np.concatenate([dense_near_zero, medium_range, sparse_range])
        representable = np.unique(representable)

    plt.figure(figsize=(10, 2))
    
    if dtype == 'float16':
        plt.scatter(representable, np.zeros_like(representable), color='blue', s=20,
                   label=f'Representable {dtype} values')
        point_density = np.ones_like(representable) * 0.1
        plt.scatter(representable, point_density, color='red', s=5, alpha=0.5,
                   label='Density visualization')
    else:
        plt.scatter(representable, np.zeros_like(representable), color='blue', s=20,
                   label=f'Representable {dtype} values')

    plt.xlim([min(values), max(values)])
    plt.title(f'Distribution of representable {dtype} values')
    plt.yticks([])
    plt.legend()
    plt.show()

# Visualize different number formats
visualize_number_line('int8')
visualize_number_line('uint8')
visualize_number_line('float16')

# %% [markdown]
# Notice how:
# - Integers are evenly spaced (like steps on a ladder)
# - Floating-point numbers are closer together near zero (like a zoom lens)

# %% [markdown]
# ## 3. Types of Quantization
#
# There are two main ways to quantize:
#
# 1. **Symmetric Quantization**: Like a mirror image around zero
#    - Simpler but less accurate
#    - Good for numbers that are balanced around zero
#
# 2. **Affine Quantization**: Like a sliding scale
#    - More accurate but more complex
#    - Good for numbers that are mostly positive or negative

# %% [markdown]
# ## 4. Let's Try Quantization!
#
# We'll create a simple quantizer that can make numbers smaller:

# %%
class SymmetricQuantizer:
    """A simple quantizer that makes numbers smaller"""

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmin = -(2**(num_bits-1) - 1)
        self.qmax = 2**(num_bits-1) - 1
        self.scale = None

    def get_scale(self, x):
        """Find the right scale to make numbers fit"""
        x_abs_max = torch.max(torch.abs(x))
        scale = x_abs_max / self.qmax
        scale = torch.max(scale, torch.tensor(1e-8))
        return scale

    def quantize(self, x):
        """Make numbers smaller"""
        self.scale = self.get_scale(x)
        x_q = torch.round(x / self.scale)
        return torch.clamp(x_q, self.qmin, self.qmax)

    def dequantize(self, x_q):
        """Make numbers bigger again"""
        if self.scale is None:
            raise ValueError("Scale is not set. Quantize first!")
        return x_q * self.scale

    def quantize_dequantize(self, x):
        """Make numbers smaller and bigger again (to see how much we lost)"""
        x_q = self.quantize(x)
        x_dq = self.dequantize(x_q)
        return x_dq

# %% [markdown]
# Let's test our quantizer with different bit sizes:

# %%
# Generate some random numbers
torch.manual_seed(42)
x = torch.randn(1000) * 5

# Test different bit sizes
bit_widths = [8, 4, 2]
plt.figure(figsize=(15, 5))

for i, bits in enumerate(bit_widths):
    quantizer = SymmetricQuantizer(num_bits=bits)
    x_dq = quantizer.quantize_dequantize(x)

    plt.subplot(1, 3, i+1)
    plt.scatter(x.numpy(), x_dq.numpy(), alpha=0.5, s=5)
    plt.plot([-15, 15], [-15, 15], 'k--', alpha=0.5)  # perfect line

    error = torch.abs(x - x_dq).mean().item()
    plt.title(f'{bits}-bit Quantization\nMAE: {error:.4f}')
    plt.xlabel('Original Value')
    plt.ylabel('Reconstructed Value')

    print(f"{bits}-bit: Scale = {quantizer.scale.item():.6f}, Mean Abs Error = {error:.6f}")

plt.tight_layout()
plt.show()

# %% [markdown]
# Notice how:
# - 8 bits: Almost perfect!
# - 4 bits: Some loss but still good
# - 2 bits: Not so good (only 4 values to work with!)

# %% [markdown]
# ## 5. Using Quantization in Real AI Models
#
# Let's see how to use quantization with a real AI model. We'll use a small language model:

# %%
# Install required packages
!pip3 install accelerate bitsandbytes

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set up 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load a small model with quantization
quantized_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B",
    torch_dtype="auto",
    quantization_config=quantization_config
)

# Try it out!
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# %% [markdown]
# ## 6. Challenge Time!
#
# Your challenge is to:
# 1. Try different bit sizes (8, 4, 2) with our quantizer
# 2. Compare the results and explain what you see
# 3. Try quantizing a different model or dataset
#
# Remember:
# - More bits = better quality but bigger size
# - Fewer bits = smaller size but lower quality
# - Choose the right balance for your needs!

# %% [markdown]
# ## 7. What's Next?
#
# - Try different quantization methods
# - Learn about quantization-aware training
# - Explore more advanced techniques
#
# Happy Hacking! 🚀
